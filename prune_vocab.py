#!/usr/bin/env python
"""Shrink a model's token-embedding table to the vocabulary its data uses.

The multilingual encoders (e.g. sentence-transformers/paraphrase-multilingual-
MiniLM-L12-v2) spend the vast majority of their parameters on a ~250k-token
embedding matrix covering 100+ languages -- ~82% of the model. We only support a
handful of locales, so most of that table is dead weight. This tool keeps only
the tokens that actually appear when tokenizing the datasets (plus all special
and byte-fallback tokens), slices the embedding rows accordingly, rebuilds the
tokenizer over the reduced vocabulary, and writes a smaller base checkpoint.

Point training at the result:

    python prune_vocab.py \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --corpus processed_datasets/training-supported.txt \
                 processed_datasets/validation-supported.txt \
                 processed_datasets/testing-supported.txt \
        --out output-models/minilm-pruned
    # then: ... --model_name output-models/minilm-pruned

Notes / assumptions:
- Works with a *fast* Unigram tokenizer (XLM-R / SentencePiece family) where the
  backend vocab index equals the token id (verified for MiniLM-L12). It remaps
  every id reference (vocab, unk_id, added_tokens, post-processor specials).
- Every context_format is covered: each dataset row's text is tokenized raw, as
  the sep/symbol re-encodings, and as the three triple sections (aa/bb stripped),
  so the kept vocab is a superset of what any format feeds the model.
- Only the input embeddings shrink; layers/hidden size are unchanged. Combine
  with fewer layers and int8 quantization for further reductions.
"""

import argparse
import json
import re

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from dotraining import reformat_context, split_context, ignoreLineCount

# Byte-fallback tokens (<0x00>..<0xFF>): always kept so byte_fallback still works
# for characters unseen in the corpus.
_BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def iter_texts(paths):
  """Yield every string worth tokenizing from the dataset files.

  Dataset rows are CSV: filename, fieldname, index, text. We yield the text in
  all the forms a model might see it: raw (bb), the sep/symbol re-encodings, and
  the three triple sections. Lines that aren't CSV are treated as raw text.
  """
  for path in paths:
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        line = line.strip()
        if not line:
          continue
        parts = line.split(",", ignoreLineCount + 1)
        text = parts[ignoreLineCount + 1] if len(parts) > ignoreLineCount + 1 else line
        yield text                                   # bb (raw prefixes)
        yield reformat_context(text, "sep")
        yield reformat_context(text, "symbol")
        for section in split_context(text):          # triple sections
          yield section


def collect_used_ids(tokenizer, texts, min_count):
  """Token ids that occur at least `min_count` times across `texts`."""
  from collections import Counter
  counts = Counter()
  batch = []
  def flush():
    if not batch:
      return
    for ids in tokenizer(batch, add_special_tokens=True)["input_ids"]:
      counts.update(ids)
    batch.clear()
  for t in texts:
    batch.append(t)
    if len(batch) >= 1000:
      flush()
  flush()
  return {tok_id for tok_id, c in counts.items() if c >= min_count}


def remap_post_processor(pp, old2new):
  """Rewrite special-token id references inside a (possibly nested) post-processor."""
  if not pp:
    return
  if pp.get("type") == "Sequence":
    for sub in pp.get("processors", []):
      remap_post_processor(sub, old2new)
  for entry in (pp.get("special_tokens") or {}).values():
    entry["ids"] = [old2new[i] for i in entry["ids"]]


def build_pruned_tokenizer(tokenizer, kept, old2new):
  """Rebuild the fast tokenizer over the reduced, reindexed vocabulary."""
  from tokenizers import Tokenizer
  spec = json.loads(tokenizer.backend_tokenizer.to_str())
  model = spec["model"]
  model["vocab"] = [model["vocab"][i] for i in kept]
  if model.get("unk_id") is not None:
    model["unk_id"] = old2new[model["unk_id"]]
  for added in spec.get("added_tokens", []):
    added["id"] = old2new[added["id"]]
  remap_post_processor(spec.get("post_processor"), old2new)

  new_backend = Tokenizer.from_str(json.dumps(spec))
  # Carry over the special-token strings so the wrapper resolves them by name.
  specials = {k: v for k, v in {
      "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token,
      "unk_token": tokenizer.unk_token, "sep_token": tokenizer.sep_token,
      "cls_token": tokenizer.cls_token, "pad_token": tokenizer.pad_token,
      "mask_token": tokenizer.mask_token,
  }.items() if v is not None}
  return PreTrainedTokenizerFast(
      tokenizer_object=new_backend,
      model_max_length=tokenizer.model_max_length,
      **specials,
  )


def remap_config_ids(config, old2new):
  """Update token-id fields on the model config to the new ids."""
  for attr in ("pad_token_id", "bos_token_id", "eos_token_id",
               "unk_token_id", "sep_token_id", "cls_token_id", "mask_token_id"):
    val = getattr(config, attr, None)
    if val is not None and val in old2new:
      setattr(config, attr, old2new[val])


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help="Base model name or path to prune.")
  ap.add_argument("--corpus", required=True, nargs="+",
                  help="Dataset file(s) whose tokens to keep (pass train+val+test).")
  ap.add_argument("--out", required=True, help="Directory to write the pruned model + tokenizer.")
  ap.add_argument("--min-count", type=int, default=1,
                  help="Keep tokens occurring at least this many times (default 1 = all seen).")
  args = ap.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModel.from_pretrained(args.model)
  if not tokenizer.is_fast:
    raise SystemExit("This tool requires a fast tokenizer.")

  vocab = json.loads(tokenizer.backend_tokenizer.to_str())["model"]["vocab"]
  old_size = len(vocab)

  used = collect_used_ids(tokenizer, iter_texts(args.corpus), args.min_count)
  # Always keep: special tokens, added tokens, and byte-fallback tokens.
  always = set(tokenizer.all_special_ids)
  always |= set(tokenizer.added_tokens_decoder.keys())
  always |= {i for i, (piece, _score) in enumerate(vocab) if _BYTE_TOKEN.match(piece)}
  kept = sorted((used | always) & set(range(old_size)))
  old2new = {old: new for new, old in enumerate(kept)}

  print(f"vocab: {old_size} -> {len(kept)} kept "
        f"({len(used)} used in corpus, {len(always)} always-kept specials/bytes)")

  # Slice the embedding rows and re-index the model.
  emb = model.get_input_embeddings()
  index = torch.tensor(kept)
  pad = emb.padding_idx
  new_emb = torch.nn.Embedding(
      len(kept), emb.embedding_dim,
      padding_idx=old2new[pad] if pad is not None and pad in old2new else None)
  with torch.no_grad():
    new_emb.weight.copy_(emb.weight.data[index])
  model.set_input_embeddings(new_emb)
  model.config.vocab_size = len(kept)
  remap_config_ids(model.config, old2new)

  new_tokenizer = build_pruned_tokenizer(tokenizer, kept, old2new)

  total = sum(p.numel() for p in model.parameters())
  print(f"pruned model params: {total/1e6:.1f}M  (~{total/1e6:.0f}MB int8)")

  model.save_pretrained(args.out)
  new_tokenizer.save_pretrained(args.out)
  print(f"wrote pruned model + tokenizer -> {args.out}")


if __name__ == "__main__":
  main()
