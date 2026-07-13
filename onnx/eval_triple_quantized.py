#!/usr/bin/env python
"""Evaluate a split + quantized triple model on the test set (total/close accuracy).

Runs the ONNX encoder (any quantized variant) + the fp32 fusion head over a test
file exactly as the deployed engine would, and reports the same total/close
accuracy metrics as dotraining.evaluate_model -- so you can confirm quantization
holds accuracy before shipping. Compares several encoder variants in one run.

    uv run --with onnxruntime python onnx/eval_triple_quantized.py \
        --split-dir quantization/autofill-tiny-supported-argo-autofillflow-lqnwd \
        --test-file testing-supported.txt \
        --variants model.onnx model_fp16.onnx model_quantized.onnx model_uint8.onnx
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import (  # noqa: E402
    fieldTypesReversedDict, fieldNamesCloseDict, split_context, ignoreLineCount)


def read_rows(path):
  """(expected_label_value, current, previous, next) per test row."""
  rows = []
  with open(path, encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if not line:
        continue
      parts = line.split(",", ignoreLineCount + 1)
      if len(parts) <= ignoreLineCount + 1:
        continue
      rows.append((int(parts[ignoreLineCount]), *split_context(parts[ignoreLineCount + 1])))
  return rows


def pooled(session, in_names, tok, texts, batch):
  """Encode texts through the ONNX encoder, attention-mask mean-pool -> [N, H]."""
  out = []
  for i in range(0, len(texts), batch):
    enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
              return_tensors="np")
    feeds = {"input_ids": enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)}
    if "token_type_ids" in in_names:
      feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
    lhs = session.run(None, feeds)[0]                       # [b, T, H]
    m = enc["attention_mask"][..., None].astype(np.float32)
    out.append((lhs * m).sum(1) / np.clip(m.sum(1), 1e-9, None))
  return np.concatenate(out, axis=0)


def evaluate(onnx_path, rows, tok, head, hj, batch):
  sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  in_names = [i.name for i in sess.get_inputs()]
  cur = torch.from_numpy(pooled(sess, in_names, tok, [r[1] for r in rows], batch))
  prev = torch.from_numpy(pooled(sess, in_names, tok, [r[2] for r in rows], batch))
  nxt = torch.from_numpy(pooled(sess, in_names, tok, [r[3] for r in rows], batch))
  with torch.no_grad():
    if "proj.weight" in head:                              # optional shared projection
      cur = F.linear(cur, head["proj.weight"], head["proj.bias"])
      prev = F.linear(prev, head["proj.weight"], head["proj.bias"])
      nxt = F.linear(nxt, head["proj.weight"], head["proj.bias"])
    sections = [cur, prev, nxt]
    if hj.get("head_interactions"):                        # neighbor difference features
      sections += [cur - prev, cur - nxt]
    fused = torch.cat(sections, dim=-1)
    h = F.gelu(F.linear(fused, head["fusion.weight"], head["fusion.bias"]))
    logits = F.linear(h, head["classifier.weight"], head["classifier.bias"])
  pred = logits.argmax(-1).tolist()

  correct = close = 0
  for (expected_val, *_), idx in zip(rows, pred):
    p = fieldTypesReversedDict.get(idx)                     # neuron idx -> label
    e = fieldTypesReversedDict.get(expected_val)
    if p is not None and p == e:
      correct += 1
      close += 1
    elif p in fieldNamesCloseDict and e in fieldNamesCloseDict[p]:
      close += 1
  n = len(rows)
  return correct / n, close / n


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--split-dir", required=True, help="Dir with encoder/ and head/ (from export_triple.py).")
  ap.add_argument("--test-file", required=True, help="Test dataset file.")
  ap.add_argument("--variants", nargs="+",
                  default=["model.onnx", "model_fp16.onnx", "model_quantized.onnx", "model_uint8.onnx"],
                  help="Encoder ONNX filenames under encoder/onnx/ to evaluate.")
  ap.add_argument("--batch-size", type=int, default=64)
  args = ap.parse_args()

  enc_dir = os.path.join(args.split_dir, "encoder")
  onnx_dir = os.path.join(enc_dir, "onnx")
  tok = AutoTokenizer.from_pretrained(enc_dir)
  head = load_file(os.path.join(args.split_dir, "head", "head.safetensors"))
  with open(os.path.join(args.split_dir, "head", "head.json"), encoding="utf-8") as fh:
    hj = json.load(fh)
  rows = read_rows(args.test_file)
  print(f"{len(rows)} test rows | variants: {args.variants}\n")
  print(f'{"variant":24s} {"size(MB)":9s} {"total":8s} {"close":8s}')
  print("-" * 52)
  base = None
  for v in args.variants:
    p = os.path.join(onnx_dir, v)
    if not os.path.exists(p):
      print(f"{v:24s}  (missing)")
      continue
    total, close = evaluate(p, rows, tok, head, hj, args.batch_size)
    if base is None:
      base = total
    mb = os.path.getsize(p) / 1e6
    delta = "" if v == args.variants[0] else f"  (Δ {(total - base) * 100:+.2f} pt)"
    print(f"{v:24s} {mb:8.1f}  {total:.4f}   {close:.4f}{delta}")


if __name__ == "__main__":
  main()
