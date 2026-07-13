#!/usr/bin/env python
"""Split a trained triple-encoder model into deployable parts.

The triple model (context_format='triple', TripleEncoderForSequenceClassification)
is not a standard HF classifier, so it can't be exported whole. But it decomposes
exactly the way the batched inference engine already runs it:

    1. a SHARED ENCODER applied once per field text  ->  pooled embedding (H)
    2. a small FUSION HEAD applied per field:
         concat[e_current, e_previous, e_next] (3H)
           -> Linear(3H->H) -> GELU -> Linear(H->num_labels) -> logits

This tool writes:
  <out>/encoder/     a plain AutoModel (+ tokenizer) -- feed to export_onnx.py
                     (--task feature-extraction) and the normal quantize pipeline.
  <out>/head/        head.safetensors (fusion.* / classifier.* weights),
                     head.json (dims, id2label, pooling/activation contract),
                     and head.onnx if torch.onnx export is available.

Example:
    uv run python export_triple.py \
        --model-dir outputs/autofill-tiny-supportedpruned-vocab-...-jzfz2 \
        --output    quantization/triple-jzfz2

Then export the encoder to ONNX with the existing tool:
    python export_onnx.py --model-dir quantization/triple-jzfz2/encoder \
        --output quantization/triple-jzfz2/encoder/onnx --task feature-extraction

Inference contract (what the engine implements):
  pooled(text) = attention-mask-weighted MEAN over the encoder's last_hidden_state
                 tokens (exclude padding).
  For field i:  logits = classifier( gelu( fusion( concat[pooled_i, pooled_{i-1},
                pooled_{i+1}] ) ) ); label = id2label[argmax(logits)].
  GELU is the exact (erf) form (torch nn.GELU default), NOT the tanh approximation.
  Concatenation order is exactly [current, previous, next].
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

# Make the repo root importable so the custom model class + its Auto-registration
# load (this script lives in onnx/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import TripleEncoderForSequenceClassification  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class _Head(nn.Module):
  """The fusion head as a standalone module: [B, 3H] -> [B, num_labels]."""

  def __init__(self, fusion, act, classifier):
    super().__init__()
    self.fusion = fusion
    self.act = act
    self.classifier = classifier

  def forward(self, fused):
    return self.classifier(self.act(self.fusion(fused)))


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model-dir", required=True, help="Directory of the saved triple model.")
  ap.add_argument("--output", required=True, help="Directory to write encoder/ and head/ into.")
  args = ap.parse_args()

  model = TripleEncoderForSequenceClassification.from_pretrained(args.model_dir)
  model.eval()
  d = model.classifier.in_features                       # head working dim (H, or proj dim)
  hidden = model.proj.in_features if model.proj is not None else d
  num_labels = model.classifier.out_features
  interactions = bool(getattr(model, "head_interactions", False))
  proj_dim = model.proj.out_features if model.proj is not None else 0
  id2label = {int(k): v for k, v in model.config.id2label.items()}

  enc_dir = os.path.join(args.output, "encoder")
  head_dir = os.path.join(args.output, "head")
  os.makedirs(enc_dir, exist_ok=True)
  os.makedirs(head_dir, exist_ok=True)

  # 1) Shared encoder as a standard model dir (+ tokenizer) for optimum export.
  model.encoder.save_pretrained(enc_dir)
  AutoTokenizer.from_pretrained(args.model_dir).save_pretrained(enc_dir)
  print(f"wrote encoder -> {enc_dir}")

  # 2) Head weights + contract. Handles the head variants: an optional shared
  # projection (proj.*) and neighbor-difference interaction features.
  from safetensors.torch import save_file
  tensors = {
      "fusion.weight": model.fusion.weight.data.contiguous(),
      "fusion.bias": model.fusion.bias.data.contiguous(),
      "classifier.weight": model.classifier.weight.data.contiguous(),
      "classifier.bias": model.classifier.bias.data.contiguous(),
  }
  if model.proj is not None:
    tensors["proj.weight"] = model.proj.weight.data.contiguous()
    tensors["proj.bias"] = model.proj.bias.data.contiguous()
  save_file(tensors, os.path.join(head_dir, "head.safetensors"))

  sections = ["current", "previous", "next"]
  if interactions:
    # differences taken in section_dim space (after the optional projection)
    sections += ["current-minus-previous", "current-minus-next"]
  pipeline = ["pool: masked_mean over encoder last_hidden_state -> H"]
  if proj_dim:
    pipeline.append(f"proj: shared Linear(H->{proj_dim}) applied to each pooled section")
  pipeline += [
      f"concat {len(sections)} sections -> {model.fusion.in_features}",
      f"fusion: Linear({model.fusion.in_features}->{d})", "gelu",
      f"classifier: Linear({d}->num_labels)"]
  head_json = {
      "hidden_size": hidden,
      "section_dim": d,
      "num_labels": num_labels,
      "head_interactions": interactions,
      "head_proj_dim": proj_dim,
      "input_order": sections,
      "input_dim": model.fusion.in_features,
      "pooling": "masked_mean",
      "activation": "gelu_erf",
      "pipeline": pipeline,
      "id2label": id2label,
      "base_model_name": model.config.base_model_name,
  }
  with open(os.path.join(head_dir, "head.json"), "w", encoding="utf-8") as fh:
    json.dump(head_json, fh, indent=2, ensure_ascii=False)
  print(f"wrote head weights + head.json -> {head_dir} "
        f"(interactions={interactions}, proj_dim={proj_dim}, fusion_in={model.fusion.in_features})")

  # 3) Optional: head.onnx (fusion+classifier over the assembled concat; the
  # engine pools/projects/assembles the concat first). Only if torch.onnx works.
  head = _Head(model.fusion, model.act, model.classifier).eval()
  dummy = torch.zeros(1, model.fusion.in_features)
  try:
    torch.onnx.export(
        head, dummy, os.path.join(head_dir, "head.onnx"),
        input_names=["fused"], output_names=["logits"],
        dynamic_axes={"fused": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)
    print(f"wrote head.onnx -> {head_dir}")
  except Exception as exc:  # onnx/onnxscript may be absent in this venv
    print(f"(skipped head.onnx: {type(exc).__name__}: {exc}) -- "
          f"the head is 2 Linears + GELU; apply head.safetensors directly.")

  print("\nNext: export the encoder to ONNX with the existing tool:")
  print(f"  python export_onnx.py --model-dir {enc_dir} "
        f"--output {os.path.join(enc_dir, 'onnx')} --task feature-extraction")


if __name__ == "__main__":
  main()
