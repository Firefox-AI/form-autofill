#!/usr/bin/env python
"""Export a Hugging Face text-classification model to ONNX, ready to quantize.

This produces ``<output>/model.onnx`` (plus the tokenizer/config files optimum
writes alongside it) so that transformers.js ``scripts.quantize`` can run over
the folder.

It also runs onnxslim on the exported graph. This is required: the torch ONNX
exporter emits a graph that ONNX Runtime's quantizer fails to shape-infer
("Inferred shape and existing shape differ"), and slimming the graph fixes it.
transformers.js's own convert.py slims for the same reason.

Run with the transformers.js scripts venv (has optimum + onnx + onnxslim):

    /tmp/transformers.js/scripts/venv/bin/python export_onnx.py \
        --model-dir outputs/autofill-tiny-supported-argo-autofillflow-jt4qd \
        --output    quantization/autofill-tiny-supported-argo-autofillflow-jt4qd/onnx
"""

import argparse
import os

import onnx
import onnxruntime as ort
import onnxslim
from optimum.exporters.onnx import main_export


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="Directory containing the PyTorch HF model to export.")
    ap.add_argument("--output", required=True,
                    help="Directory to write the ONNX model + tokenizer into (the 'onnx' folder).")
    ap.add_argument("--task", default="text-classification",
                    help="Optimum export task (default: text-classification).")
    # opset 17 is the minimum that supports a fused LayerNormalization op. The
    # torch exporter sometimes fuses layernorm; optimum then tries to
    # down-convert to its default opset (14) and crashes ("No Previous Version
    # of LayerNormalization exists"). Pinning >=17 avoids that.
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset to export with.")
    ap.add_argument("--skip-slim", action="store_true",
                    help="Skip the onnxslim pass (quantization will likely fail without it).")
    ap.add_argument("--legacy", action="store_true",
                    help="Export an encoder (feature-extraction) directly via the "
                         "legacy TorchScript exporter (torch.onnx.export dynamo=False) "
                         "instead of optimum. torch>=2.9 defaults to the dynamo "
                         "exporter, which emits a broken BertModel graph (runtime "
                         "shape errors / failed quantizer shape-inference); this "
                         "path produces a valid graph. Use for the split triple encoder.")
    args = ap.parse_args()

    # The context_format='triple' model is a custom shared-encoder architecture
    # (model_type 'triple_encoder'), not a standard text-classification model, so
    # optimum cannot export it. Fail fast with a clear message.
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, encoding="utf-8") as fh:
            if json.load(fh).get("model_type") == "triple_encoder":
                raise SystemExit(
                    "This is a monolithic triple-encoder model, which optimum "
                    "cannot export directly. Split it first with export_triple.py "
                    "(writes encoder/ + head/), then run this on the encoder/ "
                    "subdir with --task feature-extraction.")

    model_path = os.path.join(args.output, "model.onnx")
    if args.legacy:
        # Direct legacy export via the TorchScript exporter (dynamo=False). Avoids
        # optimum/torch's dynamo exporter, which is broken for BERT-family graphs
        # on torch>=2.9 (runtime shape errors / failed quantizer shape-inference).
        # feature-extraction -> AutoModel (last_hidden_state, for the triple
        # encoder); anything else -> AutoModelForSequenceClassification (logits).
        import inspect
        import torch
        os.makedirs(args.output, exist_ok=True)
        if args.task == "feature-extraction":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(args.model_dir).eval()
            out_name = "last_hidden_state"
            out_dyn = {0: "batch", 1: "seq"}
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
            out_name = "logits"
            out_dyn = {0: "batch"}                 # [batch, num_labels], no seq dim
        names, inputs = ["input_ids", "attention_mask"], [
            torch.ones(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.long)]
        if "token_type_ids" in inspect.signature(model.forward).parameters:
            names.append("token_type_ids")
            inputs.append(torch.zeros(1, 8, dtype=torch.long))
        dynamic = {n: {0: "batch", 1: "seq"} for n in names}
        dynamic[out_name] = out_dyn
        torch.onnx.export(
            model, tuple(inputs), model_path, input_names=names,
            output_names=[out_name], dynamic_axes=dynamic,
            opset_version=args.opset, dynamo=False)
    else:
        try:
            main_export(model_name_or_path=args.model_dir, output=args.output,
                        task=args.task, opset=args.opset)
        except FileNotFoundError as exc:
            # optimum has a cleanup bug: after writing model.onnx (+ external data)
            # it tries to delete a stale external-data file under the wrong name and
            # raises. The model itself is already written correctly, so tolerate it
            # only if the exported model exists and loads.
            if not os.path.exists(model_path):
                raise
            ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print(f"  (ignored optimum post-export cleanup error: {exc})")
    print(f"Exported ONNX -> {model_path}")

    if not args.skip_slim:
        # Slim in place; inline external weights so the result is a single,
        # self-contained .onnx that the quantizer can shape-infer.
        slimmed = onnxslim.slim(model_path)
        onnx.save(slimmed, model_path, save_as_external_data=False)
        print(f"Slimmed graph -> {model_path}")


if __name__ == "__main__":
    main()
