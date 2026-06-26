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
    args = ap.parse_args()

    model_path = os.path.join(args.output, "model.onnx")
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
