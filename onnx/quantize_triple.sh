#!/usr/bin/env bash
#
# ONNX-export + quantize a TRIPLE-encoder model from a Metaflow run.
#
#   ./quantize_triple.sh <metaflow-run-id> [namespace]
#
# The triple model isn't a standard classifier, so (unlike quantize_and_eval.sh)
# we split it first: the SHARED ENCODER is exported as a normal feature-extraction
# model and quantized, while the small FUSION HEAD ships as fp32 weights applied
# per-field by the inference engine (see export_triple.py for the contract).
#
# Steps:
#   1. Extract the trained model from the Metaflow run     -> outputs/<name>/
#   2. Split into encoder/ + head/                         -> quantization/<name>/{encoder,head}
#   3. Set up the transformers.js quantization toolchain   (once, cached)
#   4. Export the ENCODER to ONNX (feature-extraction)     -> .../encoder/onnx/model.onnx
#   5. Quantize the encoder                                -> .../encoder/onnx/model_*.onnx
# The head is left as quantization/<name>/head/ (head.safetensors + head.json).
#
# Example:
#   ./quantize_triple.sh argo-autofillflow-lqnwd production:autofillflow-0-egrc
#
set -euo pipefail

RUN_ID="${1:?usage: quantize_triple.sh <metaflow-run-id> [namespace]}"
NAMESPACE="${2:-production:autofillflow-0-egrc}"

TJS_DIR="${TRANSFORMERS_JS_DIR:-/tmp/transformers.js}"
TJS_REF="${TRANSFORMERS_JS_REF:-3.8.1}"
# Encoder is a feature extractor (no classifier head), so no per-channel/label
# assumptions; keep the same 8-bit-friendly per-channel default as the classifier path.
QUANT_MODES="${QUANT_MODES:-fp16 q8 int8 uint8 q4 q4f16 bnb4}"
QUANT_EXTRA="${QUANT_EXTRA:---per_channel}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="autofill-tiny-supported-${RUN_ID}"
SRC="${REPO}/outputs/${NAME}"
QDIR="${REPO}/quantization/${NAME}"
ENC="${QDIR}/encoder"
VENV_PY="${TJS_DIR}/scripts/venv/bin/python"

echo "==> [1/5] Extracting model from Metaflow run ${RUN_ID}"
( cd "$REPO" && uv run python extract_model.py \
    --run-id "$RUN_ID" --namespace "$NAMESPACE" --output "$SRC" )

echo "==> [2/5] Splitting triple model into encoder/ + head/"
( cd "$REPO" && uv run python export_triple.py --model-dir "$SRC" --output "$QDIR" )

echo "==> [3/5] Setting up transformers.js quantization toolchain (${TJS_REF})"
if [ ! -d "${TJS_DIR}/scripts" ]; then
    git clone --quiet https://github.com/huggingface/transformers.js.git "$TJS_DIR"
    ( cd "$TJS_DIR" && git checkout "$TJS_REF" -- scripts/ )
fi
if [ ! -x "$VENV_PY" ]; then
    python3 -m venv "${TJS_DIR}/scripts/venv"
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet --upgrade pip
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet -r "${TJS_DIR}/scripts/requirements.txt"
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet onnxscript scikit-learn
fi

echo "==> [4/5] Exporting ENCODER to ONNX (legacy exporter, + onnxslim)"
# --legacy: torch>=2.9's default dynamo exporter emits a broken BertModel graph
# (runtime shape errors + failed quantizer shape-inference); the legacy
# TorchScript exporter produces a valid, quantizable graph.
rm -rf "${ENC}/onnx"
"$VENV_PY" "${REPO}/export_onnx.py" --model-dir "$ENC" --output "${ENC}/onnx" \
    --task feature-extraction --legacy

echo "==> [5/5] Quantizing encoder: ${QUANT_MODES} ${QUANT_EXTRA}"
( cd "$TJS_DIR" && "$VENV_PY" -m scripts.quantize \
    --input_folder "${ENC}/onnx" --output_folder "${ENC}/onnx" --modes ${QUANT_MODES} ${QUANT_EXTRA} )

echo "==> Done."
echo "    encoder ONNX + quantized variants: ${ENC}/onnx/model*.onnx"
echo "    fusion head (fp32, apply per-field):    ${QDIR}/head/head.safetensors + head.json"
echo "    NOTE: quantized-accuracy eval for triple needs a head-aware harness (not yet wired)."
