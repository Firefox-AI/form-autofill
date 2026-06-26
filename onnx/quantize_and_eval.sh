#!/usr/bin/env bash
#
# End-to-end quantization pipeline for an autofill model trained in Metaflow.
#
#   ./quantize_and_eval.sh <metaflow-run-id> [namespace] [test_file]
#
# Steps (parameterized by run id, so it works for any AutofillFlow run):
#   1. Extract the trained model from the Metaflow run            -> outputs/<name>/
#   2. Set up the transformers.js quantization toolchain          (once, cached)
#   3. Export the model to ONNX (+ onnxslim)                      -> quantization/<name>/onnx/model.onnx
#   4. Quantize to fp16 q8 int8 uint8 q4 q4f16 bnb4               -> quantization/<name>/onnx/model_*.onnx
#   5. Evaluate every variant on the test set                    -> quantization/<name>/quantization_eval.csv
#
# Nothing is uploaded anywhere.
#
# Example:
#   ./quantize_and_eval.sh argo-autofillflow-jt4qd production:autofillflow-0-egrc testing-supported.txt
#
set -euo pipefail

RUN_ID="${1:?usage: quantize_and_eval.sh <metaflow-run-id> [namespace] [test_file]}"
NAMESPACE="${2:-production:autofillflow-0-egrc}"
TEST_FILE="${3:-testing-supported.txt}"

# transformers.js scripts checkout (the canonical quantizer). Override with env.
TJS_DIR="${TRANSFORMERS_JS_DIR:-/tmp/transformers.js}"
TJS_REF="${TRANSFORMERS_JS_REF:-3.8.1}"
QUANT_MODES="${QUANT_MODES:-fp16 q8 int8 uint8 q4 q4f16 bnb4}"
# Per-channel weight quantization (only affects the 8-bit modes q8/int8/uint8)
# dramatically improves their accuracy vs the per-tensor default -- it gives each
# output channel its own scale instead of one scale per weight tensor. Override
# with QUANT_EXTRA="" to reproduce the old per-tensor behavior.
QUANT_EXTRA="${QUANT_EXTRA:---per_channel}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="autofill-tiny-supported-${RUN_ID}"
SRC="${REPO}/outputs/${NAME}"
QDIR="${REPO}/quantization/${NAME}"
VENV_PY="${TJS_DIR}/scripts/venv/bin/python"

echo "==> [1/5] Extracting model from Metaflow run ${RUN_ID}"
( cd "$REPO" && uv run python extract_model.py \
    --run-id "$RUN_ID" --namespace "$NAMESPACE" --output "$SRC" )

echo "==> [2/5] Setting up transformers.js quantization toolchain (${TJS_REF})"
if [ ! -d "${TJS_DIR}/scripts" ]; then
    git clone --quiet https://github.com/huggingface/transformers.js.git "$TJS_DIR"
    ( cd "$TJS_DIR" && git checkout "$TJS_REF" -- scripts/ )
fi
if [ ! -x "$VENV_PY" ]; then
    python3 -m venv "${TJS_DIR}/scripts/venv"
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet --upgrade pip
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet -r "${TJS_DIR}/scripts/requirements.txt"
    # Extra deps beyond transformers.js requirements:
    #   onnxscript   - needed by the torch ONNX exporter
    #   scikit-learn - needed by eval_quantized.py for the metrics
    "${TJS_DIR}/scripts/venv/bin/pip" install --quiet onnxscript scikit-learn
fi

echo "==> [3/5] Exporting to ONNX (+ onnxslim)"
# Start from a clean onnx/ so scripts.quantize only sees the freshly exported
# base model.onnx (it quantizes every *.onnx in the folder, so leftover variants
# from a previous run would be re-quantized into garbage).
rm -rf "${QDIR}/onnx"
"$VENV_PY" "${REPO}/export_onnx.py" --model-dir "$SRC" --output "${QDIR}/onnx"

echo "==> [4/5] Quantizing: ${QUANT_MODES} ${QUANT_EXTRA}"
( cd "$TJS_DIR" && "$VENV_PY" -m scripts.quantize \
    --input_folder "${QDIR}/onnx" --output_folder "${QDIR}/onnx" --modes ${QUANT_MODES} ${QUANT_EXTRA} )

echo "==> [5/5] Evaluating variants against ${TEST_FILE}"
"$VENV_PY" "${REPO}/eval_quantized.py" \
    --model_dir "$QDIR" --test_file "${REPO}/${TEST_FILE}" \
    --output_csv "${QDIR}/quantization_eval.csv"

echo "==> Done. Results: ${QDIR}/quantization_eval.csv"
