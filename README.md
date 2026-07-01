Tools for training the local Auto Form Fill model. 

Instructions on how to run are in autofill_flow.py


Quantizing and evaluating a trained model

The training flow (autofill_flow.py) saves the model as a Metaflow artifact. To
take a trained run, quantize it to the ONNX variants used by Firefox, and score
each variant on the test set, use the one-shot pipeline (parameterized by the
Metaflow run id, so it works for any AutofillFlow run):

```
# <run-id> <metaflow-namespace> [test-file]
./quantize_and_eval.sh argo-autofillflow-jt4qd production:autofillflow-0-egrc testing-supported.txt
```

This runs the steps below and writes per-variant metrics to
`quantization/autofill-tiny-supported-<run-id>/quantization_eval.csv`. Nothing
is uploaded. The individual steps (each is a standalone script you can run on
their own) are:

1. **Extract** the model from the Metaflow run into `outputs/<name>/`:

   ```
   uv run python extract_model.py \
       --run-id argo-autofillflow-jt4qd \
       --namespace production:autofillflow-0-egrc \
       --output outputs/autofill-tiny-supported-argo-autofillflow-jt4qd
   ```

2. **Set up** the transformers.js quantizer (done automatically by the pipeline;
   the extra `onnxscript` + `scikit-learn` deps are needed for export and eval):

   ```
   git clone https://github.com/huggingface/transformers.js.git /tmp/transformers.js
   cd /tmp/transformers.js && git checkout 3.8.1 -- scripts/
   cd scripts && python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt onnxscript scikit-learn
   cd ..
   ```

3. **Export to ONNX** (PyTorch -> ONNX, with onnxslim — required, or the
   quantizer fails shape inference):

   ```
   /tmp/transformers.js/scripts/venv/bin/python export_onnx.py \
       --model-dir outputs/autofill-tiny-supported-argo-autofillflow-jt4qd \
       --output    quantization/autofill-tiny-supported-argo-autofillflow-jt4qd/onnx
   ```

4. **Quantize** to the ONNX variants. `--per_channel` is important: it gives each
   weight-output-channel its own int8 scale and recovers most of the accuracy the
   8-bit modes (q8/int8/uint8) otherwise lose to per-tensor scaling (e.g. jt4qd q8
   0.830 -> 0.878). It only affects the 8-bit modes; fp16/q4/bnb4 are unchanged.

   ```
   cd /tmp/transformers.js
   ./scripts/venv/bin/python -m scripts.quantize \
       --input_folder  <repo>/quantization/autofill-tiny-supported-argo-autofillflow-jt4qd/onnx \
       --output_folder <repo>/quantization/autofill-tiny-supported-argo-autofillflow-jt4qd/onnx \
       --modes fp16 q8 int8 uint8 q4 q4f16 bnb4 --per_channel
   ```

   Note on 8-bit choices: `q8` and `int8` are identical for this model (no Conv
   ops, so q8's auto-selection resolves to signed int8). `uint8` (unsigned,
   asymmetric) tends to score highest of the 8-bit modes, so it's the recommended
   8-bit deployment target. `fp16` is effectively lossless if size allows.

5. **Evaluate** every variant on the test set (kappa, accuracy, weighted/balanced
   accuracy per quantization, written to CSV):

   ```
   /tmp/transformers.js/scripts/venv/bin/python eval_quantized.py \
       --model_dir quantization/autofill-tiny-supported-argo-autofillflow-jt4qd \
       --test_file testing-supported.txt
   ```

Uploading a model

```
# Clone, then pull the LFS-managed ONNX files
  git clone https://huggingface.co/[username]/tinybert-address-autofill
  cd tinybert-address-autofill
  git lfs install && git lfs pull

  # Create the destination repo (only needed the first time)
  huggingface-cli repo create tinybert-address-autofill --type model --organization Mozilla

  # Repoint origin and push
  git remote set-url origin https://huggingface.co/Mozilla/tinybert-address-autofill
  git push origin main
```

