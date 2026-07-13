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

The triple-encoder model (context_format=triple)

The triple model classifies each field from three separately-encoded token
strings — the field's own tokens (`current`) plus its two neighbors
(`previous`/`next`), with the aa/bb prefixes stripped — sharing one base
transformer, followed by an MLP fusion head. It is not a standard Hugging Face
classifier, so it needs the extra pruning/split steps below. The architecture
lives in `dotraining.py` (`TripleEncoderForSequenceClassification`). The best
configuration found is a vocab-pruned multilingual MiniLM base, cut to 4 layers,
with neighbor-interaction features in the head.

1. **Prune the base model's vocabulary** (recommended). The multilingual MiniLM
   base spends ~80% of its parameters on a ~250k-token embedding table.
   `prune_vocab.py` keeps only the tokens that appear in your data, shrinking the
   model ~4x with no accuracy loss. Pass training + validation only (exclude the
   test set, so eval stays honest):

   ```
   uv run python prune_vocab.py \
       --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
       --corpus processed_datasets/training-supported.txt processed_datasets/validation-supported.txt \
       --out output-models/minilm-pruned-notest
   ```

   Push the result to Hugging Face so remote (Argo) training can pull it, then
   pass it as `--model_name`.

2. **Train.** Triple-specific `autofill_flow.py` parameters (each defaults to the
   original single-sequence behavior, so they are opt-in):

   - `--context_format triple` — use the triple architecture.
   - `--encoder_layers N` — keep N evenly-spaced transformer layers (e.g. 4 of
     12; the base is `MiniLM-L12`). 0 keeps all layers.
   - `--head_interactions True` — add neighbor-difference features
     `[cur-prev, cur-next]` to the fusion (helps the MiniLM base; did not help
     TinyBERT).
   - `--head_proj_dim D` — shared `Linear(H->D)` bottleneck before fusion (0=off).

   `argo-workflows create` bakes the code **and datasets** into the workflow
   package, so refresh the local `.txt` files (training in `processed_datasets/`,
   validation/testing at the repo root — these are read by bare name) and re-run
   `create` before `trigger` whenever the data changes:

   ```
   uv run python autofill_flow.py argo-workflows create
   uv run python autofill_flow.py argo-workflows trigger \
       --model_name rolf-mozilla/minilm-pruned-notest --context_format triple \
       --encoder_layers 4 --head_interactions True \
       --learning_rate 0.00007 --warmup_ratio 0.1 --train_batch_size 32 --num_epochs 13 \
       --train_file processed_datasets/training-supported.txt \
       --model_suffix triple-L4-interactions
   ```

3. **Export + quantize.** The triple model is split into the shared encoder (a
   plain feature-extraction model that quantizes normally) and the small fp32
   fusion head:

   ```
   ./onnx/quantize_triple.sh argo-autofillflow-<run-id> production:autofillflow-0-egrc
   ```

   This runs: extract -> `onnx/export_triple.py` (splits into `encoder/` +
   `head/`) -> `onnx/export_onnx.py --task feature-extraction --legacy` (encoder
   to ONNX) -> transformers.js quantize. Outputs land in
   `quantization/<name>/`: `encoder/onnx/model_*.onnx` (quantized shared encoder)
   and `head/` (`head.safetensors` — fusion + classifier weights, kept fp32 — and
   `head.json`, the inference contract).

   **`--legacy` flag:** torch >= 2.9 defaults to the dynamo ONNX exporter, which
   emits a broken BERT graph (runtime shape errors / failed quantizer
   shape-inference). `--legacy` forces the TorchScript exporter, which produces a
   valid, quantizable graph. Use it for the triple encoder and for any standard
   (`bb`) model on newer torch.

   **Inference contract** (`head.json`): run the encoder once per field,
   attention-mask-mean-pool each field's `last_hidden_state` to a vector, then per
   field apply the head to `concat[e_cur, e_prev, e_next (, e_cur-e_prev,
   e_cur-e_next)]` -> `Linear -> GELU -> Linear` -> argmax. GELU is the exact
   (erf) form; the concat order is `head.json`'s `input_order`. Because the
   encoder runs once per field (not 3x), triple inference costs about the same as
   the single-sequence model over a whole form.

4. **Evaluate the quantized triple model** (total/close accuracy for fp32 and each
   quantized encoder variant; reconstructs the head per `head.json`, and the fp32
   number reproduces the training-time eval):

   ```
   uv run --with onnxruntime python onnx/eval_triple_quantized.py \
       --split-dir quantization/autofill-tiny-supported-<run-id> \
       --test-file testing-supported.txt
   ```

Cross-model analysis helpers

- `onnx/classwise_int8.py` — per-class support / accuracy / f1 across several
  models (int8) written to a wide CSV.
- `data_tools/eval_quantized.py` — quantized eval for standard (`bb`) classifiers
  (accuracy / kappa per variant); also used by `quantize_and_eval.sh`.

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

