# Testing an autofill model in a real Firefox build (ml_driver)

How to verify a trained autofill classifier end-to-end against a **built
firefox-clone**, using [`ml_driver`](https://github.com/…/ml_driver) to drive the
browser. Two modes:

- **Direct-engine** (fast, no source edit): feed pre-tokenized `mlData` straight to
  the ML engine. Verifies the **model/ONNX** is correct.
- **End-to-end** (full path): let **Firefox convert the form → mlData** and run the
  model inside the browser. Verifies the **whole production path** (scanner +
  hint + model).

Both drive `dist/Nightly.app` from the firefox-clone objdir. The browser must be a
**custom build with ML features enabled** (release builds won't work).

---

## 0. Prerequisites

- firefox-clone built: binary at
  `firefox-clone/firefox/obj-<arch>/dist/Nightly.app/Contents/MacOS/firefox`
  (NOT `dist/bin/firefox` — selenium can't drive the stub).
- `ml_driver` repo with its venv: `~/Documents/GitHub/ml_driver/.venv`.
- geckodriver on `PATH` (cached at
  `~/.cache/selenium/geckodriver/mac-arm64/<ver>/geckodriver`). Without it,
  Selenium Manager auto-resolve fails with `cpio archive error: bad magic value`.
- Env for every run: `MOZ_ALLOW_EXTERNAL_ML_HUB=1`, `FASTLY_TOKEN=dummy`
  (no FxA needed for autofill ML).

## 1. Publish + quantize the model (to rolf-mozilla, not mozilla)

```bash
# extract + export ONNX + quantize all modes (int8 = model_quantized.onnx)
bash onnx/quantize_and_eval.sh argo-autofillflow-<run> production:autofillflow-0-egrc testing-weakmulti3hint.txt
# assemble repo (config.json + tokenizer.* at root, *.onnx under onnx/) and push:
#   rolf-mozilla/<name>   (mirror mozilla/tinybert-address-autofill layout)
```
Repo layout (mirror the shipped model):
```
config.json  tokenizer.json  tokenizer_config.json  [special_tokens_map.json vocab.txt]
onnx/model.onnx  onnx/model_quantized.onnx  onnx/model_int8.onnx  …
```

## 2. Point the hub at HuggingFace

rolf-mozilla models aren't on the Mozilla mirror, so set these Firefox prefs
(as `ml_prefs` on the driver, or `user.js`):
```
browser.ml.modelHubRootUrl    = "https://huggingface.co/"
browser.ml.modelHubUrlTemplate = "{model}/resolve/{revision}"   # NO {file}; filename appended automatically
```

## 3a. Direct-engine test (quick model check)

No source edit. Create the engine with your `modelId`, run on `mlData` strings:
```python
fx = FirefoxDriver(model="", auth_config={}, firefox_bin=Path(FFBIN), headless=True,
    ml_prefs={"browser.ml.modelHubRootUrl":"https://huggingface.co/",
              "browser.ml.modelHubUrlTemplate":"{model}/resolve/{revision}"})
eng = fx.create_ml_engine({"taskName":"text-classification","featureId":"formfill-classification",
    "modelId":"rolf-mozilla/<name>","modelRevision":"main","dtype":"q8","numThreads":2})
resp = fx.run_ml_engine(eng["engineId"], args=[list_of_texts], options={"topk":1})  # -> [{label,score}]
```
Input `text` = the bb-combined mlData with the `**hint` token prepended (same string
`eval_tailgold.py` builds). `dtype:"q8"` loads `onnx/model_quantized.onnx` (int8).

## 3b. End-to-end test (Firefox converts the form)

This runs the real scanner path, so Firefox builds `mlData` from the DOM and runs
the model. The single-head classifier is only used when **twoHead is off**, and its
`modelId` is **hardcoded** (no override pref), so edit it:

1. In `toolkit/components/formautofill/shared/FormAutofillML.sys.mjs`, set
   `FormFill_Config.modelId = "rolf-mozilla/<name>"`.  **Revert after testing.**
2. Rebuild: `./mach build faster` (repackages omni.ja into dist).
3. Run with these prefs (`ml_prefs`) + env:
   ```
   extensions.formautofill.useml.twoHead   = false          # use the single classifier
   extensions.formautofill.useml.features   = ["regex_hint"]  # emit **hint (match training)
   extensions.formautofill.useml.hint.fields = <default>      # must match the trained whitelist
   extensions.formautofill.useml.modelVersion = "main"        # revision for the HF repo
   browser.ml.modelHubRootUrl/UrlTemplate  = HuggingFace (step 2)
   ```
4. Drive the forms (Firefox converts each form; `reason=="ml"` fields are model-labeled):
   ```python
   res = fx.detect_form_fields(html=stripped_html, use_ml=True, use_fathom=False)
   # res["fields"][i]: fieldName (prediction), reason ("ml"), mlData (Firefox-built)
   ```
   Compare `fieldName` to ground truth `data-moz-autofill-type` on the test HTML
   (see `data_tools/build_dataset_ff.py` for the mapping helpers).

`useml.features` **must match what the model was trained with** — it's part of the
model's input contract. cckrm was trained with the hint, so `regex_hint` is
required; if training also used `select_option`/`input_attributes`, add those keys.

## Gotchas / checklist

- Binary = `.app/Contents/MacOS/firefox`, not `dist/bin/firefox`.
- geckodriver on PATH (else `cpio archive error`).
- Hub template has **no** `{file}` placeholder.
- `twoHead` defaults to **true** → set it **false** for a single-encoder model.
- The single-head `modelId` is hardcoded → edit + `mach build faster`; **revert** the
  edit before committing.
- `build_dataset_ff.py` has stale hardcoded paths (`/Users/Rrando` → your home).

See also `data_tools/ML_preprocessing_train_vs_inference.md` (why Firefox vs Python
preprocessing match) and the memory note `ml-driver-run-recipe`.
