"""Evaluate each trained A/B model on the VALIDATION set (bigger CC support than
test) and dump per-CC-field precision/recall/f1. Pulls the saved model artifact
from Metaflow and reuses dotraining.evaluate_model as-is."""
import io, os, sys, json, contextlib
from metaflow import Run, namespace
import dotraining as dt
from dotraining import Config, evaluate_model, CLASSIFICATION_REPORT

namespace(None)

# run-id -> (label, params). Params mirror the argo trigger commands.
RUNS = {
  "qlnt5": ("bb·base",       dict(modelName="huawei-noah/TinyBERT_General_4L_312D", dataVariant="-baseline", contextFormat="bb")),
  "4sjmf": ("bb·opt",        dict(modelName="huawei-noah/TinyBERT_General_4L_312D", dataVariant="-optrange", contextFormat="bb")),
  "4s8fm": ("tri-v2·base",   dict(modelName="rolf-mozilla/minilm-pruned-notest-v2", dataVariant="-baseline", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
  "rwn9t": ("tri-v2·opt",    dict(modelName="rolf-mozilla/minilm-pruned-notest-v2", dataVariant="-optrange", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
  "s5hv2": ("tri-unpr·base", dict(modelName="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dataVariant="-baseline", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
  "pggbh": ("tri-unpr·opt",  dict(modelName="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dataVariant="-optrange", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
  "mwtc8": ("tri-v3·base",   dict(modelName="rolf-mozilla/minilm-pruned-notest-v3", dataVariant="-baseline", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
  "2mrjb": ("tri-v3·opt",    dict(modelName="rolf-mozilla/minilm-pruned-notest-v3", dataVariant="-optrange", contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)),
}

out = {}
for rid, (label, params) in RUNS.items():
    print(f"=== {label} ({rid}) ===", flush=True)
    # unique suffix -> unique saveModelDir per run
    cfg = Config(modelSuffix=f"valeval-{rid}", **params)
    blob = Run(f"AutofillFlow/argo-autofillflow-{rid}").data.model_artifact
    dst = cfg.saveModelDir
    os.makedirs(dst, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dst)
    # evaluate_model is very chatty; silence its stdout, keep the return value.
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = evaluate_model(cfg, "validation")
    rep = metrics[CLASSIFICATION_REPORT]  # {field: {precision,recall,f1-score,support}}
    cc = {k: v for k, v in rep.items() if isinstance(v, dict) and k.startswith("cc-")}
    out[label] = {"overall_acc": metrics.get("accuracy"), "cc": cc}
    print(f"   overall val acc={metrics.get('accuracy')}", flush=True)

json.dump(out, open("/tmp/val_cc_eval.json", "w"), indent=2)
print("WROTE /tmp/val_cc_eval.json", flush=True)
