"""Evaluate the 8 v3-trained models on the CLEAN v2 test set (real sites, no
non-address) to see whether adding non-address negatives changed real-site
accuracy. base arms -> testing-v2base.txt (no select tokens); optrange arms ->
testing-v2opt.txt (select tokens). Reuses dotraining.evaluate_model."""
import io, os, json, zipfile, contextlib
from metaflow import Run, namespace
from dotraining import Config, evaluate_model, CLASSIFICATION_REPORT
namespace(None)

TRI = dict(contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)
# run-id -> (label, eval-variant, model params)
RUNS = {
 "t8vl7": ("bb·base","-v2base", dict(modelName="huawei-noah/TinyBERT_General_4L_312D", contextFormat="bb")),
 "rjfv5": ("bb·opt","-v2opt",  dict(modelName="huawei-noah/TinyBERT_General_4L_312D", contextFormat="bb")),
 "6r99j": ("triv2·base","-v2base", dict(modelName="rolf-mozilla/minilm-pruned-notest-v2", **TRI)),
 "ldq99": ("triv2·opt","-v2opt",  dict(modelName="rolf-mozilla/minilm-pruned-notest-v2", **TRI)),
 "swft8": ("unpr·base","-v2base", dict(modelName="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", **TRI)),
 "dvssg": ("unpr·opt","-v2opt",  dict(modelName="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", **TRI)),
 "j4jjx": ("v3·base","-v2base", dict(modelName="rolf-mozilla/minilm-pruned-notest-v3", **TRI)),
 "2m7qv": ("v3·opt","-v2opt",  dict(modelName="rolf-mozilla/minilm-pruned-notest-v3", **TRI)),
}
out = {}
for rid,(label,variant,params) in RUNS.items():
    print(f"=== {label} ({rid}) on {variant} ===", flush=True)
    cfg = Config(modelSuffix=f"v2eval-{rid}", dataVariant=variant, **params)
    blob = Run(f"AutofillFlow/argo-autofillflow-{rid}").data.model_artifact
    os.makedirs(cfg.saveModelDir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf: zf.extractall(cfg.saveModelDir)
    with contextlib.redirect_stdout(io.StringIO()):
        m = evaluate_model(cfg, "testing")
    rep = m[CLASSIFICATION_REPORT]
    cc = {k:v for k,v in rep.items() if isinstance(v,dict) and k.startswith("cc-")}
    out[label] = {"acc": m.get("accuracy"), "cc": cc}
    print(f"   v2-test acc={m.get('accuracy')}", flush=True)
json.dump(out, open("/tmp/v2test_eval.json","w"), indent=2)
print("WROTE /tmp/v2test_eval.json", flush=True)
