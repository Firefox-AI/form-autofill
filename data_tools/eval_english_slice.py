"""English-slice accuracy for a finished run, langmap-aware.

Runs the model over testing-weakmulti3hint.txt (the common test set) and reports
accuracy on three slices: overall, English, non-English. "English" = source
classify_locale()s English OR it's a Common-Crawl row whose page langdetected
English (data/cc_lang_map.json) -- so real CC-English counts, which plain
classify_locale misses. Lets the English-only TinyBERT be compared fairly to the
multilingual bb/hybrid on the English slice.

Usage: .venv/bin/python eval_english_slice.py <run-id> [test_variant]
"""
import sys, os, io, zipfile, contextlib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from metaflow import Run, namespace
from dotraining import (Config, strip_type_marker, reformat_context, dataset_path,
                        fieldTypesReversedDict, ignoreLineCount, _triple_classify,
                        classify_locale, select_device)
from transformers import pipeline
import dotraining

namespace("production:autofillflow-0-egrc")
rid = sys.argv[1]
TEST_VARIANT = sys.argv[2] if len(sys.argv) > 2 else "-weakmulti3hint"

r = Run(f"AutofillFlow/argo-autofillflow-{rid}")
# Read config params + model artifact from the TRAIN step so this works even if
# the run's own eval step hasn't finished yet (params are on every step's data).
d = r["train"].task.data
g = lambda k, dv=None: getattr(d, k, dv)
cfg = Config(
    modelName=g("model_name"), contextFormat=g("context_format", "bb"),
    dataVariant=g("data_variant", "-relabel"), encoderLayers=g("encoder_layers", 0),
    headInteractions=g("head_interactions", False), headProjDim=g("head_proj_dim", 0),
    headConcatCurrent=g("head_concat_current", True), headFullCurrent=g("head_full_current", False),
    numHeads=g("num_heads", 1), windowSize=g("window_size", 3),
    windowFusion=g("window_fusion", "attention"), pooling=g("pooling", "cls"),
    closeLabelEps=g("close_label_eps", 0.0), modelSuffix=f"eng-{rid}")

if not os.path.isdir(cfg.saveModelDir):
    blob = d.model_artifact
    os.makedirs(cfg.saveModelDir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(cfg.saveModelDir)

langmap = json.load(open("data/cc_lang_map.json"))
en_cc = {k for k, v in langmap.items() if v == "en"}
def is_english(src):
    return classify_locale(src)[2] or (src.startswith("CC_") and src in en_cc)

texts, golds, srcs = [], [], []
for line in open(dataset_path("testing" + TEST_VARIANT + ".txt"), encoding="utf-8"):
    p = line.strip().split(",", ignoreLineCount + 1)
    if len(p) < ignoreLineCount + 2:
        continue
    texts.append(reformat_context(strip_type_marker(p[ignoreLineCount + 1], cfg), cfg.contextFormat))
    golds.append(fieldTypesReversedDict[int(p[ignoreLineCount])])
    srcs.append(p[0])

with contextlib.redirect_stdout(sys.stderr):
    if cfg.contextFormat == "triple":
        res = _triple_classify(texts, cfg)
    else:
        clf = pipeline("text-classification", model=cfg.saveModelDir,
                       truncation=True, max_length=512, device=select_device())
        res = clf(texts, truncation=True, batch_size=64)
preds = [x["label"] for x in res]

slices = {"overall": [0, 0], "english": [0, 0], "non-english": [0, 0]}
for g_, p, s in zip(golds, preds, srcs):
    ok = int(g_ == p)
    slices["overall"][0] += ok; slices["overall"][1] += 1
    key = "english" if is_english(s) else "non-english"
    slices[key][0] += ok; slices[key][1] += 1

print(f"\nrun={rid}  ctx={cfg.contextFormat}  base={cfg.modelName}  data={cfg.dataVariant}")
for k, (c, n) in slices.items():
    print(f"  {k:12} {c}/{n} = {c/n:.4f}" if n else f"  {k:12} n=0")
