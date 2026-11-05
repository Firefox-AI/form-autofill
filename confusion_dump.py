"""Row-level confusion dump for a finished Argo run on the TEST split.

Rehydrates the run's Config (like eval_val_perlabel.py), runs the model over
testing<data_variant>.txt, and captures (gold, pred, score, file, fieldname,
text) for every row. Emits:
  - overall accuracy
  - error budget by gold label (support x error-rate), ranked
  - top confusion pairs (gold -> pred), ranked by count
  - per-bucket direction breakdown for the 'other' and street-address clusters
  - a CSV of every misprediction: confusion_misses_<rid>.csv

Usage: .venv/bin/python confusion_dump.py <run-id>   e.g. cckrm | tlt5d
"""
import sys, os, io, zipfile, contextlib, csv
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metaflow import Run, namespace
from dotraining import (
    Config, strip_type_marker, reformat_context, dataset_path,
    fieldTypesReversedDict, ignoreLineCount, _triple_classify,
)
from transformers import pipeline
import dotraining

namespace("production:autofillflow-0-egrc")

rid = sys.argv[1]
r = Run(f"AutofillFlow/argo-autofillflow-{rid}")
d = r["end"].task.data
g = lambda k, dv=None: getattr(d, k, dv)

cfg = Config(
    modelName=g("model_name"),
    contextFormat=g("context_format", "bb"),
    dataVariant=g("data_variant", "-relabel"),
    encoderLayers=g("encoder_layers", 0),
    headInteractions=g("head_interactions", False),
    headProjDim=g("head_proj_dim", 0),
    headConcatCurrent=g("head_concat_current", True),
    headFullCurrent=g("head_full_current", False),
    numHeads=g("num_heads", 1),
    windowSize=g("window_size", 3),
    windowFusion=g("window_fusion", "attention"),
    pooling=g("pooling", "cls"),
    genToRealRatio=g("gen_to_real_ratio", 0.0),
    ccToRealRatio=g("cc_to_real_ratio", 0.0),
    subsampleValidation=g("subsample_validation", False),
    closeLabelEps=g("close_label_eps", 0.0),
    modelSuffix=f"conf-{rid}",
)

if not os.path.isdir(cfg.saveModelDir):
    blob = r["train"].task.data.model_artifact
    os.makedirs(cfg.saveModelDir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(cfg.saveModelDir)
    print(f"extracted model -> {cfg.saveModelDir}", file=sys.stderr)

# ---- load TEST rows (mirrors evaluate_model's parsing) ----
texts, golds, files, fieldnames = [], [], [], []
path = dataset_path("testing" + cfg.dataVariant + ".txt")
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    parts = line.split(",", ignoreLineCount + 1)
    raw = strip_type_marker(parts[ignoreLineCount + 1], cfg)
    texts.append(reformat_context(raw, cfg.contextFormat))
    golds.append(fieldTypesReversedDict[int(parts[ignoreLineCount])])
    files.append(parts[0])
    fieldnames.append(parts[1])

print(f"run={rid} ctx={cfg.contextFormat} var={cfg.dataVariant} rows={len(texts)}", file=sys.stderr)

# ---- inference ----
with contextlib.redirect_stdout(sys.stderr):
    if cfg.contextFormat == "triple":
        results = _triple_classify(texts, cfg)
    else:
        clf = pipeline("text-classification", model=cfg.saveModelDir,
                       truncation=True, max_length=512, device=dotraining.select_device())
        results = clf(texts, truncation=True, batch_size=64)

preds = [res["label"] for res in results]
scores = [float(res["score"]) for res in results]

# ---- aggregate ----
N = len(golds)
correct = sum(1 for g_, p in zip(golds, preds) if g_ == p)
print(f"\n=== OVERALL: {correct}/{N} = {correct/N:.4f} ===\n")

support = Counter(golds)
errs_by_gold = Counter()
confusion = Counter()          # (gold, pred) -> count, misses only
misses = []
for g_, p, s, f, fn, t in zip(golds, preds, scores, files, fieldnames, texts):
    if g_ != p:
        errs_by_gold[g_] += 1
        confusion[(g_, p)] += 1
        misses.append((g_, p, s, f, fn, t))

total_err = sum(errs_by_gold.values())
print("=== ERROR BUDGET BY GOLD LABEL (support x error) ===")
print(f"{'gold':22}{'sup':>5}{'err':>5}{'acc':>7}{'%budget':>9}")
cum = 0
for lbl, e in errs_by_gold.most_common(18):
    sup = support[lbl]; cum += e
    print(f"{lbl:22}{sup:>5}{e:>5}{1-e/sup:>7.3f}{100*e/total_err:>8.0f}%  cum {100*cum/total_err:.0f}%")

print("\n=== TOP CONFUSION PAIRS (gold -> pred) ===")
for (gp), c in confusion.most_common(25):
    print(f"  {c:>4}  {gp[0]:22} -> {gp[1]}")

# ---- direction breakdown for key clusters ----
STREET = {"address-line1", "address-line2", "address-line3", "street-address",
          "address-housenumber", "name"}
def breakdown(title, gold_set):
    print(f"\n=== {title}: where do these golds go wrong? ===")
    sub = [(g_, p) for g_, p in zip(golds, preds) if g_ in gold_set and g_ != p]
    by = Counter((g_, p) for g_, p in sub)
    for (g_, p), c in by.most_common(20):
        print(f"  {c:>4}  {g_:22} -> {p}")
    # also: what gets wrongly predicted INTO these types (false positives)?
    print(f"  -- false positives (other golds -> these) --")
    fp = Counter((g_, p) for g_, p in zip(golds, preds)
                 if p in gold_set and g_ != p and g_ not in gold_set)
    for (g_, p), c in fp.most_common(12):
        print(f"  {c:>4}  {g_:22} -> {p}")

breakdown("'other' bucket", {"other"})
breakdown("street-address cluster", STREET)

# ---- write all misses to CSV ----
out = f"confusion_misses_{rid}.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["gold", "pred", "score", "file", "fieldname", "text"])
    for m in sorted(misses, key=lambda x: (x[0], x[1], -x[2])):
        w.writerow(m)
print(f"\nwrote {len(misses)} misses -> {out}", file=sys.stderr)
