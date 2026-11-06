"""Out-of-fold (OOF) outlier / mislabel finder for k-fold CV runs.

Confident-learning style label cleaning: for each fold i, load the model trained
with --fold_index i (which never saw fold i) and predict on fold i's held-out
rows. Aggregating over all folds gives every training row an OOF prediction from
a model that did NOT train on it. Rows where the OOF prediction disagrees with
the stored label -- especially at high confidence -- are the most likely
mislabeled or genuine outliers.

Prereq: train K models with numFolds=K, foldIndex=0..K-1 (same modelSuffix,
dataVariant, foldSeed), so output-models/autofill-tiny-all<suffix>-fold{i}of{K}
each exist.

Usage:
  .venv/bin/python oof_outliers.py --suffix -ende --folds 5 --variant -wm3en \
      [--context bb] [--conf 0.0]
Writes: oof_predictions.csv (every row) and oof_suspects.csv (label != OOF pred,
sorted by model confidence descending).
"""
import sys, os, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotraining import (Config, fold_of, strip_type_marker, reformat_context,
                        dataset_path, fieldTypesReversedDict, ignoreLineCount,
                        _triple_classify, select_device)
from transformers import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="", help="model_suffix used for the fold runs")
ap.add_argument("--folds", type=int, required=True)
ap.add_argument("--variant", default="", help="dataVariant (training<variant>.txt)")
ap.add_argument("--context", default="bb", choices=["bb", "triple"])
ap.add_argument("--fold-seed", type=int, default=20261106)
ap.add_argument("--conf", type=float, default=0.0, help="min confidence to flag a suspect")
args = ap.parse_args()

train_path = dataset_path("training" + args.variant + ".txt")
all_rows = []
with open(train_path, encoding="utf-8") as fh:
    for line in fh:
        p = line.rstrip("\n").split(",", ignoreLineCount + 1)
        if len(p) < ignoreLineCount + 2:
            continue
        all_rows.append({"src": p[0], "field": p[1] if ignoreLineCount == 2 else "",
                         "gold": fieldTypesReversedDict[int(p[ignoreLineCount])],
                         "raw": p[ignoreLineCount + 1]})

oof = []   # (src, field, gold, pred, conf)
for i in range(args.folds):
    cfg = Config(contextFormat=args.context, dataVariant=args.variant,
                 numFolds=args.folds, foldIndex=i, foldSeed=args.fold_seed,
                 modelSuffix=args.suffix)
    model_dir = cfg.saveModelDir
    if not os.path.isdir(model_dir):
        sys.exit(f"missing fold model: {model_dir} (train fold {i} first)")
    held = [r for r in all_rows if fold_of(r["src"], cfg) == i]
    texts = [reformat_context(strip_type_marker(r["raw"], cfg), args.context) for r in held]
    print(f"fold {i}: {len(held)} held-out rows, model {model_dir}", file=sys.stderr, flush=True)
    if args.context == "triple":
        res = _triple_classify(texts, cfg)
    else:
        clf = pipeline("text-classification", model=model_dir, truncation=True,
                       max_length=512, device=select_device())
        res = clf(texts, truncation=True, batch_size=64)
    for r, out in zip(held, res):
        oof.append((r["src"], r["field"], r["gold"], out["label"], float(out["score"])))

# every training row now has an OOF prediction
with open("oof_predictions.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["src", "field", "gold", "oof_pred", "conf"])
    w.writerows(oof)

suspects = sorted([o for o in oof if o[2] != o[3] and o[4] >= args.conf],
                  key=lambda x: -x[4])
with open("oof_suspects.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["src", "field", "gold", "oof_pred", "conf"])
    w.writerows(suspects)

n = len(oof)
acc = sum(1 for o in oof if o[2] == o[3]) / max(n, 1)
print(f"\nOOF over {n} training rows: accuracy {acc:.3f}")
print(f"suspects (label != OOF pred, conf >= {args.conf}): {len(suspects)} "
      f"({100*len(suspects)/max(n,1):.1f}%) -> oof_suspects.csv")
print("top high-confidence disagreements (likely mislabels):")
for s in suspects[:20]:
    print(f"  conf {s[4]:.2f}  gold={s[2]:16} oof={s[3]:16} {s[0]}/{s[1]}")
