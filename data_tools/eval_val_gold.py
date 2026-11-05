"""Evaluate model(s) on the GOLD, human/o4-labeled portion of the validation set
-- the rows that come from data/training (real QA/real forms), excluding the
synthetic GEN_* and common-crawl CC_* rows. Unlike the small testing set, this
subset covers the long tail (gr/fi/ru/id/ua/pl/...), so it's the right ruler for
multilingual work. Reports per-language and aggregate accuracy for each model.

Usage:
  uv run python data_tools/eval_val_gold.py --val validation-relabelpy.txt \
      base=/tmp/m_prvt4 weak=/tmp/m_sjqpt
"""
import sys, os, re, argparse
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotraining import ignoreLineCount, strip_type_marker, reformat_context, Config

_CFG = Config(contextFormat="bb")
# languages the multilingual weak-labeling augments (for the aug/non-aug split)
AUG = {"de", "es", "fr", "it", "nl", "pt", "pl", "tr", "ru", "ja", "jp",
       "zh", "ko", "fa", "hi", "ml"}


def locale(fn):
    m = re.match(r'GEN_([a-z]{2})-', fn) or re.match(r'QA_([A-Z]{2})_', fn) or re.match(r'([A-Z]{2})_', fn)
    return m.group(1).lower() if m else "??"


def load_gold(val_file):
    rows = []
    for l in open(val_file, encoding="utf-8"):
        if not l.strip():
            continue
        fn = l.split(",", 1)[0]
        if fn.startswith("GEN_") or fn.startswith("CC_"):   # keep only data/training gold
            continue
        d = l.rstrip("\n").split(",", ignoreLineCount + 1)
        rows.append((locale(fn), int(d[ignoreLineCount]),
                     reformat_context(strip_type_marker(d[ignoreLineCount + 1], _CFG), "bb")))
    return rows


def predict(mdir, texts):
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=512, padding=True, return_tensors="pt")
            out += m(**enc).logits.argmax(-1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="validation-relabelpy.txt",
                    help="default val file; override per model with tag=dir@valfile")
    ap.add_argument("models", nargs="+",
                    help="tag=model_dir  (or tag=model_dir@val_file for hint models "
                         "that need a hint-containing val); first = baseline")
    args = ap.parse_args()
    # each spec: tag, model_dir, val_file (per-model; hint models need their hint val)
    specs = []
    for s in args.models:
        tag, rest = s.split("=", 1)
        mdir, _, vf = rest.partition("@")
        specs.append((tag, mdir, vf or args.val))
    tags = [t for t, _, _ in specs]
    # rows aligned by language across models (same gold forms/labels in each val file)
    st = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # lang -> tag -> [correct,n]
    ns = {}
    for tag, mdir, vf in specs:
        rows = load_gold(vf)
        preds = predict(mdir, [r[2] for r in rows])
        ns[tag] = len(rows)
        for (lc, g, _), p in zip(rows, preds):
            st[lc][tag][0] += p == g
            st[lc][tag][1] += 1
    print(f"gold (data/training) val rows per model: {ns}")
    if len(set(ns.values())) != 1:
        print("  NOTE: row counts differ across val files (hint vs no-hint) -- per-language "
              "accuracy is still comparable; treat aggregate Δ as approximate.")
    hdr = f"{'lang':5}{'n':>4}" + "".join(f"{t:>8}" for t in tags)
    if len(tags) > 1: hdr += f"{'Δ':>6}  aug"
    print(hdr)
    for lc in sorted(st, key=lambda k: -st[k][tags[0]][1]):
        n = st[lc][tags[0]][1]
        cells = "".join(f"{100*st[lc][t][0]/n:7.0f}%" for t in tags)
        line = f"{lc:5}{n:4}{cells}"
        if len(tags) > 1:
            d = 100*(st[lc][tags[-1]][0]-st[lc][tags[0]][0])/n
            line += f"{d:+5.0f}  {'aug' if lc in AUG else ''}"
        print(line)

    def agg(langs, tag):
        c = sum(st[l][tag][0] for l in langs); n = sum(st[l][tag][1] for l in langs)
        return c, n, 100*c/max(n, 1)
    for name, langs in (("ALL", list(st)),
                        ("AUGMENTED", [l for l in st if l in AUG]),
                        ("non-augmented", [l for l in st if l not in AUG])):
        parts = []
        for t in tags:
            _, n, a = agg(langs, t); parts.append(f"{t} {a:.1f}")
        d = agg(langs, tags[-1])[2] - agg(langs, tags[0])[2] if len(tags) > 1 else 0
        print(f"{name:14} n={agg(langs, tags[0])[1]:4}  " + "  ".join(parts) + (f"  Δ{d:+.1f}" if len(tags) > 1 else ""))


if __name__ == "__main__":
    main()
