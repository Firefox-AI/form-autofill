"""Override analysis for the regex-hint experiments (batch 16).

The key question is not just overall accuracy but WHETHER the model learned to
use the regex hint intelligently: confirm it when right, supply an answer when
the regex is silent, and OVERRIDE it when the context disagrees. This buckets
every test field by the regex outcome vs ground truth and reports each model's
accuracy per bucket, plus the postal-code and name focus metrics.

Buckets (per field, from the regex hint token baked into testing-relabelhint.txt):
  correct : regex fired and equals GT
  wrong   : regex fired but != GT   (override target -- must not blindly copy)
  silent  : regex silent (**hintnone) (recall target -- ML must supply)

Usage:
  uv run --with onnxruntime python data_tools/hint_override_analysis.py \
     B0=<dir>:testing-relabelpy.txt \
     B1=<dir>:testing-relabelhint.txt \
     B2=<dir>:testing-relabelhintdrop.txt

Each arg is TAG=<model_dir>:<test_file>. The regex outcome + GT are read from
testing-relabelhint.txt (rows align by index across the parallel test files, all
built from the same HTML in the same order).
"""
import sys, os, json
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotraining import ignoreLineCount, strip_type_marker, reformat_context, Config

HINT_TEST = "testing-relabelhint.txt"
_CFG = Config(contextFormat="bb")


_NONE = {"--NONE--", "other", "", None}


def _canon(name):
    """Canonicalize the no-autofill class: the dataset's fieldname column uses
    '--NONE--' while the model's id2label uses 'other'."""
    return "--NONE--" if name in _NONE else name


def _strip(name):
    return (name or "").replace("-", "")


def load_rows(test_file):
    """Return (texts, labels) from a bb test file."""
    texts, labels = [], []
    for l in open(test_file, encoding="utf-8"):
        if not l.strip():
            continue
        d = l.rstrip("\n").split(",", ignoreLineCount + 1)
        labels.append(int(d[ignoreLineCount]))
        texts.append(reformat_context(strip_type_marker(d[ignoreLineCount + 1], _CFG), "bb"))
    return texts, labels


def load_regex_and_gt():
    """Per test field: (gt_name, regex_hint_name) from testing-relabelhint.txt."""
    out = []
    for l in open(HINT_TEST, encoding="utf-8"):
        if not l.strip():
            continue
        d = l.rstrip("\n").split(",", ignoreLineCount + 1)
        gt = d[1]
        toks = d[ignoreLineCount + 1].split(" ")
        hint = ""  # leading **hint<x> = this field's own regex hint
        for t in toks:
            if t.startswith("**hint"):
                hint = t[len("**hint"):]
                break
            if t.startswith("aa") or t.startswith("bb"):
                break
        out.append((gt, hint))  # hint is hyphen-stripped class or "none"/""
    return out


def predict(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label = json.load(open(os.path.join(model_dir, "config.json")))["id2label"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=512,
                      padding=True, return_tensors="pt")
            preds += m(**enc).logits.argmax(-1).tolist()
    return [id2label[str(p)] for p in preds], id2label


def bucket_of(gt, hint):
    if not hint or hint == "none":
        return "silent"
    return "correct" if hint == _strip(gt) else "wrong"


def main():
    specs = []
    for a in sys.argv[1:]:
        tag, rest = a.split("=", 1)
        mdir, tfile = rest.split(":", 1)
        specs.append((tag, mdir, tfile))
    meta = load_regex_and_gt()
    gts = [_canon(g) for g, _ in meta]
    buckets = [bucket_of(g, h) for g, h in meta]
    bcount = defaultdict(int)
    for b in buckets:
        bcount[b] += 1
    print("field buckets (from regex vs GT):",
          {k: bcount[k] for k in ("correct", "wrong", "silent")}, "total", len(meta))

    results = {}
    for tag, mdir, tfile in specs:
        texts, labels = load_rows(tfile)
        assert len(texts) == len(meta), f"{tfile}: {len(texts)} rows != {len(meta)}"
        pred_names, _ = predict(mdir, texts)
        pred_names = [_canon(p) for p in pred_names]
        overall = np.mean([p == g for p, g in zip(pred_names, gts)])
        per_bucket = defaultdict(lambda: [0, 0])
        for p, g, b in zip(pred_names, gts, buckets):
            per_bucket[b][0] += int(p == g)
            per_bucket[b][1] += 1
        # postal-code + name focus (P/R)
        def pr(cls):
            tp = sum(1 for p, g in zip(pred_names, gts) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(pred_names, gts) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(pred_names, gts) if p != cls and g == cls)
            P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
            return P, R, tp, fp, fn
        results[tag] = (overall, per_bucket, pr)
        print(f"\n=== {tag} ({mdir}) ===")
        print(f"  overall {overall:.4f}")
        for b in ("correct", "wrong", "silent"):
            c, n = per_bucket[b]
            print(f"  bucket {b:8} acc {c/max(n,1):.3f} ({c}/{n})")
        for cls in ("postal-code", "name", "given-name", "family-name"):
            P, R, tp, fp, fn = pr(cls)
            print(f"  {cls:12} P={P:.3f} R={R:.3f} (tp{tp} fp{fp} fn{fn})")

    if len(results) > 1:
        print("\n=== DELTAS vs first tag (override health: 'wrong'/'silent' should rise) ===")
        base = specs[0][0]
        bo, bb_, bpr = results[base]
        for tag, *_ in specs[1:]:
            o, pb, _ = results[tag]
            row = [f"overallΔ {o-bo:+.4f}"]
            for b in ("correct", "wrong", "silent"):
                c, n = pb[b]; bc, bn = bb_[b]
                row.append(f"{b}Δ {c/max(n,1)-bc/max(bn,1):+.3f}")
            print(f"  {tag} vs {base}: " + "  ".join(row))


if __name__ == "__main__":
    main()
