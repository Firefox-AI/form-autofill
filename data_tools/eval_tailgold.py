"""Evaluate model(s) on the held-out TAIL-language gold eval set
(testing-tailgold.txt / testing-tailgold-hint.txt), built by build_weak_dataset
from o4-audited disagreement fields in the mined CC forms. Rows look like:
  TAILGOLD_<lang>_<si>_<ri>,<label>,<label_id>,<mldata>
Reports per-language and aggregate accuracy for each model. Point a hint model at
its hint val with tag=dir@testing-tailgold-hint.txt.

Usage:
  uv run python data_tools/eval_tailgold.py --val testing-tailgold.txt \
      base=/tmp/labeler_889f8 n984f=/tmp/m_n984f
"""
import sys, os, argparse
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotraining import ignoreLineCount, strip_type_marker, reformat_context, Config

_CFG = Config(contextFormat="bb")


def load_rows(val_file):
    rows = []
    for l in open(val_file, encoding="utf-8"):
        if not l.strip():
            continue
        fn = l.split(",", 1)[0]
        lang = fn.split("_")[1] if fn.startswith("TAILGOLD_") else "??"
        d = l.rstrip("\n").split(",", ignoreLineCount + 1)
        rows.append((lang, int(d[ignoreLineCount]),
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
    ap.add_argument("--val", default="testing-tailgold.txt")
    ap.add_argument("models", nargs="+", help="tag=model_dir[@val_file]; first = baseline")
    args = ap.parse_args()
    specs = []
    for s in args.models:
        tag, rest = s.split("=", 1)
        mdir, _, vf = rest.partition("@")
        specs.append((tag, mdir, vf or args.val))
    tags = [t for t, _, _ in specs]
    st = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    ns = {}
    for tag, mdir, vf in specs:
        rows = load_rows(vf)
        preds = predict(mdir, [r[2] for r in rows])
        ns[tag] = len(rows)
        for (lc, g, _), p in zip(rows, preds):
            st[lc][tag][0] += p == g
            st[lc][tag][1] += 1
    print(f"tail-gold rows per model: {ns}")
    hdr = f"{'lang':5}{'n':>4}" + "".join(f"{t:>10}" for t in tags)
    if len(tags) > 1:
        hdr += f"{'Δ':>6}"
    print(hdr)
    for lc in sorted(st, key=lambda k: -st[k][tags[0]][1]):
        n = st[lc][tags[0]][1]
        cells = "".join(f"{100*st[lc][t][0]/n:9.0f}%" for t in tags)
        line = f"{lc:5}{n:4}{cells}"
        if len(tags) > 1:
            d = 100 * (st[lc][tags[-1]][0] - st[lc][tags[0]][0]) / n
            line += f"{d:+5.0f}"
        print(line)

    def agg(tag):
        c = sum(st[l][tag][0] for l in st); n = sum(st[l][tag][1] for l in st)
        return 100 * c / max(n, 1), n
    parts = []
    for t in tags:
        a, n = agg(t); parts.append(f"{t} {a:.1f}")
    d = agg(tags[-1])[0] - agg(tags[0])[0] if len(tags) > 1 else 0
    print(f"{'ALL':5}n={agg(tags[0])[1]:4}  " + "  ".join(parts) + (f"  Δ{d:+.1f}" if len(tags) > 1 else ""))


if __name__ == "__main__":
    main()
