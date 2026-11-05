"""Tail-gold eval for TRIPLE-encoder models (TripleEncoderForSequenceClassification),
which can't load through AutoModel. Reuses dotraining._triple_classify. Same
tail-gold files/format as eval_tailgold.py. Point a token-hint triple at
testing-tailgold-hint.txt (the **hint token stays in the section text when the
model was trained with hint_embed off, i.e. config.hint_classes is 0/None).

Usage:
  uv run python data_tools/eval_tailgold_triple.py \
      kbns9=/tmp/m_kbns9 \
      triple_hint=/tmp/m_413980@testing-tailgold-hint.txt
"""
import sys, os, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotraining import ignoreLineCount, strip_type_marker, Config, _triple_classify

_CFG = Config(contextFormat="triple")


def load_rows(val_file):
    rows = []
    for l in open(val_file, encoding="utf-8"):
        if not l.strip():
            continue
        fn = l.split(",", 1)[0]
        lang = fn.split("_")[1] if fn.startswith("TAILGOLD_") else "??"
        d = l.rstrip("\n").split(",", ignoreLineCount + 1)
        # strip the type marker; split_context (inside _triple_classify) handles aa/bb.
        rows.append((lang, int(d[ignoreLineCount]),
                     strip_type_marker(d[ignoreLineCount + 1], _CFG)))
    return rows


class _Cfg:
    """_triple_classify only reads .modelName (tokenizer) and .saveModelDir (model)."""
    def __init__(self, mdir):
        self.modelName = mdir
        self.saveModelDir = mdir


def predict(mdir, texts):
    return [r["label"] for r in _triple_classify(texts, _Cfg(mdir))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="testing-tailgold.txt")
    ap.add_argument("models", nargs="+", help="tag=model_dir[@val_file]; first = baseline")
    args = ap.parse_args()
    # map label string -> id from the first model's config, to score _triple_classify
    # (it returns id2label strings; compare label strings, not ids, across models)
    import json
    specs = []
    for s in args.models:
        tag, rest = s.split("=", 1)
        mdir, _, vf = rest.partition("@")
        specs.append((tag, mdir, vf or args.val))
    tags = [t for t, _, _ in specs]
    st = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    ns = {}
    for tag, mdir, vf in specs:
        id2label = {int(k): v for k, v in json.load(open(os.path.join(mdir, "config.json")))["id2label"].items()}
        rows = load_rows(vf)
        preds = predict(mdir, [r[2] for r in rows])   # preds are label STRINGS
        ns[tag] = len(rows)
        for (lc, g, _), p in zip(rows, preds):
            st[lc][tag][0] += (id2label.get(g) == p)
            st[lc][tag][1] += 1
    print(f"tail-gold rows per model: {ns}")
    hdr = f"{'lang':5}{'n':>5}" + "".join(f"{t:>14}" for t in tags)
    print(hdr)
    for lc in sorted(st, key=lambda k: -st[k][tags[0]][1]):
        n = st[lc][tags[0]][1]
        cells = "".join(f"{100*st[lc][t][0]/n:13.0f}%" for t in tags)
        print(f"{lc:5}{n:5}{cells}")

    def agg(tag):
        c = sum(st[l][tag][0] for l in st); n = sum(st[l][tag][1] for l in st)
        return 100 * c / max(n, 1), n
    parts = [f"{t} {agg(t)[0]:.1f}" for t in tags]
    print(f"{'ALL':5}{agg(tags[0])[1]:5}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
