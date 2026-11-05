"""Re-encode a single-neighbor (aa/bb) dataset into the wide-context "~<offset>"
format at window=1, so context_format=attention can train on it WITHOUT a
Firefox wide-context rebuild. prev field -> "~-1<tok>", next -> "~1<tok>",
current field unprefixed. After parse_wide_context strips the "~-1"/"~1", the
per-offset buckets hold exactly the same clean tokens as triple's cur/prev/next,
so the pruned base vocab already covers them.

Usage: build_wide1.py <base-variant>   e.g. build_wide1.py -relabel
       -> training-relabelw1.txt / validation-relabelw1.txt / testing-relabelw1.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import split_context, ignoreLineCount  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "-relabel"


def rebuild(cur, prev, nxt):
    return " ".join(cur.split()
                    + ["~-1" + t for t in prev.split()]
                    + ["~1" + t for t in nxt.split()])


for split in ("training", "validation", "testing"):
    src = f"{split}{VARIANT}.txt"
    out = f"{split}{VARIANT}w1.txt"
    n = 0
    with open(src, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as w:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            d = line.split(",", ignoreLineCount + 1)
            if len(d) < ignoreLineCount + 2:
                continue
            head = d[: ignoreLineCount + 1]
            cur, prev, nxt = split_context(d[ignoreLineCount + 1])
            w.write(",".join(head + [rebuild(cur, prev, nxt)]) + "\n")
            n += 1
    print(f"{src} -> {out} ({n} rows)")
