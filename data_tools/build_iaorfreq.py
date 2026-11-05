"""Post-process -iaor into -iaorfreq: the strip_frequent cleanup (drop tokens
present in >=85% of a form's fields, for forms with >=3 fields), mirroring the
FF _stripSharedTokens logic. No FF regen needed (deterministic on -iaor tokens).
Structural markers (**..., a...b option ranges) are never counted or removed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import split_context, ignoreLineCount  # noqa: E402

MIN_FIELDS = 3
FRACTION = 0.85


def is_structural(tok):
    return tok.startswith("**") or "..." in tok


def rebuild(cur, prev, nxt):
    return " ".join(cur + ["bb" + t for t in prev] + ["aa" + t for t in nxt])


for split in ("training", "validation", "testing"):
    rows = [l.rstrip("\n") for l in open(f"{split}-iaor.txt", encoding="utf-8") if l.strip()]
    forms, cur_file, group = [], None, []
    for line in rows:
        fname = line.split(",", 1)[0]
        if fname != cur_file and cur_file is not None:
            forms.append(group); group = []
        cur_file = fname
        group.append(line)
    if group:
        forms.append(group)

    out = []
    tok_before = tok_after = 0
    forms_stripped = 0
    for group in forms:
        parsed = []
        for line in group:
            d = line.split(",", ignoreLineCount + 1)
            cur, prev, nxt = split_context(d[ignoreLineCount + 1])
            parsed.append((d[: ignoreLineCount + 1], cur.split(), prev.split(), nxt.split()))

        strip = set()
        if len(parsed) >= MIN_FIELDS:
            count = {}
            for _h, cur, _p, _n in parsed:
                for t in {t for t in cur if not is_structural(t)}:
                    count[t] = count.get(t, 0) + 1
            cutoff = FRACTION * len(parsed)
            strip = {t for t, c in count.items() if c >= cutoff}
            if strip:
                forms_stripped += 1

        for head, cur, prev, nxt in parsed:
            tok_before += len(cur) + len(prev) + len(nxt)
            line = rebuild(
                [t for t in cur if t not in strip],
                [t for t in prev if t not in strip],
                [t for t in nxt if t not in strip],
            )
            tok_after += len(line.split())
            out.append(",".join(head + [line]))

    open(f"{split}-iaorfreq.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"{split}: {len(forms)} forms ({forms_stripped} stripped), {len(rows)} rows | "
          f"tokens {tok_before} -> {tok_after} ({100*(tok_before-tok_after)/tok_before:.1f}% cut)")
