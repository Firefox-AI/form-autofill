"""Post-process the -iaor dataset into two mlData-cleanup variants (no Firefox
regen needed), for A/B against the -iaor baseline:

  -iaornocommon : per form (>=3 fields), drop tokens that appear in EVERY field's
                  own section -- form-level boilerplate like "ctl00"/"form1" that
                  can't discriminate fields. Removed from own + neighbor sections.
                  Structural tokens ("**..." type/attr markers and "..." option
                  ranges) are protected and never stripped.
  -iaordedup    : within each field, keep only the first occurrence of each token
                  (collapses the id+name+label overlap, e.g. "name name" -> "name").

Rows are grouped by form (contiguous same-file rows). split_context is reused so
own/prev/next match dotraining exactly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import split_context, ignoreLineCount  # noqa: E402

# Base dataset variant to post-process (e.g. "-iaor", "-relabel"). Reads
# {split}{VARIANT}.txt, writes {split}{VARIANT}nocommon/dedup/both.txt.
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "-iaor"


def is_structural(tok):
    return tok.startswith("**") or "..." in tok


def dedup(tokens):
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def rebuild(cur, prev, nxt):
    return " ".join(cur + ["bb" + t for t in prev] + ["aa" + t for t in nxt])


def process(split):
    src = f"{split}{VARIANT}.txt"
    rows = [l.rstrip("\n") for l in open(src, encoding="utf-8") if l.strip()]
    # group contiguous rows by filename
    forms, cur_file, group = [], None, []
    for line in rows:
        fname = line.split(",", 1)[0]
        if fname != cur_file and cur_file is not None:
            forms.append(group)
            group = []
        cur_file = fname
        group.append(line)
    if group:
        forms.append(group)

    out_nc, out_dd, out_both = [], [], []
    tok_before = tok_after_nc = tok_after_dd = tok_after_both = 0
    forms_stripped = 0

    for group in forms:
        parsed = []  # (prefix3cols, cur[], prev[], nxt[])
        for line in group:
            d = line.split(",", ignoreLineCount + 1)
            head = d[: ignoreLineCount + 1]
            cur, prev, nxt = split_context(d[ignoreLineCount + 1])
            parsed.append((head, cur.split(), prev.split(), nxt.split()))

        # common tokens across all fields' own sections (>=3 fields), non-structural
        common = set()
        if len(parsed) >= 3:
            own_sets = [
                {t for t in cur if not is_structural(t)} for _h, cur, _p, _n in parsed
            ]
            common = set.intersection(*own_sets) if own_sets else set()
            if common:
                forms_stripped += 1

        for head, cur, prev, nxt in parsed:
            tok_before += len(cur) + len(prev) + len(nxt)
            # nocommon: drop the common set from every section
            nc = rebuild(
                [t for t in cur if t not in common],
                [t for t in prev if t not in common],
                [t for t in nxt if t not in common],
            )
            # dedup: per-section unique (each section is one field's own tokens)
            dd = rebuild(dedup(cur), dedup(prev), dedup(nxt))
            # both: drop common set, then dedup
            both = rebuild(
                dedup([t for t in cur if t not in common]),
                dedup([t for t in prev if t not in common]),
                dedup([t for t in nxt if t not in common]),
            )
            tok_after_nc += len(nc.split())
            tok_after_dd += len(dd.split())
            tok_after_both += len(both.split())
            out_nc.append(",".join(head + [nc]))
            out_dd.append(",".join(head + [dd]))
            out_both.append(",".join(head + [both]))

    open(f"{split}{VARIANT}nocommon.txt", "w", encoding="utf-8").write("\n".join(out_nc) + "\n")
    open(f"{split}{VARIANT}dedup.txt", "w", encoding="utf-8").write("\n".join(out_dd) + "\n")
    open(f"{split}{VARIANT}both.txt", "w", encoding="utf-8").write("\n".join(out_both) + "\n")
    print(f"{split}: {len(forms)} forms, {len(rows)} rows | tokens {tok_before} -> "
          f"nocommon {tok_after_nc} ({100*(tok_before-tok_after_nc)/tok_before:.1f}% cut, "
          f"{forms_stripped} forms stripped) | dedup {tok_after_dd} "
          f"({100*(tok_before-tok_after_dd)/tok_before:.1f}% cut) | both {tok_after_both} "
          f"({100*(tok_before-tok_after_both)/tok_before:.1f}% cut)")


for split in ("training", "validation", "testing"):
    process(split)
