"""Build the -iaorctl control variant from the -iaor dataset by removing ONLY
the new tokens (input-attribute + option-range), keeping everything else. This
gives a field-for-field-identical baseline so an -iaor vs -iaorctl A/B isolates
exactly the token additions (same corpus, same split, same FF field selection).

Stripped (after removing any leading aa/bb neighbor prefix):
  - **maxlen<N>   (input maxlength)
  - **inputmode<mode>
  - <first>...<last>  (select option range; any token containing "...")
Kept: pre-existing **<type> markers (**tel, **select-one, ...) and all words.
"""
import re

PREFIX = re.compile(r"^(aa|bb)")


def is_new(tok):
    base = PREFIX.sub("", tok)
    return (
        base.startswith("**maxlen")
        or base.startswith("**inputmode")
        or "..." in base
    )


for split in ("training", "validation", "testing"):
    src = f"{split}-iaor.txt"
    dst = f"{split}-iaorctl.txt"
    nrows = ntok = nstrip = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                out.write(line)
                continue
            toks = parts[3].split()
            kept = [t for t in toks if not is_new(t)]
            ntok += len(toks)
            nstrip += len(toks) - len(kept)
            parts[3] = " ".join(kept)
            out.write(",".join(parts) + "\n")
            nrows += 1
    print(f"{dst}: {nrows} rows, stripped {nstrip}/{ntok} tokens")
