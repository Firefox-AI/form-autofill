"""Split the regenerated train-pool mlData into training/validation using the
frozen file-level manifest (seed 20260824), so the -iaor variant stays directly
comparable to the existing -supported runs. Testing is generated separately."""
import sys

MANIFEST = "datasets/v4_window3/split/validation_files.txt"  # canonical 562-file split
POOL = "trainpool-iaor.txt"
TRAIN_OUT = "training-iaor.txt"
VAL_OUT = "validation-iaor.txt"

val_files = {l.strip() for l in open(MANIFEST, encoding="utf-8") if l.strip()}
ntr = nva = 0
seen_val = set()
with open(POOL, encoding="utf-8") as f, \
     open(TRAIN_OUT, "w", encoding="utf-8") as tr, \
     open(VAL_OUT, "w", encoding="utf-8") as va:
    for line in f:
        if not line.strip():
            continue
        base = line.split(",", 1)[0]
        if base in val_files:
            va.write(line); nva += 1; seen_val.add(base)
        else:
            tr.write(line); ntr += 1
missing = val_files - seen_val
print(f"train rows={ntr}  val rows={nva}  (val files matched {len(seen_val)}/{len(val_files)})")
if missing:
    print(f"WARNING: {len(missing)} manifest files not found in pool, e.g. {sorted(missing)[:5]}",
          file=sys.stderr)
