#!/usr/bin/env bash
# Build the RELABELED dataset (o4-mini name/label corrections applied to cc+gen,
# plus the new 2026 multilingual batch merged into data/common_crawl) via the real
# Firefox scanner (ml_driver). Feature config = the -iaor winner:
# select_option + input_attributes (source_tags dropped, was neutral-negative).
# Same corpus layout + frozen validation manifest as -iaor. --resume-safe.
set -euo pipefail
cd /Users/mozilla/Documents/GitHub/ml-form-autofill

# Firefox ML build: override with FIREFOX_BIN, else auto-pick the newest objdir.
FF_TREE=/Users/mozilla/Documents/GitHub/firefox
if [ -z "${FIREFOX_BIN:-}" ]; then
  FIREFOX_BIN=$(ls -td "$FF_TREE"/obj-*/dist/Nightly.app/Contents/MacOS/firefox 2>/dev/null | head -1)
fi
export FIREFOX_BIN
[ -x "$FIREFOX_BIN" ] || { echo "MISSING Firefox binary: $FIREFOX_BIN"; exit 1; }
echo "FIREFOX_BIN=$FIREFOX_BIN"
export MOZ_ALLOW_EXTERNAL_ML_HUB=1
export FASTLY_TOKEN=dummy
export FF_EXTRA_PREFS='{"extensions.formautofill.useml.features": "[\"select_option\", \"input_attributes\"]"}'
ML_DRIVER=/Users/mozilla/Documents/GitHub/ml_driver
# build_dataset_ff.py imports form_autofill_accuracy (examples/) and ml_driver (src/).
export PYTHONPATH="$ML_DRIVER/examples:$ML_DRIVER/src${PYTHONPATH:+:$PYTHONPATH}"
PY=$ML_DRIVER/.venv/bin/python
MANIFEST="datasets/v4_window3/split/validation_files.txt"
[ -f "$MANIFEST" ] || { echo "MISSING manifest $MANIFEST"; exit 1; }

echo "### [1/3] train pool: data/training + data/generated + data/common_crawl (merged 2026)"
$PY data_tools/build_dataset_ff.py \
  --input data/training data/generated data/common_crawl \
  --output trainpool-relabel.txt --resume

echo "### [2/3] testing: data/testing"
$PY data_tools/build_dataset_ff.py \
  --input data/testing \
  --output testing-relabel.txt --resume

echo "### [3/3] split train pool by frozen validation manifest"
$PY - <<PYEOF
val = {l.strip() for l in open("$MANIFEST", encoding="utf-8") if l.strip()}
ntr = nva = 0
with open("trainpool-relabel.txt", encoding="utf-8") as f, \
     open("training-relabel.txt", "w", encoding="utf-8") as tr, \
     open("validation-relabel.txt", "w", encoding="utf-8") as va:
    for line in f:
        if not line.strip():
            continue
        (va if line.split(",", 1)[0] in val else tr).write(line)
        if line.split(",", 1)[0] in val: nva += 1
        else: ntr += 1
print(f"train {ntr}  val {nva}")
PYEOF

echo "### DONE"
wc -l training-relabel.txt validation-relabel.txt testing-relabel.txt
