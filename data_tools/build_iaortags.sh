#!/usr/bin/env bash
# Regenerate the -iaortags dataset: select_option + input_attributes + source_tags
# (machine id/name tokens prefixed "attr"). Same corpus/split as -iaor so it's a
# clean A/B for the source_tags feature. Uses the firefox-clone build + updated
# ml_driver. --resume-safe (skip-and-restart on per-file errors).
set -euo pipefail
cd /Users/Rrando/Documents/GitHub/ml-form-autofill

export FIREFOX_BIN=/Users/Rrando/Documents/GitHub/firefox-clone/firefox/obj-aarch64-apple-darwin25.6.0/dist/Nightly.app/Contents/MacOS/firefox
export MOZ_ALLOW_EXTERNAL_ML_HUB=1
export FASTLY_TOKEN=dummy
export FF_EXTRA_PREFS='{"extensions.formautofill.useml.features": "[\"source_tags\", \"select_option\", \"input_attributes\"]"}'
PY=/Users/Rrando/Documents/GitHub/ml_driver/.venv/bin/python

echo "### [1/3] train pool: data/training + data/generated + data/common_crawl"
$PY data_tools/build_dataset_ff.py \
  --input data/training data/generated data/common_crawl \
  --output trainpool-iaortags.txt --resume

echo "### [2/3] testing: data/testing"
$PY data_tools/build_dataset_ff.py \
  --input data/testing \
  --output testing-iaortags.txt --resume

echo "### [3/3] split train pool by frozen manifest"
$PY - <<'PY'
MANIFEST = "datasets/v4_window3/split/validation_files.txt"
val = {l.strip() for l in open(MANIFEST, encoding="utf-8") if l.strip()}
ntr = nva = 0
with open("trainpool-iaortags.txt", encoding="utf-8") as f, \
     open("training-iaortags.txt", "w", encoding="utf-8") as tr, \
     open("validation-iaortags.txt", "w", encoding="utf-8") as va:
    for line in f:
        if not line.strip():
            continue
        if line.split(",", 1)[0] in val:
            va.write(line); nva += 1
        else:
            tr.write(line); ntr += 1
print(f"train {ntr}  val {nva}")
PY

echo "### DONE"
wc -l training-iaortags.txt validation-iaortags.txt testing-iaortags.txt
