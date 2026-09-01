#!/usr/bin/env bash
# Regenerate the -iaor dataset variant (inputAttributes + selectOptionRange, with
# blank-option skipping; legacy ±1 bb/aa context) via the real Firefox scanner,
# then split the train pool by the frozen manifest. Idempotent: --resume skips
# files already written, so re-running after a crash continues where it stopped.
set -euo pipefail
cd /Users/Rrando/Documents/GitHub/ml-form-autofill

export FIREFOX_BIN=/Users/Rrando/Documents/GitHub/firefox/obj-aarch64-apple-darwin25.6.0/dist/Nightly.app/Contents/MacOS/firefox
export MOZ_ALLOW_EXTERNAL_ML_HUB=1
export FASTLY_TOKEN=dummy
export FF_EXTRA_PREFS='{"extensions.formautofill.useml.inputAttributes": true, "extensions.formautofill.useml.selectOptionRange": true, "extensions.formautofill.useml.contextWindow": 1}'
PY=/Users/Rrando/Documents/GitHub/ml_driver/.venv/bin/python

echo "### [1/3] train pool: data/training + data/generated + data/common_crawl"
$PY data_tools/build_dataset_ff.py \
  --input data/training data/generated data/common_crawl \
  --output trainpool-iaor.txt --resume

echo "### [2/3] testing: data/testing"
$PY data_tools/build_dataset_ff.py \
  --input data/testing \
  --output testing-iaor.txt --resume

echo "### [3/3] split train pool by frozen manifest"
$PY data_tools/split_iaor.py

echo "### DONE"
wc -l trainpool-iaor.txt training-iaor.txt validation-iaor.txt testing-iaor.txt
