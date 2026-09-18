#!/usr/bin/env python
"""Build a dotraining .txt dataset from labeled HTML forms (Python port).

Takes directories of labeled HTML (each fillable field annotated with
`data-moz-autofill-type`) and emits the CSV rows dotraining.py consumes:

    <filename>,<fieldname>,<label_id>,<mlData>

- filename  : source HTML basename (debug reference).
- fieldname : the field's data-moz-autofill-type, or "--NONE--" for a fillable
              field with no/unknown type (the "other" class).
- label_id  : fieldTypesDict[type], or 1 ("other") for --NONE--.
- mlData    : the aa/bb-baked context string, produced by ff_preprocess (a
              faithful port of Firefox FormAutofillHeuristics.tokenizeElements).

This reproduces the "HTML->txt processing" originally done by Firefox's
FormAutofillMLTest.sys.mjs (JS). `--type-marker training` keeps the literal
'**email'/'**select-one' input-type markers (matching the existing .txt and the
post-Bug-2063017 inference, which emits them verbatim); 'inference' strips them
via WORD_RE (the pre-fix behavior).

    python build_dataset.py --input data/testing --output /tmp/testing.txt
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_preprocess import fillable_fields, tokenize_elements

# Label normalization -- an exact copy of getAdjustedFieldName
# (FormAutofillMLTest.sys.mjs / form_autofill_accuracy.adjusted_field_name),
# which the Firefox-driven builder (build_dataset_ff.py) applies before writing
# the label column: a "supported" type is kept, else mapped to its nominal
# supported type, else "" -> the "--NONE--"/other negative class. Without this,
# e.g. a "floor" field (which FF labels address-line2) or "country-name"
# (-> country) would get the wrong label and diverge from the FF dataset.
SUPPORTED_FIELDS = {
    "given-name", "family-name", "name", "additional-name", "street-address",
    "address-line1", "address-line2", "address-line3", "address-level1",
    "address-level2", "address-level3", "address-housenumber",
    "address-extra-housesuffix", "postal-code", "country", "tel",
    "tel-country-code", "tel-national", "tel-area-code", "tel-local",
    "tel-local-prefix", "tel-local-suffix", "tel-extension", "organization",
    "bday", "bday-day", "bday-month", "bday-year", "email", "cc-name",
    "cc-given-name", "cc-additional-name", "cc-family-name", "cc-number",
    "cc-exp", "cc-exp-month", "cc-exp-year", "cc-csc", "cc-type", "sex",
}
NOMINAL_MAP = {
    "phonetic-given-name": "given-name", "phonetic-family-name": "family-name",
    "phonetic-name": "name", "address-lookup": "street-address",
    "street": "street-address", "address-streetname": "address-line1",
    "postal-code-lookup": "postal-code", "postal-code-and-city": "postal-code",
    "postal-code-or-suburb": "postal-code", "country-name": "country",
    "apartment": "address-line2", "floor": "address-line2",
    "stair": "address-line2", "building": "address-line2",
}


def adjusted_field_name(raw):
    if raw in SUPPORTED_FIELDS:
        return raw
    return NOMINAL_MAP.get(raw, "")


def load_field_types(dotraining_path):
    """Parse fieldTypesDict (name->id) from dotraining.py without importing it
    (avoids pulling in torch/transformers)."""
    src = open(dotraining_path, encoding="utf-8").read()
    m = re.search(r"fieldTypesDict\s*=\s*(\{.*?\n\})", src, re.S)
    if not m:
        raise SystemExit(f"fieldTypesDict not found in {dotraining_path}")
    return ast.literal_eval(m.group(1))


def build(paths, field_types, type_marker):
    """Yield '<file>,<fieldname>,<label>,<mlData>' rows for every fillable field."""
    other_id = field_types.get("other", 1)
    for path in paths:
        fname = os.path.basename(path)
        try:
            soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "html.parser")
        except Exception:
            continue
        fields = fillable_fields(soup)
        if not fields:
            continue
        ml = tokenize_elements(soup, type_marker=type_marker)
        for el, data in zip(fields, ml):
            raw = el.get("data-moz-autofill-type")
            adj = adjusted_field_name(raw) if raw else ""
            if adj and adj in field_types:
                name, label = adj, field_types[adj]
            else:
                name, label = "--NONE--", other_id
            yield f"{fname},{name},{label},{data}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="Directory(ies) of labeled *.html.")
    ap.add_argument("--output", required=True, help="Output .txt path.")
    ap.add_argument("--type-marker", choices=("training", "inference"), default="training",
                    help="'training' keeps literal '**<type>' markers (default, matches the "
                         "existing .txt); 'inference' strips them via WORD_RE.")
    ap.add_argument("--dotraining", default=os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    field_types = load_field_types(args.dotraining)
    import glob
    paths = []
    for d in args.input:
        paths += sorted(glob.glob(os.path.join(d, "*.html")))
    rows = list(build(paths, field_types, args.type_marker))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{len(paths)} html files -> {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
