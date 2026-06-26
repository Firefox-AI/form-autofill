"""Offline tests (no OpenAI API) for the generation pipeline.

Run with:  uv run python -m gen.test_offline
Exits non-zero on any failure.
"""

from __future__ import annotations

import glob
import os
import random

from gen.dedup import DedupIndex, signature_from_spec
from gen.params import sample_params
from gen.render import render_form
from gen.validate import (
    NONFILL,
    load_taxonomy,
    validate_form_spec,
    validate_html,
    valid_types,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO_ROOT, "samples")

_failures = 0


def check(cond: bool, msg: str) -> None:
    global _failures
    status = "PASS" if cond else "FAIL"
    if not cond:
        _failures += 1
    print(f"[{status}] {msg}")


def test_taxonomy():
    tax = load_taxonomy()
    check(len(tax) == 66, f"taxonomy has 66 types (got {len(tax)})")
    check("given-name" in tax and "cc-number" in tax, "key types present")


def test_existing_samples_pass_gate():
    """The gate must accept hand-built real samples — proves it matches reality.

    We test the clean hand-authored forms (small files). Full scraped pages can
    contain unlabeled noise inputs that are out of scope for the gate, so we
    focus on files under ~6KB which are the canonical clean shape we generate.
    """
    files = sorted(glob.glob(os.path.join(SAMPLES, "training", "*.html")))
    clean = [f for f in files if os.path.getsize(f) < 6000]
    check(len(clean) > 5, f"found {len(clean)} clean sample files to check")
    bad = []
    for path in clean:
        with open(path, encoding="utf-8", errors="replace") as fh:
            html = fh.read()
        # Some reference files (fields.html) have dup ids / odd shapes; only
        # require that any file WITH labeled fields parses and yields valid types.
        res = validate_html(html)
        if not res:
            bad.append((os.path.basename(path), res.reason))
    check(not bad, f"all clean samples pass the gate (failures: {bad[:5]})")


def test_render_roundtrip():
    """A synthetic spec renders to HTML that passes the gate and round-trips."""
    spec = {
        "locale": "de-DE",
        "site_domain": "shop.example.de",
        "purpose": "checkout",
        "fields": [
            {"autofill_type": "given-name", "label_text": "Vorname",
             "placeholder_text": None, "element": "input_text",
             "select_options": None},
            {"autofill_type": "family-name", "label_text": "Nachname",
             "placeholder_text": None, "element": "input_text",
             "select_options": None},
            {"autofill_type": "postal-code", "label_text": "PLZ",
             "placeholder_text": "z.B. 10115", "element": "input_text",
             "select_options": None},
            {"autofill_type": "country-name", "label_text": "Land",
             "placeholder_text": None, "element": "select",
             "select_options": ["Deutschland", "Österreich", "Schweiz"]},
            {"autofill_type": NONFILL, "label_text": "Passwort",
             "placeholder_text": None, "element": "input_password",
             "select_options": None},
            {"autofill_type": NONFILL, "label_text": "Absenden",
             "placeholder_text": None, "element": "button",
             "select_options": None},
        ],
    }
    check(bool(validate_form_spec(spec)), "sample spec passes spec validation")

    for style in ("span_before", "label_for", "placeholder_only", "aria_label",
                  "nested_label", "label_plus_placeholder", "mixed"):
        rng = random.Random(1)
        html = render_form(spec, markup_style=style, name_style="machine_colon",
                           rng=rng, note="test")
        res = validate_html(html, expected_label_count=4)
        check(bool(res), f"render style={style} passes gate ({res.reason})")
        check("data-moz-autofill-type=\"given-name\"" in html,
              f"style={style}: label attribute emitted")
        check('type="password"' in html and html.count("data-moz-autofill-type") == 4,
              f"style={style}: password NOT labeled (4 labels total)")


def test_bad_specs_rejected():
    bad_type = {
        "fields": [{"autofill_type": "phone-number", "label_text": "Phone",
                    "placeholder_text": None, "element": "input_tel",
                    "select_options": None}]
    }
    check(not validate_form_spec(bad_type), "invalid type rejected")

    nonfill_labeled = {
        "fields": [{"autofill_type": "given-name", "label_text": "x",
                    "placeholder_text": None, "element": "input_password",
                    "select_options": None}]
    }
    check(not validate_form_spec(nonfill_labeled),
          "password element with real type rejected")

    label_eq_code = {
        "fields": [{"autofill_type": "given-name", "label_text": "given-name",
                    "placeholder_text": None, "element": "input_text",
                    "select_options": None}]
    }
    check(not validate_form_spec(label_eq_code), "label==code rejected")


def test_dedup():
    spec = {"fields": [
        {"autofill_type": "email", "label_text": "Email"},
        {"autofill_type": "given-name", "label_text": "Name"},
    ]}
    idx = DedupIndex()
    sig = signature_from_spec(spec)
    check(idx.add_if_new(sig), "first signature is new")
    check(not idx.add_if_new(sig), "identical signature is a duplicate")


def test_sampling_deterministic():
    a = sample_params(7, 1234)
    b = sample_params(7, 1234)
    check(a.field_types == b.field_types and a.locale == b.locale,
          "sample_params is deterministic for (index, seed)")
    c = sample_params(8, 1234)
    check(c.field_types != a.field_types or c.locale != a.locale or c.purpose != a.purpose,
          "different index yields different params")
    # Every sampled field type must be a real taxonomy type.
    valid = valid_types()
    allgood = all(t in valid for i in range(50)
                  for t in sample_params(i, 99).field_types)
    check(allgood, "all sampled field types are valid taxonomy types")


if __name__ == "__main__":
    test_taxonomy()
    test_existing_samples_pass_gate()
    test_render_roundtrip()
    test_bad_specs_rejected()
    test_dedup()
    test_sampling_deterministic()
    print()
    if _failures:
        print(f"{_failures} check(s) FAILED")
        raise SystemExit(1)
    print("All checks passed.")
