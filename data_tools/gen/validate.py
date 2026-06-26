"""Taxonomy loading and the hard validation gate for generated samples.

Every generated HTML file must pass `validate_html` before it is written to the
dataset. The gate guarantees that synthetic data is indistinguishable, in format
and label correctness, from the hand-built real samples (see samples/training/at.html).

The 66 field types are the single source of truth in dotraining.py's
`fieldTypesDict`. We extract them by parsing the source with `ast` rather than
importing the module, because dotraining.py imports transformers/torch at module
load time (heavy, and unnecessary here).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from functools import lru_cache

from bs4 import BeautifulSoup

# The custom attribute that carries the ground-truth label inline in the HTML.
LABEL_ATTR = "data-moz-autofill-type"

# Sentinel returned by the LLM for fields that must NOT carry a label
# (passwords, submit buttons, search boxes, etc.).
NONFILL = "__nonfill__"

# Element kinds the LLM may request; maps to how render.py emits markup.
# These are the only values allowed in FieldSpec.element.
ELEMENTS = {
    "input_text",
    "input_email",
    "input_tel",
    "input_number",
    "select",
    "input_password",
    "button",
}

# Elements that are never autofillable and therefore must never be labeled.
NONFILL_ELEMENTS = {"input_password", "button"}

def _find_dotraining() -> str:
    """Locate dotraining.py (the taxonomy source). Searches this package's dir
    and the directories above it, so the tools work whether they live in the
    repo root or in a data_tools/ subdirectory next to the parent's dotraining.py.
    """
    here = os.path.dirname(os.path.abspath(__file__))      # .../gen
    candidates = [
        os.path.dirname(here),                              # data_tools/
        os.path.dirname(os.path.dirname(here)),             # parent repo (ml-form-autofill/)
    ]
    for d in candidates:
        path = os.path.join(d, "dotraining.py")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "dotraining.py not found (looked in "
        + ", ".join(candidates) + "). The taxonomy is read from it.")


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, int]:
    """Return fieldTypesDict from dotraining.py without importing the module.

    Parsing with ast avoids triggering dotraining.py's top-level transformers
    import while keeping dotraining.py the single source of truth.
    """
    dotraining = _find_dotraining()
    with open(dotraining, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=dotraining)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "fieldTypesDict":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict) or not value:
                        raise ValueError("fieldTypesDict did not parse to a dict")
                    return value
    raise ValueError(f"Could not find fieldTypesDict in {dotraining}")


@lru_cache(maxsize=1)
def valid_types() -> frozenset[str]:
    """The set of 66 valid autofill field-type strings."""
    return frozenset(load_taxonomy())


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:  # allows `if validate_html(...):`
        return self.ok


def _fail(reason: str) -> ValidationResult:
    return ValidationResult(False, reason)


OK = ValidationResult(True)


def validate_html(html: str, expected_label_count: int | None = None) -> ValidationResult:
    """The hard gate. Returns a falsy ValidationResult (with a reason) on any failure.

    Checks, in order:
      1. The document parses and has the required structural wrapper.
      2. Every data-moz-autofill-type value is one of the 66 valid types.
      3. Only <input>/<select> elements are labeled.
      4. Password/submit/search elements are never labeled.
      5. No label-text leakage / empty-document guards.
      6. (optional) labeled-field count matches what the spec intended.
    """
    if not html or not html.strip():
        return _fail("empty document")
    if LABEL_ATTR in html and "<form" not in html:
        return _fail("labels present but no <form>")

    soup = BeautifulSoup(html, "html.parser")

    if soup.find("form") is None:
        return _fail("missing <form>")
    # charset meta keeps generated files consistent with the real corpus.
    if soup.find("meta") is None:
        return _fail("missing <meta> charset")

    valid = valid_types()
    labeled = soup.select(f"[{LABEL_ATTR}]")
    for el in labeled:
        value = el.get(LABEL_ATTR, "")
        if value not in valid:
            return _fail(f"invalid label type: {value!r}")
        if el.name not in ("input", "select"):
            return _fail(f"label on non-input element <{el.name}>")
        if el.name == "input":
            input_type = (el.get("type") or "text").lower()
            if input_type in ("password", "submit", "button", "search", "hidden", "reset"):
                return _fail(f"label on non-autofillable input type={input_type}")

    # Passwords and submit buttons must exist sometimes, but never carry a label.
    for el in soup.find_all("input"):
        input_type = (el.get("type") or "text").lower()
        if input_type in ("password", "submit", "search") and el.has_attr(LABEL_ATTR):
            return _fail(f"input type={input_type} must not be labeled")

    if expected_label_count is not None and len(labeled) != expected_label_count:
        return _fail(
            f"labeled-field count {len(labeled)} != expected {expected_label_count}"
        )

    if len(labeled) == 0:
        return _fail("no labeled fields")

    return OK


def validate_form_spec(spec: dict) -> ValidationResult:
    """Validate the structured JSON returned by the LLM before rendering.

    This is defense-in-depth: the json_schema enum already constrains
    autofill_type, but we re-check here so a schema regression can't slip
    bad data through, and we enforce the non-fill element/label coupling.
    """
    fields = spec.get("fields")
    if not isinstance(fields, list) or not fields:
        return _fail("spec has no fields")

    valid = valid_types()
    seen_labels: list[str] = []
    labeled_count = 0
    for i, field in enumerate(fields):
        element = field.get("element")
        if element not in ELEMENTS:
            return _fail(f"field {i}: bad element {element!r}")

        autofill_type = field.get("autofill_type")
        is_nonfill = autofill_type == NONFILL
        if not is_nonfill and autofill_type not in valid:
            return _fail(f"field {i}: invalid autofill_type {autofill_type!r}")

        # Coupling: non-fill elements must be __nonfill__, and vice versa.
        if element in NONFILL_ELEMENTS and not is_nonfill:
            return _fail(f"field {i}: {element} must be __nonfill__")
        if element not in NONFILL_ELEMENTS and is_nonfill:
            # A text field marked nonfill is allowed (e.g. a search box), but a
            # nonfill text field with no purpose is just noise; permit it.
            pass

        label = (field.get("label_text") or "").strip()
        if not is_nonfill:
            labeled_count += 1
            if not label:
                return _fail(f"field {i}: empty label for {autofill_type}")
            if label == autofill_type:
                return _fail(f"field {i}: label equals type code {autofill_type!r}")
            if LABEL_ATTR in label or LABEL_ATTR in (field.get("placeholder_text") or ""):
                return _fail(f"field {i}: label leakage")
            seen_labels.append(label.lower())

    if labeled_count == 0:
        return _fail("spec has no labeled fields")
    # Guard against the model emitting the same label for everything.
    if len(seen_labels) >= 3 and len(set(seen_labels)) == 1:
        return _fail("all labels identical")

    return OK
