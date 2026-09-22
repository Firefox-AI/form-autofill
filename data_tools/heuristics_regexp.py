"""Python port of Firefox's regex field-name heuristic (the "regex-heuristic"
prediction, i.e. the control / useMl=false field name).

The composed rule sources are dumped verbatim from Firefox's own
HeuristicsRegExp.getRules()/getLabelRules() (branch fr_perf) into
heuristics_data/rules.json via Node, so the patterns are guaranteed identical to
what ships. This module reproduces the matcher:
  FormAutofillHeuristics._getPossibleFieldNames / _findMatchedFieldNames /
  _matchRegexp / testRegex   (control mode: useML=false -> all field names).

regex_hint(el, soup) -> the primary regex-recommended field name for an element
(or "" when the regex is silent). Field-name normalization (adjusted_field_name)
is applied by the caller, matching the ML label pipeline.
"""
from __future__ import annotations

import json
import os
import unicodedata

import regex

from ff_preprocess import get_label_strings  # FF _getElementLabelStrings port

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heuristics_data", "rules.json")
with open(_DATA, encoding="utf-8") as _f:
    _RAW = json.load(_f)

# Compile once. The `regex` module (unlike stdlib `re`) permits duplicate group
# names, which the composed rules contain (multiple "(?<neg>...)" across rulesets).
def _compile(src):
    return regex.compile(src)

RULES = {k: _compile(v) for k, v in _RAW["rules"].items()}
LABEL_RULES = {k: _compile(v) for k, v in _RAW["labelRules"].items()}
RULES_ORDER = _RAW["rulesOrder"]

# isCreditCardField / isPassportField split (FormAutofillUtils).
_CC_FIELDS = {
    "cc-name", "cc-csc", "cc-number", "cc-exp-month", "cc-exp-year", "cc-exp",
    "cc-type", "cc-given-name", "cc-additional-name", "cc-family-name",
}
def _is_cc(name):
    return name.startswith("cc-")
def _is_passport(name):
    return name.startswith("passport-")

# Candidate lists in RULES key order (FormAutofillHeuristics lazy getters).
CREDIT_CARD_FIELDNAMES = [n for n in RULES_ORDER if _is_cc(n)]
ADDRESS_FIELDNAMES = [n for n in RULES_ORDER if not _is_cc(n) and not _is_passport(n)]

CC_TYPE, ADDR_TYPE = "cc", "addr"

# _getPossibleFieldNames type filters.
_FIELDNAMES_FOR_SEARCH = {
    "address-level1", "address-level2", "address-line1", "address-line2",
    "address-line3", "street-address", "postal-code",
}
_FIELDNAMES_FOR_SELECT = {
    "address-level1", "address-level2", "country", "cc-exp-month",
    "cc-exp-year", "cc-exp", "cc-type", "tel-country-code",
}
_FIELDNAMES_FOR_TEXTAREA = {"street-address"}


def _element_type(el):
    if el.name == "select":
        return "select"
    if el.name == "textarea":
        return "textarea"
    return (el.get("type") or "text").strip().lower()


def get_element_strings(el):
    """FF _getElementStrings: [id, name, placeholder.trim()]."""
    ph = el.get("placeholder")
    return [el.get("id"), el.get("name"), ph.strip() if ph else ph]


def get_possible_field_names(el):
    """FF _getPossibleFieldNames (control mode: autocomplete stripped, so both
    CC and address candidates, then filtered by element type)."""
    names = list(CREDIT_CARD_FIELDNAMES) + list(ADDRESS_FIELDNAMES)
    t = _element_type(el)
    if el.name == "input" and t == "search":
        names = [n for n in names if n in _FIELDNAMES_FOR_SEARCH]
    elif el.name == "select":
        names = [n for n in names if n in _FIELDNAMES_FOR_SELECT]
    elif el.name == "textarea":
        names = [n for n in names if n in _FIELDNAMES_FOR_TEXTAREA]
    return names


def test_regex(compiled, s):
    """FF testRegex: a match counts unless the only thing it captured is the
    negative-lookbehind stand-in group 'neg' (the (?<neg>...) workaround)."""
    if not s:
        return False
    for m in compiled.finditer(s):
        try:
            negs = set(m.capturesdict().get("neg", []))
        except Exception:
            negs = set()
        parts = [m.group(0)] + list(m.groups())
        if any(p and p not in negs for p in parts):
            return True
    return False


def _match_regexp(el, compiled, soup, attribute=True, label=True):
    if compiled is None:
        return False
    if attribute:
        for s in get_element_strings(el):
            if s and test_regex(compiled, s.lower()):
                return True
    if label:
        for s in get_label_strings(el, soup):
            if s and test_regex(compiled, s.lower()):
                return True
    return False


def find_matched_field_names(el, field_names, soup):
    """FF _findMatchedFieldNames: try RULES (attributes+labels) then LABEL_RULES
    (labels only), in candidate order, collecting up to one addr + one cc name."""
    if not field_names:
        return []
    fields = [(n, CC_TYPE if _is_cc(n) else ADDR_TYPE) for n in field_names]
    matched = []
    found_type = ""
    attribute = True
    for rules in (RULES, LABEL_RULES):
        hit_two = False
        for name, typ in fields:
            if found_type == typ:
                continue
            if name not in rules:
                continue
            if not _match_regexp(el, rules[name], soup, attribute=attribute):
                continue
            found_type = typ
            matched.append(name)
            if len(matched) == 2:
                hit_two = True
                break
        if hit_two:
            break
        attribute = False  # LABEL_RULES pass: labels only
    return matched


def regex_hint(el, soup):
    """Primary regex-heuristic field name for an element ("" if none)."""
    matched = find_matched_field_names(el, get_possible_field_names(el), soup)
    return matched[0] if matched else ""
