"""Python port of Firefox's ML field-context preprocessing.

Faithfully reproduces FormAutofillHeuristics.sys.mjs (Bug 2035232 / D305753):
turn a form's fields into the per-field `mlData` context string the autofill
model consumes -- lowercased, camelCase-split, word-tokenized (words >=3 chars),
with an input-type marker and `bb`(previous)/`aa`(next) neighbor baking.

Ported 1:1 from toolkit/components/formautofill/shared/FormAutofillHeuristics.sys.mjs:
  WORD_RE = /\\s*([\\p{L}\\p{N}]+)/u          (letters/digits only)
  ADJACENT_BEFORE_PREFIX = "bb"  (previous field)
  ADJACENT_AFTER_PREFIX  = "aa"  (next field)
  tokenizeWords / splitMixedCase / tokenizeAttributes / tokenizeElements

Use split_context() (dotraining) on the result to get the three bare strings for
the 'triple' format.

NOTE (train/inference skew): the shipped inference routes the "**"+type marker
through tokenizeWords, whose WORD_RE strips the "**" -> a plain type word
("**email" -> "email"). The training .txt files, however, contain literal
"**email"/"**select-one" tokens, so the generator that produced them preserved
the marker. `type_marker` selects which behavior to reproduce:
  "inference" (default) -> matches the DEPLOYED model's actual input
  "training"            -> matches the tokens in the training .txt
"""
from __future__ import annotations

import regex  # \p{...} unicode classes, matching the JS /u regex

WORD_RE = regex.compile(r"[\p{L}\p{N}]+")
_MIXED_RE = regex.compile(r"([\p{Ll}\p{N}]*)(\p{Lu}*)")
ADJACENT_BEFORE_PREFIX = "bb"
ADJACENT_AFTER_PREFIX = "aa"
_NONFIELD_TYPES = {"hidden", "submit", "image", "button", "reset"}


def tokenize_words(text: str, words: list[str]) -> None:
    """Lowercase, split into letter/digit runs, keep words with >=3 chars."""
    if not text:
        return
    text = text.lower()
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if len(w) >= 3:            # FF: "Ignore short words"
            words.append(w)


def split_mixed_case(text: str) -> str:
    """Insert a space between a lower/digit run and an Upper run (addressLine
    -> address Line). FF applies this to id and name only."""
    if not text:
        return text
    return _MIXED_RE.sub(r"\1 \2", text)


def element_type(el) -> str:
    if el.name == "select":
        return "select-multiple" if el.has_attr("multiple") else "select-one"
    if el.name == "textarea":
        return "textarea"
    return (el.get("type") or "text").lower()


def get_label_strings(el, soup) -> list[str]:
    """Approximate FormAutofillHeuristics._getElementLabelStrings: explicit
    <label for>/ancestor <label>, else nearby preceding text, plus aria-label."""
    out: list[str] = []
    fid = el.get("id")
    found = False
    if fid:
        for lab in soup.find_all("label", attrs={"for": fid}):
            t = lab.get_text(" ", strip=True)
            if t:
                out.append(t); found = True
    parent = el.find_parent("label")
    if parent:
        t = parent.get_text(" ", strip=True)
        if t:
            out.append(t); found = True
    if not found:
        prev = el.find_previous(["label", "span", "td", "th", "p"])
        if prev:
            t = prev.get_text(" ", strip=True)
            if t:
                out.append(t[:120])
    aria = el.get("aria-label")
    if aria:
        out.append(aria)
    return out


def tokenize_attributes(el, soup, type_marker: str = "inference") -> list[str]:
    """FF tokenizeAttributes: id, name (both split-mixed-case), placeholder,
    labels, then the input-type marker for non-text fields."""
    words: list[str] = []
    tokenize_words(split_mixed_case(el.get("id", "")), words)
    tokenize_words(split_mixed_case(el.get("name", "")), words)
    tokenize_words(el.get("placeholder", ""), words)
    for label in get_label_strings(el, soup):
        tokenize_words(label, words)
    etype = element_type(el)
    if etype != "text":
        if type_marker == "training":
            words.append("**" + etype)          # literal marker, as in the .txt
        else:
            tokenize_words("**" + etype, words)  # inference: WORD_RE strips "**"
    return words


def fillable_fields(soup) -> list:
    """Fields counted for the size filter / fed to the model, in document
    order (text-like inputs, selects, textareas; excludes hidden/submit/etc)."""
    out = []
    for el in soup.find_all(["input", "select", "textarea"]):
        if el.name == "input" and (el.get("type") or "text").lower() in _NONFIELD_TYPES:
            continue
        out.append(el)
    return out


def tokenize_elements(soup, type_marker: str = "inference") -> list[str]:
    """FF tokenizeElements: each field's own words + bb-prefixed previous +
    aa-prefixed next, whitespace-joined. Returns one mlData string per field,
    in document order."""
    fields = fillable_fields(soup)
    per_field_words = [tokenize_attributes(el, soup, type_marker) for el in fields]
    out = []
    for i, words in enumerate(per_field_words):
        combined = list(words)
        if i > 0:
            combined += [ADJACENT_BEFORE_PREFIX + w for w in per_field_words[i - 1]]
        if i < len(per_field_words) - 1:
            combined += [ADJACENT_AFTER_PREFIX + w for w in per_field_words[i + 1]]
        out.append(" ".join(combined))
    return out
