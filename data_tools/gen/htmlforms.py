"""Parse real form HTML, extract labelable fields with context, inject labels.

Used by label_common_crawl.py to turn scraped Common Crawl forms into
ground-truth training samples. The LLM classifies each candidate field; this
module owns the DOM work: finding the field's visible label, counting fields
for the size filter, and writing the data-moz-autofill-type attribute back onto
the real element so the output matches the existing inline-label format.
"""

from __future__ import annotations

from urllib.parse import urlparse

from bs4 import BeautifulSoup

from gen.validate import LABEL_ATTR

# Input types we never label (not personal-info text fields). Counted for the
# size filter only if they're real fields (see fillable_count).
_NONFIELD_TYPES = {"hidden", "submit", "image", "button", "reset"}

# Input types we CAN attach a label to. validate_html only permits labels on
# <input>/<select>, and rejects labeled password/submit/search/etc., so we keep
# injection to genuinely textual inputs plus <select>.
_INJECTABLE_INPUT_TYPES = {"text", "email", "tel", "number", "url", "date", ""}


def _input_type(el) -> str:
    return (el.get("type") or "text").lower()


def fillable_count(soup) -> int:
    """Count 'real' user-fillable fields (for the size filter).

    Includes text-like inputs, selects, and textareas; excludes hidden/submit/
    image/button/reset.
    """
    n = 0
    for el in soup.find_all(["input", "select", "textarea"]):
        if el.name == "input" and _input_type(el) in _NONFIELD_TYPES:
            continue
        n += 1
    return n


def _label_text_for(el, soup) -> str:
    """Best-effort visible label for a field, mirroring how the autofill model
    derives field text: explicit <label for>, wrapping <label>, then a nearby
    preceding label/span/td/th.
    """
    fid = el.get("id")
    if fid:
        lab = soup.find("label", attrs={"for": fid})
        if lab:
            t = lab.get_text(" ", strip=True)
            if t:
                return t
    parent_label = el.find_parent("label")
    if parent_label:
        t = parent_label.get_text(" ", strip=True)
        if t:
            return t
    # Preceding label-ish element.
    prev = el.find_previous(["label", "span", "td", "th", "p"])
    if prev:
        t = prev.get_text(" ", strip=True)
        if t:
            return t[:120]
    return ""


def _nearby_text(el) -> str:
    """Text of the field's container, for extra context."""
    container = el.find_parent(["div", "td", "p", "li", "fieldset"]) or el.parent
    if container is None:
        return ""
    return container.get_text(" ", strip=True)[:160]


def _select_options(el) -> str:
    if el.name != "select":
        return ""
    opts = [o.get_text(strip=True) for o in el.find_all("option")][:8]
    return ", ".join(o for o in opts if o)


def extract_candidates(soup) -> list[dict]:
    """Return injectable candidate fields (text inputs + selects), in document
    order, each as a context dict for the LLM. Element refs are kept under 'el'.
    """
    fields: list[dict] = []
    idx = 0
    for el in soup.find_all(["input", "select"]):
        if el.name == "input" and _input_type(el) not in _INJECTABLE_INPUT_TYPES:
            continue
        fields.append({
            "index": idx,
            "el": el,
            "tag": el.name,
            "type": _input_type(el) if el.name == "input" else "select",
            "label": _label_text_for(el, soup),
            "placeholder": el.get("placeholder", ""),
            "name": el.get("name", ""),
            "id": el.get("id", ""),
            "options": _select_options(el),
            "nearby": _nearby_text(el),
        })
        idx += 1
    return fields


def inject_label(field: dict, autofill_type: str) -> bool:
    """Write the ground-truth attribute onto the real element.

    Returns True if applied. Guards against labeling element kinds that
    validate_html would reject (so a stray LLM label can't fail the whole form).
    """
    el = field["el"]
    if el.name == "select":
        el[LABEL_ATTR] = autofill_type
        return True
    if el.name == "input" and _input_type(el) in _INJECTABLE_INPUT_TYPES:
        el[LABEL_ATTR] = autofill_type
        return True
    return False


def guess_domain(soup) -> str:
    """Pull a plausible site domain from any absolute URL in the form."""
    for attr in ("action", "href", "src"):
        for el in soup.find_all(attrs={attr: True}):
            val = el.get(attr, "")
            if val.startswith("http"):
                host = urlparse(val).netloc
                if host:
                    return host.lstrip("www.")
    return "common-crawl"


def wrap_form(form_inner_html: str, site: str, note: str) -> str:
    """Wrap a form's inner HTML in the canonical sample scaffold."""
    note_html = f"<p><i>{note}</i></p>\n" if note else ""
    return (
        "<html><head>\n"
        '<meta http-equiv="content-type" content="text/html; charset=UTF-8">\n'
        "</head>\n\n"
        "<body>\n\n"
        "<fieldset>\n"
        f"<h3>{site}</h3>\n"
        f"{note_html}\n"
        "<form>\n"
        f"{form_inner_html}\n"
        "</form>\n\n"
        "</fieldset></body></html>\n"
    )
