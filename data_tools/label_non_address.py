#!/usr/bin/env python
"""Annotate non-address HTML forms with `data-moz-autofill-type` so the genuinely
autofillable fields (email / name / tel) get a real label and everything else
stays the negative "other" class.

data/non_address/*.html carry NO annotations. Most fields are negatives (search,
login, newsletter, coupon, ...), but many forms still contain an email or name
field that IS autofillable regardless of the surrounding form. We label those
using, in priority order:

  1. the `autocomplete` attribute (authoritative author intent),
  2. `<input type=email|tel>` (HTML5 semantic type),
  3. a careful name/id/placeholder/aria-label regex, with a guard that sends
     username / login / search / password / display-name to "other".

Only the injected `data-moz-autofill-type` is added; nothing else changes, so
mlData tokenization (id/name/placeholder/labels/**type) is unaffected. The
annotated copies feed build_dataset_ff.py exactly like the address datasets.

    python label_non_address.py --input data/non_address --output data/non_address_labeled
"""
import argparse, glob, os, re
from collections import Counter

AC_MAP = {"email": "email", "name": "name", "given-name": "given-name",
          "family-name": "family-name", "additional-name": "additional-name",
          "tel": "tel", "organization": "organization"}
SKIP_TYPES = {"hidden", "submit", "button", "image", "reset", "checkbox",
              "radio", "file", "search"}
NEG = re.compile(r"user\s*name|login|screen\s*name|nick|display\s*name|"
                 r"file\s*name|host\s*name|search|coupon|promo|password", re.I)


def _attr(tag, name):
    m = (re.search(rf'{name}\s*=\s*"([^"]*)"', tag, re.I) or
         re.search(rf"{name}\s*=\s*'([^']*)'", tag, re.I))
    return (m.group(1).strip().lower() if m else "")


def _tagname(tag):
    m = re.match(r"<\s*(\w+)", tag)
    return m.group(1).lower() if m else ""


def label_field(tag):
    """Return a supported field-type name, or None (-> other)."""
    for tok in _attr(tag, "autocomplete").split():
        if tok in AC_MAP:
            return AC_MAP[tok]
    if _tagname(tag) == "input":
        t = _attr(tag, "type")
        if t == "email":
            return "email"
        if t == "tel":
            return "tel"
    blob = " ".join(_attr(tag, a) for a in ("name", "id", "placeholder", "aria-label"))
    if not blob.strip():
        return None
    if NEG.search(blob):
        return "email" if re.search(r"e-?mail", blob) else None
    if re.search(r"e-?mail", blob):
        return "email"
    if re.search(r"first[\s_-]*name|given[\s_-]*name|vorname|prénom|nombre", blob):
        return "given-name"
    if re.search(r"last[\s_-]*name|family[\s_-]*name|surname|nachname", blob):
        return "family-name"
    if re.search(r"(full[\s_-]*)?name\b|your[\s_-]*name|\bnom\b", blob):
        return "name"
    if re.search(r"phone|\btel\b|mobile|telefon|telephone", blob):
        return "tel"
    return None


def annotate(html, counts):
    def repl(m):
        tag = m.group(0)
        if _tagname(tag) == "input" and _attr(tag, "type") in SKIP_TYPES:
            return tag
        if "data-moz-autofill-type" in tag.lower():
            return tag
        lbl = label_field(tag)
        if not lbl:
            return tag
        counts[lbl] += 1
        return re.sub(r"^<(\w+)", rf'<\1 data-moz-autofill-type="{lbl}"', tag, count=1)
    return re.sub(r"<(?:input|select|textarea)\b[^>]*>", repl, html, flags=re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    counts = Counter()
    files = sorted(glob.glob(os.path.join(args.input, "*.html")))
    for p in files:
        html = open(p, encoding="utf-8", errors="ignore").read()
        out = annotate(html, counts)
        open(os.path.join(args.output, os.path.basename(p)), "w", encoding="utf-8").write(out)
    print(f"{len(files)} files -> {args.output}")
    print(f"labels injected: {dict(counts.most_common())}  total={sum(counts.values())}")


if __name__ == "__main__":
    main()
