"""Render a FormSpec (structured JSON from the LLM) into a labeled HTML file.

Code — not the LLM — owns the markup and the data-moz-autofill-type attribute.
The output mirrors the canonical clean-form shape used throughout the corpus
(see samples/training/at.html): a charset <meta>, a <fieldset> with an <h3>
site header, and a <form> of <div>-wrapped fields.

The LLM only supplies surface text (labels, placeholders, option lists). Markup
style and the messy name/id attributes are applied here so we can vary them
systematically for diversity without trusting the model to do it.
"""

from __future__ import annotations

import html as _html
import random
import string

from gen.validate import LABEL_ATTR, NONFILL

# How a field's label is attached to its control. Frequencies are tuned in
# params.py to roughly match the real corpus (label_for and placeholder common).
MARKUP_STYLES = (
    "span_before",        # <span>Label</span><input>
    "label_for",          # <label for=id>Label</label><input id=id>
    "placeholder_only",   # <input placeholder="Label">
    "aria_label",         # <input aria-label="Label">
    "nested_label",       # <label>Label<input></label>
    "label_plus_placeholder",
)

# How name/id attributes look. Real forms range from semantic to machine junk.
NAME_STYLES = ("machine_colon", "random_hash", "semantic", "mixed")

_ELEMENT_INPUT_TYPE = {
    "input_text": "text",
    "input_email": "email",
    "input_tel": "tel",
    "input_number": "number",
    "input_password": "password",
}


def _esc(text: str) -> str:
    return _html.escape(text or "", quote=True)


def _rand_token(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


# Realistic name/id attribute values per type, the way real forms actually name
# fields. CRITICAL: these must NOT contain the autofill-type code itself — using
# the code (e.g. name="address_level1") leaks the label into the tokens, which is
# trivially predictive in training but absent from real forms, hurting
# generalization. Types not listed fall back to an opaque hash (no leak).
NAME_HINTS: dict[str, list[str]] = {
    "given-name": ["firstName", "fname", "first_name", "givenName", "billing_first_name"],
    "family-name": ["lastName", "lname", "last_name", "surname", "billing_last_name"],
    "additional-name": ["middleName", "middle_name", "mname"],
    "name": ["fullName", "yourName", "name", "contactName"],
    "honorific-prefix": ["title", "salutation", "prefix"],
    "nickname": ["nickname", "displayName", "handle"],
    "email": ["email", "emailAddress", "email_address", "user_email", "mail"],
    "tel": ["phone", "telephone", "phoneNumber", "mobile", "tel"],
    "tel-country-code": ["phoneCountryCode", "countryCode"],
    "tel-area-code": ["areaCode"],
    "organization": ["company", "companyName", "organization", "business"],
    "organization-title": ["jobTitle", "position", "role"],
    "street-address": ["address", "streetAddress", "addr", "street"],
    "address-line1": ["address1", "addressLine1", "addr1", "street1"],
    "address-line2": ["address2", "addressLine2", "addr2", "street2"],
    "address-line3": ["address3", "addressLine3"],
    "address-level1": ["state", "province", "region", "county", "stateProvince"],
    "address-level2": ["city", "town", "locality"],
    "address-level3": ["district", "suburb", "neighborhood"],
    "address-streetname": ["streetName", "street", "strasse"],
    "address-housenumber": ["houseNumber", "streetNumber", "number", "hausnummer"],
    "postal-code": ["zip", "zipCode", "postalCode", "postcode", "zip_code"],
    "country": ["country", "countryCode", "country_id"],
    "country-name": ["country", "countryName"],
    "apartment": ["apt", "apartment", "suite", "unit", "flat"],
    "floor": ["floor", "etage"],
    "cc-name": ["cardName", "cardholderName", "nameOnCard", "ccName"],
    "cc-number": ["cardNumber", "ccNumber", "card_number", "cardNo"],
    "cc-exp": ["cardExpiry", "expiry", "expiration", "exp"],
    "cc-exp-month": ["expMonth", "ccExpMonth", "exp_month"],
    "cc-exp-year": ["expYear", "ccExpYear", "exp_year"],
    "cc-csc": ["cvv", "cvc", "csc", "securityCode", "cardCode"],
    "cc-type": ["cardType", "ccType"],
    "bday": ["birthday", "dob", "dateOfBirth", "birthdate"],
    "bday-day": ["birthDay", "dobDay"],
    "bday-month": ["birthMonth", "dobMonth"],
    "bday-year": ["birthYear", "dobYear"],
    "sex": ["gender", "sex"],
    "id-number": ["idNumber", "nationalId", "taxId"],
    "vat-number": ["vat", "vatNumber", "vatId"],
    "loginname": ["username", "userId", "login", "account"],
}


def _gen_name(rng: random.Random, style: str, autofill_type: str, index: int) -> str:
    """Produce a realistic (often messy) name/id attribute value.

    Never embeds the autofill-type code (that would leak the label); 'semantic'
    and 'mixed' draw from realistic NAME_HINTS, falling back to an opaque hash.
    """
    if style == "machine_colon":
        parts = ["r", "r", str(rng.randint(1, 9)), "m", "r", str(index + 1),
                 "cr", str(rng.randint(1, 3)), "c", "pr", "1",
                 "container", "fs", "i", "input"]
        return ":".join(parts)
    if style == "random_hash":
        return "id" + _rand_token(rng, rng.randint(2, 5))
    # semantic / mixed: realistic field name, no type-code leak.
    hints = NAME_HINTS.get(autofill_type)
    if not hints:
        return "id" + _rand_token(rng, rng.randint(2, 5))
    base = rng.choice(hints)
    if style == "mixed" and rng.random() < 0.3:
        base = f"{base}{rng.randint(1, 9)}"   # small realistic suffix, not noise
    return base


def _render_control(field: dict, *, name: str, el_id: str,
                    markup_style: str, label_text: str) -> str:
    """Render the <input>/<select>/<button> control element itself."""
    element = field["element"]
    autofill_type = field["autofill_type"]
    attrs: list[str] = []

    if element == "select":
        tag = "select"
    elif element == "button":
        tag = "button"
    else:
        tag = "input"
        attrs.append(f'type="{_ELEMENT_INPUT_TYPE.get(element, "text")}"')

    attrs.append(f'name="{_esc(name)}"')
    attrs.append(f'id="{_esc(el_id)}"')

    if markup_style in ("placeholder_only", "label_plus_placeholder"):
        placeholder = field.get("placeholder_text") or label_text
        if tag == "input":
            attrs.append(f'placeholder="{_esc(placeholder)}"')
    elif field.get("placeholder_text") and tag == "input":
        attrs.append(f'placeholder="{_esc(field["placeholder_text"])}"')

    if markup_style == "aria_label":
        attrs.append(f'aria-label="{_esc(label_text)}"')

    # The ground-truth label, applied only to autofillable controls.
    if autofill_type != NONFILL:
        attrs.append(f'{LABEL_ATTR}="{_esc(autofill_type)}"')

    attr_str = " ".join(attrs)

    if tag == "select":
        options = field.get("select_options") or []
        opts = ['  <option value=""></option>']
        opts += [f"  <option>{_esc(o)}</option>" for o in options]
        return f"<select {attr_str}>\n" + "\n".join(opts) + "\n</select>"
    if tag == "button":
        return f'<button {attr_str} type="submit">{_esc(label_text or "Submit")}</button>'
    return f"<input {attr_str}>"


def _render_field(field: dict, *, rng: random.Random, markup_style: str,
                  name_style: str, index: int) -> str:
    """Render one field (label + control) wrapped in a <div>."""
    autofill_type = field["autofill_type"]
    label_text = (field.get("label_text") or "").strip()
    name = _gen_name(rng, name_style, autofill_type, index)
    # id is an opaque hash so the (realistic) name word isn't double-counted in tokens.
    el_id = _gen_name(rng, "random_hash", autofill_type, index)

    # Buttons / passwords don't get a visible field label row treatment.
    if field["element"] == "button":
        control = _render_control(field, name=name, el_id=el_id,
                                  markup_style="placeholder_only", label_text=label_text)
        return f"  <div>\n    {control}\n  </div>"

    control = _render_control(field, name=name, el_id=el_id,
                              markup_style=markup_style, label_text=label_text)

    if markup_style == "span_before":
        body = f"<span>{_esc(label_text)}</span>\n    {control}"
    elif markup_style == "label_for":
        body = f'<label for="{_esc(el_id)}">{_esc(label_text)}</label>\n    {control}'
    elif markup_style == "nested_label":
        body = f"<label>{_esc(label_text)}\n    {control}\n    </label>"
    elif markup_style in ("placeholder_only", "aria_label"):
        body = control  # label lives inside the control's attributes
    else:  # label_plus_placeholder
        body = f'<label for="{_esc(el_id)}">{_esc(label_text)}</label>\n    {control}'

    return f"  <div>\n    {body}\n  </div>"


def render_form(spec: dict, *, markup_style: str, name_style: str,
                rng: random.Random, note: str = "") -> str:
    """Render a full standalone HTML file from a FormSpec.

    `markup_style` may be a single style applied to all fields, or "mixed" to
    pick a per-field style (more realistic for messy real-world forms).
    """
    site = spec.get("site_domain") or "example.com"
    fields = spec.get("fields", [])

    rendered: list[str] = []
    for i, field in enumerate(fields):
        style = (rng.choice([s for s in MARKUP_STYLES if s != "mixed"])
                 if markup_style == "mixed" else markup_style)
        rendered.append(_render_field(field, rng=rng, markup_style=style,
                                      name_style=name_style, index=i))

    note_html = f"<p><i>{_esc(note)}</i></p>\n" if note else ""
    body = "\n".join(rendered)
    return (
        "<html><head>\n"
        '<meta http-equiv="content-type" content="text/html; charset=UTF-8">\n'
        "</head>\n\n"
        "<body>\n\n"
        "<fieldset>\n"
        f"<h3>{_esc(site)}</h3>\n"
        f"{note_html}\n"
        '<form method="post">\n'
        f"{body}\n"
        "</form>\n\n"
        "</fieldset></body></html>\n"
    )
