"""OpenAI client wrapper, JSON schemas, and the generation/verification calls.

The LLM never emits HTML. It returns a FormSpec (typed JSON) whose
`autofill_type` is constrained to an enum built from dotraining.py's 66 field
types plus the `__nonfill__` sentinel, so an out-of-taxonomy label is
structurally impossible. render.py turns the FormSpec into HTML.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from gen.params import GenParams, SELECT_TYPES
from gen.validate import ELEMENTS, NONFILL, valid_types

# Rough per-1M-token prices (USD) for cost estimation; update as needed.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# Only transient failures are retried. Auth/permission/bad-request (4xx other
# than 429) fail fast so misconfiguration surfaces immediately.
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, resp_usage) -> None:
        if resp_usage is None:
            return
        self.prompt_tokens += getattr(resp_usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(resp_usage, "completion_tokens", 0) or 0
        self.calls += 1

    def cost(self, model: str) -> float:
        pin, pout = PRICING.get(model, (0.0, 0.0))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000


def _field_spec_schema() -> dict:
    enum = sorted(valid_types()) + [NONFILL]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "autofill_type": {"type": "string", "enum": enum},
            "label_text": {"type": "string"},
            "placeholder_text": {"type": ["string", "null"]},
            "element": {"type": "string", "enum": sorted(ELEMENTS)},
            "select_options": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": [
            "autofill_type", "label_text", "placeholder_text",
            "element", "select_options",
        ],
    }


def _form_spec_schema() -> dict:
    return {
        "name": "form_spec",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "locale": {"type": "string"},
                "site_domain": {"type": "string"},
                "purpose": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": _field_spec_schema(),
                },
            },
            "required": ["locale", "site_domain", "purpose", "fields"],
        },
    }


_SYSTEM = (
    "You generate realistic web-form field metadata for ML training data used by "
    "a browser autofill field detector. You will be given a locale, a form "
    "purpose, and a list of field-type codes to include. For each code, "
    "produce the natural-language LABEL and PLACEHOLDER text that a real website "
    "in that locale's language would show for that field, matching the purpose's "
    "context (e.g. billing vs shipping wording). Rules:\n"
    "- Echo each given field-type code verbatim in autofill_type; never translate "
    "or change the codes, never invent new ones.\n"
    "- Order the fields the way a real website in that locale would present them "
    "for that purpose; you MAY reorder the codes from the list (e.g. family name "
    "before given name, or postal code before city, where that is the local "
    "convention). Keep every given code; do not add or drop fields.\n"
    "- Write label/placeholder text in the locale's language with realistic, "
    "varied wording (abbreviations, formal/informal register).\n"
    "- Choose a sensible element per field; use 'select' with localized "
    "select_options for dropdown-style fields (countries, months, card type).\n"
    "- Propose a plausible fictitious site_domain consistent with the locale.\n"
    "- When asked, append a password field (element input_password) and/or a "
    "submit button (element button), each with autofill_type '__nonfill__'."
)


def _build_user_prompt(params: GenParams) -> str:
    lines = [
        f"locale: {params.locale}",
        f"purpose: {params.purpose}",
        "Field-type codes to include (one FieldSpec each; order them as a real "
        "site in this locale would, not necessarily the order listed here):",
    ]
    for t in params.field_types:
        hint = " (render as a select with localized options)" if t in SELECT_TYPES else ""
        lines.append(f"  - {t}{hint}")
    if params.include_password:
        lines.append("Then append a password field: autofill_type '__nonfill__', "
                     "element input_password, localized label.")
    if params.include_submit:
        lines.append("Then append a submit button: autofill_type '__nonfill__', "
                     "element button, localized button label.")
    return "\n".join(lines)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _chat_json(client: AsyncOpenAI, model: str, system: str, user: str,
                     schema: dict, seed: int, usage: Usage) -> dict:
    resp = await client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=0.85,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    usage.add(resp.usage)
    return json.loads(resp.choices[0].message.content)


async def request_form_spec(client: AsyncOpenAI, model: str, params: GenParams,
                            usage: Usage) -> dict:
    """Mode A (hybrid): ask the LLM for localized text for a planned field set."""
    return await _chat_json(
        client, model, _SYSTEM, _build_user_prompt(params),
        _form_spec_schema(), seed=params.seed, usage=usage,
    )


_MUTATE_SYSTEM = (
    "You rewrite a real web form into a realistic VARIANT for a different "
    "locale/site for ML training data. You are given the form's fields as an "
    "ordered list of (autofill_type code, original label). Produce a new "
    "FormSpec in the TARGET locale's language. Rules:\n"
    "- Keep the SAME autofill_type codes in the SAME ORDER; never change or "
    "invent codes. You may translate/reword the labels and placeholders.\n"
    "- You may keep all fields. Do not drop labeled fields.\n"
    "- Choose sensible elements; use 'select' with localized options where apt.\n"
    "- Propose a plausible fictitious site_domain for the target locale."
)


async def request_mutation(client: AsyncOpenAI, model: str, anchor_fields: list[dict],
                           target_locale: str, seed: int, usage: Usage) -> dict:
    """Mode B (anchored mutation): rewrite a real form into a variant.

    anchor_fields: [{"autofill_type": str, "label": str}] in document order.
    The verified anchor type is carried forward by the caller after return.
    """
    lines = [f"target locale: {target_locale}", "Original fields (code, label):"]
    for f in anchor_fields:
        lines.append(f"  - {f['autofill_type']} | {f.get('label', '')}")
    return await _chat_json(
        client, model, _MUTATE_SYSTEM, "\n".join(lines),
        _form_spec_schema(), seed=seed, usage=usage,
    )


_VERIFY_SCHEMA = {
    "name": "label_verdicts",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "match": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                    },
                    "required": ["index", "match"],
                },
            }
        },
        "required": ["verdicts"],
    },
}

_VERIFY_SYSTEM = (
    "You audit ML training labels. Given a locale and a list of "
    "(index, label_text, autofill_type) rows, decide for each whether the "
    "label_text, in that locale's language, plausibly denotes a form field of "
    "that autofill_type. Answer yes / no / uncertain per row."
)


async def verify_form(client: AsyncOpenAI, model: str, locale: str,
                      fields: list[dict], usage: Usage) -> dict[int, str]:
    """Second-pass semantic check of label<->type plausibility.

    Returns {field_index: 'yes'|'no'|'uncertain'} for labeled (non-nonfill) fields.
    """
    rows = [(i, f.get("label_text", ""), f["autofill_type"])
            for i, f in enumerate(fields) if f["autofill_type"] != NONFILL]
    if not rows:
        return {}
    user = f"locale: {locale}\nrows:\n" + "\n".join(
        f"  - index={i} | label={lbl!r} | type={t}" for i, lbl, t in rows
    )
    result = await _chat_json(
        client, model, _VERIFY_SYSTEM, user, _VERIFY_SCHEMA,
        seed=12345, usage=usage,
    )
    return {v["index"]: v["match"] for v in result.get("verdicts", [])}


# --------------------------------------------------------------------------- #
# Labeling real (Common Crawl) form fields.
# Unlike generation, here the HTML already exists and the LLM CLASSIFIES each
# field into the taxonomy. '__skip__' means "not a personal-info autofill field"
# (search, quantity, coupon, password, captcha, message, etc.) -> left unlabeled.
# --------------------------------------------------------------------------- #
SKIP = "__skip__"


def _label_schema() -> dict:
    enum = sorted(valid_types()) + [SKIP]
    return {
        "name": "field_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "index": {"type": "integer"},
                            "autofill_type": {"type": "string", "enum": enum},
                        },
                        "required": ["index", "autofill_type"],
                    },
                }
            },
            "required": ["labels"],
        },
    }


_LABEL_SYSTEM = (
    "You label the fields of a REAL web form to create ground-truth training data "
    "for a browser autofill field detector. You are given the form's fields, each "
    "with an index and context (associated label text, placeholder, name/id "
    "attributes, input type, option samples, and nearby text). Field text may be "
    "in any language. For each index, choose the SINGLE best autofill field-type "
    "code from the allowed enum that describes the personal/address/contact/"
    "payment data the field collects. Rules:\n"
    "- Use the context to decide; the visible label/placeholder is the strongest "
    "signal, then name/id, then nearby text.\n"
    "- Return '__skip__' for any field that is NOT a personal-info autofill field: "
    "search boxes, quantities, coupon/promo codes, captchas, passwords, free-text "
    "message/comment boxes, newsletter checkboxes, search filters, etc.\n"
    "- A login/username field that is an email address is 'email'; a non-email "
    "account name is 'loginname'.\n"
    "- Address lines: decide by whether the form has a SECOND address line — "
    "either another street line OR an 'Apartment / Suite / Unit / Flat / Floor' "
    "field. If there IS a second line, label the FIRST as 'address-line1' and the "
    "second as 'address-line2' (or 'apartment' when it is specifically an "
    "apartment/suite/unit/flat). If there is only ONE combined address field with "
    "no second line, use 'street-address' (labels like 'Address', 'Street address', "
    "'Straße und Hausnummer'). This matches our training-set convention.\n"
    "- Administrative levels: a state/province/region/prefecture/county field "
    "(incl. dropdowns listing them) is 'address-level1'; city/town is "
    "'address-level2'. Do not label these as address-line*.\n"
    "- Return exactly one entry per provided index."
)


async def request_field_labels(client: AsyncOpenAI, model: str,
                               fields: list[dict], usage: Usage,
                               seed: int = 7) -> dict[int, str]:
    """Classify real form fields. `fields` is a list of context dicts with keys:
    index, type, label, placeholder, name, id, options, nearby.

    Returns {index: autofill_type_or_'__skip__'}.
    """
    if not fields:
        return {}
    lines = ["Fields:"]
    for f in fields:
        opts = f.get("options") or ""
        if opts:
            opts = f" | options={opts[:120]!r}"
        lines.append(
            f"  - index={f['index']} | tag={f.get('tag','input')} "
            f"type={f.get('type','text')} | label={ (f.get('label') or '')[:100]!r} "
            f"| placeholder={(f.get('placeholder') or '')[:60]!r} "
            f"| name={(f.get('name') or '')[:60]!r} id={(f.get('id') or '')[:40]!r}"
            f"{opts} | nearby={(f.get('nearby') or '')[:120]!r}"
        )
    result = await _chat_json(
        client, model, _LABEL_SYSTEM, "\n".join(lines), _label_schema(),
        seed=seed, usage=usage,
    )
    return {item["index"]: item["autofill_type"] for item in result.get("labels", [])}


# --------------------------------------------------------------------------- #
# Label-accuracy audit: judge existing labels against the autofill spec.
# Used by evaluate_labels.py to measure whether a sample of forms is correctly
# labeled (and thus whether re-labeling is warranted).
# --------------------------------------------------------------------------- #
_AUDIT_SCHEMA = {
    "name": "label_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "verdict": {"type": "string",
                                    "enum": ["correct", "incorrect", "unsure"]},
                        "suggested": {"type": "string"},
                    },
                    "required": ["index", "verdict", "suggested"],
                },
            }
        },
        "required": ["fields"],
    },
}

_AUDIT_SYSTEM_TEMPLATE = (
    "You audit autofill field labels that are used as ground-truth ML training "
    "data. Below is the authoritative autocomplete specification.\n\n"
    "=== SPECIFICATION ===\n{spec}\n=== END SPECIFICATION ===\n\n"
    "You will be given the fields of a SINGLE web form. Each field lists its "
    "visible context (label / placeholder / name / nearby text) and the autofill "
    "TOKEN that was assigned to it. For each field, decide whether the assigned "
    "token correctly describes the data that field collects, per the spec.\n"
    "Rules:\n"
    "- Judge by the field's real meaning from its context; the visible label is "
    "the strongest signal. Field text may be in ANY language.\n"
    "- Some assigned tokens are valid Mozilla/Firefox extensions BEYOND the "
    "standard spec (e.g. address-housenumber, address-streetname, apartment, "
    "floor, stair, building, block, address-extra, postal-code-and-city, "
    "id-number, vat-number, loginname, phonetic-given-name, phonetic-family-name, "
    "reference-point). Treat these as CORRECT when they accurately describe the "
    "field — do NOT mark a field incorrect merely because its token is not in the "
    "specification text.\n"
    "- Mark 'incorrect' only when the token does not match the field's meaning; "
    "then put the token you would assign in 'suggested'. Use 'unsure' when the "
    "context is too ambiguous to tell. For 'correct'/'unsure', set suggested to ''.\n"
    "- Return exactly one entry per provided index."
)


async def audit_form_labels(client: AsyncOpenAI, model: str, spec: str,
                            fields: list[dict], usage: Usage,
                            lang: str = "") -> list[dict]:
    """Judge a form's labels against the spec.

    `fields`: list of {index, label, placeholder, name, autofill_type}.
    Returns a list of {index, verdict, suggested}.
    """
    if not fields:
        return []
    lines = [f"form language hint: {lang or 'unknown'}", "Fields:"]
    for f in fields:
        lines.append(
            f"  - index={f['index']} | assigned_token={f['autofill_type']} "
            f"| label={(f.get('label') or '')[:100]!r} "
            f"| placeholder={(f.get('placeholder') or '')[:60]!r} "
            f"| name={(f.get('name') or '')[:60]!r}"
        )
    system = _AUDIT_SYSTEM_TEMPLATE.format(spec=spec)
    result = await _chat_json(
        client, model, system, "\n".join(lines), _AUDIT_SCHEMA,
        seed=7, usage=usage,
    )
    return result.get("fields", [])
