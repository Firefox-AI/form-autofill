"""Sampled generation parameters that drive diversity.

Each form is built from an independently-sampled `GenParams`: a locale, a form
purpose (which maps to a base field-set), a markup style, and a name-messiness
profile. Rare field types are injected at an elevated rate to rebalance the
real corpus, which is heavily skewed toward postal-code/address-level2/tel/email.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from gen.render import MARKUP_STYLES, NAME_STYLES

# Locale codes drive the language of LLM-generated labels and the filename
# prefix. Weighted toward locales already present in the corpus, plus some
# under-represented ones to broaden language coverage.
LOCALES = [
    ("en-US", 10), ("en-GB", 6), ("de-DE", 8), ("de-AT", 3), ("fr-FR", 7),
    ("es-ES", 6), ("it-IT", 5), ("nl-NL", 3), ("da-DK", 3), ("sv-SE", 2),
    ("fi-FI", 2), ("pt-PT", 2), ("pt-BR", 3), ("pl-PL", 3), ("cs-CZ", 2),
    ("hu-HU", 2), ("ro-RO", 2), ("el-GR", 2), ("tr-TR", 2), ("ru-RU", 2),
    ("uk-UA", 2), ("ja-JP", 3), ("ca-ES", 1), ("nb-NO", 2), ("bg-BG", 1),
]

PURPOSES = (
    "checkout", "registration", "billing", "shipping",
    "payment", "profile", "contact",
)

# Base field-set per purpose. Optional fields (see OPTIONAL_FIELDS) are added
# probabilistically on top of these. Order here is a sensible default; it gets
# lightly shuffled within groups during sampling.
PURPOSE_FIELDS: dict[str, list[str]] = {
    "checkout": [
        "given-name", "family-name", "street-address", "address-level2",
        "postal-code", "country-name", "email", "tel",
    ],
    "registration": [
        "given-name", "family-name", "email", "tel", "bday",
    ],
    # Single combined address field -> street-address (the safe, spec-correct
    # choice). address-line1/2 only read correctly with generic "Address line N"
    # labels, which the LLM rarely produces, so we avoid them in templates and
    # let real address-line1/2 examples come from mutation anchors + common_crawl.
    "billing": [
        "cc-name", "street-address", "postal-code",
        "address-level2", "address-level1", "country-name",
    ],
    "shipping": [
        "name", "street-address", "postal-code",
        "address-level2", "country-name",
    ],
    "payment": [
        "cc-name", "cc-number", "cc-exp-month", "cc-exp-year", "cc-csc", "cc-type",
    ],
    "profile": [
        "given-name", "family-name", "additional-name", "nickname",
        "email", "tel", "organization", "bday",
    ],
    "contact": [
        "name", "email", "tel", "street-address", "address-level2", "postal-code",
    ],
}

# Optional fields that may be sprinkled in per purpose, with their probability.
OPTIONAL_FIELDS: dict[str, list[tuple[str, float]]] = {
    # A second address line (address-line2 or apartment) promotes the primary
    # field to address-line1 via _apply_address_scheme; their combined rate sets
    # the address-line1-vs-street-address mix (~half, matching the dataset).
    "checkout": [("organization", 0.3), ("address-line2", 0.3), ("apartment", 0.2),
                 ("honorific-prefix", 0.2), ("address-level1", 0.3)],
    "registration": [("organization", 0.25), ("sex", 0.2), ("nickname", 0.2)],
    "billing": [("vat-number", 0.4), ("organization", 0.35), ("tel", 0.3),
                ("address-line2", 0.3), ("apartment", 0.2)],
    "shipping": [("address-line2", 0.3), ("apartment", 0.35), ("floor", 0.3),
                 ("building", 0.2), ("reference-point", 0.2), ("address-level3", 0.25)],
    "payment": [("street-address", 0.2), ("postal-code", 0.2)],
    "profile": [("organization-title", 0.3), ("phonetic-given-name", 0.15),
                ("phonetic-family-name", 0.15), ("id-number", 0.2)],
    "contact": [("organization", 0.3), ("country-name", 0.4)],
}

# Rare types under-represented in the real corpus. We force at least one into a
# fraction of forms so the model gets more signal for them.
RARE_TYPES = [
    "vat-number", "id-number", "bday-day", "bday-month", "bday-year",
    "floor", "apartment", "building", "stair", "block", "address-level3",
    "address-level4", "reference-point", "phonetic-given-name",
    "phonetic-family-name", "phonetic-name", "address-streetname",
    "address-extra", "honorific-suffix", "organization-title",
    "tel-country-code", "tel-area-code", "tel-extension", "cc-additional-name",
]

# Types best rendered as <select> rather than free text.
SELECT_TYPES = {
    "country-name", "country", "cc-type", "cc-exp-month", "cc-exp-year",
    "bday-month", "bday-day", "bday-year", "address-level1", "sex",
    "honorific-prefix",
}


@dataclass
class GenParams:
    index: int
    seed: int
    locale: str
    purpose: str
    field_types: list[str]
    markup_style: str
    name_style: str
    include_password: bool
    include_submit: bool
    rare_injected: list[str] = field(default_factory=list)

    @property
    def short_domain_hint(self) -> str:
        return self.purpose


# A "second address line" field. Per the hand-labeled dataset, the primary
# address field is address-line1 when one of these is also present (~92% of
# address-line1 forms have one), and street-address when it stands alone
# (~70% of street-address forms have no second line).
SECOND_LINE_TOKENS = {"address-line2", "address-line3", "apartment", "floor", "building"}


def _apply_address_scheme(types: list[str]) -> None:
    """Enforce the dataset's address convention in place: a primary address field
    is 'address-line1' when a second-line field is present, else 'street-address'.
    """
    has_second = any(t in SECOND_LINE_TOKENS for t in types)
    primary = "address-line1" if has_second else "street-address"
    for i, t in enumerate(types):
        if t in ("street-address", "address-line1"):
            types[i] = primary


def _weighted_choice(rng: random.Random, choices: list[tuple[str, int]]) -> str:
    population = [c for c, _ in choices]
    weights = [w for _, w in choices]
    return rng.choices(population, weights=weights, k=1)[0]


def sample_params(index: int, master_seed: int) -> GenParams:
    """Draw an independent, reproducible parameter set for form `index`."""
    seed = master_seed ^ (index * 0x9E3779B1)
    rng = random.Random(seed)

    locale = _weighted_choice(rng, LOCALES)
    purpose = rng.choice(PURPOSES)

    types = list(PURPOSE_FIELDS[purpose])
    for opt, prob in OPTIONAL_FIELDS.get(purpose, []):
        if rng.random() < prob and opt not in types:
            types.append(opt)

    # Inject rare types into ~35% of forms (1-2 of them) to rebalance.
    rare_injected: list[str] = []
    if rng.random() < 0.35:
        k = rng.randint(1, 2)
        for rare in rng.sample(RARE_TYPES, k=min(k, len(RARE_TYPES))):
            if rare not in types:
                types.append(rare)
                rare_injected.append(rare)

    # Resolve the primary address token (address-line1 vs street-address) to
    # match the dataset convention, based on whether a second line is present.
    _apply_address_scheme(types)

    # Field order is left to the LLM, which orders fields by local convention
    # (see _SYSTEM in gen/llm.py) — more realistic than a blind code-side shuffle.

    # Email-confirm duplication shows up often in real forms.
    if "email" in types and rng.random() < 0.25:
        types.insert(types.index("email") + 1, "email")

    markup_style = rng.choices(
        MARKUP_STYLES + ("mixed",),
        weights=[6, 8, 5, 4, 3, 4, 5],  # label_for & placeholder common; mixed frequent
        k=1,
    )[0]
    name_style = rng.choices(NAME_STYLES, weights=[3, 3, 2, 2], k=1)[0]

    return GenParams(
        index=index,
        seed=seed,
        locale=locale,
        purpose=purpose,
        field_types=types,
        markup_style=markup_style,
        name_style=name_style,
        include_password=(purpose in ("registration", "profile") and rng.random() < 0.6),
        include_submit=(rng.random() < 0.4),
        rare_injected=rare_injected,
    )
