"""Extract candidate web forms from Common Crawl WARC files into a CSV.

This is a script-friendly port of notebooks/Common Crawl Form.ipynb from the
smart-tab-grouping repo, generalized so the caller can request a *list* of
languages (e.g. en + de) instead of just English, and so the output CSV carries
a few extra columns (url, tld, lang, title, num_fields) alongside the form HTML.

Each emitted row is a single <form>'s inner HTML (one form per row), which is
the format label_common_crawl.py consumes.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from warcio.archiveiterator import ArchiveIterator

from gen.htmlforms import fillable_count, _label_text_for

# Deterministic language detection (langdetect is randomized by default).
DetectorFactory.seed = 0

# Control bytes (incl. NUL) that appear in raw crawl HTML and break CSV parsers.
# Keep tab/newline/carriage-return; strip the rest.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Tags that are pure noise for field detection and bloat the stored HTML.
_STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe",
               "template", "link", "meta", "canvas")

# Default address/contact/payment substrings (matched case-insensitively
# against a form's inner HTML — labels, placeholders, name/id/autocomplete
# attributes). Keeping a form only if it contains at least one of these is how
# the original dataset used the word "street" to drop noisy search/login/
# newsletter forms; this generalizes that across the supported languages and
# adds credit-card forms. Curated to avoid short, noisy substrings (e.g. bare
# "card", "cap", "via ", "cep"). The labeler filters further.
FORM_KEYWORDS = (
    # --- Address / generic + autocomplete attribute tokens ---
    "street", "address", "postal", "postcode", "zip",
    "straße", "strasse", "adresse", "postleitzahl", "plz",   # German
    "straat", "adres", "woonplaats",                          # Dutch
    "indirizzo", "provincia",                                 # Italian
    "endereço", "endereco", "morada",                         # Portuguese
    "dirección", "direccion", "calle",                        # Spanish
    "住所", "郵便番号", "都道府県",                              # Japanese
    "code postal", "ville", "rue",                            # French
    "posta kodu", "şehir", "adres",                           # Turkish (adres also pl/nl)
    "kod pocztowy", "ulica", "miasto", "miejscowość",         # Polish
    "адрес", "индекс", "город", "улица", "почтовый",          # Russian
    "주소", "우편번호",                                        # Korean
    "地址", "邮编", "邮政编码", "郵遞區號",                      # Chinese (CN + TW)
    "पता", "पिन",                                             # Hindi
    "آدرس", "کد پستی",                                        # Persian
    "വിലാസം", "പിന്‍കോഡ്",                                     # Malayalam
    # --- Credit card / payment ---
    "credit card", "card number", "cardnumber",
    "cc-number", "cc-csc", "cc-exp", "cardnum", "cvv", "cvc",  # EN + autocomplete
    "kreditkarte", "kartennummer",                            # German
    "creditcard", "kaartnummer",                              # Dutch
    "carta di credito", "numero carta",                       # Italian
    "cartão", "cartao",                                       # Portuguese
    "tarjeta",                                                # Spanish
    "クレジットカード", "カード番号",                           # Japanese
)


def sanitize(text: str) -> str:
    """Strip control bytes (incl. NUL) that corrupt CSV round-tripping.

    Applied to every text field written to the CSV (form HTML, title, url),
    not just the form — a NUL in any column breaks the parser and shifts rows.
    """
    return _CTRL_RE.sub("", text or "")


# Physical-address detection (no LLM), for the --exclude-address mode. Word
# boundaries avoid false hits ("state" in "estate"/"States"); the email guard
# avoids excluding "Email Address" fields (login/newsletter forms we WANT).
_ADDR_KEYWORDS = re.compile(
    r"\b(street|postal|postcode|zip|city|state|province|county|suburb|"
    r"prefecture|region|shipping|billing|town)\b", re.I)
_ADDR_AUTOCOMPLETE = (
    "street-address", "address-line1", "address-line2", "address-line3",
    "address-level1", "address-level2", "address-level3", "address-level4",
    "postal-code", "country", "country-name")


def form_has_address_field(soup) -> bool:
    """True if any field looks like a physical-address field (label/name/id/
    placeholder/autocomplete). Email fields are exempt from the bare-'address'
    rule so 'Email Address' doesn't count as a physical address."""
    for el in soup.find_all(["input", "select", "textarea"]):
        ac = (el.get("autocomplete") or "").lower()
        if any(tok in ac for tok in _ADDR_AUTOCOMPLETE):
            return True
        ctx = " ".join(filter(None, [
            _label_text_for(el, soup), el.get("placeholder", ""),
            el.get("name", ""), el.get("id", "")])).lower()
        itype = (el.get("type") or "text").lower()
        is_email = itype == "email" or "email" in ctx or "e-mail" in ctx
        if not is_email and "address" in ctx:
            return True
        if _ADDR_KEYWORDS.search(ctx):
            return True
    return False


def collapse_ws(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines/tabs) to single spaces.

    Inter-tag whitespace is insignificant for HTML parsing, but an embedded
    newline inside a CSV field is exactly what splits a row and misaligns the
    columns. Collapsing makes every stored value single-line, so each CSV row is
    one physical line and round-trips through pandas no matter what the HTML
    contains.
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_form_inner(form) -> str:
    """Return a form's inner HTML with scripts/styles/comments and control
    bytes removed, and whitespace collapsed to single-line. `form` is a bs4 Tag
    (mutated in place — caller discards it).
    """
    for tag in form.find_all(_STRIP_TAGS):
        tag.decompose()
    for comment in form.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    return collapse_ws(sanitize(form.decode_contents()))


@dataclass
class ExtractConfig:
    languages: tuple[str, ...] = ("en",)      # primary subtags, lowercased
    max_forms: int = 4000
    min_fields: int = 3
    max_fields: int = 25
    tlds: tuple[str, ...] | None = None        # None = allow any TLD
    require_any: tuple[str, ...] = ()          # keep form only if it contains one
    per_page_limit: int = 5                    # cap forms taken from one page
    per_domain_limit: int = 0                  # cap forms taken from one domain (0 = no cap)
    max_html_chars: int = 50000                # skip forms whose cleaned HTML exceeds this
    allow_single_input: bool = False           # also keep forms with exactly 1 field
    single_input_fraction: float = 0.10        # ...but cap them at this fraction of max_forms
    exclude_address: bool = False              # drop forms that contain any address-type field
    # Targeted mining: keep a form only if the ported regex labeler assigns one
    # of these normalized field types to some field (e.g. for under-covered
    # fields: address-line2/3, address-housenumber). Empty = no field-type gate.
    require_field_types: tuple[str, ...] = ()
    # Dedup: skip any form whose sha1(inner_html)[:8] is in this set. Matches the
    # downstream CC_<domain>_<hash>.html naming (label_common_crawl), so seeding
    # it from existing filenames avoids re-pulling forms we already have.
    exclude_hashes: frozenset = frozenset()

    def __post_init__(self):
        self.languages = tuple(l.split("-")[0].lower() for l in self.languages)
        if not isinstance(self.exclude_hashes, frozenset):
            self.exclude_hashes = frozenset(self.exclude_hashes)


def tld_of(url: str) -> str:
    host = urlparse(url).netloc
    return host.rsplit(".", 1)[-1].lower() if "." in host else ""


def domain_of(url: str) -> str:
    """Host without a leading 'www.', lowercased — the key for per-domain caps
    (groups www.x.com and x.com; keeps distinct subdomains separate)."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def detect_language(html_tag, page_text: str) -> str:
    """Resolve a page's primary language subtag.

    Prefers a confident langdetect result on the visible text; falls back to the
    declared <html lang> attribute when detection fails (very short pages).
    """
    try:
        return detect(page_text[:5000]).split("-")[0].lower()
    except LangDetectException:
        declared = (html_tag.get("lang", "") if html_tag else "").split("-")[0]
        return declared.lower()


def _form_text(form_html: str) -> str:
    return form_html.lower()


def form_hash(inner_html: str) -> str:
    """Identity hash for a cleaned form, matching the CC_<domain>_<hash>.html
    naming in label_common_crawl (sha1 of the inner HTML, first 8 hex)."""
    return hashlib.sha1(inner_html.encode("utf-8", "ignore")).hexdigest()[:8]


def load_existing_hashes(html_globs=(), csv_paths=()) -> set:
    """Collect form hashes already present, to seed ExtractConfig.exclude_hashes.

    Reads the <hash> straight out of existing CC_<domain>_<hash>.html filenames,
    and (for prior extraction CSVs that aren't yet saved as files) recomputes it
    from the 'form' column. Both use the same sha1(inner)[:8] key as mining.
    """
    import csv
    import glob
    hashes: set = set()
    for pattern in html_globs:
        for path in glob.glob(pattern):
            stem = os.path.basename(path)
            if stem.endswith(".html"):
                stem = stem[:-5]
            tail = stem.rsplit("_", 1)[-1]
            if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
                hashes.add(tail)
    for cp in csv_paths:
        with open(cp, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("form"):
                    hashes.add(form_hash(row["form"]))
    return hashes


# Lazily-initialized (fillable_fields, regex_hint, norm) so importing ccextract
# stays cheap for callers that don't use require_field_types.
_LABELER = None


def _form_field_types(inner_soup) -> set:
    """Normalized regex field-type labels present in a form (ported Firefox
    heuristic). Used by the require_field_types gate for targeted mining."""
    global _LABELER
    if _LABELER is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from ff_preprocess import fillable_fields
        from heuristics_regexp import regex_hint
        from build_dataset import load_field_types, adjusted_field_name
        ft = load_field_types(
            os.path.join(os.path.dirname(__file__), "..", "..", "dotraining.py"))

        def _norm(raw):
            a = adjusted_field_name(raw) if raw else ""
            return a if a and a in ft else ""

        _LABELER = (fillable_fields, regex_hint, _norm)
    fillable_fields, regex_hint, _norm = _LABELER
    types = set()
    for el in fillable_fields(inner_soup):
        t = _norm(regex_hint(el, inner_soup))
        if t:
            types.add(t)
    return types


def iter_warc_forms(warc_path: str, cfg: ExtractConfig):
    """Yield candidate-form dicts from one WARC file (does not enforce max_forms).

    A cheap `b'<form'` substring check skips the BeautifulSoup parse for the vast
    majority of pages that have no form at all.
    """
    with open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue
            ctype = record.http_headers.get("Content-Type", "") if record.http_headers else ""
            if "text/html" not in ctype:
                continue
            payload = record.content_stream().read()
            if b"<form" not in payload.lower():
                continue

            url = sanitize(record.rec_headers.get("WARC-Target-URI") or "")
            if cfg.tlds is not None and tld_of(url) not in cfg.tlds:
                continue

            soup = BeautifulSoup(payload, "html.parser")
            forms = soup.find_all("form")
            if not forms:
                continue

            lang = detect_language(soup.find("html"), soup.get_text(" ", strip=True))
            if lang not in cfg.languages:
                continue

            title = collapse_ws(sanitize(
                soup.title.get_text(strip=True) if soup.title else ""))[:200]

            taken = 0
            for fi, form in enumerate(forms):
                if taken >= cfg.per_page_limit:
                    break
                inner = clean_form_inner(form)
                if not inner:
                    continue
                if len(inner) > cfg.max_html_chars:
                    continue
                if cfg.require_any and not any(
                        kw in _form_text(inner) for kw in cfg.require_any):
                    continue
                # Dedup against forms we already have (cheap string hash first).
                fhash = form_hash(inner)
                if fhash in cfg.exclude_hashes:
                    continue
                inner_soup = BeautifulSoup(inner, "html.parser")
                if cfg.exclude_address and form_has_address_field(inner_soup):
                    continue
                n_fields = fillable_count(inner_soup)
                in_range = cfg.min_fields <= n_fields <= cfg.max_fields
                single_ok = cfg.allow_single_input and n_fields == 1
                if not (in_range or single_ok):
                    continue
                # Targeted mining: keep only forms that actually contain one of
                # the requested field types (per the ported regex labeler).
                if cfg.require_field_types and not (
                        _form_field_types(inner_soup) & set(cfg.require_field_types)):
                    continue
                taken += 1
                yield {
                    "url": url,
                    "tld": tld_of(url),
                    "lang": lang,
                    "title": title,
                    "form_index": fi,
                    "num_fields": n_fields,
                    "form": inner,
                    "form_hash": fhash,
                }


@dataclass
class ExtractStats:
    by_lang: dict = field(default_factory=dict)
    by_tld: dict = field(default_factory=dict)
    total: int = 0
    single_input: int = 0

    def add(self, row: dict):
        self.total += 1
        self.by_lang[row["lang"]] = self.by_lang.get(row["lang"], 0) + 1
        self.by_tld[row["tld"]] = self.by_tld.get(row["tld"], 0) + 1


def extract_forms(warc_paths: list[str], cfg: ExtractConfig,
                  on_progress=None) -> tuple[list[dict], ExtractStats]:
    """Scan WARC files in order, collecting up to cfg.max_forms candidate forms."""
    rows: list[dict] = []
    stats = ExtractStats()
    # Cap single-input forms so they make up no more than the configured fraction.
    single_cap = (int(cfg.max_forms * cfg.single_input_fraction)
                  if cfg.allow_single_input else 0)
    single_count = 0
    per_domain: dict[str, int] = {}
    for path in warc_paths:
        if len(rows) >= cfg.max_forms:
            break
        for row in iter_warc_forms(path, cfg):
            # Cap forms per domain so one big site can't dominate the sample.
            if cfg.per_domain_limit:
                dom = domain_of(row["url"])
                if per_domain.get(dom, 0) >= cfg.per_domain_limit:
                    continue
                per_domain[dom] = per_domain.get(dom, 0) + 1
            if row["num_fields"] == 1:
                if single_count >= single_cap:
                    continue
                single_count += 1
            rows.append(row)
            stats.add(row)
            if on_progress and stats.total % 50 == 0:
                on_progress(path, stats.total)
            if len(rows) >= cfg.max_forms:
                break
    stats.single_input = single_count
    return rows, stats
