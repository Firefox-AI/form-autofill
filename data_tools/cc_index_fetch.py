"""Fetch WARC records listed by cc_index_query.py (one byte-range per page) and
keep those whose forms contain an under-covered target field.

Only German (or whatever the index query selected) pages are fetched -- ~65KB
per record instead of 1GB WARCs -- so single-language mining is orders of
magnitude cheaper than brute-force WARC scanning. Dedupes vs existing corpus and
within-run, caps forms per domain, throttles fetches.

Usage: .venv/bin/python -I data_tools/cc_index_fetch.py [target] [candidates_csv]
"""
import sys, os, csv, io, time, subprocess, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from gen.ccextract import (ExtractConfig, clean_form_inner, form_hash, domain_of,
                           load_existing_hashes, _form_field_types, tld_of,
                           form_has_address_field)
from gen.htmlforms import fillable_count
from collections import Counter

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CAND = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cc_warc/index_candidates.csv"
PER_DOMAIN = 3
# GATE modes:
#   keyword (default): keep any form whose HTML MENTIONS an address/postal term
#     (multilingual). Does NOT require the regex to label fields -> keeps MESSY forms
#     whose fields the regex can't parse (the high-value training cases), not just the
#     clean/English-labeled ones. The labeling pipeline assigns types later.
#   address: keep only if the regex labeler finds an address field (stricter, biased).
#   target:  narrow under-covered-field gate (address-line2/3, tel-split...).
GATE = os.environ.get("GATE", "keyword")

# Multilingual address/postal keywords (German-forward for this batch). A form
# that mentions any of these is an address form even if its field labels are too
# messy for the regex to type. Generic + several languages so it is not English-only.
ADDR_KEYWORDS = [
    # German
    "straße", "strasse", "str.", "adresse", "anschrift", "postleitzahl", "plz",
    "ort", "stadt", "wohnort", "hausnummer", "lieferadresse", "rechnungsadresse",
    "lieferung", "versandadresse", "bundesland",
    # generic / English
    "address", "street", "postal", "postcode", "zip", "city", "state", "province",
    "country", "county", "suburb", "town",
    # a few other languages (reuse across non-German crawls)
    "adres", "adresse", "indirizzo", "dirección", "direccion", "endereço", "endereco",
    "code postal", "ville", "rue", "住所", "郵便番号", "адрес", "индекс", "город",
]
def has_address_keyword(inner_lower):
    return any(k in inner_lower for k in ADDR_KEYWORDS)
FIELDS = {"address-line1", "address-line2", "address-line3", "address-housenumber",
          "tel-national", "tel-area-code", "tel-local",
          "tel-local-prefix", "tel-local-suffix", "tel-extension"}
# Broader address/postal field set, recorded for info under the address gate.
ADDR_FIELDS = FIELDS | {"street-address", "postal-code", "address-level1",
                        "address-level2", "address-level3", "country", "country-name"}
cfg = ExtractConfig(min_fields=3, max_fields=25)   # only for the field-count bounds

seen = load_existing_hashes(
    html_globs=[os.path.join(HERE, "..", "data", "common_crawl", "CC_*.html")],
    csv_paths=[p for p in (
        [os.path.join(HERE, "..", "data", "cc_mined_2026", f"cc_forms_shard{i}.csv")
         for i in range(1, 6)]
        + ["/tmp/cc_warc/mined_campaign.csv"]   # also skip the all-language batch we just mined
    ) if os.path.exists(p)])
print(f"{len(seen)} existing hashes", flush=True)

def fetch_record(warc_filename, offset, length):
    # NOTE: curl without -f returns CloudFront 503 "SlowDown" XML bodies as a
    # success, so we must validate the gzip magic ourselves and back off --
    # trusting the exit code silently drops every throttled record.
    url = "https://data.commoncrawl.org/" + warc_filename
    rng = f"{offset}-{offset+length-1}"
    for attempt in range(6):
        blob = subprocess.run(["curl", "-sS", "--max-time", "120", "-r", rng, url],
                              capture_output=True).stdout
        if blob[:2] == b"\x1f\x8b":        # valid gzip (a WARC record member)
            return blob
        time.sleep(5 * (attempt + 1))      # throttled -> back off
    return None

per_domain = Counter(); by_field = Counter(); kept = 0
fh = open("/tmp/cc_warc/mined_index.csv", "w", newline="", encoding="utf-8")
w = csv.writer(fh)
w.writerow(["url", "tld", "lang", "num_fields", "form_hash", "target_fields", "form"])

rows = list(csv.DictReader(open(CAND, encoding="utf-8")))
print(f"{len(rows)} candidate records", flush=True)
for i, c in enumerate(rows):
    if kept >= TARGET:
        break
    dom = domain_of(c["url"])
    if per_domain[dom] >= PER_DOMAIN:
        continue
    blob = fetch_record(c["warc_filename"], int(c["offset"]), int(c["length"]))
    if not blob:
        continue
    try:
        rec = next(ArchiveIterator(io.BytesIO(blob)))
        payload = rec.content_stream().read()
    except Exception:
        continue
    soup = BeautifulSoup(payload, "html.parser")
    for form in soup.find_all("form"):
        inner = clean_form_inner(form)
        if not inner or len(inner) > cfg.max_html_chars:
            continue
        h = form_hash(inner)
        if h in seen:
            continue
        isoup = BeautifulSoup(inner, "html.parser")
        n = fillable_count(isoup)
        if not (cfg.min_fields <= n <= cfg.max_fields):
            continue
        types = _form_field_types(isoup)   # ported Firefox regex labeler (info only in keyword mode)
        tf = sorted(types & ADDR_FIELDS)   # recorded for info regardless of gate
        if GATE == "keyword":
            # Loose, multilingual, label-agnostic: keep messy address forms too.
            if not has_address_keyword(inner.lower()):
                continue
        elif GATE == "address":
            if not tf:
                continue
        else:  # target
            tf = sorted(types & FIELDS)
            if not tf:
                continue
        seen.add(h); per_domain[dom] += 1
        for t in tf:
            by_field[t] += 1
        w.writerow([c["url"], tld_of(c["url"]), c["content_languages"], n, h, "|".join(tf), inner])
        kept += 1
        break   # one form per page is plenty
    if (i + 1) % 50 == 0:
        print(f"  fetched {i+1}/{len(rows)}, kept {kept}", flush=True)
    time.sleep(0.3)   # be gentle with CloudFront

fh.close()
print(f"\nDONE: kept {kept} -> /tmp/cc_warc/mined_index.csv")
print("by field:", dict(by_field.most_common()))
print(f"distinct domains: {len(per_domain)}")
