"""Fast German address-form mining by WARC scan.

Why this can beat the index path once CloudFront throttles: WARC prefixes are a
few LARGE sequential GETs (far better under a per-IP request-rate limit than the
index path's thousands of tiny range requests), and the scan is made cheap with
a byte-level German-keyword PRE-FILTER that skips HTML parsing on ~all pages.

Tricks:
  * raw-bytes keyword pre-filter BEFORE any BeautifulSoup/lxml parse or langdetect
  * lxml parser; no langdetect (German keyword presence is the language signal)
  * big prefixes from WARCs spread across the crawl; prefix deleted after scan
  * dedup vs existing corpus + prior batches; global per-domain cap

Usage: .venv/bin/python -I data_tools/mine_german_warc.py [target] [prefix_mb] [max_warcs]
Output: /tmp/cc_warc/german_from_warc.csv
"""
import sys, os, io, csv, gzip, subprocess, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0
from gen.ccextract import clean_form_inner, form_hash, domain_of, load_existing_hashes, tld_of
from gen.htmlforms import fillable_count
from ff_preprocess import fillable_fields, get_label_strings
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name
from collections import Counter
import re as _re

_FT = load_field_types(os.path.join(HERE, "..", "dotraining.py"))
_ADDR_TYPES = {"postal-code", "street-address", "address-line1", "address-line2",
               "address-line3", "address-level1", "address-level2", "address-level3",
               "address-housenumber"}
# Distinctive postal/street tokens on a field's own label/name/id/placeholder/autocomplete.
# Precise on purpose (no bare 'ort'/'address'/'adresse') so we gate on real address INPUTS.
_FIELD_ADDR = _re.compile(r"(\bplz\b|postleitzahl|stra[sß]e|hausnummer|\bpostal\s*code\b|"
                          r"postcode|\bzip\b|\bstreet\b|address-?line|address-?level)", _re.I)
def _norm(r):
    a = adjusted_field_name(r) if r else ""
    return a if a and a in _FT else ""
def form_has_address_input(isoup):
    """True iff some fillable field is a real address/postal INPUT (regex-typed or
    its label/name is a distinctive postal/street token)."""
    for el in fillable_fields(isoup):
        if _norm(regex_hint(el, isoup)) in _ADDR_TYPES:
            return True
        ctx = " ".join(get_label_strings(el, isoup)) + " " + (el.get("name") or "") + \
              " " + (el.get("id") or "") + " " + (el.get("placeholder") or "") + \
              " " + (el.get("autocomplete") or "")
        if _FIELD_ADDR.search(ctx):
            return True
    return False

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 900
PREFIX_MB = int(sys.argv[2]) if len(sys.argv) > 2 else 600
MAX_WARCS = int(sys.argv[3]) if len(sys.argv) > 3 else 120
PER_DOMAIN = 3
WDIR = "/tmp/cc_warc"
OUT = os.environ.get("OUT_CSV", os.path.join(WDIR, "german_from_warc.csv"))

# Distinctive German address/postal tokens (utf-8, lowercase) for the raw-bytes
# pre-filter -- present in labels/placeholders of German address forms. Kept
# distinctive to avoid matching everything (no bare "ort").
# Broad address prefilter (incl. ambiguous adresse/ort/stadt/land) to maximize
# German RECALL -- precision now comes from a full-PAGE langdetect=='de' check
# (reliable, unlike substring tokens or short form-label langdetect).
# Postal/street byte prefilter: pages with these are likely to carry an address
# INPUT (precision now enforced per-field by form_has_address_input), so skip the
# broad ort/stadt/land/address tokens that dragged in non-address German forms.
BYTE_HINTS = [b"stra\xc3\x9fe", b"strasse", b"postleitzahl", b"plz", b"hausnummer",
              b"lieferadresse", b"rechnungsadresse"]
def form_has_kw(s):
    return any(k in s for k in FORM_KW)

seen = load_existing_hashes(
    html_globs=[os.path.join(HERE, "..", "data", "common_crawl", "CC_*.html")],
    csv_paths=[p for p in (
        [os.path.join(HERE, "..", "data", "cc_mined_2026", f"cc_forms_shard{i}.csv") for i in range(1, 6)]
        + ["/tmp/cc_warc/mined_campaign.csv", "/tmp/cc_warc/german_from_index.csv",
           os.path.join(HERE, "..", "data", "cc_german_2026", "german_forms.csv"),
           os.path.join(HERE, "..", "data", "cc_german_2026", "german_address_forms.csv")]  # prior German batches
    ) if os.path.exists(p)])
print(f"{len(seen)} existing hashes (incl. prior batches)", flush=True)

# WARC_OFFSET skips the first N step-sampled WARCs so a continuation run pulls
# FRESH prefixes instead of re-downloading ones already scanned.
OFFSET = int(os.environ.get("WARC_OFFSET", "0"))
all_paths = gzip.open(os.path.join(WDIR, "warc.paths.gz"), "rt").read().split()
step = max(1, len(all_paths) // (MAX_WARCS + OFFSET))
warcs = all_paths[::step][OFFSET:OFFSET + MAX_WARCS]
print(f"{len(warcs)} WARCs selected (offset {OFFSET}), {PREFIX_MB}MB prefixes", flush=True)

per_domain = Counter(); by_field = Counter(); kept = 0
fh = open(OUT, "w", newline="", encoding="utf-8"); w = csv.writer(fh)
w.writerow(["url", "tld", "num_fields", "form_hash", "keywords", "warc", "form"])

def dl(path, dest):
    url = "https://data.commoncrawl.org/" + path
    for a in range(6):
        rc = subprocess.run(["curl", "-sS", "--max-time", "900", "-r", f"0-{PREFIX_MB*1024*1024-1}",
                             url, "-o", dest], capture_output=True).returncode
        if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            return True
        time.sleep(8 * (a + 1))
    return False

t0 = time.time(); scanned_pages = 0; prefiltered = 0
for wi, path in enumerate(warcs):
    if kept >= TARGET:
        break
    dest = os.path.join(WDIR, "gcur.warc.gz")
    if os.path.exists(dest):
        os.remove(dest)
    td = time.time()
    if not dl(path, dest):
        print(f"[{wi+1}/{len(warcs)}] dl failed", flush=True); continue
    dl_s = time.time() - td
    got = 0; ts = time.time()
    try:
        with open(dest, "rb") as stream:
            for rec in ArchiveIterator(stream):
                if rec.rec_type != "response":
                    continue
                payload = rec.content_stream().read()
                scanned_pages += 1
                low = payload.lower()
                if b"<form" not in low:
                    continue
                if not any(h in low for h in BYTE_HINTS):   # <-- cheap pre-filter, no parse
                    continue
                prefiltered += 1
                url = rec.rec_headers.get("WARC-Target-URI") or ""
                dom = domain_of(url)
                if per_domain[dom] >= PER_DOMAIN:
                    continue
                soup = BeautifulSoup(payload, "lxml")
                # Reliable language gate: langdetect on the FULL page text (long, so
                # accurate -- unlike short form labels). Only runs on prefiltered pages.
                try:
                    if detect(soup.get_text(" ", strip=True)[:3000]) != "de":
                        continue
                except LangDetectException:
                    continue
                for form in soup.find_all("form"):
                    inner = clean_form_inner(form)
                    if not inner or len(inner) > 50000:
                        continue
                    h = form_hash(inner)
                    if h in seen:
                        continue
                    isoup = BeautifulSoup(inner, "lxml")
                    n = fillable_count(isoup)
                    if not (3 <= n <= 25):
                        continue
                    if not form_has_address_input(isoup):   # real address/postal INPUT required
                        continue
                    seen.add(h); per_domain[dom] += 1
                    w.writerow([url, tld_of(url), n, h, "", path.rsplit("/", 1)[-1], inner])
                    kept += 1; got += 1
                    break
                if kept >= TARGET:
                    break
    except Exception:
        pass
    fh.flush(); os.remove(dest)
    print(f"[{wi+1}/{len(warcs)}] +{got} total={kept} | dl {dl_s:.0f}s scan {time.time()-ts:.0f}s "
          f"| pages {scanned_pages} prefilt {prefiltered} | {time.time()-t0:.0f}s", flush=True)

fh.close()
print(f"\nDONE: {kept} German forms -> {OUT}  ({time.time()-t0:.0f}s)")
print("distinct domains:", len(per_domain))
