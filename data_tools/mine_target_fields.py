"""Smoke test for TARGETED Common-Crawl mining.

Scans local CC forms, runs the ported Firefox regex labeler on every fillable
field, and KEEPS only forms that contain >=1 under-covered target field
(address-line1/2/3, address-housenumber, tel-split). Reports the hit rate (how
selective the filter is -> how many raw forms you'd crawl per 1,000 kept), which
target fields drive the keeps, and the language mix of kept forms.

Usage: .venv/bin/python -I data_tools/mine_target_fields.py
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bs4 import BeautifulSoup
from ff_preprocess import fillable_fields
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name
from collections import Counter

FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))
def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""

TARGET = {
    "address-line1", "address-line2", "address-line3", "address-housenumber",
    "tel-national", "tel-area-code", "tel-local",
    "tel-local-prefix", "tel-local-suffix", "tel-extension",
}

langmap = json.load(open("data/cc_lang_map.json")) if os.path.exists("data/cc_lang_map.json") else {}

files = sorted(glob.glob("data/common_crawl/*.html"))
scanned = with_fields = kept = 0
target_field_hits = Counter()   # which target fields drive keeps
kept_lang = Counter()
kept_target_fields = 0
examples = []

for path in files:
    fn = os.path.basename(path)
    try:
        soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "html.parser")
    except Exception:
        continue
    scanned += 1
    labels = [norm(regex_hint(el, soup)) for el in fillable_fields(soup)]
    labels = [l for l in labels if l]
    if labels:
        with_fields += 1
    hits = [l for l in labels if l in TARGET]
    if hits:
        kept += 1
        kept_target_fields += len(hits)
        for h in set(hits):
            target_field_hits[h] += 1
        kept_lang[langmap.get(fn, "?")] += 1
        if len(examples) < 8:
            examples.append((fn, langmap.get(fn, "?"), Counter(hits)))

rate = kept / scanned if scanned else 0
print(f"scanned={scanned}  with-any-labeled-field={with_fields}  KEPT(>=1 target)={kept}")
print(f"hit rate = {100*rate:.1f}%  ->  ~{round(1/rate) if rate else 0} raw forms crawled per 1 kept "
      f"(~{round(1000/rate) if rate else 0} to reach 1,000 targeted)")
print(f"avg target fields per kept form = {kept_target_fields/kept:.1f}" if kept else "")

print("\nforms containing each target field (of kept):")
for lbl, n in target_field_hits.most_common():
    print(f"  {lbl:22}{n:>5}")

print("\nlanguage mix of kept forms (top 12):")
for lg, n in kept_lang.most_common(12):
    print(f"  {lg:6}{n:>5}")

print("\nexample kept forms:")
for fn, lg, c in examples:
    print(f"  [{lg}] {fn}: {dict(c)}")
