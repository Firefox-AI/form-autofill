"""Targeted Common-Crawl mining run.

Scans a (possibly byte-range-truncated) WARC, keeping only forms that contain an
under-covered target field (per the ported regex labeler), capping forms per
domain, and deduping against forms we already have. Reports yield by target
field / language / domain and writes the kept forms to a CSV ready for labeling.

Usage: .venv/bin/python -I data_tools/run_targeted_mine.py <warc.gz> [max_forms]
"""
import sys, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)            # data_tools (ff_preprocess, heuristics_regexp, build_dataset)
sys.path.insert(0, os.path.join(HERE, ".."))  # repo root
from bs4 import BeautifulSoup
from gen.ccextract import (ExtractConfig, iter_warc_forms, load_existing_hashes,
                           domain_of, _form_field_types)
from collections import Counter

warc = sys.argv[1]
max_forms = int(sys.argv[2]) if len(sys.argv) > 2 else 300

TARGET = ("address-line1", "address-line2", "address-line3", "address-housenumber",
          "tel-national", "tel-area-code", "tel-local",
          "tel-local-prefix", "tel-local-suffix", "tel-extension")

print("loading existing form hashes for dedup ...", flush=True)
existing = load_existing_hashes(
    html_globs=[os.path.join(HERE, "..", "data", "common_crawl", "CC_*.html")],
    csv_paths=[p for p in [
        os.path.join(HERE, "..", "data", "cc_mined_2026", f"cc_forms_shard{i}.csv")
        for i in range(1, 6)] if os.path.exists(p)],
)
print(f"  {len(existing)} existing hashes", flush=True)

cfg = ExtractConfig(
    languages=("en", "de", "es", "fr", "it", "nl", "pt", "ja", "pl", "ru", "tr"),
    max_forms=max_forms,
    min_fields=3, max_fields=25,
    per_page_limit=3,
    per_domain_limit=2,
    require_field_types=TARGET,
    exclude_hashes=existing,
)

kept = []
per_domain = {}
by_field = Counter()
by_lang = Counter()
dom_counter = Counter()
try:
    for row in iter_warc_forms(warc, cfg):
        dom = domain_of(row["url"])
        if per_domain.get(dom, 0) >= cfg.per_domain_limit:
            continue
        per_domain[dom] = per_domain.get(dom, 0) + 1
        types = _form_field_types(BeautifulSoup(row["form"], "html.parser"))
        row["target_fields"] = sorted(types & set(TARGET))
        for t in row["target_fields"]:
            by_field[t] += 1
        by_lang[row["lang"]] += 1
        dom_counter[dom] += 1
        kept.append(row)
        if len(kept) % 20 == 0:
            print(f"  kept {len(kept)} ...", flush=True)
        if len(kept) >= max_forms:
            break
except Exception as e:
    print(f"scan stopped: {type(e).__name__}: {str(e)[:80]} (expected at a truncated prefix)", flush=True)

print(f"\n=== KEPT {len(kept)} target forms ===")
print("by target field:", dict(by_field.most_common()))
print("by language:", dict(by_lang.most_common()))
print(f"distinct domains: {len(dom_counter)}  (per-domain cap {cfg.per_domain_limit})")
print("top domains:", dict(dom_counter.most_common(8)))

out = "/tmp/cc_warc/mined_targeted.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["url", "tld", "lang", "title", "num_fields", "form_hash",
                "target_fields", "form"])
    for r in kept:
        w.writerow([r["url"], r["tld"], r["lang"], r["title"], r["num_fields"],
                    r["form_hash"], "|".join(r["target_fields"]), r["form"]])
print(f"\nwrote {len(kept)} forms -> {out}")
