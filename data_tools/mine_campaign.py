"""Scaled targeted Common-Crawl mining campaign.

Downloads byte-range prefixes from many distinct WARC files (throttled with
backoff), scans each for forms containing an under-covered target field, dedupes
globally (vs existing corpus AND within-run), caps forms per domain across the
whole campaign, and accumulates until TARGET forms are kept. Each prefix is
deleted after scanning to bound disk. Writes kept forms incrementally.

Usage: .venv/bin/python -I data_tools/mine_campaign.py [target] [prefix_mb] [max_warcs]
"""
import sys, os, csv, gzip, subprocess, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from bs4 import BeautifulSoup
from gen.ccextract import (ExtractConfig, iter_warc_forms, load_existing_hashes,
                           domain_of, _form_field_types)
from collections import Counter

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
PREFIX_MB = int(sys.argv[2]) if len(sys.argv) > 2 else 350
MAX_WARCS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
PER_DOMAIN = 3
WDIR = "/tmp/cc_warc"
PATHS_GZ = os.path.join(WDIR, "warc.paths.gz")
OUT = os.path.join(WDIR, "mined_campaign.csv")

FIELDS = ("address-line1", "address-line2", "address-line3", "address-housenumber",
          "tel-national", "tel-area-code", "tel-local",
          "tel-local-prefix", "tel-local-suffix", "tel-extension")

print("loading existing hashes (dedup) ...", flush=True)
seen_hashes = load_existing_hashes(
    html_globs=[os.path.join(HERE, "..", "data", "common_crawl", "CC_*.html")],
    csv_paths=[p for p in [os.path.join(HERE, "..", "data", "cc_mined_2026",
               f"cc_forms_shard{i}.csv") for i in range(1, 6)] if os.path.exists(p)])
print(f"  {len(seen_hashes)} existing", flush=True)

# Spread WARC choices across the crawl's segments for domain/topic diversity.
all_paths = gzip.open(PATHS_GZ, "rt").read().split()
step = max(1, len(all_paths) // MAX_WARCS)
warc_paths = all_paths[::step][:MAX_WARCS]
print(f"selected {len(warc_paths)} WARCs (of {len(all_paths)}), prefix {PREFIX_MB}MB", flush=True)

cfg = ExtractConfig(
    languages=("en", "de", "es", "fr", "it", "nl", "pt", "ja", "pl", "ru", "tr", "sv", "da", "cs"),
    max_forms=10 ** 9, min_fields=3, max_fields=25,
    per_page_limit=3, require_field_types=FIELDS,
    exclude_hashes=seen_hashes,          # note: a frozenset snapshot; we also track per-run below
)

per_domain = Counter()
by_field = Counter()
by_lang = Counter()
kept = 0
fh = open(OUT, "w", newline="", encoding="utf-8")
writer = csv.writer(fh)
writer.writerow(["url", "tld", "lang", "title", "num_fields", "form_hash",
                 "target_fields", "warc", "form"])

def dl_prefix(path, dest):
    url = "https://data.commoncrawl.org/" + path
    for attempt in range(6):
        rc = subprocess.run(
            ["curl", "-sS", "--retry", "6", "--retry-delay", "6", "--retry-all-errors",
             "--max-time", "600", "-r", f"0-{PREFIX_MB*1024*1024-1}", url, "-o", dest],
            capture_output=True).returncode
        if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            return True
        time.sleep(10 * (attempt + 1))   # back off harder on repeated failure
    return False

t0 = time.time()
for wi, path in enumerate(warc_paths):
    if kept >= TARGET:
        break
    dest = os.path.join(WDIR, "cur.warc.gz")
    if os.path.exists(dest):
        os.remove(dest)
    if not dl_prefix(path, dest):
        print(f"[{wi+1}/{len(warc_paths)}] download failed, skipping", flush=True)
        continue
    warc_name = path.rsplit("/", 1)[-1]
    got = 0
    cfg.exclude_hashes = frozenset(seen_hashes)   # refresh with everything kept so far
    try:
        for row in iter_warc_forms(dest, cfg):
            h = row["form_hash"]
            if h in seen_hashes:
                continue
            dom = domain_of(row["url"])
            if per_domain[dom] >= PER_DOMAIN:
                continue
            tf = sorted(_form_field_types(BeautifulSoup(row["form"], "html.parser")) & set(FIELDS))
            if not tf:
                continue
            seen_hashes.add(h)
            per_domain[dom] += 1
            for t in tf:
                by_field[t] += 1
            by_lang[row["lang"]] += 1
            writer.writerow([row["url"], row["tld"], row["lang"], row["title"],
                             row["num_fields"], h, "|".join(tf), warc_name, row["form"]])
            kept += 1
            got += 1
            if kept >= TARGET:
                break
    except Exception as e:
        pass  # truncated prefix end
    fh.flush()
    os.remove(dest)
    print(f"[{wi+1}/{len(warc_paths)}] {warc_name[-30:]}: +{got} (total {kept}, {time.time()-t0:.0f}s)", flush=True)

fh.close()
print(f"\n=== DONE: {kept} target forms -> {OUT} ===")
print("by field:", dict(by_field.most_common()))
print("by lang:", dict(by_lang.most_common()))
print(f"distinct domains: {len(per_domain)} (cap {PER_DOMAIN}/domain)")
