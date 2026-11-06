"""Query the Common Crawl columnar (Parquet) index for candidate pages, by
language and/or TLD, writing their exact WARC record locations.

Anonymous + free: the index parquet file list is published at
<crawl>/cc-index-table.paths.gz and the files are served over HTTPS from
data.commoncrawl.org (no AWS creds, no requester-pays). DuckDB reads them with
predicate pushdown; CloudFront 503-throttles bursts, so threads are capped and
HTTP retries/backoff are on.

Output: /tmp/cc_warc/index_candidates.csv (warc_filename, offset, length, url,
content_languages) -> feed to cc_index_fetch.py.

Usage: .venv/bin/python -I data_tools/cc_index_query.py <crawl> <lang_iso3> [max] [tlds]
  e.g. cc_index_query.py CC-MAIN-2024-33 deu 5000 de,at,ch
"""
import sys, os, gzip, subprocess, csv
import duckdb

CRAWL = sys.argv[1] if len(sys.argv) > 1 else "CC-MAIN-2024-33"
LANG = sys.argv[2] if len(sys.argv) > 2 else "deu"       # ISO-639-3 as stored in the index
MAXN = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
TLDS = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ()
WDIR = "/tmp/cc_warc"
os.makedirs(WDIR, exist_ok=True)

# 1) fetch the published parquet file list for this crawl
paths_gz = os.path.join(WDIR, f"{CRAWL}-index.paths.gz")
if not os.path.exists(paths_gz):
    subprocess.run(["curl", "-sS", "--retry", "6", "--retry-delay", "5", "--max-time", "60",
                    f"https://data.commoncrawl.org/crawl-data/{CRAWL}/cc-index-table.paths.gz",
                    "-o", paths_gz], check=True)
urls = ["https://data.commoncrawl.org/" + l.strip()
        for l in gzip.open(paths_gz, "rt") if "subset=warc/" in l]
print(f"{CRAWL}: {len(urls)} index parquet files", flush=True)

# 2) query with pushdown; CloudFront-friendly settings
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
for s in ["SET threads=2;", "SET http_retries=10;", "SET http_retry_wait_ms=2000;",
          "SET http_retry_backoff=2;", "SET http_keep_alive=true;"]:
    try:
        con.execute(s)
    except Exception:
        pass

where = ["content_mime_type='text/html'", "fetch_status=200",
         f"contains(content_languages, '{LANG}')"]
if TLDS:
    where.append("url_host_tld IN (" + ",".join(f"'{t}'" for t in TLDS) + ")")
# Bias toward address/checkout FORM pages (a plain language query returns mostly
# blogs/articles). Cheap parquet string-contains, not the expensive CDX regex.
# Env URL_HINTS overrides; covers several languages' checkout/contact words.
URL_HINTS = os.environ.get("URL_HINTS", "").split(",") if os.environ.get("URL_HINTS") else [
    "checkout", "kasse", "warenkorb", "bestell", "kontakt", "anmeld", "register",
    "konto", "account", "address", "adresse", "shipping", "versand", "lieferung"]
if URL_HINTS:
    ors = " OR ".join(f"contains(url, '{h}')" for h in URL_HINTS)
    where.append(f"({ors})")
q = f"""SELECT warc_filename, warc_record_offset AS roff, warc_record_length AS rlen,
               url, content_languages
        FROM read_parquet(?)
        WHERE {' AND '.join(where)} LIMIT {MAXN}"""
print("querying index (CloudFront throttles -> can take minutes) ...", flush=True)
rows = con.execute(q, [urls]).fetchall()

out = os.path.join(WDIR, "index_candidates.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["warc_filename", "offset", "length", "url", "content_languages"])
    w.writerows(rows)
print(f"wrote {len(rows)} candidate records -> {out}", flush=True)
