#!/usr/bin/env python3
"""Extract English Common Crawl forms that contain NO address-type fields.

For collecting non-address form contexts (login, search, newsletter, contact-
without-address, surveys, etc.). Forms are cleaned and wrapped exactly like the
other sample folders, but NOT labeled (no data-moz-autofill-type) and NOT sent
to any LLM. Output goes to samples/non_address/ as NA_*.html.

Address detection is heuristic (gen.ccextract.form_has_address_field) — it drops
any form with a street/city/state/postal/country/etc. field, while keeping
'Email Address' fields (so login/newsletter forms survive).

Usage:
  python extract_non_address.py \\
      --warc-glob "/Users/Rrando/Documents/common_crawl/CC-MAIN-*.warc.gz" \\
      --count 400
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

from bs4 import BeautifulSoup

from gen.ccextract import ExtractConfig, iter_warc_forms
from gen.htmlforms import wrap_form

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(REPO_ROOT, "samples", "non_address")


def signature(form_html: str) -> str:
    """Structural signature (field tags/types/names) to drop near-duplicate
    forms (e.g. the same login/newsletter template across many sites)."""
    soup = BeautifulSoup(form_html, "html.parser")
    parts = []
    for el in soup.find_all(["input", "select", "textarea"]):
        itype = (el.get("type") or "text").lower()
        if el.name == "input" and itype in ("hidden", "submit", "image", "button", "reset"):
            continue
        key = (el.get("name") or el.get("id") or el.get("placeholder") or "").lower()
        parts.append(f"{el.name}:{itype}:{key}")
    return hashlib.sha1("|".join(sorted(parts)).encode("utf-8")).hexdigest()


def short_domain(domain: str) -> str:
    return "".join(c for c in domain.split(".")[0] if c.isalnum())[:16] or "site"


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warc", nargs="*", default=[])
    p.add_argument("--warc-glob", default=None)
    p.add_argument("--count", type=int, default=400, help="number of forms to write")
    p.add_argument("--languages", nargs="+", default=["en"])
    p.add_argument("--min-fields", type=int, default=2)
    p.add_argument("--max-fields", type=int, default=25)
    p.add_argument("--per-page-limit", type=int, default=2)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    warc_paths = list(args.warc)
    if args.warc_glob:
        warc_paths += sorted(glob.glob(args.warc_glob))
    warc_paths = [p for p in warc_paths if p and os.path.exists(p)]
    if not warc_paths:
        print("ERROR: no WARC files found (use --warc and/or --warc-glob).",
              file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    cfg = ExtractConfig(
        languages=tuple(args.languages),
        min_fields=args.min_fields,
        max_fields=args.max_fields,
        per_page_limit=args.per_page_limit,
        require_any=(),            # no keyword requirement
        exclude_address=True,      # the whole point: drop address forms
    )
    print(f"Scanning {len(warc_paths)} WARC file(s) for {args.count} "
          f"non-address {cfg.languages} forms ({cfg.min_fields}-{cfg.max_fields} fields)")

    seen: set[str] = set()
    written = 0
    for path in warc_paths:
        if written >= args.count:
            break
        for row in iter_warc_forms(path, cfg):
            sig = signature(row["form"])
            if sig in seen:
                continue
            seen.add(sig)
            html = wrap_form(row["form"], site=(row["tld"] or "common-crawl"),
                             note="Common Crawl form, other context, unlabeled.")
            h = hashlib.sha1(row["form"].encode("utf-8", "ignore")).hexdigest()[:8]
            fname = f"NA_{short_domain(row['tld'])}_{h}.html"
            with open(os.path.join(args.out, fname), "w", encoding="utf-8") as fh:
                fh.write(html)
            written += 1
            if written % 50 == 0:
                print(f"  {written}/{args.count} written "
                      f"(current file: {os.path.basename(path)})", flush=True)
            if written >= args.count:
                break

    print(f"\nDone: wrote {written} non-address forms to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
