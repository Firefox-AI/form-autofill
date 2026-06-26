#!/usr/bin/env python3
"""Extract candidate web forms from Common Crawl WARC files into a richer CSV.

A script port of notebooks/Common Crawl Form.ipynb, generalized to accept a
list of desired languages and to emit extra context columns. The output CSV
(url, tld, lang, title, form_index, num_fields, form) is consumed downstream by
label_common_crawl.py, which reads the `form` column.

Examples:
  # English + German, up to 4000 forms, scanning several WARC segments
  python extract_forms.py \\
      --warc /Users/Rrando/Documents/common_crawl/CC-MAIN-...-0000{0,2,3,4,5}.warc.gz \\
      --languages en de --max-forms 4000 \\
      --out ./data/cc_forms_en_de.csv

  # Quick smoke test
  python extract_forms.py --warc <one.warc.gz> --max-forms 20 --out /tmp/sample.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from gen.ccextract import FORM_KEYWORDS, ExtractConfig, extract_forms


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warc", nargs="*", default=[],
                   help="one or more WARC (.warc.gz) file paths")
    p.add_argument("--warc-glob", default=None,
                   help="glob for WARC files (alternative/addition to --warc)")
    p.add_argument("--languages", nargs="+", default=["en"],
                   help="desired language primary subtags, e.g. en de fr")
    p.add_argument("--max-forms", type=int, default=4000,
                   help="total candidate forms to collect across all WARC files")
    p.add_argument("--min-fields", type=int, default=3)
    p.add_argument("--max-fields", type=int, default=25)
    p.add_argument("--tlds", nargs="*", default=None,
                   help="restrict to these TLDs (default: any)")
    p.add_argument("--require-any", nargs="*", default=list(FORM_KEYWORDS),
                   help="keep a form only if its HTML contains one of these "
                        "substrings (case-insensitive). Defaults to a multilingual "
                        "address/contact/payment keyword set; pass with no values to "
                        "disable filtering, or pass your own words to override.")
    p.add_argument("--per-page-limit", type=int, default=5,
                   help="max forms taken from a single page")
    p.add_argument("--max-html-chars", type=int, default=50000,
                   help="skip forms whose cleaned inner HTML exceeds this length")
    p.add_argument("--allow-single-input", action="store_true",
                   help="also keep some forms with exactly 1 fillable field "
                        "(otherwise excluded by --min-fields)")
    p.add_argument("--single-input-fraction", type=float, default=0.10,
                   help="max fraction of collected forms allowed to be single-input")
    p.add_argument("--out", required=True, help="output CSV path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    warc_paths = list(args.warc)
    if args.warc_glob:
        warc_paths += sorted(glob.glob(args.warc_glob))
    warc_paths = [p for p in warc_paths if p]
    missing = [p for p in warc_paths if not os.path.exists(p)]
    if not warc_paths:
        print("ERROR: no WARC files given (use --warc and/or --warc-glob).",
              file=sys.stderr)
        return 2
    if missing:
        print(f"ERROR: WARC file(s) not found: {missing}", file=sys.stderr)
        return 2

    cfg = ExtractConfig(
        languages=tuple(args.languages),
        max_forms=args.max_forms,
        min_fields=args.min_fields,
        max_fields=args.max_fields,
        tlds=tuple(args.tlds) if args.tlds else None,
        require_any=tuple(k.lower() for k in args.require_any),
        per_page_limit=args.per_page_limit,
        max_html_chars=args.max_html_chars,
        allow_single_input=args.allow_single_input,
        single_input_fraction=args.single_input_fraction,
    )

    print(f"Scanning {len(warc_paths)} WARC file(s) for up to {cfg.max_forms} "
          f"forms in languages={cfg.languages} "
          f"(fields {cfg.min_fields}-{cfg.max_fields}, max {cfg.per_page_limit}/page)")
    print(f"require-any keywords ({len(cfg.require_any)}): "
          f"{list(cfg.require_any) if cfg.require_any else 'DISABLED'}")

    def progress(path, n):
        print(f"  {n} forms collected (current file: {os.path.basename(path)})",
              flush=True)

    rows, stats = extract_forms(warc_paths, cfg, on_progress=progress)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = ["url", "tld", "lang", "title", "form_index", "num_fields", "form"]
    pd.DataFrame(rows, columns=cols).to_csv(args.out, index=True)

    print("\n--- Summary ---")
    print(f"collected: {stats.total} forms -> {args.out}")
    if cfg.allow_single_input:
        pct = (100 * stats.single_input / stats.total) if stats.total else 0
        print(f"single-input forms: {stats.single_input} ({pct:.1f}%)")
    print(f"by language: {dict(sorted(stats.by_lang.items(), key=lambda x: -x[1]))}")
    top_tlds = dict(sorted(stats.by_tld.items(), key=lambda x: -x[1])[:12])
    print(f"top TLDs: {top_tlds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
