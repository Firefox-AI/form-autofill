#!/usr/bin/env python3
"""Apply audited label corrections back into the HTML files, in place.

Reads a verdict dump (from evaluate_labels.py --dump) and, for every actionable
correction, rewrites that field's data-moz-autofill-type attribute in the source
HTML. "Actionable" mirrors the error report: an 'incorrect' verdict whose
suggested value is a single valid taxonomy token, different from the assigned
token, not password-related, and not a fieldNamesCloseDict close-match.

Edits are surgical — only the attribute VALUE of the Nth labeled element changes,
so diffs stay tiny — with two safety gates:
  * the count of `data-moz-autofill-type="..."` occurrences must equal the number
    of labeled elements BeautifulSoup sees (else the file is skipped), and
  * the Nth occurrence's current value must equal the verdict's `assigned` (else
    that single correction is skipped) — so a stale dump can't corrupt a file.

Usage:
  python apply_corrections.py --dump /tmp/v_training_o4.jsonl --dir <html_dir> --dry-run
  python apply_corrections.py --dump /tmp/v_training_o4.jsonl --dir <html_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

from bs4 import BeautifulSoup

from gen.validate import LABEL_ATTR, load_taxonomy
from build_error_report import load_close_pairs

# Captures: (1) 'data-moz-autofill-type=' (2) the quote char (3) the value.
ATTR_RE = re.compile(r'(data-moz-autofill-type\s*=\s*)(["\'])(.*?)\2')


def applicable_corrections(dump_path: str) -> dict:
    """file -> {field_index: (assigned, suggested)} for corrections worth applying."""
    valid = set(load_taxonomy())
    close = load_close_pairs()
    by_file: dict[str, dict] = collections.defaultdict(dict)
    for line in open(dump_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("verdict") != "incorrect":
            continue
        a = r["assigned"]
        s = (r.get("suggested") or "").strip()
        if not s or s == a:
            continue
        if "password" in (a + " " + s).lower():
            continue
        if frozenset((a, s)) in close:
            continue
        if " " in s or s not in valid:            # only clean single-token relabels
            continue
        by_file[r["file"]][r["index"]] = (a, s)
    return by_file


def apply_to_file(path: str, corrections: dict, dry_run: bool):
    """Returns (status, n_changed). status in ok / mismatch / missing."""
    if not os.path.exists(path):
        return "missing", 0
    raw = open(path, encoding="utf-8", errors="replace").read()
    n_labeled = len(BeautifulSoup(raw, "html.parser").select(f"[{LABEL_ATTR}]"))
    matches = list(ATTR_RE.finditer(raw))
    if len(matches) != n_labeled:
        # Regex occurrences don't line up 1:1 with parsed elements (e.g. the
        # string appears in text/script, or an unquoted attribute) — don't risk
        # a misaligned edit.
        return "mismatch", 0

    edits, applied = [], []
    for idx, (assigned, suggested) in corrections.items():
        if not (0 <= idx < len(matches)):
            continue
        m = matches[idx]
        if m.group(3) != assigned:      # value drifted from the audited one — skip
            continue
        edits.append((m.start(), m.end(),
                      f"{m.group(1)}{m.group(2)}{suggested}{m.group(2)}"))
        applied.append((assigned, suggested))

    if edits and not dry_run:
        for start, end, new in sorted(edits, reverse=True):
            raw = raw[:start] + new + raw[end:]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw)
    return "ok", applied


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="verdict JSONL from evaluate_labels.py --dump")
    p.add_argument("--dir", required=True, help="directory of the source HTML forms")
    p.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    by_file = applicable_corrections(args.dump)
    pairs = collections.Counter()
    files_changed = fields_changed = skipped_mismatch = skipped_value = 0
    for fname, corrs in sorted(by_file.items()):
        status, applied = apply_to_file(os.path.join(args.dir, fname), corrs, args.dry_run)
        if status == "mismatch":
            skipped_mismatch += 1
            continue
        if status == "missing":
            continue
        skipped_value += len(corrs) - len(applied)
        if applied:
            files_changed += 1
            fields_changed += len(applied)
            for a, s in applied:
                pairs[(a, s)] += 1

    mode = "DRY RUN — no files written" if args.dry_run else "applied"
    print(f"=== corrections {mode} ===")
    print(f"files with corrections: {len(by_file)} | files changed: {files_changed} "
          f"| fields changed: {fields_changed}")
    print(f"skipped (value drift): {skipped_value} | skipped files (count mismatch): {skipped_mismatch}")
    print("top corrections (assigned -> suggested):")
    for (a, s), n in pairs.most_common(15):
        print(f"   {n:4d}  {a} -> {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
