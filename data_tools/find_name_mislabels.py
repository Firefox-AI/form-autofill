#!/usr/bin/env python3
"""Find primary name-mislabel candidates by pure structure (no LLM).

Heuristic (per Rolf): a field labeled as a *component* name
(given-name / family-name / additional-name) that is the SOLE name-type field
on its form -- or has NO adjacent name-type field -- was almost certainly a
full-name box that should be labeled `name`. That is the classic mislabel that
makes autofill drop the given or family part (the live "users editing
family-name" bug).

Two input modes:
  --dir  <labeled-html-dir>   one form per .html file, fields carry
                              data-moz-autofill-type (e.g. data/labeled_2026)
  --txt  <training .txt>      per-field rows grouped by form; label = column 1
                              (e.g. testing-supported.txt, *-supported.ffgen.txt)

Emits a CSV of candidates (file, field_index, assigned, reason, identifying
text, and html_path when known) sorted by reason then language.

Usage:
  python find_name_mislabels.py --dir data/labeled_2026 --out name_mislabels_2026.csv
  python find_name_mislabels.py --txt testing-supported.txt --out name_mislabels_train.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

NAME_GROUP = {"name", "given-name", "family-name", "additional-name"}
COMPONENT = {"given-name", "family-name", "additional-name"}

try:
    from dotraining import ignoreLineCount
except Exception:
    ignoreLineCount = 1  # column 0 = filename, column 1 = label; context after


def _candidates_from_sequence(types: list[str]):
    """Given a form's fields as an ordered list of autofill-type strings, yield
    (index, assigned, reason) for each component-name field that has no partner.

    A given-name should pair with a family-name (and vice versa). A component
    name with NO complementary component on the form is almost certainly a
    full-name box that should be `name`. Being the SOLE name field is the
    strongest form of that. Partner search ignores gap/phonetic fields, so a
    legit `given-name --NONE-- family-name` or JP `given / phonetic / family`
    split is NOT flagged."""
    has = {t: (t in types) for t in NAME_GROUP}
    total_name = sum(1 for t in types if t in NAME_GROUP)
    partner = {"given-name": "family-name", "family-name": "given-name"}
    for i, t in enumerate(types):
        if t not in COMPONENT:
            continue
        if t in partner:
            if has[partner[t]]:
                continue  # has its complementary component -> legit split
            reason = ("sole name field on form" if total_name == 1
                      else f"lone {t} (no {partner[t]} on form)")
            yield i, t, reason
        else:  # additional-name orphaned without both given & family
            if not (has["given-name"] and has["family-name"]):
                yield i, t, "orphan additional-name (no given+family pair)"


# ---------- HTML mode ----------
def scan_html_dir(path: str):
    from bs4 import BeautifulSoup
    rows = []
    for fp in sorted(glob.glob(os.path.join(path, "*.html"))):
        html = open(fp, encoding="utf-8", errors="replace").read()
        soup = BeautifulSoup(html, "html.parser")
        els = [e for e in soup.find_all(["input", "select", "textarea"])
               if e.get("data-moz-autofill-type")]
        types = [e.get("data-moz-autofill-type").strip() for e in els]
        for i, assigned, reason in _candidates_from_sequence(types):
            e = els[i]
            ident = " | ".join(filter(None, [
                f"id={e.get('id')}" if e.get("id") else "",
                f"name={e.get('name')}" if e.get("name") else "",
                f"ph={e.get('placeholder')}" if e.get("placeholder") else "",
                f"ac={e.get('autocomplete')}" if e.get("autocomplete") else "",
            ]))
            rows.append({
                "file": os.path.basename(fp), "field_index": i,
                "assigned": assigned, "reason": reason,
                "n_name_fields": sum(1 for t in types if t in NAME_GROUP),
                "form_types": " ".join(types)[:200],
                "identifying_text": ident,
                "html_path": os.path.abspath(fp),
            })
    return rows


# ---------- training .txt mode ----------
def scan_txt(path: str):
    rows, cur, group = [], None, []

    def flush(fname, grp):
        types = [lbl for lbl, _ctx in grp]
        for i, assigned, reason in _candidates_from_sequence(types):
            rows.append({
                "file": fname, "field_index": i,
                "assigned": assigned, "reason": reason,
                "n_name_fields": sum(1 for t in types if t in NAME_GROUP),
                "form_types": " ".join(types)[:200],
                "identifying_text": grp[i][1][:160],
                "html_path": "",
            })

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        d = line.split(",", ignoreLineCount + 1)
        if len(d) < ignoreLineCount + 2:
            continue
        fname, label, ctx = d[0], d[1], d[ignoreLineCount + 1]
        if fname != cur and cur is not None:
            flush(cur, group)
            group = []
        cur = fname
        group.append((label, ctx))
    if group:
        flush(cur, group)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="directory of labeled .html forms")
    g.add_argument("--txt", help="training/testing .txt (rows grouped by form)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = scan_html_dir(args.dir) if args.dir else scan_txt(args.txt)
    rows.sort(key=lambda r: (r["reason"], r["assigned"], r["file"]))

    cols = ["file", "field_index", "assigned", "reason", "n_name_fields",
            "identifying_text", "form_types", "html_path"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    by_reason = Counter(r["reason"] for r in rows)
    by_assigned = Counter(r["assigned"] for r in rows)
    src = args.dir or args.txt
    print(f"{len(rows)} candidate mislabels in {src} -> {args.out}")
    print("  by reason:  ", dict(by_reason))
    print("  by assigned:", dict(by_assigned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
