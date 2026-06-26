#!/usr/bin/env python3
"""Label real Common Crawl form HTML into autofill training samples.

Reads a CSV whose `form` column holds the inner HTML of a real web form, keeps
forms with a reasonable number of fillable fields, uses the OpenAI API to
classify each field into the 66-type autofill taxonomy, injects the resulting
data-moz-autofill-type attributes onto the real elements, and writes each form
as a standalone HTML file into samples/common_crawl/ (same format as
samples/generated/).

Usage:
  python label_common_crawl.py --csv forms.csv --analyze   # field-count histogram, no API
  python label_common_crawl.py --csv forms.csv             # label + write all passing forms
  python label_common_crawl.py --review 10                 # inspect already-labeled output

Requires OPENAI_API_KEY in the environment or repo-root .env.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import sys

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from gen.dedup import DedupIndex, signature_from_html
from gen.htmlforms import (
    extract_candidates,
    fillable_count,
    guess_domain,
    inject_label,
    wrap_form,
)
from gen.llm import SKIP, Usage, PRICING, request_field_labels, verify_form
from gen.params import SECOND_LINE_TOKENS
from gen.validate import LABEL_ATTR, validate_html

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(REPO_ROOT, "samples")
DEFAULT_OUT = os.path.join(SAMPLES_DIR, "common_crawl")


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def short_domain(domain: str) -> str:
    return "".join(c for c in domain.split(".")[0] if c.isalnum())[:16] or "site"


async def label_one(client, args, row_index: int, form_html: str,
                    dedup: DedupIndex, usage: Usage, lang: str = "") -> dict:
    rec = {"index": row_index, "lang": lang}
    soup = BeautifulSoup(str(form_html), "html.parser")

    n_fields = fillable_count(soup)
    if not (args.min_fields <= n_fields <= args.max_fields):
        rec.update(status="skipped", reason=f"field count {n_fields} out of range")
        return rec

    candidates = extract_candidates(soup)
    if not candidates:
        rec.update(status="skipped", reason="no labelable candidates")
        return rec

    try:
        verdicts = await request_field_labels(
            client, args.model,
            [{k: v for k, v in c.items() if k != "el"} for c in candidates],
            usage,
        )
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        return rec

    by_index = {c["index"]: c for c in candidates}
    applied: list[tuple[dict, str]] = []   # (candidate, autofill_type), in label order
    for idx, autofill_type in verdicts.items():
        if autofill_type == SKIP:
            continue
        field = by_index.get(idx)
        if field and inject_label(field, autofill_type):
            applied.append((field, autofill_type))

    # Verification pass: a stronger model checks each label<->type against the
    # form's language, and we strip any label it rejects. Real-form labels are
    # LLM-assigned (the model owns them here), so this is the main quality gate.
    if args.verify and applied:
        try:
            vfields = [{"label_text": (f.get("label") or f.get("placeholder")
                                       or f.get("name") or ""),
                        "autofill_type": t} for f, t in applied]
            verdict = await verify_form(
                client, args.verify_model, lang or "unknown", vfields, usage)
            removed = 0
            for pos, (field, _t) in enumerate(applied):
                if verdict.get(pos) == "no":
                    del field["el"][LABEL_ATTR]
                    removed += 1
            rec["verify_removed"] = removed
        except Exception as exc:  # noqa: BLE001 - don't fail the form on verify error
            rec["verify_error"] = f"{type(exc).__name__}: {exc}"

    # Structural address-scheme enforcement (mirrors gen.params._apply_address_scheme,
    # matching the hand-labeled dataset): the primary address field is
    # address-line1 when a second line (address-line2/apartment/...) is present,
    # else street-address. Deterministic, so it doesn't depend on the LLM
    # following the prompt rule. (Form-level; multi-section forms are rare in CC.)
    labeled_els = soup.select(f"[{LABEL_ATTR}]")
    has_second = any(e.get(LABEL_ATTR) in SECOND_LINE_TOKENS for e in labeled_els)
    primary = "address-line1" if has_second else "street-address"
    for e in labeled_els:
        if e.get(LABEL_ATTR) in ("street-address", "address-line1"):
            e[LABEL_ATTR] = primary

    labeled = len(soup.select(f"[{LABEL_ATTR}]"))
    # Single-input forms can only ever have one label, so they require just 1;
    # multi-field forms keep the normal min_labeled bar.
    required = 1 if n_fields == 1 else args.min_labeled
    if labeled < required:
        rec.update(status="skipped",
                   reason=f"only {labeled} autofill fields after verify (need {required})")
        return rec

    domain = guess_domain(soup)
    html = wrap_form(str(soup), site=domain,
                     note="Common Crawl form, LLM-labeled.")

    v = validate_html(html)
    if not v:
        rec.update(status="rejected", reason=f"html: {v.reason}")
        return rec

    sig = signature_from_html(html)
    if not dedup.add_if_new(sig):
        rec.update(status="rejected", reason="duplicate signature")
        return rec

    # Content-hash suffix keeps filenames stable and collision-free across
    # batches (different CSVs reuse row indices), unlike a row-index suffix.
    h = hashlib.sha1(str(form_html).encode("utf-8", "ignore")).hexdigest()[:8]
    fname = f"CC_{short_domain(domain)}_{h}.html"
    with open(os.path.join(args.out, fname), "w", encoding="utf-8") as fh:
        fh.write(html)
    rec.update(status="accepted", filename=fname, fields=n_fields, labeled=labeled)
    return rec


def analyze(df: pd.DataFrame, args) -> int:
    import collections
    dist = collections.Counter()
    for html in df["form"]:
        n = fillable_count(BeautifulSoup(str(html), "html.parser"))
        bucket = ("0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5
                  else "6-10" if n <= 10 else "11-20" if n <= 20
                  else "21-30" if n <= 30 else "31+")
        dist[bucket] += 1
    in_range = sum(args.min_fields <= fillable_count(
        BeautifulSoup(str(h), "html.parser")) <= args.max_fields for h in df["form"])
    print(f"rows: {len(df)}")
    print(f"fillable-field buckets: {dict(dist)}")
    print(f"forms within [{args.min_fields},{args.max_fields}]: {in_range}")
    return 0


async def run(df: pd.DataFrame, args) -> int:
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")

    dedup = DedupIndex()
    print(f"Indexed {dedup.load_corpus(SAMPLES_DIR)} existing sample files for dedup.")

    usage = Usage()
    sem = asyncio.Semaphore(args.concurrency)
    counts: dict[str, int] = {}

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    async def worker(row_index: int, form_html: str, lang: str):
        async with sem:
            rec = await label_one(client, args, row_index, form_html, dedup,
                                  usage, lang=lang)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            append_jsonl(manifest_path, rec)
            tag = {"accepted": "OK ", "skipped": "skip",
                   "rejected": "rej", "error": "ERR"}.get(rec["status"], "?")
            extra = rec.get("filename") or rec.get("reason", "")
            rm = rec.get("verify_removed")
            print(f"  [{row_index}] {tag} {extra}"
                  + (f" (verify -{rm})" if rm else ""))
            return rec

    langs = df["lang"] if "lang" in df.columns else [""] * len(df)
    await asyncio.gather(*(worker(int(i), h, str(l) if pd.notna(l) else "")
                           for i, h, l in
                           zip(df[df.columns[0]], df["form"], langs)))

    print("\n--- Summary ---")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    # Generation/verification share the token pool; report cost across both models.
    gen_cost = usage.cost(args.model)
    print(f"API calls={usage.calls} tokens(in/out)="
          f"{usage.prompt_tokens}/{usage.completion_tokens} "
          f"approx cost(gen model)~${gen_cost:.2f}"
          + (f" + verify on {args.verify_model}" if args.verify else ""))
    print(f"Output: {args.out}")
    return 0


def review(args) -> int:
    files = sorted(glob.glob(os.path.join(args.out, "CC_*.html")))
    if not files:
        print(f"No labeled files in {args.out}")
        return 1
    for path in files[:args.review]:
        print(f"\n=== {os.path.basename(path)} ===")
        soup = BeautifulSoup(open(path, encoding="utf-8").read(), "html.parser")
        for el in soup.select(f"[{LABEL_ATTR}]"):
            lab = ""
            if el.get("id"):
                lf = soup.find("label", attrs={"for": el.get("id")})
                if lf:
                    lab = lf.get_text(" ", strip=True)
            lab = lab or el.get("placeholder") or el.get("name") or ""
            print(f"  {el.get(LABEL_ATTR):<20} <- {lab!r}")
    return 0


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None,
                   help="CSV of candidate forms (from extract_forms.py); must have a "
                        "'form' column. Required unless --review.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--verify", action="store_true", default=True,
                   help="run the label<->type verification pass (default on)")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--verify-model", default="gpt-4o",
                   help="model for the verification pass")
    p.add_argument("--min-fields", type=int, default=3,
                   help="minimum fillable fields to keep a form")
    p.add_argument("--max-fields", type=int, default=25,
                   help="maximum fillable fields to keep a form")
    p.add_argument("--min-labeled", type=int, default=2,
                   help="drop forms where fewer than this many fields get labeled")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="only process the first N rows of the CSV (0 = all)")
    p.add_argument("--analyze", action="store_true",
                   help="print field-count histogram and exit (no API)")
    p.add_argument("--review", type=int, default=0,
                   help="print label tables for N labeled files and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.review:
        return review(args)

    if not args.csv:
        print("ERROR: --csv is required (path to a candidate-forms CSV with a "
              "'form' column, e.g. from extract_forms.py).", file=sys.stderr)
        return 2
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.csv)
    if "form" not in df.columns:
        print(f"ERROR: CSV has no 'form' column (cols={list(df.columns)})",
              file=sys.stderr)
        return 2
    if args.limit:
        df = df.head(args.limit)

    if args.analyze:
        return analyze(df, args)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (.env or environment).", file=sys.stderr)
        return 2
    return asyncio.run(run(df, args))


if __name__ == "__main__":
    raise SystemExit(main())
