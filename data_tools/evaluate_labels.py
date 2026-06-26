#!/usr/bin/env python3
"""Audit the labeling accuracy of generated / common_crawl sample forms.

Samples N forms from each directory, sends each form's labeled fields together
with the full autofill specification (gen/autofill_spec.txt) to a strong model,
and asks it to judge whether each assigned data-moz-autofill-type token is
correct. Reports per-directory accuracy and the most common mislabels so we can
decide whether the forms need re-labeling with a clearer prompt.

Usage:
  python evaluate_labels.py                       # 100 from generated + 100 from common_crawl
  python evaluate_labels.py --n 50 --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import random
import sys

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from gen.llm import Usage, audit_form_labels
from gen.validate import LABEL_ATTR

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(REPO_ROOT, "gen", "autofill_spec.txt")
SAMPLES = os.path.join(REPO_ROOT, "samples")
DIRS = {
    "generated": os.path.join(SAMPLES, "generated"),
    "common_crawl": os.path.join(SAMPLES, "common_crawl"),
}


def lang_from_name(fname: str) -> str:
    """GEN_<locale>_... encodes locale; CC_ files don't, so return ''."""
    base = os.path.basename(fname)
    if base.startswith("GEN_"):
        return base.split("_")[1]
    return ""


def _field_label(el, soup) -> str:
    """Best visible label for a field, scoped to its own container so we never
    grab unrelated page text (e.g. the form's synthetic-sample note)."""
    fid = el.get("id")
    if fid:
        lf = soup.find("label", attrs={"for": fid})
        if lf and lf.get_text(strip=True):
            return lf.get_text(" ", strip=True)
    par = el.find_parent("label")
    if par and par.get_text(strip=True):
        return par.get_text(" ", strip=True)
    # A <label>/<span> within the field's immediate container.
    container = el.find_parent(["div", "p", "td", "li", "fieldset"])
    if container:
        for tag in container.find_all(["label", "span"]):
            t = tag.get_text(" ", strip=True)
            if t:
                return t[:120]
    if el.get("aria-label"):
        return el.get("aria-label")
    return ""


def extract_labeled_fields(soup) -> list[dict]:
    """Pull each labeled element with its best visible context, in order."""
    fields = []
    for i, el in enumerate(soup.select(f"[{LABEL_ATTR}]")):
        fields.append({
            "index": i,
            "autofill_type": el.get(LABEL_ATTR),
            "label": _field_label(el, soup),
            "placeholder": el.get("placeholder", ""),
            "name": el.get("name", ""),
        })
    return fields


async def audit_dir(client, model, spec, name, paths, usage, sem) -> dict:
    results = {"name": name, "forms": 0, "fields": 0,
               "correct": 0, "incorrect": 0, "unsure": 0,
               "mislabels": collections.Counter(), "examples": [], "records": []}

    async def one(path):
        async with sem:
            html = open(path, encoding="utf-8", errors="replace").read()
            soup = BeautifulSoup(html, "html.parser")
            fields = extract_labeled_fields(soup)
            if not fields:
                return None
            try:
                verdicts = await audit_form_labels(
                    client, model, spec, fields, usage,
                    lang=lang_from_name(path))
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {os.path.basename(path)}: {type(exc).__name__}: {exc}")
                return None
            by_idx = {f["index"]: f for f in fields}
            return os.path.basename(path), by_idx, verdicts

    outs = await asyncio.gather(*(one(p) for p in paths))
    for out in outs:
        if not out:
            continue
        fname, by_idx, verdicts = out
        results["forms"] += 1
        for v in verdicts:
            f = by_idx.get(v["index"])
            if not f:
                continue
            results["fields"] += 1
            results[v["verdict"]] = results.get(v["verdict"], 0) + 1
            results["records"].append({
                "dir": name, "file": fname, "assigned": f["autofill_type"],
                "label": f["label"], "placeholder": f["placeholder"],
                "verdict": v["verdict"], "suggested": v.get("suggested", ""),
            })
            if v["verdict"] == "incorrect":
                pair = (f["autofill_type"], v.get("suggested", "") or "?")
                results["mislabels"][pair] += 1
                if len(results["examples"]) < 12:
                    results["examples"].append(
                        f"{fname}: {f['autofill_type']} -> {v.get('suggested','?')} "
                        f"(label={f['label'][:40]!r})")
    return results


def report(r: dict) -> None:
    n = r["fields"] or 1
    print(f"\n=== {r['name']} ===")
    print(f"forms audited: {r['forms']} | fields audited: {r['fields']}")
    print(f"  correct:   {r['correct']:4d}  ({100*r['correct']/n:.1f}%)")
    print(f"  incorrect: {r['incorrect']:4d}  ({100*r['incorrect']/n:.1f}%)")
    print(f"  unsure:    {r['unsure']:4d}  ({100*r['unsure']/n:.1f}%)")
    if r["mislabels"]:
        print("  top mislabels (assigned -> suggested):")
        for (a, s), c in r["mislabels"].most_common(12):
            print(f"      {c:3d}x  {a} -> {s}")
    if r["examples"]:
        print("  examples:")
        for ex in r["examples"]:
            print(f"      {ex}")


async def run(args) -> int:
    spec = open(SPEC_PATH, encoding="utf-8").read()
    rng = random.Random(args.seed)
    usage = Usage()
    sem = asyncio.Semaphore(args.concurrency)

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    all_results = []
    for name, d in DIRS.items():
        files = sorted(glob.glob(os.path.join(d, "*.html")))
        if not files:
            print(f"[{name}] no files found in {d}, skipping")
            continue
        if args.require_type:
            needle = f'{LABEL_ATTR}="{args.require_type}"'
            files = [p for p in files
                     if needle in open(p, encoding="utf-8", errors="replace").read()]
            print(f"[{name}] {len(files)} forms contain {args.require_type!r}")
        sample = rng.sample(files, k=min(args.n, len(files)))
        print(f"[{name}] auditing {len(sample)} of {len(files)} forms...")
        all_results.append(
            await audit_dir(client, args.model, spec, name, sample, usage, sem))

    print("\n" + "=" * 60)
    for r in all_results:
        report(r)
    print(f"\njudge model={args.model}  API calls={usage.calls}  "
          f"tokens(in/out)={usage.prompt_tokens}/{usage.completion_tokens}  "
          f"cost~${usage.cost(args.model):.2f}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r in all_results:
                for rec in r["records"]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"per-field verdicts written to {args.dump}")
    return 0


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="forms to sample per directory")
    p.add_argument("--model", default="gpt-4o", help="judge model")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dump", default=None, help="write per-field verdicts to this jsonl")
    p.add_argument("--require-type", default=None,
                   help="only audit forms that contain this autofill token")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    if not os.path.exists(SPEC_PATH):
        print(f"ERROR: spec not found at {SPEC_PATH}", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (.env or environment).", file=sys.stderr)
        return 2
    return asyncio.run(run(parse_args(argv if argv is not None else sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
