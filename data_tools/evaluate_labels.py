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
from gen.params import SECOND_LINE_TOKENS
from gen.validate import LABEL_ATTR


def apply_address_scheme(fields: list[dict], verdicts: list[dict]) -> list[dict]:
    """Deterministic post-process of the GPT-4 verdicts for the primary address
    field, matching the generator/labeler convention: the primary address field
    is 'address-line1' when a second line (address-line2/apartment/…) is adjacent
    to it, else 'street-address'.

    Uses the IMMEDIATE NEIGHBORS (previous/next labeled field) rather than the
    whole form, so a form with multiple address sections is corrected per section
    (e.g. a shipping block with line1+line2 next to a single billing field).
    """
    types = [f["autofill_type"] for f in fields]   # in document order
    n = len(types)
    vmap = {v["index"]: v for v in verdicts}
    for k, f in enumerate(fields):
        if types[k] not in ("address-line1", "street-address"):
            continue
        neighbors = ([types[k - 1]] if k > 0 else []) + ([types[k + 1]] if k < n - 1 else [])
        has_second = any(nb in SECOND_LINE_TOKENS for nb in neighbors)
        correct = "address-line1" if has_second else "street-address"
        v = vmap.get(f["index"])
        if v is None:
            v = {"index": f["index"]}
            verdicts.append(v)
            vmap[f["index"]] = v
        if types[k] == correct:
            v["verdict"], v["suggested"] = "correct", ""
        else:
            v["verdict"], v["suggested"] = "incorrect", correct
    return verdicts

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
    """Best visible label for a field. Ordered from most- to least-reliable so a
    section heading in an ancestor can't override the field's own label/
    placeholder (which caused e.g. a 'First name' field to be read as its
    section's 'Street Edit' heading)."""
    # 1. Explicit <label for=id> association.
    fid = el.get("id")
    if fid:
        lf = soup.find("label", attrs={"for": fid})
        if lf and lf.get_text(strip=True):
            return lf.get_text(" ", strip=True)
    # 2. Wrapping <label>.
    par = el.find_parent("label")
    if par and par.get_text(strip=True):
        return par.get_text(" ", strip=True)
    # 3. The field's OWN attributes — specific and reliable.
    for attr in ("placeholder", "aria-label", "title"):
        if el.get(attr) and el.get(attr).strip():
            return el.get(attr).strip()
    # 4. A <label>/<span> in the field's IMMEDIATE wrapper — but only when that
    #    wrapper holds a single field, so the label is unambiguously this field's.
    #    This handles both "label then input" and "input then label" layouts
    #    without grabbing a neighbor's label (which caused an off-by-one when
    #    labels followed inputs) or a section heading from a big ancestor.
    parent = el.parent
    if parent is not None and len(parent.find_all(["input", "select", "textarea"])) <= 1:
        lab = parent.find(["label", "span"])
        if lab and lab.get_text(strip=True):
            return lab.get_text(" ", strip=True)[:120]
    return ""


def extract_labeled_fields(soup) -> list[dict]:
    """Pull each labeled element with its best visible context, in order."""
    fields = []
    for i, el in enumerate(soup.select(f"[{LABEL_ATTR}]")):
        element = el.name
        if el.name == "input":
            element = "input:" + (el.get("type") or "text").lower()
        options = ""
        if el.name == "select":
            opts = [o.get_text(strip=True)
                    for o in el.find_all("option") if o.get_text(strip=True)]
            options = ", ".join(opts[:14])   # first 14 only, to bound prompt size
        fields.append({
            "index": i,
            "autofill_type": el.get(LABEL_ATTR),
            "label": _field_label(el, soup),
            "placeholder": el.get("placeholder", ""),
            "name": el.get("name", ""),
            "element": element,
            "options": options,
        })
    return fields


_NAME_COMPONENTS = ("given-name", "family-name", "additional-name")


def apply_name_scheme(fields: list[dict], verdicts: list[dict]) -> list[dict]:
    """Reconcile the combined 'name' vs split given/family-name convention, both
    directions:
      - split components (given/family) are correct when the form splits the name
        (has both) -> don't 'correct' them to the combined 'name';
      - a standalone 'name' field is a correct full-name field when NO name
        component is adjacent to it -> don't 'correct' it into a component.
    A form can contain both a split section and standalone name fields (e.g. a
    recipient name), so the split check is form-level and the standalone check is
    neighbor-based.
    """
    types = [f["autofill_type"] for f in fields]
    n = len(types)
    form_split = "given-name" in types and "family-name" in types
    vmap = {v["index"]: v for v in verdicts}
    for k, f in enumerate(fields):
        v = vmap.get(f["index"])
        if not v or v.get("verdict") != "incorrect":
            continue
        t, s = types[k], (v.get("suggested") or "")
        if t in _NAME_COMPONENTS and s == "name" and form_split:
            v["verdict"], v["suggested"] = "correct", ""
        elif t == "name" and s in _NAME_COMPONENTS:
            neigh = [types[j] for j in (k - 1, k + 1) if 0 <= j < n]
            if not any(nb in _NAME_COMPONENTS for nb in neigh):
                v["verdict"], v["suggested"] = "correct", ""
    return verdicts


async def audit_dir(client, model, spec, name, paths, usage, sem, overrides=True) -> dict:
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
            if overrides:   # deterministic convention overrides (off to test prompt alone)
                verdicts = apply_address_scheme(fields, verdicts)
                verdicts = apply_name_scheme(fields, verdicts)
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
                "dir": name, "file": fname, "index": v["index"],
                "assigned": f["autofill_type"],
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

    dirs = {os.path.basename(os.path.normpath(args.dir)): args.dir} if args.dir else DIRS
    all_results = []
    for name, d in dirs.items():
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
            await audit_dir(client, args.model, spec, name, sample, usage, sem,
                            overrides=args.overrides))

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
    p.add_argument("--n", type=int, default=100,
                   help="forms to sample per directory (use a large value to audit all)")
    p.add_argument("--no-overrides", dest="overrides", action="store_false", default=True,
                   help="disable the deterministic address/name convention overrides "
                        "(to test whether the prompt+model handle them alone)")
    p.add_argument("--dir", default=None,
                   help="audit this single directory of labeled HTML instead of the "
                        "built-in generated/common_crawl dirs")
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
