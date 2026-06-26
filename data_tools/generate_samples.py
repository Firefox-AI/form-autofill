#!/usr/bin/env python3
"""Expand the autofill HTML training set ~10x using the OpenAI API.

The LLM only produces structured field metadata (localized label text + a
field-type drawn from dotraining.py's 66-type enum); this script renders the
HTML and owns the data-moz-autofill-type ground-truth attribute, so labels are
correct by construction. Two modes are mixed:

  * hybrid  (default ~70%): build a planned field set, ask the LLM for localized
              labels, render.
  * mutation (~30%):        take a real labeled form and ask the LLM to rewrite
              it for a new locale, carrying the verified labels forward.

Output goes to samples/generated/ as GEN_*.html so it can be globbed in or out
of training without touching the real sample dirs.

Usage:
  python generate_samples.py --count 50 --dry-run        # plan + cost estimate, no API
  python generate_samples.py --count 50                  # generate 50
  python generate_samples.py --count 4500 --resume       # resume a large run
  python generate_samples.py --review 20                 # eyeball accepted files

Requires OPENAI_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import sys

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from gen.dedup import DedupIndex, signature_from_html, signature_from_spec
from gen.llm import (
    Usage,
    PRICING,
    request_form_spec,
    request_mutation,
    verify_form,
)
from gen.params import GenParams, sample_params
from gen.render import render_form
from gen.validate import LABEL_ATTR, NONFILL, validate_form_spec, validate_html

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(REPO_ROOT, "samples")
TRAINING_DIR = os.path.join(SAMPLES_DIR, "training")
DEFAULT_OUT = os.path.join(SAMPLES_DIR, "generated")


# --------------------------------------------------------------------------- #
# Manifest (resumability + audit trail)
# --------------------------------------------------------------------------- #
def load_manifest(path: str) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["index"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Mode B helpers: anchor extraction + label carry-forward
# --------------------------------------------------------------------------- #
def extract_anchor_fields(html: str) -> list[dict]:
    """Pull (autofill_type, nearby label) pairs from a real sample, in order."""
    soup = BeautifulSoup(html, "html.parser")
    fields: list[dict] = []
    for el in soup.select(f"[{LABEL_ATTR}]"):
        label = ""
        prev = el.find_previous(["span", "label"])
        if prev and prev.get_text(strip=True):
            label = prev.get_text(strip=True)
        elif el.get("placeholder"):
            label = el.get("placeholder")
        fields.append({"autofill_type": el.get(LABEL_ATTR, ""), "label": label})
    return fields


def carry_forward_anchor(spec: dict, anchor_fields: list[dict]) -> dict:
    """Force the verified anchor types onto the mutated spec, by position.

    Guarantees label correctness regardless of what the LLM returned. We align
    to the anchor length and keep only those positions.
    """
    fields = spec.get("fields", [])
    n = min(len(fields), len(anchor_fields))
    aligned = []
    for i in range(n):
        f = dict(fields[i])
        f["autofill_type"] = anchor_fields[i]["autofill_type"]
        aligned.append(f)
    spec["fields"] = aligned
    return spec


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #
def output_filename(params: GenParams, site_domain: str) -> str:
    short = "".join(c for c in site_domain.split(".")[0] if c.isalnum())[:16] or "site"
    return f"GEN_{params.locale}_{params.purpose}_{short}_{params.index:05d}.html"


# --------------------------------------------------------------------------- #
# Per-form generation
# --------------------------------------------------------------------------- #
async def generate_one(client, args, params: GenParams, anchors: list[str],
                       dedup: DedupIndex, usage: Usage) -> dict:
    """Generate, validate, (optionally) verify, and write a single form.

    Returns a manifest record dict (status in accepted/rejected/error).
    """
    rng = random.Random(params.seed)
    mode = "mutation" if rng.random() < args.mutate_ratio and anchors else "hybrid"

    rec = {"index": params.index, "mode": mode, "locale": params.locale,
           "purpose": params.purpose, "seed": params.seed}

    try:
        if mode == "mutation":
            anchor_path = rng.choice(anchors)
            with open(anchor_path, encoding="utf-8", errors="replace") as fh:
                anchor_fields = extract_anchor_fields(fh.read())
            if not anchor_fields:
                rec.update(status="rejected", reason="anchor had no labeled fields")
                return rec
            spec = await request_mutation(
                client, args.model, anchor_fields, params.locale, params.seed, usage)
            spec = carry_forward_anchor(spec, anchor_fields)
            rec["anchor"] = os.path.basename(anchor_path)
        else:
            spec = await request_form_spec(client, args.model, params, usage)

        # Gate 1: structured spec validation.
        v = validate_form_spec(spec)
        if not v:
            rec.update(status="rejected", reason=f"spec: {v.reason}")
            return rec

        # Optional semantic verification of label<->type plausibility.
        rare_present = bool(params.rare_injected)
        do_verify = args.verify and (rare_present or rng.random() < args.verify_sample)
        if do_verify:
            verdicts = await verify_form(
                client, args.verify_model, params.locale, spec["fields"], usage)
            bad = [i for i, m in verdicts.items() if m == "no"]
            if bad:
                rec.update(status="rejected", reason=f"verify flagged fields {bad}",
                           verified=True)
                return rec
            rec["verified"] = True

        # Render HTML (code owns the label attribute).
        note = "Synthetic sample (LLM-augmented)."
        html = render_form(spec, markup_style=params.markup_style,
                           name_style=params.name_style, rng=rng, note=note)

        # Gate 2: HTML validation.
        expected = sum(1 for f in spec["fields"] if f["autofill_type"] != NONFILL)
        v = validate_html(html, expected_label_count=expected)
        if not v:
            rec.update(status="rejected", reason=f"html: {v.reason}")
            return rec

        # Gate 3: dedup against corpus + previously generated.
        sig = signature_from_spec(spec)
        if not dedup.add_if_new(sig):
            rec.update(status="rejected", reason="duplicate signature")
            return rec

        site_domain = spec.get("site_domain") or "example.com"
        fname = output_filename(params, site_domain)
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as fh:
            fh.write(html)
        rec.update(status="accepted", filename=fname, fields=expected,
                   rare=params.rare_injected)
        return rec

    except Exception as exc:  # noqa: BLE001 - record and continue the batch
        rec.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        return rec


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    rejects_path = os.path.join(args.out, "rejects.jsonl")

    manifest = load_manifest(manifest_path) if args.resume else {}
    done = {i for i, r in manifest.items() if r.get("status") == "accepted"}

    dedup = DedupIndex()
    corpus_n = dedup.load_corpus(SAMPLES_DIR)
    print(f"Indexed {corpus_n} existing sample files for dedup.")

    anchors = glob.glob(os.path.join(TRAINING_DIR, "*.html"))

    todo = [i for i in range(args.count) if i not in done]
    if args.resume and done:
        print(f"Resuming: {len(done)} already accepted, {len(todo)} remaining.")

    # Cost estimate up front.
    pin, pout = PRICING.get(args.model, (0.0, 0.0))
    est = len(todo) * (350 * pin + 450 * pout) / 1_000_000
    print(f"Model={args.model}  forms={len(todo)}  est. cost ~${est:.2f} "
          f"(verify adds ~10% sample on {args.verify_model})")

    if args.dry_run:
        for i in todo[:20]:
            p = sample_params(i, args.seed)
            print(f"  [{i}] {p.locale} {p.purpose} "
                  f"fields={len(p.field_types)} markup={p.markup_style} "
                  f"name={p.name_style} rare={p.rare_injected}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        print("Dry run: no API calls made.")
        return 0

    usage = Usage()
    sem = asyncio.Semaphore(args.concurrency)
    counts = {"accepted": 0, "rejected": 0, "error": 0}

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    async def worker(i: int):
        async with sem:
            if args.max_cost and usage.cost(args.model) >= args.max_cost:
                return None
            params = sample_params(i, args.seed)
            rec = await generate_one(client, args, params, anchors, dedup, usage)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            append_jsonl(manifest_path, rec)
            if rec["status"] != "accepted":
                append_jsonl(rejects_path, rec)
            tag = {"accepted": "OK ", "rejected": "skip", "error": "ERR"}[rec["status"]]
            extra = rec.get("filename") or rec.get("reason", "")
            print(f"  [{i}] {tag} {rec['mode']:<8} {extra}")
            return rec

    await asyncio.gather(*(worker(i) for i in todo))

    print("\n--- Summary ---")
    print(f"accepted={counts['accepted']} rejected={counts['rejected']} "
          f"error={counts['error']}")
    print(f"API calls={usage.calls} tokens(in/out)="
          f"{usage.prompt_tokens}/{usage.completion_tokens} "
          f"cost~${usage.cost(args.model):.2f}")
    print(f"Output: {args.out}")
    return 0


def review(args) -> int:
    """Print a label/type table for a sample of accepted files for eyeballing."""
    files = glob.glob(os.path.join(args.out, "GEN_*.html"))
    if not files:
        print(f"No generated files in {args.out}")
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(files, k=min(args.review, len(files)))
    for path in sample:
        print(f"\n=== {os.path.basename(path)} ===")
        with open(path, encoding="utf-8") as fh:
            soup = BeautifulSoup(fh.read(), "html.parser")
        for el in soup.select(f"[{LABEL_ATTR}]"):
            label = ""
            prev = el.find_previous(["span", "label"])
            if prev and prev.get_text(strip=True):
                label = prev.get_text(strip=True)
            elif el.get("placeholder"):
                label = el.get("placeholder")
            elif el.get("aria-label"):
                label = el.get("aria-label")
            print(f"  {el.get(LABEL_ATTR):<22} <- {label!r}")
    return 0


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=50,
                   help="number of forms to (attempt to) generate")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                   help="generation model")
    p.add_argument("--verify-model", default="gpt-4o", help="verification model")
    p.add_argument("--mutate-ratio", type=float, default=0.30,
                   help="fraction of forms produced by anchored mutation")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234, help="master RNG seed")
    p.add_argument("--max-cost", type=float, default=None,
                   help="hard stop once estimated spend (USD) is reached")
    p.add_argument("--verify", action="store_true", default=True,
                   help="run the semantic verification pass (default on)")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--verify-sample", type=float, default=0.10,
                   help="fraction of non-rare forms to verify")
    p.add_argument("--dry-run", action="store_true",
                   help="show the plan + cost estimate without calling the API")
    p.add_argument("--resume", action="store_true",
                   help="skip indices already accepted in the manifest")
    p.add_argument("--review", type=int, default=0,
                   help="print label/type tables for N accepted files and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Load the repo-root .env explicitly so the key is found regardless of CWD.
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.review:
        return review(args)

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (put it in .env or the environment).",
              file=sys.stderr)
        return 2

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
