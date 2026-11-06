"""LLM-label every field of every (deduped) form across one or more CSVs.

Writes one JSONL line per form with ground-truth field types, keeping the form
HTML so the result is reusable for training-row construction. Forms are deduped
by form_hash across inputs and tagged with which input set(s) they belong to.
Concurrent batches for speed; hard $ budget cap (needs model in gen.llm PRICING).

Usage:
  .venv/bin/python data_tools/llm_label_forms.py --out data/cc_german_2026/labeled_terra.jsonl \
      --model gpt-5.6-terra --budget 18 \
      --set all:data/cc_german_2026/german_forms.csv \
      --set address:data/cc_german_2026/german_address_forms.csv
"""
import sys, os, csv, json, argparse, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
csv.field_size_limit(10 ** 7)
from bs4 import BeautifulSoup
from collections import Counter
from ff_preprocess import fillable_fields, get_label_strings, element_type
from build_dataset import load_field_types, adjusted_field_name
from gen.llm import Usage, request_field_labels
from openai import AsyncOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", required=True, help="tag:path.csv (repeatable)")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-5.6-terra")
ap.add_argument("--budget", type=float, default=18.0)
ap.add_argument("--batch", type=int, default=30)
ap.add_argument("--concurrency", type=int, default=6)
args = ap.parse_args()
FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))

def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else (raw or "other")

def fctx(el, soup, idx):
    opts = ""
    if el.name == "select":
        opts = " ".join((o.get_text() or o.get("value") or "").strip() for o in el.find_all("option"))[:120]
    return {"index": idx, "tag": el.name, "type": element_type(el),
            "label": " ".join(get_label_strings(el, soup))[:100],
            "placeholder": el.get("placeholder") or "", "name": el.get("name") or "",
            "id": el.get("id") or "", "options": opts, "nearby": ""}

# dedup across sets, tag membership
forms = {}   # form_hash -> record
for spec in args.set:
    tag, path = spec.split(":", 1)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        h = r.get("form_hash")
        if not h:
            continue
        if h not in forms:
            forms[h] = {"form_hash": h, "url": r.get("url", ""), "tld": r.get("tld", ""),
                        "sets": set(), "form": r.get("form", "")}
        forms[h]["sets"].add(tag)
forms = list(forms.values())

# flatten fields with back-references to their form
items = []   # (form_idx, fctx)
for fi, rec in enumerate(forms):
    soup = BeautifulSoup(f"<html><body>{rec['form']}</body></html>", "html.parser")
    rec["_fields"] = []
    for el in fillable_fields(soup):
        rec["_fields"].append({"id": el.get("id") or "", "name": el.get("name") or "",
                               "type": element_type(el), "label": None})
        items.append((fi, len(rec["_fields"]) - 1, fctx(el, soup, 0)))
print(f"{len(forms)} forms, {len(items)} fields -> {args.model} (budget ${args.budget})", flush=True)

async def main():
    client = AsyncOpenAI(); usage = Usage()
    sem = asyncio.Semaphore(args.concurrency)
    dist = Counter(); done = [0]; stop = [False]
    batches = [items[s:s + args.batch] for s in range(0, len(items), args.batch)]
    async def run_batch(batch):
        if stop[0] or usage.cost(args.model) >= args.budget:
            stop[0] = True; return
        async with sem:
            ctxs = [dict(it[2], index=k) for k, it in enumerate(batch)]
            try:
                labs = await request_field_labels(client, args.model, ctxs, usage)
            except Exception as e:
                print("batch error:", repr(e)[:120], flush=True); return
            for k, (fi, fj, _c) in enumerate(batch):
                lab = norm(labs.get(k, "other"))
                forms[fi]["_fields"][fj]["label"] = lab
                dist[lab] += 1
            done[0] += len(batch)
            if done[0] % 1500 < args.batch:
                print(f"  {done[0]}/{len(items)}  cost=${usage.cost(args.model):.2f}", flush=True)
    # run with bounded concurrency
    pending = [run_batch(b) for b in batches]
    await asyncio.gather(*pending)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in forms:
            fh.write(json.dumps({"form_hash": rec["form_hash"], "url": rec["url"],
                                 "tld": rec["tld"], "sets": sorted(rec["sets"]),
                                 "fields": rec["_fields"], "form": rec["form"]},
                                ensure_ascii=False) + "\n")
    print(f"\nlabeled {done[0]} fields | cost ${usage.cost(args.model):.2f} -> {args.out}")
    tot = sum(dist.values())
    for lab, n in dist.most_common(30):
        print(f"  {lab:22}{n:>6} ({100*n/max(tot,1):.0f}%)")

asyncio.run(main())
