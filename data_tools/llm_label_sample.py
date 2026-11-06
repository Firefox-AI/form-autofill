"""LLM-label a random sample of mined forms (ground truth) to measure the field
distribution and the real per-field cost, then extrapolate to the full set.

Independent of the consensus ML model: every field of each sampled form is sent
to the LLM in batches. Budget-capped. Requires OPENAI_API_KEY in the env.

Usage: .venv/bin/python data_tools/llm_label_sample.py <csv> [n_forms] [model] [budget]
"""
import sys, os, csv, random, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
csv.field_size_limit(10 ** 7)
from bs4 import BeautifulSoup
from collections import Counter
from ff_preprocess import fillable_fields, get_label_strings, element_type
from build_dataset import load_field_types, adjusted_field_name
from gen.llm import Usage, request_field_labels
from openai import AsyncOpenAI

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/cc_german_2026/german_forms.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MODEL = sys.argv[3] if len(sys.argv) > 3 else "o4-mini"
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
BATCH = 30
FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))

def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else (raw or "other")

def field_context(el, soup, idx):
    opts = ""
    if el.name == "select":
        opts = " ".join((o.get_text() or o.get("value") or "").strip() for o in el.find_all("option"))[:120]
    return {"index": idx, "tag": el.name, "type": element_type(el),
            "label": " ".join(get_label_strings(el, soup))[:100],
            "placeholder": el.get("placeholder") or "", "name": el.get("name") or "",
            "id": el.get("id") or "", "options": opts, "nearby": ""}

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
random.seed(20261106)
sample = random.sample(rows, min(N, len(rows)))

# flatten all fields of the sample into LLM batches
items = []
for r in sample:
    soup = BeautifulSoup(f"<html><body>{r['form']}</body></html>", "html.parser")
    for el in fillable_fields(soup):
        items.append(field_context(el, soup, 0))
print(f"sample: {len(sample)} forms, {len(items)} fields -> labeling with {MODEL} (budget ${BUDGET})", flush=True)

async def main():
    client = AsyncOpenAI()
    usage = Usage()
    dist = Counter(); done = 0
    for start in range(0, len(items), BATCH):
        if usage.cost(MODEL) >= BUDGET:
            print(f"budget hit at {start} fields", flush=True); break
        chunk = [dict(it, index=k) for k, it in enumerate(items[start:start + BATCH])]
        labels = await request_field_labels(client, MODEL, chunk, usage)
        for k in range(len(chunk)):
            dist[norm(labels.get(k, "other"))] += 1
        done = start + len(chunk)
        print(f"  {done}/{len(items)}  cost=${usage.cost(MODEL):.3f}", flush=True)
    cost = usage.cost(MODEL)
    tot = sum(dist.values())
    print(f"\n=== LLM-labeled {tot} fields | cost ${cost:.3f} | ${cost/max(tot,1):.5f}/field ===")
    for lab, n in dist.most_common(28):
        print(f"  {lab:22}{n:>5} ({100*n/tot:.0f}%)")
    full = 28074
    print(f"\nextrapolated FULL-SET cost (28,074 fields, all-LLM): ${cost/max(tot,1)*full:.2f} with {MODEL}")
    print("(consensus path is far cheaper: only the ~24% disagree bucket hits the LLM)")

asyncio.run(main())
