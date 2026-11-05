"""LLM audit of the consensus weak-labeler's DISAGREEMENT bucket.

For every field where the ML model and the regex disagree (or only the regex
fired), ask an LLM (default o4-mini) for the ground-truth autofill type, then
tally whether the LLM sides with the model, the regex, or neither. This both
(a) measures the true precision of each labeler on the hard cases and (b) yields
gold labels for those fields. Hard-capped by a dollar budget (default $10) via
the shared Usage cost tracker; stops early when the budget is hit.

Requires OPENAI_API_KEY. Usage:
  uv run python data_tools/audit_llm.py --csv data/cc_mined_2026/cc_forms_shard*.csv \
      --model /tmp/labeler_889f8 --llm o4-mini --budget 10 --batch 30
"""
import sys, os, glob, json, argparse, asyncio
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ff_preprocess import fillable_fields, tokenize_elements, get_label_strings, element_type
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name
from dotraining import reformat_context, strip_type_marker, Config
from gen.llm import Usage, request_field_labels
from openai import AsyncOpenAI

_CFG = Config(contextFormat="bb")
FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))


def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""


def field_context(el, soup, idx):
    labels = get_label_strings(el, soup)
    opts = ""
    if el.name == "select":
        opts = " ".join((o.get_text() or o.get("value") or "").strip()
                        for o in el.find_all("option"))[:120]
    return {
        "index": idx, "tag": el.name, "type": element_type(el),
        "label": " ".join(labels)[:100], "placeholder": el.get("placeholder") or "",
        "name": el.get("name") or "", "id": el.get("id") or "", "options": opts,
        "nearby": "",
    }


def collect_disagreements(model_dir, csv_paths):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label = {int(k): v for k, v in
                json.load(open(os.path.join(model_dir, "config.json")))["id2label"].items()}
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    items = []
    for p in [x for pat in csv_paths for x in glob.glob(pat)]:
        with open(p, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                html = row.get("form") or ""
                if not html.strip():
                    continue
                lang = row.get("lang", "?")
                soup = BeautifulSoup(f"<html><body>{html}</body></html>", "html.parser")
                fields = fillable_fields(soup)
                if not fields:
                    continue
                ml = tokenize_elements(soup, type_marker="training")
                with torch.no_grad():
                    for i in range(0, len(ml), 64):
                        enc = tok([reformat_context(strip_type_marker(t, _CFG), "bb")
                                   for t in ml[i:i + 64]], truncation=True, max_length=512,
                                  padding=True, return_tensors="pt")
                        idxs = model(**enc).logits.argmax(-1).tolist()
                        for j, k in enumerate(idxs):
                            el = fields[i + j]
                            m = norm(id2label[k]); r = norm(regex_hint(el, soup))
                            if (m and r and m != r) or (r and not m):
                                items.append({"lang": lang, "model": m or None,
                                              "regex": r or None,
                                              "ctx": field_context(el, soup, len(items))})
    return items


async def run_audit(items, llm, budget, batch):
    client = AsyncOpenAI()
    usage = Usage()
    verdicts = []
    for start in range(0, len(items), batch):
        if usage.cost(llm) >= budget:
            print(f"budget ${budget} reached at {start} fields"); break
        chunk = items[start:start + batch]
        ctxs = [dict(it["ctx"], index=k) for k, it in enumerate(chunk)]
        labels = await request_field_labels(client, llm, ctxs, usage)
        for k, it in enumerate(chunk):
            llm_lab = norm(labels.get(k, ""))
            verdicts.append({**{x: it[x] for x in ("lang", "model", "regex")},
                             "llm": llm_lab or None})
        print(f"  audited {min(start+batch,len(items))}/{len(items)}  cost=${usage.cost(llm):.2f}", flush=True)
    return verdicts, usage.cost(llm)


def report(verdicts):
    t = defaultdict(int); per = defaultdict(lambda: defaultdict(int))
    for v in verdicts:
        if not v["llm"]:
            t["llm_skip"] += 1; continue
        who = ("model" if v["llm"] == v["model"] else
               "regex" if v["llm"] == v["regex"] else "neither")
        t[who] += 1; per[v["lang"]][who] += 1
    tot = t["model"] + t["regex"] + t["neither"]
    print(f"\n=== audit verdicts on {tot} judged disagreements (skipped {t['llm_skip']}) ===")
    print(f"  LLM sides with MODEL : {t['model']} ({100*t['model']/max(tot,1):.0f}%)")
    print(f"  LLM sides with REGEX : {t['regex']} ({100*t['regex']/max(tot,1):.0f}%)")
    print(f"  LLM says NEITHER     : {t['neither']} ({100*t['neither']/max(tot,1):.0f}%)")
    print(f"\n{'lang':6}{'model':>7}{'regex':>7}{'neither':>8}")
    for lang in sorted(per, key=lambda k: -sum(per[k].values())):
        p = per[lang]; print(f"{lang:6}{p['model']:7}{p['regex']:7}{p['neither']:8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm", default="o4-mini")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--out", default="data/cc_mined_2026/audit_verdicts.jsonl")
    args = ap.parse_args()
    # Load OPENAI_API_KEY from a repo-root .env if not already in the environment.
    if not os.environ.get("OPENAI_API_KEY"):
        for envp in (".env", os.path.join(os.path.dirname(__file__), "..", ".env")):
            if os.path.exists(envp):
                for line in open(envp, encoding="utf-8"):
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY") and "=" in line:
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set (checked env and .env).")
    print("collecting disagreement fields...")
    items = collect_disagreements(args.model, args.csv)
    print(f"{len(items)} disagreement fields to audit (budget ${args.budget}, {args.llm})")
    verdicts, spent = asyncio.run(run_audit(items, args.llm, args.budget, args.batch))
    with open(args.out, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    report(verdicts)
    print(f"\nspent ${spent:.2f} of ${args.budget}; verdicts -> {args.out}")


if __name__ == "__main__":
    main()
