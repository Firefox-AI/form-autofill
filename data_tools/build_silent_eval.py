"""Diagnostic: does the full-scripts tokenizer help on fields where the REGEX IS
SILENT (no rule fires) in non-Latin tail languages -- the one place the regex-hint
feature can't help, so only script vocab can. tail-gold.txt is the regex-*disagreement*
set (regex always fired), so it can't answer this; this builds the complementary set.

Collects fields from the mined CC forms where norm(regex_hint)=="" (silent) in the
target langs, o4-labels them for independent ground truth (the model's own label
would be circular), then writes two tail-gold-format files:
  silent-tailgold.txt       (plain mlData; for no-hint models)
  silent-tailgold-hint.txt  (mlData with selective hints; the silent focus field's
                             own hint is **hintnone -- exactly what a hint model
                             sees in production for a regex-silent field)

Eval afterwards with data_tools/eval_tailgold.py.
"""
import sys, os, glob, json, argparse, asyncio, csv as _csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bs4 import BeautifulSoup

from ff_preprocess import (fillable_fields, tokenize_elements, hint_token,
                           get_label_strings, element_type)
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name, DEFAULT_HINT_SUPPRESS
from gen.llm import Usage, request_field_labels
from openai import AsyncOpenAI

FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))
_csv.field_size_limit(10 ** 7)


def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""


def field_ctx(el, soup, idx):
    opts = ""
    if el.name == "select":
        opts = " ".join((o.get_text() or o.get("value") or "").strip()
                        for o in el.find_all("option"))[:120]
    return {"index": idx, "tag": el.name, "type": element_type(el),
            "label": " ".join(get_label_strings(el, soup))[:100],
            "placeholder": el.get("placeholder") or "", "name": el.get("name") or "",
            "id": el.get("id") or "", "options": opts, "nearby": ""}


def load_env_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    for envp in (".env", os.path.join(os.path.dirname(__file__), "..", ".env")):
        if os.path.exists(envp):
            for line in open(envp, encoding="utf-8"):
                line = line.strip()
                if line.startswith("OPENAI_API_KEY") and "=" in line:
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


async def run_audit(items, llm, budget, batch):
    client = AsyncOpenAI()
    usage = Usage()
    labels = []
    for start in range(0, len(items), batch):
        if usage.cost(llm) >= budget:
            print(f"budget ${budget} reached at {start}"); break
        chunk = items[start:start + batch]
        ctxs = [dict(it["ctx"], index=k) for k, it in enumerate(chunk)]
        got = await request_field_labels(client, llm, ctxs, usage)
        for k in range(len(chunk)):
            labels.append(norm(got.get(k, "")))
        print(f"  audited {min(start+batch,len(items))}/{len(items)}  cost=${usage.cost(llm):.2f}", flush=True)
    # pad if budget cut short
    labels += [""] * (len(items) - len(labels))
    return labels, usage.cost(llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=sorted(glob.glob("data/cc_mined_2026/cc_forms_shard*.csv")))
    ap.add_argument("--langs", default="ru,ja,fa,zh,ko,hi")
    ap.add_argument("--cap", type=int, default=100, help="max silent fields per lang")
    ap.add_argument("--llm", default="o4-mini")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=30)
    args = ap.parse_args()
    target = {x.strip() for x in args.langs.split(",") if x.strip()}
    load_env_key()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set (env or .env).")

    per_lang = defaultdict(int)
    items = []   # each: {lang, ctx, ml_plain, ml_hint, tag}
    for si, shard in enumerate(args.shards):
        for ri, row in enumerate(_csv.DictReader(open(shard, newline="", encoding="utf-8"))):
            lang = row.get("lang", "?")
            if lang not in target or per_lang[lang] >= args.cap:
                continue
            html = row.get("form") or ""
            if not html.strip():
                continue
            soup = BeautifulSoup(f"<html><body>{html}</body></html>", "html.parser")
            fields = fillable_fields(soup)
            if not fields:
                continue
            regex_labels = [norm(regex_hint(el, soup)) for el in fields]
            hints = [hint_token(r if (r and r not in DEFAULT_HINT_SUPPRESS) else "")
                     for r in regex_labels]
            ml_plain = tokenize_elements(soup, type_marker="training")
            ml_hint = tokenize_elements(soup, type_marker="training", hints=hints)
            for fi, (el, r) in enumerate(zip(fields, regex_labels)):
                if r:                       # regex fired -> not silent, skip
                    continue
                ctx = field_ctx(el, soup, len(items))
                if not (ctx["label"] or ctx["name"] or ctx["placeholder"]):
                    continue                # nothing for o4 (or model) to read
                if per_lang[lang] >= args.cap:
                    break
                per_lang[lang] += 1
                items.append({"lang": lang, "ctx": ctx,
                              "ml_plain": ml_plain[fi], "ml_hint": ml_hint[fi],
                              "tag": f"SILENT_{lang}_{si}_{ri}_{fi}"})
    print(f"collected {len(items)} regex-silent fields  by lang: {dict(per_lang)}")

    labels, spent = asyncio.run(run_audit(items, args.llm, args.budget, args.batch))
    plain, hint = [], []
    kept = defaultdict(int)
    for it, lab in zip(items, labels):
        if not lab:
            continue
        kept[it["lang"]] += 1
        plain.append(f"{it['tag']},{lab},{FT[lab]},{it['ml_plain']}")
        hint.append(f"{it['tag']},{lab},{FT[lab]},{it['ml_hint']}")
    open("silent-tailgold.txt", "w", encoding="utf-8").write("\n".join(plain) + "\n")
    open("silent-tailgold-hint.txt", "w", encoding="utf-8").write("\n".join(hint) + "\n")
    print(f"\nspent ${spent:.2f}; wrote {len(plain)} labeled rows -> silent-tailgold(.hint).txt")
    print(f"by lang: {dict(kept)}")


if __name__ == "__main__":
    main()
