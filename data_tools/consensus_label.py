"""Consensus weak-labeler: label real Common Crawl forms with the EXISTING ML
model cross-checked by the ported regex heuristic -- no per-field LLM cost.

For each fillable field of each form:
  * ML label  = a no-hint bb classifier's argmax over the ff_preprocess mlData
                (independent of the regex), with its softmax confidence.
  * regex label = heuristics_regexp.regex_hint (adjusted to the taxonomy).

Consensus routing (per field):
  keep(agree)      : ml == regex (both a real type)      -> high-confidence weak label
  keep(model_only) : regex silent AND ml_conf >= --conf  -> recall path (uncovered langs)
  audit(disagree)  : ml != regex, both real  OR  regex-only fires -> LLM/o4 bucket
  drop             : both silent / low confidence

Outputs (JSONL): --out-keep (weak-labeled), --out-audit (disagreements), plus a
per-language yield/agreement report. The LLM is only ever needed for the audit
bucket, and even that is optional.

Usage:
  uv run python data_tools/consensus_label.py --model /tmp/hint_B0 \
      --csv /tmp/cc_forms_shard*.csv --conf 0.90
"""
import sys, os, glob, json, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ff_preprocess import fillable_fields, tokenize_elements
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name
from dotraining import reformat_context, strip_type_marker, Config

_CFG = Config(contextFormat="bb")
FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))


def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""


def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label = {int(k): v for k, v in
                json.load(open(os.path.join(model_dir, "config.json")))["id2label"].items()}
    return tok, model, id2label


def predict(tok, model, id2label, texts):
    """Return [(label, confidence)] for a list of mlData strings."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok([reformat_context(strip_type_marker(t, _CFG), "bb") for t in texts[i:i + 64]],
                      truncation=True, max_length=512, padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            out += [(id2label[int(j)], float(c)) for j, c in zip(idx.tolist(), conf.tolist())]
    return out


def iter_forms(csv_paths):
    """Yield (lang, form_html) from the extract_forms CSVs."""
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                html = row.get("form") or ""
                if html.strip():
                    yield row.get("lang", "?"), html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="no-hint bb model dir (independent labeler)")
    ap.add_argument("--csv", nargs="+", required=True, help="extract_forms CSV(s)")
    ap.add_argument("--conf", type=float, default=0.90, help="model-only keep threshold")
    ap.add_argument("--out-keep", default="/tmp/consensus_keep.jsonl")
    ap.add_argument("--out-audit", default="/tmp/consensus_audit.jsonl")
    args = ap.parse_args()

    tok, model, id2label = load_model(args.model)
    csvs = [p for pat in args.csv for p in glob.glob(pat)]
    keep = open(args.out_keep, "w", encoding="utf-8")
    audit = open(args.out_audit, "w", encoding="utf-8")
    # per-lang: forms, fields, agree, model_only, disagree, drop
    st = defaultdict(lambda: defaultdict(int))

    for lang, html in iter_forms(csvs):
        soup = BeautifulSoup(f"<html><body>{html}</body></html>", "html.parser")
        fields = fillable_fields(soup)
        if not fields:
            continue
        ml = tokenize_elements(soup, type_marker="training")   # no hint -> independent
        preds = predict(tok, model, id2label, ml)
        st[lang]["forms"] += 1
        kept_fields = []
        for el, (mlab, conf) in zip(fields, preds):
            m = norm(mlab)
            r = norm(regex_hint(el, soup))
            st[lang]["fields"] += 1
            fid = el.get("id") or el.get("name") or ""
            if m and r and m == r:
                st[lang]["agree"] += 1
                kept_fields.append({"id": fid, "label": m, "src": "agree", "conf": round(conf, 3)})
            elif not r and m and conf >= args.conf:
                st[lang]["model_only"] += 1
                kept_fields.append({"id": fid, "label": m, "src": "model_only", "conf": round(conf, 3)})
            elif (m and r and m != r) or (r and not m):
                st[lang]["disagree"] += 1
                audit.write(json.dumps({"lang": lang, "id": fid, "ml": m or None,
                                        "regex": r or None, "conf": round(conf, 3)}) + "\n")
            else:
                st[lang]["drop"] += 1
        if kept_fields:
            keep.write(json.dumps({"lang": lang, "fields": kept_fields}) + "\n")
    keep.close(); audit.close()

    print(f"{'lang':6}{'forms':>7}{'fields':>8}{'agree':>7}{'mdlOnly':>8}{'audit':>7}{'drop':>6}{'keep%':>7}")
    agg = defaultdict(int)
    for lang in sorted(st, key=lambda k: -st[k]["fields"]):
        s = st[lang]; kept = s["agree"] + s["model_only"]
        pct = 100 * kept / max(s["fields"], 1)
        print(f"{lang:6}{s['forms']:7}{s['fields']:8}{s['agree']:7}{s['model_only']:8}"
              f"{s['disagree']:7}{s['drop']:6}{pct:6.0f}%")
        for k, v in s.items():
            agg[k] += v
    tot = agg["fields"]; kept = agg["agree"] + agg["model_only"]
    print(f"\nTOTAL: {agg['forms']} forms, {tot} fields -> KEEP {kept} "
          f"(agree {agg['agree']} + model-only {agg['model_only']}), "
          f"AUDIT {agg['disagree']}, drop {agg['drop']}")
    print(f"weak labels -> {args.out_keep}   audit bucket -> {args.out_audit}")
    print(f"LLM needed only for the {agg['disagree']} audit fields "
          f"({100*agg['disagree']/max(tot,1):.0f}% of fields), and even that is optional.")


if __name__ == "__main__":
    main()
