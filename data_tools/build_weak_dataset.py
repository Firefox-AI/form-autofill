"""Build training rows from the mined CC forms using the best available label
per field, then merge with the Python gold base (-relabelpy) into a new variant.

Per-field label priority:
  1. gold      : an LLM-audited disagreement (audit_verdicts.jsonl, consumed in
                 the SAME deterministic order the audit collected them).
  2. agree     : model == regex (both a real type).
  3. model_only: regex silent AND model confident (>= --conf), EXCEPT for the
                 languages where the audit showed the model is unreliable
                 (zh/ko/hi -> drop model_only; keep only agree + gold there).
Fields with no trusted label emit no row (they still provide bb/aa neighbor
context in every emitted row's mlData). Capped to --cap fields/language (whole
forms kept). mlData is the Python ff_preprocess tokenization (see module notes on
Firefox vs Python).

Output: processed_datasets/training<variant>.txt = training-relabelpy.txt + the
new rows; validation/testing<variant>.txt are copied from -relabelpy (gold only).
"""
import sys, os, glob, json, argparse, csv as _csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ff_preprocess import fillable_fields, tokenize_elements, hint_token
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name, DEFAULT_HINT_SUPPRESS
from dotraining import reformat_context, strip_type_marker, Config

_CFG = Config(contextFormat="bb")
FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))
OTHER = FT.get("other", 1)
WEAK_MODEL_LANGS = {"zh", "ko", "hi"}   # model unreliable (audit) -> no model_only
_csv.field_size_limit(10 ** 7)


def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="ordered shard CSVs (audit order)")
    ap.add_argument("--verdicts", default="data/cc_mined_2026/audit_verdicts.jsonl")
    ap.add_argument("--model", default="/tmp/labeler_889f8")
    ap.add_argument("--conf", type=float, default=0.90)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--base", default="training-relabelpy.txt")
    ap.add_argument("--variant", default="-weakmulti")
    ap.add_argument("--hint", action="store_true",
                    help="prepend selective regex-hint tokens to the mined rows' mlData "
                         "(pair with --base training-relabelhintsel.txt)")
    ap.add_argument("--eval-langs", default="",
                    help="comma list of languages whose o4-AUDITED fields to HOLD OUT into a "
                         "gold eval file (excluded from training) -- e.g. zh,ko,fa,hi,ru,pl,tr")
    ap.add_argument("--eval-out", default="testing-tailgold.txt")
    args = ap.parse_args()
    eval_langs = {x.strip() for x in args.eval_langs.split(",") if x.strip()}
    eval_rows = []

    verdicts = [json.loads(l) for l in open(args.verdicts, encoding="utf-8") if l.strip()]
    vptr = 0
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    id2label = {int(k): v for k, v in
                json.load(open(os.path.join(args.model, "config.json")))["id2label"].items()}

    per_lang = defaultdict(int)
    rows = []
    stats = defaultdict(lambda: defaultdict(int))
    for si, shard in enumerate(args.shards):
        for ri, row in enumerate(_csv.DictReader(open(shard, newline="", encoding="utf-8"))):
            html = row.get("form") or ""
            lang = row.get("lang", "?")
            if not html.strip():
                continue
            soup = BeautifulSoup(f"<html><body>{html}</body></html>", "html.parser")
            fields = fillable_fields(soup)
            if not fields:
                continue
            ml_nohint = tokenize_elements(soup, type_marker="training")   # labeler is no-hint
            regex_labels = [norm(regex_hint(el, soup)) for el in fields]
            if args.hint:
                # selective hint token per field (same rule as -relabelhintsel):
                # emit **hint<class> for a hintable regex class, else **hintnone.
                hints = [hint_token(r if (r and r not in DEFAULT_HINT_SUPPRESS) else "")
                         for r in regex_labels]
                ml_out = tokenize_elements(soup, type_marker="training", hints=hints)
            else:
                ml_out = ml_nohint
            # model predictions (on the no-hint mlData, matching the labeler)
            preds = []
            with torch.no_grad():
                for i in range(0, len(ml_nohint), 64):
                    enc = tok([reformat_context(strip_type_marker(t, _CFG), "bb") for t in ml_nohint[i:i + 64]],
                              truncation=True, max_length=512, padding=True, return_tensors="pt")
                    probs = torch.softmax(model(**enc).logits, dim=-1)
                    conf, idx = probs.max(dim=-1)
                    preds += list(zip(idx.tolist(), conf.tolist()))
            form_rows = []
            for el, mldata, r, (pk, pconf) in zip(fields, ml_out, regex_labels, preds):
                m = norm(id2label[pk])
                label = None
                if m and r and m == r:
                    label = m; stats[lang]["agree"] += 1
                elif not r and m:                      # model_only candidate (regex silent)
                    if lang in WEAK_MODEL_LANGS or pconf < args.conf:
                        stats[lang]["drop_weakmodel" if lang in WEAK_MODEL_LANGS else "drop_lowconf"] += 1
                    else:
                        label = m; stats[lang]["model_only"] += 1
                elif (m and r and m != r) or (r and not m):   # disagreement -> audited verdict
                    v = verdicts[vptr]; vptr += 1
                    # safety: order must match the audit collection
                    assert v["lang"] == lang and (v["model"] or "") == (m or "") and (v["regex"] or "") == (r or ""), \
                        f"verdict misalignment at {vptr}: {v} vs lang={lang} m={m} r={r}"
                    llm = norm(v.get("llm"))
                    if llm and lang in eval_langs:
                        # hold out this o4-gold field into the tail eval; not trained on.
                        eval_rows.append((f"TAILGOLD_{lang}_{si}_{ri}", llm, FT[llm], mldata))
                        stats[lang]["heldout_eval"] += 1
                        continue
                    if llm:
                        label = llm; stats[lang]["gold"] += 1
                    else:
                        stats[lang]["gold_skip"] += 1
                if label:
                    # CC_ prefix -> classified as the 'cc' source (they ARE common-crawl),
                    # so they don't inflate n_real and the gen/cc subsample caps stay
                    # identical to the baseline (clean attribution of any gain).
                    form_rows.append((f"CC_weak_{lang}_{si}_{ri}", label, FT[label], mldata))
            if not form_rows:
                continue
            if per_lang[lang] >= args.cap:             # cap by language (whole forms)
                stats[lang]["capped_forms"] += 1
                continue
            per_lang[lang] += len(form_rows)
            rows.extend(form_rows)

    assert vptr == len(verdicts), f"consumed {vptr} verdicts != {len(verdicts)} (order drift!)"

    # write the new rows + concat with the gold base
    os.makedirs("processed_datasets", exist_ok=True)
    weak_lines = [f"{f},{n},{lid},{d}" for (f, n, lid, d) in rows]
    open(f"/tmp/weak_rows{args.variant}.txt", "w", encoding="utf-8").write("\n".join(weak_lines) + "\n")
    base = open(args.base, encoding="utf-8").read().rstrip("\n")
    out = f"training{args.variant}.txt"
    open(out, "w", encoding="utf-8").write(base + "\n" + "\n".join(weak_lines) + "\n")
    import shutil
    base_suffix = os.path.basename(args.base)[len("training"):]   # e.g. -relabelpy.txt
    for split in ("validation", "testing"):
        src = f"{split}{base_suffix}"
        if os.path.exists(src):
            shutil.copy(src, f"{split}{args.variant}.txt")
        else:
            print(f"  WARN: {src} not found; {split}{args.variant}.txt not written")

    if eval_rows:
        with open(args.eval_out, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{fn},{n},{lid},{d}" for (fn, n, lid, d) in eval_rows) + "\n")
        from collections import Counter
        ec = Counter(r[0].split("_")[1] for r in eval_rows)
        print(f"HELD-OUT tail gold eval: {len(eval_rows)} fields -> {args.eval_out}  by lang: {dict(ec)}")
    print(f"new weak+gold rows: {len(rows)}  (base {args.base}: {base.count(chr(10))+1} rows)")
    print(f"{'lang':5}{'agree':>7}{'model_only':>11}{'gold':>6}{'goldskip':>9}{'dropWM':>8}{'kept':>7}")
    total_kept = defaultdict(int)
    for f, n, lid, d in rows:
        total_kept[f.split("_")[2]] += 1   # CC_weak_<lang>_<si>_<ri>
    for lang in sorted(stats, key=lambda k: -total_kept[k]):
        s = stats[lang]
        print(f"{lang:5}{s['agree']:7}{s['model_only']:11}{s['gold']:6}{s['gold_skip']:9}"
              f"{s['drop_weakmodel']:8}{total_kept[lang]:7}")
    print(f"\nwrote {out} (+ validation/testing{args.variant}.txt copied from -relabelpy gold)")


if __name__ == "__main__":
    main()
