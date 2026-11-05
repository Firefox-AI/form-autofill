"""Regex-as-weak-labeler over real Common Crawl forms, focused on non-English
(long-tail) languages. For each form, runs the ported Firefox regex heuristic
(heuristics_regexp.regex_hint) on every fillable field and emits a weak label.
Reports per-language yield and, where a gold data-moz-autofill-type exists,
the regex's agreement with it (weak-label precision) -- the signal for whether
regex weak-labeling is viable to scale for that language. Writes weak-labeled
records to /tmp/weak_labeled_tail.jsonl.

Scale-out note: for a raw WARC/CSV dump (vs the ~4k local files) shard the input
and fan out; at this local scale a single pass runs in seconds.
"""
import sys, os, glob, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bs4 import BeautifulSoup
from ff_preprocess import fillable_fields
from heuristics_regexp import regex_hint
from build_dataset import load_field_types, adjusted_field_name

FT = load_field_types(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"))
def norm(raw):
    a = adjusted_field_name(raw) if raw else ""
    return a if a and a in FT else ""

# filename -> lang from the CC manifest
lang_of = {}
for l in open("data/common_crawl_2026_manifest.jsonl", encoding="utf-8"):
    try:
        r = json.loads(l); lang_of[r["filename"]] = r.get("lang", "?")
    except Exception:
        pass

NONLATIN = re.compile(r"[^\x00-\x7f]")
from collections import defaultdict
stats = defaultdict(lambda: {"forms":0, "regex_fields":0, "native_fields":0,
                             "gold_fields":0, "agree":0})
out = open("/tmp/weak_labeled_tail.jsonl", "w", encoding="utf-8")
nrec = 0
for path in sorted(glob.glob("data/common_crawl/*.html")):
    fn = os.path.basename(path)
    lang = lang_of.get(fn, "?")
    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "html.parser")
    fields = fillable_fields(soup)
    recs = []
    for el in fields:
        h = norm(regex_hint(el, soup))
        if not h:
            continue
        blob = " ".join(str(v) for v in [el.get("id"), el.get("name"),
                el.get("placeholder"), el.get("aria-label")] if v)
        native = bool(NONLATIN.search(blob))
        gold = norm(el.get("data-moz-autofill-type"))
        recs.append({"id": el.get("id") or el.get("name"), "regex": h,
                     "native": native, "gold": gold or None})
        stats[lang]["regex_fields"] += 1
        if native: stats[lang]["native_fields"] += 1
        if gold:
            stats[lang]["gold_fields"] += 1
            if gold == h: stats[lang]["agree"] += 1
    if recs:
        stats[lang]["forms"] += 1
        out.write(json.dumps({"file": fn, "lang": lang, "fields": recs}) + "\n")
        nrec += 1
out.close()
print(f"weak-labeled {nrec} forms -> /tmp/weak_labeled_tail.jsonl\n")
print(f"{'lang':6}{'forms':>7}{'regexFields':>12}{'nativeScript':>13}{'gold∩':>7}{'agree%':>8}")
tot=defaultdict(int)
for lang in sorted(stats, key=lambda k:-stats[k]["regex_fields"]):
    s=stats[lang]; ag = 100*s["agree"]/s["gold_fields"] if s["gold_fields"] else 0
    print(f"{lang:6}{s['forms']:7}{s['regex_fields']:12}{s['native_fields']:13}{s['gold_fields']:7}{ag:7.1f}%")
    for k in s: tot[k]+=s[k]
# non-English aggregate
nonen={k:v for k,v in stats.items() if k not in ("en","?")}
ff=sum(v["forms"] for v in nonen.values()); rf=sum(v["regex_fields"] for v in nonen.values())
nf=sum(v["native_fields"] for v in nonen.values()); gf=sum(v["gold_fields"] for v in nonen.values())
ag=sum(v["agree"] for v in nonen.values())
print(f"\nNON-ENGLISH tail: {ff} forms, {rf} regex-labeled fields ({nf} via native script), "
      f"weak-label agreement with gold {100*ag/max(gf,1):.1f}% (n={gf})")
