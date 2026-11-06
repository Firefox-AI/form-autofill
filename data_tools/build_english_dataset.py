"""Build an English-only train/val dataset from a weakmulti3hint base.

"English" = a row whose source classify_locale()s as English (GEN_en*, EN_*,
QA-<english cc>, bare en/us/...) OR a Common-Crawl row (CC_*) whose page was
detected English by langdetect (data/cc_lang_map.json). The latter recovers the
real English CC forms that classify_locale can't see (CC filenames carry no
language tag).

Writes training-<out>.txt and validation-<out>.txt filtered to English, and
testing-<out>.txt as a FULL copy of the base test set (so evaluation still runs
on everything and the English-vs-rest comparison stays visible).

Usage: .venv/bin/python -I data_tools/build_english_dataset.py [base_variant] [out_variant]
   default base=-weakmulti3hint  out=-wm3en
"""
import sys, os, json, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotraining import classify_locale

BASE = sys.argv[1] if len(sys.argv) > 1 else "-weakmulti3hint"
OUT = sys.argv[2] if len(sys.argv) > 2 else "-wm3en"
LANGMAP = "data/cc_lang_map.json"

langmap = json.load(open(LANGMAP))
en_cc = {k for k, v in langmap.items() if v == "en"}

def is_english(src):
    if classify_locale(src)[2]:
        return True
    return src.startswith("CC_") and src in en_cc

def filter_file(split):
    src_path = f"{split}{BASE}.txt"
    out_path = f"{split}{OUT}.txt"
    kept = tot = cc_en = loc_en = 0
    with open(src_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            tot += 1
            s = line.split(",", 1)[0]
            if classify_locale(s)[2]:
                fout.write(line); kept += 1; loc_en += 1
            elif s.startswith("CC_") and s in en_cc:
                fout.write(line); kept += 1; cc_en += 1
    print(f"{split:11} {src_path} -> {out_path}: kept {kept}/{tot}  (locale-en={loc_en}, cc-en={cc_en})")
    return kept

filter_file("training")
filter_file("validation")
# Test stays FULL so eval covers all locales.
shutil.copyfile(f"testing{BASE}.txt", f"testing{OUT}.txt")
print(f"testing      copied FULL base -> testing{OUT}.txt")
