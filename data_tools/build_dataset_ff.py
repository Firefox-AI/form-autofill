#!/usr/bin/env python
"""Build a dotraining .txt dataset by driving the REAL Firefox scanner (ml_driver).

Unlike build_dataset.py (a Python reimplementation of the tokenizer that can't
reproduce Firefox's field selection/ordering), this drives Firefox: each field's
`mlData` comes straight from FormAutofillHeuristics.tokenizeElements, so the field
SET, ORDER, neighbor (aa/bb) context, and `**` markers are guaranteed-faithful.

Per field it emits `<file>,<fieldname>,<label_id>,<mlData>`, mapping the field
(by inspect-id, else id/name) to its ground-truth `data-moz-autofill-type`.

Env: FIREFOX_BIN, MOZ_ALLOW_EXTERNAL_ML_HUB=1, FASTLY_TOKEN=dummy, LIMIT (opt).
    FIREFOX_BIN=... python build_dataset_ff.py --input data/testing --output out.txt
"""
import argparse, ast, glob, json, os, re, sys, time
from collections import defaultdict

os.environ.setdefault("MOZ_ALLOW_EXTERNAL_ML_HUB", "1")
os.environ.setdefault("FASTLY_TOKEN", "dummy")
sys.path.insert(0, "/Users/Rrando/Documents/GitHub/ml_driver/examples")
sys.path.insert(0, "/Users/Rrando/Documents/GitHub/ml_driver/src")
import form_autofill_accuracy as H  # noqa: E402  (attr, strip_autocomplete, retries)
from ml_driver.firefox.driver import FirefoxDriver  # noqa: E402


def field_types():
    src = open(os.path.join(os.path.dirname(__file__), "..", "dotraining.py"), encoding="utf-8").read()
    return ast.literal_eval(re.search(r"fieldTypesDict\s*=\s*(\{.*?\n\})", src, re.S).group(1))


def raw_ground_truth(raw):
    """Ordered [(key, raw_data-moz-autofill-type)] for annotated fields."""
    out = []
    for m in re.finditer(r"<(?:input|select|textarea)\b[^>]*>", raw, re.I):
        tag = m.group(0)
        t = H.attr(tag, "data-moz-autofill-type")
        if not t:
            continue
        iid = H.attr(tag, "data-moz-autofill-inspect-id")
        key = ("iid", iid) if iid else ("idname", f'{H.attr(tag, "id")}/{H.attr(tag, "name")}')
        out.append((key, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Append to an existing --output, skipping files already "
                         "present (per-file flush means covered files are complete).")
    args = ap.parse_args()
    FT = field_types()

    # Extra Firefox prefs (JSON) merged at launch, e.g. to enable the
    # select-option-range mlData token:
    #   FF_EXTRA_PREFS='{"extensions.formautofill.useml.selectOptionRange": true}'
    extra_prefs = json.loads(os.environ.get("FF_EXTRA_PREFS", "{}"))
    if extra_prefs:
        print(f"extra ml_prefs: {extra_prefs}", flush=True)
    def make_driver():
        return FirefoxDriver(model="", auth_config={},
            firefox_bin=None if not os.environ.get("FIREFOX_BIN") else __import__("pathlib").Path(os.environ["FIREFOX_BIN"]),
            headless=True, log_level="WARNING", ml_prefs=extra_prefs)

    fx = make_driver()

    paths = []
    for d in args.input:
        paths += sorted(glob.glob(os.path.join(d, "*.html")))
    lim = int(os.environ.get("LIMIT", "0"))
    if lim:
        paths = paths[:lim]

    def detect(stripped, wait):
        res = fx.detect_form_fields(html=stripped, use_ml=True, use_fathom=False, script_timeout=600)
        t = 0
        while wait and not any(f.get("mlData") for f in res["fields"]) and t < H.FIRST_USE_RETRIES:
            time.sleep(H.FIRST_USE_WAIT_S)
            res = fx.detect_form_fields(html=stripped, use_ml=True, use_fathom=False, script_timeout=600)
            t += 1
        return res

    # Incremental write: flush after every file so a long run survives a crash.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    covered = set()
    if args.resume and os.path.exists(args.output):
        for ln in open(args.output, encoding="utf-8"):
            if ln.strip():
                covered.add(ln.split(",", 1)[0])
        print(f"resume: {len(covered)} files already covered, skipping them", flush=True)
    fh = open(args.output, "a" if covered else "w", encoding="utf-8")
    nrows, wait, skipped = 0, True, 0
    for n, path in enumerate(paths, 1):
        if os.path.basename(path) in covered:
            continue
        raw = open(path, encoding="utf-8", errors="ignore").read()
        by_key = defaultdict(list)
        for k, t in raw_ground_truth(raw):
            by_key[k].append(t)
        # A single page can hang navigation (300s selenium timeout) or otherwise
        # error; skip it rather than aborting the whole run. It stays uncovered,
        # so a later --resume pass with a fresh browser can retry it.
        try:
            res = detect(H.strip_autocomplete(raw), wait); wait = False
        except Exception as exc:
            skipped += 1
            print(f"  SKIP {os.path.basename(path)}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:100]}", flush=True)
            # A timeout can leave the browser session wedged, which would make
            # every subsequent file burn the full timeout too. Restart it so one
            # bad page can't cascade.
            try:
                fx.close()
            except Exception:
                pass
            fx = make_driver()
            wait = True  # fresh browser needs the first-use ML warm-up wait again
            continue
        seen = defaultdict(int)
        fname = os.path.basename(path)
        for fd in res["fields"]:
            ml = fd.get("mlData")
            if not ml:
                continue
            iid = fd.get("inspectId")
            key = ("iid", iid) if iid else ("idname", fd.get("identifier"))
            lst = by_key.get(key, [])
            i = seen[key]; seen[key] += 1
            t = lst[i] if i < len(lst) else None
            # Normalize to Firefox's supported field set (getAdjustedFieldName /
            # D305753): unsupported types (id-number, vat-number, loginname, ...)
            # and NOMINAL_MAP aliases collapse to the canonical supported type or
            # to "other". Matches how the shipped -supported data was labeled;
            # without this, raw ids 63-66 overflow the model's 66-class head.
            adj = H.adjusted_field_name(t) if t else ""
            if adj and adj in FT:
                name, label = adj, FT[adj]
            else:
                name, label = "--NONE--", FT.get("other", 1)
            fh.write(f"{fname},{name},{label},{ml}\n"); nrows += 1
        fh.flush()
        if n % 100 == 0:
            print(f"  ...{n}/{len(paths)} files, {nrows} rows", flush=True)
    fh.close()
    try:
        fx.close()
    except Exception:
        pass
    print(f"{len(paths)} files -> {nrows} rows -> {args.output} "
          f"({skipped} skipped due to errors)")


if __name__ == "__main__":
    main()
