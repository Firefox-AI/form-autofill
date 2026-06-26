# data_tools — autofill training-data generation & extraction

Tools for building labeled HTML form samples for the Firefox autofill
field-detection model. They produce HTML files in the same inline-label format
as the hand-labeled `samples/` corpus (each autofillable field carries a
`data-moz-autofill-type="<token>"` attribute), so the output flows through the
existing HTML→txt processing and `dotraining.py` unchanged.

Three sources of data:
1. **Synthetic** — LLM-generated forms (`generate_samples.py`).
2. **Common Crawl, labeled** — real scraped forms classified by an LLM (`extract_forms.py` → `label_common_crawl.py`).
3. **Common Crawl, non-address (unlabeled)** — real forms with no address fields, for other contexts (`extract_non_address.py`).

Plus an audit tool (`evaluate_labels.py`) that scores label accuracy against the autofill spec.

---

## Layout

```
data_tools/
├── generate_samples.py      # synthetic form generation (OpenAI)
├── extract_forms.py         # WARC → CSV of candidate forms (address/payment)
├── label_common_crawl.py    # CSV of real forms → labeled HTML (OpenAI)
├── extract_non_address.py   # WARC → unlabeled non-address HTML files
├── evaluate_labels.py       # audit label accuracy vs the spec (OpenAI judge)
└── gen/                     # shared package
    ├── validate.py          # taxonomy loader + HTML/label validation gate
    ├── render.py            # FormSpec → HTML (markup styles, realistic field names)
    ├── params.py            # sampling: locales, purposes, field-set templates, address scheme
    ├── llm.py               # OpenAI client + all prompts/schemas (generate, label, verify, audit)
    ├── htmlforms.py         # parse real forms: field extraction, label injection, cleaning
    ├── ccextract.py         # WARC iteration, language detect, keyword/address filtering
    ├── dedup.py             # signature-based near-duplicate detection
    ├── autofill_spec.txt    # the MDN/WHATWG autocomplete spec (used by the audit)
    └── test_offline.py      # offline tests (no API)
```

---

## Setup

**1. Python dependencies** (not part of the parent repo's training deps):

```bash
pip install "openai>=1.40" "beautifulsoup4>=4.12" langdetect warcio tenacity pandas python-dotenv
```

**2. OpenAI key** — create `data_tools/.env`:

```
OPENAI_API_KEY=sk-...
# optional: OPENAI_MODEL=gpt-4o-mini
```

**3. Taxonomy source** — the 66 field types are read (via `ast`, not imported) from
`dotraining.py`. The loader (`gen/validate.py:_find_dotraining`) searches
`data_tools/` and its parent, so the repo-root `dotraining.py` is found automatically.

**4. Sample directories** — the generators read/write under `data_tools/samples/`:
- `samples/training/` — used as **mutation anchors** by `generate_samples.py` and for **dedup** (real forms to avoid duplicating). Point to (or symlink) the HTML sample corpus.
- `samples/testing/`, `samples/validation/` — used by `evaluate_labels.py` as ground truth.
- Output lands in `samples/generated/`, `samples/common_crawl/`, `samples/non_address/`.

All tools run as modules from the `data_tools/` directory, e.g. `python generate_samples.py …`.

---

## Tools

### 1. `generate_samples.py` — synthetic forms

LLM produces only structured field metadata (localized label text + a field type
constrained to the 66-token enum); **code renders the HTML and owns the
`data-moz-autofill-type` attribute**, so labels can't be hallucinated. Two modes
are mixed: *hybrid* (build a planned field set, ask for localized labels) and
*mutation* (rewrite a real anchor form into a new locale, carrying its verified
labels forward). A gpt-4o verification pass strips any field whose label↔type
looks wrong.

```bash
python generate_samples.py --count 50 --dry-run     # plan + cost estimate, no API
python generate_samples.py --count 6000              # generate (~3,800 survive filters)
python generate_samples.py --count 6000 --resume     # crash-safe resume via manifest
python generate_samples.py --review 20               # print label tables for accepted files
```

Key flags: `--count` (attempts), `--out` (default `samples/generated`), `--model`
(default `gpt-4o-mini`), `--verify-model` (default `gpt-4o`), `--no-verify`,
`--verify-sample` (fraction of non-rare forms verified), `--mutate-ratio`
(default 0.30), `--concurrency` (default 8), `--seed`, `--max-cost`, `--resume`.
Output: `GEN_<locale>_<purpose>_<domain>_<index>.html` + `manifest.jsonl`/`rejects.jsonl`.

### 2. `extract_forms.py` — WARC → candidate CSV

Streams Common Crawl `.warc.gz` segments and writes a CSV of candidate forms
(one row per `<form>`, columns: `url, tld, lang, title, form_index, num_fields,
form`). Filters by language (langdetect), field count, an address/payment
**keyword** gate, and cleaned-HTML size. No LLM.

```bash
python extract_forms.py \
  --warc-glob "/path/CC-MAIN-*.warc.gz" \
  --languages en de nl it pt es ja --per-page-limit 2 --max-forms 4000 \
  --out /path/cc_forms.csv
```

Key flags: `--warc`/`--warc-glob`, `--languages`, `--max-forms`, `--min-fields`/`--max-fields`,
`--require-any` (keyword filter; defaults to the multilingual address+credit-card
set — pass empty to disable), `--tlds`, `--per-page-limit`, `--max-html-chars`,
`--allow-single-input` + `--single-input-fraction` (allow a capped share of 1-field forms).

### 3. `label_common_crawl.py` — CSV → labeled HTML

Reads the `form` column of a CSV (from `extract_forms.py`), filters by field
count, has the LLM **classify** each field into the taxonomy (or `__skip__`),
injects `data-moz-autofill-type`, runs the gpt-4o verification pass, then writes
cleaned wrapped HTML to `samples/common_crawl/`.

```bash
python label_common_crawl.py --csv /path/cc_forms.csv --analyze        # field-count histogram, no API
python label_common_crawl.py --csv /path/cc_forms.csv --min-fields 1   # label + write
python label_common_crawl.py --review 10                               # inspect labels
```

Key flags: `--csv`, `--out` (default `samples/common_crawl`), `--model`,
`--verify-model`, `--no-verify`, `--min-fields`/`--max-fields`, `--min-labeled`
(single-input forms need 1, others need this many), `--concurrency`, `--limit`
(first N rows), `--analyze`, `--review`. Output: `CC_<domain>_<hash>.html`.

### 4. `extract_non_address.py` — non-address forms (unlabeled)

Like `extract_forms.py` but keeps only forms with **no address-type field**
(detected heuristically per field — `Email Address` is exempt so login/newsletter
forms survive) and writes cleaned, **unlabeled** wrapped HTML. No LLM.

```bash
python extract_non_address.py \
  --warc-glob "/path/CC-MAIN-*.warc.gz" --count 400
```

Key flags: `--warc`/`--warc-glob`, `--count`, `--languages` (default `en`),
`--min-fields`/`--max-fields` (default 2–25), `--per-page-limit`, `--out`
(default `samples/non_address`). Output: `NA_<domain>_<hash>.html`.

### 5. `evaluate_labels.py` — label-accuracy audit

Samples forms from `samples/generated` and `samples/common_crawl`, sends each
form's labels plus the **full autofill spec** to gpt-4o, and reports per-directory
accuracy and the most common mislabels. Read-only — never modifies data.

```bash
python evaluate_labels.py --n 100                              # 100 from each dir
python evaluate_labels.py --n 120 --require-type address-line1 \
       --dump /tmp/verdicts.jsonl                              # focus one field type
```

Key flags: `--n` (forms per dir), `--model` (default `gpt-4o`), `--concurrency`,
`--seed`, `--require-type` (only audit forms containing this token), `--dump`
(write per-field verdicts to JSONL).

---

## End-to-end workflow

```
# Synthetic
python generate_samples.py --count 6000            # → samples/generated/

# Common Crawl (labeled)
python extract_forms.py --warc-glob "…CC-MAIN-*.warc.gz" --languages en de … --out cc.csv
python label_common_crawl.py --csv cc.csv --min-fields 1   # → samples/common_crawl/

# Common Crawl (non-address, unlabeled)
python extract_non_address.py --warc-glob "…CC-MAIN-*.warc.gz" --count 400  # → samples/non_address/

# (optional) audit label quality
python evaluate_labels.py --n 100

# then: run the project's HTML→txt processing over samples/** and train with dotraining.py
```

---

## Conventions & design notes

- **Labels are correct by construction (synthetic):** the field type comes from
  code/templates and is enum-constrained; the LLM only writes label text.
- **Address-line vs street-address:** matches the hand-labeled corpus — a primary
  address field is `address-line1` only when a second line (`address-line2`/
  `apartment`/`floor`/…) is present, otherwise `street-address`. Enforced
  structurally in both `gen/params.py` (`_apply_address_scheme`) and
  `label_common_crawl.py`, so it doesn't depend on the LLM complying.
- **Field names are realistic, not type codes:** `gen/render.py:NAME_HINTS` maps
  types to real-world `name`/`id` values (e.g. `address-level1` → `state`/
  `province`/`county`). It must **never** embed the type code itself — doing so
  leaks the label into the feature tokens (trivially predictive in training,
  absent from real forms) and hurts generalization.
- **Dedup:** `gen/dedup.py` rejects near-duplicate forms by structural signature,
  including against the existing `samples/` corpus (prevents leaking near-copies
  of `testing/` into training).
- **Verification pass:** generation and labeling both run a gpt-4o check that
  strips implausible label↔type pairs (using the form's detected language).

## Testing

```bash
python -m gen.test_offline      # offline checks (taxonomy, validation gate, render, dedup) — no API
```
