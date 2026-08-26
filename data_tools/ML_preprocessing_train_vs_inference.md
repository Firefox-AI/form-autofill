# ML Autofill preprocessing: training-data generation vs. inference

**Audience:** anyone touching the autofill ML data pipeline or model.
**Bottom line:** training-data generation and shipped inference use the **same
tokenization**. The only observable difference is a **cosmetic input-type
marker** (`**email` in the training text vs. `email` at inference). It does not
affect the informative tokens, and end-to-end ml_driver tests on real HTML forms
produce high-quality results — i.e. there is **no meaningful divergence** on the
tokens that carry signal.

---

## The shared pipeline (shipped inference)

Source (searchfox, mozilla-central):
- [`FormAutofillHeuristics.sys.mjs`](https://searchfox.org/mozilla-central/source/toolkit/components/formautofill/shared/FormAutofillHeuristics.sys.mjs)
  — builds the per-field `mlData` context string.
- [`FormAutofillML.sys.mjs`](https://searchfox.org/mozilla-central/source/toolkit/components/formautofill/shared/FormAutofillML.sys.mjs)
  — consumes `mlData`; `splitContext` strips the `bb`/`aa` prefixes into the
  three plain strings the triple encoder embeds.

Key definitions ([searchfox](https://searchfox.org/mozilla-central/source/toolkit/components/formautofill/shared/FormAutofillHeuristics.sys.mjs)):
- `WORD_RE = /\s*([\p{L}\p{N}]+)/u` — line 19 (matches **only** letters/digits).
- `ADJACENT_BEFORE_PREFIX = "bb"` (previous field) — line 22.
- `ADJACENT_AFTER_PREFIX  = "aa"` (next field) — line 23.
- Methods (searchfox search — line numbers drift):
  [`tokenizeWords`](https://searchfox.org/mozilla-central/search?q=tokenizeWords&path=FormAutofillHeuristics.sys.mjs),
  [`splitMixedCase`](https://searchfox.org/mozilla-central/search?q=splitMixedCase&path=FormAutofillHeuristics.sys.mjs),
  [`tokenizeAttributes`](https://searchfox.org/mozilla-central/search?q=tokenizeAttributes&path=FormAutofillHeuristics.sys.mjs),
  [`tokenizeElements`](https://searchfox.org/mozilla-central/search?q=tokenizeElements&path=FormAutofillHeuristics.sys.mjs),
  [`_getElementLabelStrings`](https://searchfox.org/mozilla-central/search?q=_getElementLabelStrings&path=FormAutofillHeuristics.sys.mjs).

### Per-field tokens (`tokenizeAttributes`)
For each field, in this order:
1. `splitMixedCase(id)` → `tokenizeWords`
2. `splitMixedCase(name)` → `tokenizeWords`
3. `placeholder` → `tokenizeWords` (no mixed-case split)
4. each label string (`_getElementLabelStrings`: explicit `<label for>`/ancestor
   `<label>`, else nearby text, plus `aria-label`) → `tokenizeWords`
5. input-type marker for non-`text` fields → `tokenizeWords("**" + type)`

`tokenizeWords`: lowercase → keep each `[\p{L}\p{N}]+` run **≥ 3 chars**.
`splitMixedCase`: insert a space between a lower/digit run and an Upper run
(`addressLine` → `address line`).

### Neighbor baking (`tokenizeElements`)
Each field's `mlData` = its own tokens + the **previous** field's tokens each
prefixed `bb` + the **next** field's tokens each prefixed `aa`, whitespace-joined:

```
first name aaidwgzne aaid6ssb3 aasurname
```

For the `triple` format, `splitContext` strips `bb`/`aa` back into three bare
strings (current / previous / next), each encoded independently.

---

## The one difference: the `**type` marker

| | Training `.txt` (generator) | Shipped inference |
|---|---|---|
| email input | `email address **email` | `email address email` |
| `<select>` | `country region **select-one` | `country region select one` |

**Why:** at inference the marker is routed through `tokenizeWords`, whose
`WORD_RE` matches only letters/digits, so it strips the `**` (and splits on the
`-`). The training generator emitted the marker as a **literal token** instead
(the `.txt` files contain `**email` / `**select-one` verbatim).

**Impact: negligible.** The type-carrying word (`email`, `tel`, `select`…) is
present in *both* representations; only the `**` decoration differs. The model
generalizes across it — confirmed by end-to-end ml_driver runs on real forms.
This is not a quality-regression cause. (If we ever want them byte-identical,
either push the literal marker at inference or strip it during generation.)

---

## Still to confirm against D305753 (couldn't read the patch directly)

The Phabricator diff is JS-rendered and `FormAutofillMLTest.sys.mjs` isn't in
mozilla-central, so the generator side above is reconstructed from the shipped
`tokenizeElements` + the observable `.txt` contents. Worth a 2-minute check by
someone with the patch open, in case the generator differs elsewhere:
- Does it include the **`autocomplete`** attribute? (Inference has it
  commented out — see the `// stringText.add(element.autocompleteInfo…)` line in
  `tokenizeAttributes`.)
- Same label extraction and field selection?
- Any field-count filter? (We now want **any form with ≥3 fields**, matching
  the live rule that ML is only invoked at 3+ fields — no training-specific
  filter.)

If the generator is just `tokenizeElements` with the literal marker, then the
`**` marker is the *entire* difference.

---

## Python port

`data_tools/ff_preprocess.py` is a 1:1 port of the above (`tokenizeWords`,
`splitMixedCase`, `tokenizeAttributes`, `tokenizeElements`, label extraction,
field selection). It has a `type_marker` flag: `"inference"` (default, strips
`**`, matches the deployed model's actual input) or `"training"` (literal
marker, matches the `.txt`). Feed the output through `dotraining.split_context`
for the triple format.
