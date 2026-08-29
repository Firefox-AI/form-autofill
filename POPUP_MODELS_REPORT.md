# Autofill ML — popup-data models report

Covers every model trained since the new **popup (select-option-range) data
format** was introduced, the best bb + multi-head models, their **quantized**
(per-channel rescaled q8) accuracy, per-field credit-card accuracy, and example
training rows.

## 1. The popup data format

For a `<select>` field, its option domain is baked into `mlData` as a single
`first...last` token (visible option text, 16 chars/side, whitespace-collapsed),
gated behind `extensions.formautofill.useml.selectOptionRange`. Example rows
(from `DE_B378b.html`, a German checkout form — the token follows `**select-one`):

```
cc-type,61,      …kartentyp **select-one visa...mastercard aa…
cc-exp-month,58, …monat **select-one 01...12 bb…**number aa…**select-one aa2020...2039
cc-exp-year,59,  …jahr  **select-one 2020...2039 bb…**select-one bb01...12 aa…
country,29,      …land  **select-one ...zypern bb…            (…unitedstates...zimbabwe on another form)
```

## 2. All models trained (popup era)

4 architectures × 4 dataset variants. **val** = validation accuracy (large,
reliable support); **test** = real-site QA set (thin CC support, noisy).

| architecture | v1 (no-popup) | v2 (**popup**) | v3base (+non-addr) | v3 (**popup**+non-addr) |
|---|---|---|---|---|
| bb (TinyBERT) | .896 / .872 | .903 / .878 | .894 / .880 | .909 / .871 |
| triple · pruned-v2 | .910 / .901 | .914 / .900 | .910 / .902 | .916 / .901 |
| triple · unpruned (250k) | .911 / .899 | .913 / .899 | .911 / .912 | .914 / .908 |
| triple · **reprune-v3** | .909 / .906 | **.916** / .900 | .909 / .907 | .915 / .905 |

*(val / test accuracy. reprune-v3 = MiniLM re-pruned on our corpus, vocab 15,701,
keeps the popup tokens in-vocab; pruned-v2 UNKs them.)*

## 3. Best models — popup data, no non-address (v2)

| | run | base model | val acc | test acc | f1 |
|---|---|---|---|---|---|
| **bb** | 4sjmf | huawei-noah/TinyBERT_General_4L_312D | 0.9028 | 0.8777 | 0.874 |
| **multi-head** | 2mrjb | rolf-mozilla/minilm-pruned-notest-v3 (triple, 4 layers) | **0.9158** | 0.900 | 0.897 |

## 4. Quantization (q8, per-channel rescale + reduce_range)

Rescaling = per-channel weight quantization (each output channel its own scale);
`--reduce_range` required with it (Bug 2064781). int8 weights verified within ±64.

| model | fp32 val acc | q8 val acc | Δ |
|---|---|---|---|
| bb (4sjmf) | 0.9028 | 0.8964 | −0.64 pt |
| multi-head (2mrjb) | 0.9158 | 0.9167 | +0.09 pt (lossless) |

## 5. Per-field credit-card accuracy — fp32 → q8 (validation)

f1, with validation support n. Quantization is effectively lossless per field
(all Δ within ±0.01 = noise):

| CC field | n | bb fp32→q8 | multi-head fp32→q8 |
|---|---|---|---|
| cc-number | 68 | .94 → .94 | .97 → .97 |
| cc-name | 105 | .94 → .93 | .94 → .94 |
| cc-exp-month ◧ | 64 | .98 → .98 | .98 → .98 |
| cc-exp-year ◧ | 63 | .98 → .97 | .98 → .98 |
| cc-csc | 62 | .98 → .98 | .97 → .97 |
| cc-type ◧ | 49 | 1.00 → 1.00 | .98 → .98 |

◧ = select field (carries the popup range token).

## 6. Key findings

- **Quantization is safe**: q8 with per-channel rescaling costs the multi-head
  model ~0 and the bb model ~0.6 pt overall; per-field CC accuracy is unchanged.
  The multi-head (reprune-v3) at **0.917 quantized** is the strongest option.
- **The multi-head model beats bb** by ~1.3 pt overall and is quantization-robust.
- **Popup data helps in-distribution** (validation, +0.3–1.5 pt) but the gain does
  not clearly transfer to the small real-site test set (see prior A/B analysis).
- Per-field CC accuracy is high across the board (.93–1.00 f1) and preserved
  through quantization.

### Artifacts
- Quantized bb: `onnx/quantization/autofill-tiny-supported-argo-autofillflow-4sjmf/onnx/model_quantized.onnx`
- Quantized multi-head: `onnx/quantization/autofill-tiny-supported-argo-autofillflow-2mrjb/{encoder/onnx/model_quantized.onnx, head/}`
- Datasets: `datasets/v1_base` .. `datasets/v3_optrange_nonaddr`
