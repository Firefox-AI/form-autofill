# Training run registry

All Argo runs are `AutofillFlow/argo-autofillflow-<id>`. Common defaults unless noted:
base `rolf-mozilla/minilm-pruned-notest-v2`, `-supported` data, 15 epochs, batch 8,
weight_decay 0, warmup 0. `subsample_validation` = gen/cc ratio also applied to validation.
Metrics are `acc/close` (accuracy / closeAccuracy); val = subsampled validation, test = data/testing (real QA sites).

## 1. GEN-cut sweep (triple, eps0.1, subsample_validation, -supported)
Purpose: does cutting synthetic GEN help, and does subsampled-validation track test?

| id | GEN ratio | val | test |
|----|-----------|-----|------|
| l2sn9 | 2.0 (~40% cut) | 0.9315/0.9525 | **0.9042/0.9249** |
| fxqdj | 1.0 (~70%) | 0.9315/0.9501 | 0.8989/0.9178 |
| bqt9z | 1.0 seed2 | 0.9270/0.9469 | 0.9029/0.9225 |
| pfnjm | 0.5 (~85%) | 0.9118/0.9360 | 0.9013/0.9196 |
| m27bh | 0.5 seed2 | 0.9110/0.9355 | 0.8972/0.9178 |
| 8gg6m | 0.5 **NO-EPS** | 0.9131/0.9331 | **0.9070/0.9253** |
| z6d8m | 0.5 NO-EPS seed2 | 0.9108/0.9312 | 0.9042/0.9210 |

Finding: GEN cut safe-to-positive; **eps HURTS test** (no-eps > eps at g0.5); g2.0 best eps run.

## 2. Attention arch (g0.5, eps0.1, -w3, proj64 learned w3)
| id | layers | val | test |
|----|--------|-----|------|
| m9mwc | 12L | 0.8356/0.8772 | 0.8550/0.8889 |
| 4nq48 | 4L | 0.8074/0.8488 | 0.8224/0.8717 |

Finding: attention collapses under heavy GEN cut (data-hungry random head). Dead end here.

## 3. Token A/B — input-attrs + option-range (triple, no-eps, g2.0, subsample_val)
`-iaor` = new tokens (**maxlen/**inputmode + option-range, blank-skipped); `-iaorctl` = same fields, tokens stripped.
Base config = WD0/15ep/b8.

| id | data | val | test |
|----|------|-----|------|
| m9z8m | iaor s1 | 0.9114/0.9303 | 0.8989/0.9195 |
| x5d68 | iaor s2 | 0.9141/0.9330 | 0.8939/0.9136 |
| xx25b | ctl s1 | 0.9105/0.9286 | 0.9050/0.9214 |
| h8vdz | ctl s2 | 0.9066/0.9262 | 0.9025/0.9206 |

Finding: base config — tokens help val (+0.4pt) but hurt test (−0.7pt) = overfitting. Per-field (val): tel +2.2pt, tel-country-code +10pt, cc-exp-year +3.6pt.

## 4. Regularization sweep (triple, iaor, no-eps, g2.0, v2 base) — fix overfitting
| id | config | val | test |
|----|--------|-----|------|
| 99xjd | wd0.01 | 0.9163/0.9349 | 0.9023/0.9197 |
| rp9lq | wd0.05 | 0.9127/0.9338 | 0.9071/0.9233 |
| hkzxp | wd0.1 | 0.9143/0.9336 | 0.9054/0.9235 |
| **244cb** | **6ep-b32-wd0.05** | **0.9262/0.9435** | **0.9073/0.9256** |
| 5bhlj | 6ep-b64-wd0.05 | 0.9169/0.9365 | 0.9035/0.9227 |
| t9v9h | 6ep-b64-wd0.05 seed2 | 0.9174/0.9374 | 0.9058/0.9248 |

Finding: regularization is a clear win (WD was 0 by default). **244cb (6ep + batch32 + wd0.05) = best run of the session** (val 0.9262/0.9435, test 0.9073/0.9256). batch32 > batch64. Validates the overfitting hypothesis + "smaller GEN → change sizes".

## 5. Regularized control (triple, -iaorctl, 6ep-b64-wd0.05, v2 base)
| id | data | val | test |
|----|------|-----|------|
| mlrpn | ctl s1 | 0.9053/0.9257 | 0.9031/0.9229 |
| 45n9f | ctl s2 | 0.9101/0.9305 | 0.9035/0.9218 |

## 6. Monolithic v1 (context_format bb, no-eps, g2.0, v2 base)
| id | config | val | test |
|----|--------|-----|------|
| 4lrpj | iaor base WD0 | 0.9040/0.9255 | 0.8897/0.9174 |
| frtwx | iaor wd0.05 | 0.9004/0.9231 | 0.8817/0.9111 |
| xcpbn | iaor 6ep-b32-wd0.05 | 0.9125/0.9360 | 0.8926/0.9159 |
| c6s8d | ctl wd0.05 | 0.8829/0.9092 | 0.8842/0.9117 |

Finding: monolithic v1 loses to triple by ~1.5pt test (best bb xcpbn 0.8926 vs best triple 0.9073). Not competitive.

## 7. Re-pruned base A/B (triple, 6ep-b64-wd0.05, base `rolf-mozilla/minilm-pruned-notest-iaor`)
Base re-pruned on -iaor train+val (16007 vocab vs v2's 12455). Isolates the new tokens on a vocab that actually covers them.

| id | data | val | test |
|----|------|-----|------|
| mf4jn | iaor s1 | 0.9204/0.9415 | 0.9048/0.9233 |
| nhdx7 | iaor s2 | 0.9154/0.9374 | 0.9037/0.9227 |
| pn2kk | ctl s1 | 0.9127/0.9338 | 0.9056/0.9245 |
| 5wg5j | ctl s2 | 0.9116/0.9303 | 0.9027/0.9218 |

Finding (2v2 means): iaor val 0.9179 vs ctl 0.9122 (**+0.57pt val**); test iaor 0.9043 vs ctl 0.9042 (**dead tie**). Re-pruning closed the test gap the tokens took on v2. Per-field (val): tel-country-code +7.7pt, cc-exp-year +4pt, bday-month/day/year +14–37pt.

## Best config + untested follow-up
Session best = **244cb** (triple, iaor, 6ep + batch32 + wd0.05, v2 base). The re-pruned iaor base helped iaor but those runs used batch64. **Untested winning combo: re-pruned iaor base + 6ep-b32-wd0.05.**

## 8. BEST combo: re-pruned iaor base + 6ep-b32-wd0.05 (triple, no-eps, g2.0)
Stacks the two wins: re-pruned vocab (batch 7) + best regularization (batch 4, batch32).

| id | data | val | test |
|----|------|-----|------|
| wtjzr | iaor s1 | 0.9176/0.9376 | **0.9090/0.9266** |
| 42fx8 | iaor s2 | 0.9189/0.9389 | 0.9037/0.9243 |
| 469wv | ctl s1 | 0.9163/0.9358 | **0.9100/0.9273** |
| 8qvvf | ctl s2 | 0.9123/0.9327 | 0.9050/0.9252 |

Finding (2v2 means): iaor val 0.9183 vs ctl 0.9143 (**+0.4pt val**); test iaor 0.9064 vs ctl 0.9075 (~tie). batch32 edged batch64 on test (0.9064 vs 0.9043). **469wv (control) = highest test of session, 0.9100/0.9273.** Tokens: consistent +val, test-neutral.

## 9. Shippable 4-layer retrains (both arches, iaor base, 6ep-b32-wd0.05, --encoder_layers 4)
Batches 1-8 all used the DEFAULT encoder_layers=0 (all 12 layers) → wtjzr int8 = 27MB, NOT shippable
(shipped target is 4-layer). These are the shippable-size retrains of the winning recipe.

| id | arch | data | val | test |
|----|------|------|-----|------|
| **v9pdg** | triple | iaor s1 | 0.9088/0.9290 | 0.9012/0.9227 |
| xsvrk | triple | iaor s2 | 0.9090/0.9308 | 0.9010/0.9220 |
| v4kkl | triple | ctl s1 | 0.8980/0.9200 | 0.9040/0.9241 |
| 7hsf4 | triple | ctl s2 | 0.9046/0.9257 | 0.9029/0.9222 |
| **9fkp9** | bb | iaor s1 | 0.9022/0.9244 | 0.8857/0.9107 |
| 654jl | bb | iaor s2 | 0.8980/0.9207 | 0.8869/0.9094 |

Finding (4L, 2-seed means): triple iaor val 0.9089 / test 0.9011; triple ctl val 0.9013 / test 0.9035 (tokens +0.76pt val, ~test-neutral). bb iaor val 0.9001 / test 0.8863 → **triple beats bb by ~1.5pt test at 4L too**. 4L ~0.5pt behind 12L on test (0.9011 vs 0.9064). Ship candidates: triple **v9pdg**, bb **9fkp9**.

Poller: eval_4l.py (/tmp/eval_4l.json). 12L wtjzr int8 (27MB) kept as reference.

## Quantization
- Triple wtjzr (12L, iaor base): int8 encoder 27MB + fp32 head 1.8MB (onnx/quantization/...-wtjzr/).
  Per-channel + reduce_range; int8 range verified (146 tensors within +-64, Bug 2064781). Toolchain:
  /tmp/transformers.js @3.8.1, venv rebuilt (transformers 4.49, optimum, onnx 1.17).
- 4-layer ship candidates (per-channel int8, int8-range verified, eval on testing-iaor):
  - triple v9pdg: fp32 0.9012 -> int8 0.9021/0.9237 (+0.08pt); size 13.8MB encoder + 1.8MB fp32 head ~= 15.6MB.
  - bb 9fkp9: fp32 0.8857 -> int8 0.8846 (-0.11pt); size 13.8MB single model.
  - Both lossless under int8. Triple beats bb by ~1.7pt at equal shippable size. Both dominated by the
    16007-vocab embedding (~1.4MB bigger than shipped v2's 12455). bb int8 eval: /tmp/bb_int8_eval.py
    (model needs token_type_ids=zeros); triple: onnx/eval_triple_quantized.py --variants model.onnx model_int8.onnx.

## 10. mlData cleanup A/B (triple, iaor base, 4L, 6ep-b32-wd0.05)
Post-process cleanups of -iaor (no FF regen; data_tools/build_iaor_variants.py). Baseline = wtjzr/42fx8.
- -iaornocommon: drop tokens common to ALL fields in 3+ forms (kills ctl00/form1 boilerplate; **/... protected).
  Cut 10% train tokens (1568/4993 forms), 1.7% test.
- -iaordedup: dedup repeated words within each field (name name -> name). Cut 18.6% train, 32.8% test tokens.

- -iaorboth: common-strip AND dedup. Cut 27.5% val / 33.8% test tokens.

| id | variant | val | test |
|----|---------|-----|------|
| 6sjfs / m5klt | nocommon | pending | pending |
| d2b5c / 42vxq | dedup | pending | pending |
| ct9f5 / khlsj | both | 0.912/— | 0.900/— |

RESULT (2-seed mean overall val/test): baseline 0.918/0.906; nocommon 0.912/0.905; dedup 0.907/0.899;
both 0.912/0.900. **NEGATIVE — all cleanups mildly hurt; dedup worst.** tel-country-code test regresses
monotonically (0.880→0.840→0.800→0.770 as more is stripped). Interpretation: for a mean-pooled encoder the
id==name repetition usefully up-weights the strongest tokens; deduping rebalances toward weaker/noisier
tokens (labels, spurious nearby-text). Common tokens also load-bearing. DON'T SHIP — keep raw tokenization.
Only benefit was ~30% shorter sequences, not worth −0.5 to −0.7pt test.

### 10b. Same cleanups on bb (monolithic) — ARCHITECTURE-DEPENDENT
bb runs (iaor base, 4L, same recipe): baseline 9fkp9/654jl; nocommon kknm9/r5kst; dedup tln42/v6zdx; both g6s72/zjwp9.
Overall val/test (2-seed): baseline 0.900/0.886; nocommon 0.905/0.887; dedup 0.897/0.883; both 0.897/0.877.
Verdict FLIPS for bb: nocommon is slightly POSITIVE (+0.5/+0.1, lifts tel-country-code/cc-exp on test); dedup
mildly negative but far less than triple (−0.3 vs −1.1 val). Mechanism confirmed: bb's full self-attention
doesn't lean on id/name repetition the way the mean-pooled triple does, so common-strip is fine/helpful for bb;
dedup's residual harm (rebalancing toward noisier label/nearby-text tokens) is shared. bb still trails triple
absolutely (test 0.886 vs 0.906), so no change to the ship recommendation.

## 11. source_tags A/B (triple + bb, 4L, 6ep-b32-wd0.05)
FF feature `source_tags` (firefox-clone branch ml-source-tags): tag own id/name tokens with "attr" so the
model tells machine attrs from user-facing text; neighbors (aa/bb) untagged. Learnable alternative to the
failed dedup/nocommon cleanups. Needs FF regen (provenance lost in -iaor): data_tools/build_iaortags.sh ->
-iaortags (41050/4550/4758, matches -iaor). Re-pruned base rolf-mozilla/minilm-pruned-notest-iaortags
(16069 vocab, +62 vs -iaor). Baseline = -iaor on base-iaor (wtjzr/42fx8 triple, 9fkp9/654jl bb).

| id | arch | val | test |
|----|------|-----|------|
| vw7dj / qflmp | triple tags | pending | pending |
| bmh7x / fnrqq | bb tags | pending | pending |

Poller: eval_tags.py. KEY question: does tagging recover what dedup lost (email/name/cc-exp) by reweighting
machine tokens instead of removing them? Watch overall + macro(>=15) + those fields.

## 12. strip_frequent A/B (triple + bb, 4L, 6ep-b32-wd0.05, base -iaor)
FF feature strip_frequent (firefox-clone branch ml-strip-common): drop tokens in >=85% of a form's fields
(3+ fields); broader than strip_common(=nocommon, 100%). Also added strip_common to FF as real code.
Dataset -iaorfreq via data_tools/build_iaorfreq.py (post-process of -iaor, no regen). Cut 12.3% train /
13.3% val / 6.9% test tokens (vs nocommon 10%/-/1.7% -> ~4x more on test). Base -iaor (strip only removes
tokens). Baseline = -iaor (wtjzr/42fx8 triple, 9fkp9/654jl bb); strip_common already tested (=nocommon:
bb +0.4 overall/+0.9 macro test, triple negative).

| id | arch | val | test |
|----|------|-----|------|
| 7df89 / v82m9 | triple freq | 0.9104 | 0.9047 |
| lgcdg / bzcdv | bb freq | 0.9039 | 0.8852 |

RESULT vs 4L baseline (v9pdg/xsvrk triple 0.9089/0.9011; 9fkp9/654jl bb 0.9001/0.8863):
triple freq +0.15 val / +0.36 test; bb freq +0.38 val / -0.11 test. Broader stripping ~= strip_common on
triple, slightly worse than strip_common on bb test. strip_common is the better of the two.

### CONFOUND CORRECTION to batches 10/10b (IMPORTANT)
Batch 10's "cleanups HURT triple" was WRONG -- it compared 4-layer cleanup runs against a 12-LAYER baseline
(wtjzr/42fx8, encoder_layers=0). Re-checked at equal 4L (baseline v9pdg/xsvrk = 0.9089/0.9011):
- strip_common (6sjfs/m5klt): 0.9117/0.9050 -> **+0.28 val / +0.39 test** (POSITIVE, both arches: bb +0.44/+0.09).
- strip_frequent: +0.15/+0.36 triple; +0.38/-0.11 bb.
- dedup (d2b5c/42vxq): 0.9070/0.8987 -> -0.19/-0.24 (mildly negative even at 4L; the -1.1 was the layer confound).
So strip_common is a real win for BOTH arches (was mis-labeled bb-only). dedup mildly negative. source_tags (§) neutral-to-negative.

## 13. close_label_eps on strip_common winners — NEGATIVE (eps doesn't help closeAccuracy)
strip_common + eps0.1 (triple d4nhl/x4v5b, bb 6hmn6/fbz89), 4L 6ep-b32-wd0.05, base -iaor, -iaornocommon.
vs no-eps winners (6sjfs triple, r5kst bb). acc/close 2-seed(eps) vs 1-seed(no-eps):
- triple: no-eps 0.9121/0.9321 val, 0.9048/0.9241 test | eps 0.9093/0.9321 val, 0.9033/0.9236 test
  -> closeAccuracy DEAD TIE (val 0.9321=0.9321), strict acc -0.15/-0.28. eps buys nothing on close.
- bb: no-eps 0.9055/0.9275 val, 0.8880/0.9121 test | eps 0.8998/0.9262 val, 0.8857/0.9115 test
  -> eps DOWN on every metric.
Third confirmation eps's closeAccuracy gain doesn't generalize. Ship models keep close_label_eps=0.

## PUBLISHED MODELS (rolf-mozilla, int8, model cards w/ run ids)
All 4L, base minilm-pruned-notest-iaor, 6ep-b32-wd0.05, gen2.0, subsample_val, eps0, per-channel int8.
int8 ~= fp32 for all (verified on each model's test set).
- autofill-triple-iaor-int8       <- v9pdg  (-iaor):        test acc 0.9012 fp / 0.9021 int8; 13.8MB enc + 1.8MB head
- autofill-bb-iaor-int8           <- 9fkp9  (-iaor):        test acc 0.8857 fp / 0.8846 int8; 13.8MB
- autofill-triple-stripcommon-int8<- 6sjfs  (-iaornocommon): test acc 0.9048 fp / 0.9037 int8; 13.8MB + 1.8MB  <- BEST triple
- autofill-bb-stripcommon-int8    <- r5kst  (-iaornocommon): test acc 0.8880 fp / 0.8865 int8; 13.8MB          <- BEST bb
Ship pick: triple strip_common (6sjfs). FF code: firefox-clone branch ml-strip-common (strip_common/strip_frequent).

## 14. RELABEL dataset (o4-mini name/label corrections + new 2026 multilingual CC batch)
Dataset `-relabel` (training 54009 / val 4550 / test 4758). Built via build_relabel.sh (ml_driver,
FF features select_option+input_attributes, single-neighbor). Corpus = data/{training,generated,common_crawl}
with common_crawl now = old 1686 + 2412 new audited CC-MAIN-2026-34 forms (en/de/es/ja/it/nl/pt) merged in.
o4-mini audit applied 366 field corrections (cc 37, gen 131, 2026 batch 198); lone-component→name prompt rule
([[relabel-audit-o4mini]]). name-class rebalanced: full `name` 1459→2551 (+75%), name:given 0.49→0.62.
Repruned base rolf-mozilla/minilm-pruned-notest-relabel (vocab 18473, +JP vs -iaor 16069). Val/test frozen for A/B vs -iaor.

| run | arch | base | args |
|-----|------|------|------|
| **b67p7** | bb | huawei-noah/TinyBERT_General_4L_312D (native vocab, NO custom tokenizer) | bb, 4L native, 8ep-b32, lr7e-5, wd0.05, g2.0, subsample_val |
| **d5tsg** | triple | minilm-pruned-notest-relabel | triple, 4L, 6ep-b32-wd0.05, head_interactions, lr7e-5, g2.0, subsample_val |

Both triggered 2026-09-07, namespace production:autofillflow-0-egrc. Baseline to beat: -iaor v9pdg (triple 0.9012 test) / 9fkp9 (bb). Watch: `name` recall on fixed test vs -iaor.

Results (b67p7 bb-tinybert, done): test total 0.871 / close 0.895 (EN 0.922, nonEN 0.860); name P0.735 R0.725 (n69).
Validation per-label (eval_val_perlabel.py, rehydrates run config): b67p7 val total 0.895/close 0.919, name P0.787 **R0.867** (n166); vs OLD -iaor val name: bb 9fkp9 R0.783, triple v9pdg R0.898. bb-tinybert name recall +8.4pt over old bb.

### 14b. strip_common variant (-relabelnocommon, reprocess only — no FF rebuild)
build_iaor_variants.py now takes a base-variant arg; `-relabel` → training/validation/testing-relabelnocommon.txt
(11.4%/10.1%/1.7% token cut; drops per-form tokens common to EVERY field's own section, structural **/... protected).
Pruned base reused (nocommon is a token-subset → superset vocab). Triggered 2026-09-07:

| run | arch | base | args |
|-----|------|------|------|
| **vb24c** | bb | TinyBERT_General_4L_312D | bb, native, 8ep-b32, -relabelnocommon |
| **74nzg** | bb | minilm-pruned-notest-relabel | bb, 4L, 6ep-b32-wd0.05, -relabelnocommon |
| **qq6s7** | triple | minilm-pruned-notest-relabel | triple, 4L, 6ep-b32-wd0.05, head_interactions, -relabelnocommon |

A/B vs the plain -relabel runs. NOTE: minilm base must be PUBLIC (workers have no HF token); first 4 minilm runs
(d5tsg/ss8d4/74nzg/qq6s7) died on 401 (private repo) — re-triggered after flipping public. Final run-ids below.

### 14c. FINAL results — all 6 -relabel runs (test n=69 name / val n=166 name)
| model | run | test tot/close | val tot/close | val name P/R | given/family R |
|-------|-----|----------------|---------------|--------------|----------------|
| bb-tiny plain        | b67p7 | 0.871/0.895 | 0.895/0.919 | 0.787/0.867 | 0.955/0.974 |
| bb-tiny strip_common | vb24c | 0.880/0.905 | 0.897/0.921 | 0.775/**0.934** | — |
| bb-intl plain        | 8wpwd | 0.887/0.912 | 0.904/0.927 | 0.754/**0.940** | 0.946/0.971 |
| bb-intl strip_common | dqd2c | 0.884/0.911 | 0.907/0.933 | 0.788/0.916 | 0.958/0.977 |
| triple plain         | gj8ch | 0.897/0.918 | 0.912/0.935 | 0.796/0.892 | 0.974/0.980 |
| triple strip_common  | m6rf9 | **0.900**/0.919 | 0.911/0.933 | 0.822/0.861 | 0.974/0.987 |

Baselines: triple -iaor v9pdg val name R0.898; bb -iaor 9fkp9 val name R0.783.
KEY: relabel lifted val name recall +13-16pt over old bb (0.783 -> 0.86-0.94); bb-intl plain best recall 0.940,
now exceeds old triple. strip_common: helps bb-tiny name (0.867->0.934); trades recall->precision on bb-intl/triple.
Best overall acc = triple strip_common m6rf9 (0.900). eval_val_perlabel.py rehydrates run config for val per-label.

### 14d. Attention retry on +30% data (Path A: window_size=1 proxy, no FF wide rebuild)
Batch 2 attention collapsed from data starvation (g0.5). Retry with relabel (+30% real, g2.0). Attention needs
`~<offset>` wide tokens (parse_wide_context); -relabel is aa/bb, so build_wide1.py re-encodes aa/bb -> ~-1/~1 =
same ±1 info as triple, no Firefox rebuild (-relabelw1 datasets). This tests attention FUSION vs triple add/subtract
on identical ±1 neighbors. (Full wide ±3 = Path B, needs restoring the removed wide-context FF code + rebuild.)

| run | config |
|-----|--------|
| **c6c7k** | attention, window_size 1, num_heads 1, proj0, 4L, 6ep-b32-wd0.05, base relabel |
| **gppln** | attention, window_size 1, num_heads 1, head_proj_dim 64, 4L, 6ep-b32-wd0.05, base relabel |

Baseline to beat: triple plain gj8ch (test 0.897, val name R0.892). Prior attention (g0.5): m9mwc 12L test 0.855.

## CRF / seq2seq probe (crf_probe.py) — NEGATIVE
Frozen triple emissions + empirical label transitions + Viterbi: best λ=0.1 = argmax (+0.02pt noise), any real transition weight HURTS tel+cc. Joint label decoding gives no lift here → don't build CRF/seq2seq; ceiling is emission/data quality.

---
Pollers: `eval_all.py` (batches 3-6, /tmp/eval_all.json). Datasets: `-supported` (old), `-iaor`/`-iaorctl` (new, build_iaor.sh + strip_iaor_control.py). Split: frozen manifest seed 20260824.

### 14e. GEN-ratio sweep (root-cause: CC is 80-95% synthetic; g2.0 starved it)
Diagnosis: v0.3.2 CC regression vs v0.3.1 = the GEN cut. v0.3.1 had NO cut (all synthetic);
relabel used gen_to_real_ratio=2.0. CC fields are 80-95% GEN (cc-type 95%, cc-number 80%,
cc-exp 85%, cc-name 92%) so g2.0 removed most CC training signal. NOT audit (97 cc-name->name
filtered by name<->cc-name close-gate), NOT quantization (reduce_range/±64, fp32==int8), counts flat.
Sweep g4/g8 to recover CC without no-cut (which hurts real-site generalization):

| ratio | triple | bb-intl | bb-tinybert |
|-------|--------|---------|-------------|
| g4 | nzgn4 | h5htt | 6hgng |
| g8 | kbns9 | 6wwsd | p5gzm |

Baselines (g2.0): triple gj8ch, bb-intl 8wpwd, bb-tiny b67p7. Reference: v0.3.1 (no cut) cc-number 0.981,
cc-exp-year 1.0. Watch: CC recovery vs name/real-site tradeoff. FF loads models from GCS moz-model-hub
(NOT HF); modelVersion pref = revision on a fixed modelId (tinybert single-head / embed+head twoHead).

### 14f. CC-targeted GEN split (gen-cc source: keep all credit-card synthetic)
Root cause was the uniform GEN cut starving CC (80-95% synthetic). New "gen-cc" source in
_subsample_sources (dotraining) = GEN forms with any cc-* field (853 forms/7469 rows: cc-name 798,
cc-number 459, cc-exp 422+, cc-type 352). genCcToRealRatio pref (autofill_flow --gen_cc_to_real_ratio)
keeps them (0=all) while gen_to_real_ratio still cuts the non-cc GEN (17861 rows). Sweep 14e showed
triple g8 (kbns9 0.9103/0.9266) already beats v0.3.1 (0.9037/0.9222) on total/close/name; targeted split
aims to fully recover CC (v0.3.1 cc-number 0.981) at a leaner non-cc GEN. Runs (gen2.0 + gencc0):
triple khh8z, bb-intl hplrn, bb-tiny kqgxc. Requires re-create (dotraining/flow baked).

### 14g. Targeted-cut sweet-spot sweep (gen_cc=0 keep all CC + light/moderate non-cc cut)
NOTE: with gen-cc split, non-cc gen=17861 rows / n_real=7558 -> cut only bites below ratio 2.36
(g4/g8 uniform = no cut; both keep all 25330 gen -> g4-vs-g8 diff was seed noise). gencc-g2 cut non-cc
15% (name 0.710, cc-type 0.714); g8/no-cut name 0.768 cc-type 0.571. Sweep the middle:
| non-cc cut | triple | bb-intl |
| g2.2 (~7%) | f8xv6 | m2pkr |
| g1.5 (~37%) | 9ggvs | pzkmc |
Find ratio balancing name recall (priority) vs cc-type. Ship candidates so far: triple g8 kbns9, bb-intl g4 h5htt (exported to HF, no-cut).

### 15. Store-locator negatives + postal-code mislabel fix (fix v0.2.7 postal over-firing)
Root cause of v0.2.7 (bb-intl g4 h5htt) 3x postal-code modification rate: postal-code training rows
included store-locator/pickup "search" fields mislabeled postal-code (model learned locator search ->
postal-code). Fixes on the -relabel corpus:
- 20 mislabeled fields -> --NONE-- : 11 store-locator/pickup search (hervis q, ikea/id.html pc_shipping,
  elefant StoreFinder_City, goodlife store_locator_list, GB/US GEN pickup) + 6 GEN synthetic garbage
  (Free Pickup / Search for Branch / Estimated Business Days / Find Pickup Location / Delivery Timeframe)
  + 3 earlier (bauhaus pickup, peacocks store finder, bestshoponline zip). 17 were postal-code.
- 27 synthetic store-locator/pickup negatives (9 langs x finder/locator/pickup), postcode-named
  type=search field left unlabeled -> --NONE-- (36 rows).

Dataset rebuilt WITHOUT Firefox (both objdirs broken: `firefox` on wrong branch granite_embeddings,
`firefox-clone` fr_perf objdir won't launch). Ported the select_option/input_attributes features +
LabelUtils.findNearbyText/extractLabelStrings + ELIGIBLE_INPUT_TYPES + adjusted_field_name into
data_tools/ff_preprocess.py + build_dataset.py (mlData now byte-identical to FF). Built trainpool from
the FF .bak: kept exact FF rows, surgically flipped the 22 changed field labels (matched by stable
id/name tokens), appended the 27 Python-tokenized negatives. training-relabel 54045 / val 4550 /
testing 4758 (unchanged).

Triggered 2026-09-18, namespace production:autofillflow-0-egrc, base rolf-mozilla/minilm-pruned-notest-relabel:
| model | run | config | replaces |
|-------|-----|--------|----------|
| bb-intl g4     | argo-autofillflow-889f8 | bb, 4L, 6ep-b32-wd0.05, gen_cc=4, subsample_val | v0.2.7 h5htt |
| triple-intl g8 | argo-autofillflow-gsmrp | triple, 4L, head_interactions, lr7e-5 warmup0.1, gen_cc=8, subsample_val | v0.3.3 kbns9 |
Watch: postal-code precision AND recall (esp on contact-only/postal-alone forms) vs h5htt/kbns9; name recall must hold.

### 16. Regex-heuristic hint as an ML input feature (design + experiments)
Idea: feed the regex heuristic's recommended field name to the model so it can
CORRECT it (regex = high precision, low recall; ML's value is recall).
Enabling port: data_tools/heuristics_regexp.py + heuristics_data/rules.json
(rules dumped verbatim from Firefox HeuristicsRegExp.getRules via node, matcher
reproduces _findMatchedFieldNames/_matchRegexp/testRegex). regex_hint(el,soup).

Analysis A (regex-alone on test n=4979): overall 78.4%; coverage 76.9%.
Per-class precision: postal-code P0.989/R0.925, email P0.938/R0.976,
family-name P0.990, address-level2 P0.976, country P0.975. Weak/absent:
tel-national/bday/sex (no rules -> ML must supply), cc-number P0.315 (over-fires
"^number$"), address-line1/2/street disambiguation. => hint is a strong,
high-precision, complementary signal; especially clean for postal-code.

Representation 1 (token): ff_preprocess.hint_token + tokenize_elements(hints=...);
build_dataset.py --hint/--hint-dropout. '**hint<class>' prepended per field ->
propagates to neighbors (bb/aa). Subword-splits like existing **markers (no
tokenizer change; deployable via an mlData reorder in FF). Controlled datasets
(all Python-built; only diff = hint): -relabelpy (baseline), -relabelhint,
-relabelhintdrop (dropout 0.3, train-only; val/test use true hints).

bb-intl runs (g4 config, 2026-09-18):
| exp | run | data_variant |
|-----|-----|--------------|
| B0 baseline no-hint | argo-autofillflow-prvt4 | -relabelpy |
| B1 hint token       | argo-autofillflow-ptjwh | -relabelhint |
| B2 hint token+drop  | argo-autofillflow-jnqrp | -relabelhintdrop |
Watch: total/close, postal-code P+R, name recall, and override buckets
{regex-correct/wrong/silent} ML+hint vs ML-alone. Representation 2 (arch
categorical hint-embedding) gated on B1/B2 results (token is far cheaper to ship;
subword mechanism removes the 'pollution' argument for arch-embed).

### 16b. RESULTS — regex-hint token experiments (bb-intl, test n=4979)
| exp | run | total | close | postal-code P/R | nonEN |
|-----|-----|-------|-------|-----------------|-------|
| B0 no-hint    | prvt4 | 0.8891 | 0.9162 | 0.970/0.960 | 0.881 |
| B1 hint       | ptjwh | 0.8954 | 0.9205 | 0.978/0.965 | 0.890 |
| B2 hint+drop0.3 | jnqrp | 0.8940 | 0.9189 | 0.971/0.968 | 0.884 |

Override buckets (regex vs GT; correct 2970 / wrong 857 / silent 1152):
| exp | correct | wrong(override) | silent(recall) |
|-----|---------|------|--------|
| B0 | 0.945 | 0.737 | 0.857 |
| B1 | 0.963 (+.017) | 0.674 (-.063) | 0.886 (+.030) |
| B2 | 0.960 (+.014) | 0.695 (-.042) | 0.872 (+.015) |

VERDICT: hint HELPS overall (+0.63pt total, +0.42 close) and specifically lifts postal-code
precision (0.970->0.978, our over-fire target), silent-field recall, and non-English. BUT it
DEGRADES override on the 17% of fields where the regex is WRONG (wrong-bucket -6.3pt) — the model
partly copies bad regex predictions (concentrated in noisy classes: cc-number P0.808->0.767,
address-line disambig). Dropout (B2) partially restores override (-4.2 vs -6.3) at a small overall
cost. Net positive but with a real override tax. Next: class-selective hint (inject only high-
precision regex classes: postal-code/email/family-name/level2/country; drop noisy cc-number/
address-line) to keep gains without the override regression; or higher dropout.

### 15b. RESULTS — postal-fix retrains (test)
| model | run | total | close | vs live |
|-------|-----|-------|-------|---------|
| bb-intl g4 (locator-fixed) | 889f8 | 0.8928 | 0.9161 | ~= h5htt (accuracy preserved) |
| triple g8 (locator-fixed)  | gsmrp | 0.9069 | 0.9243 | ~= kbns9 0.9103/0.9266 (within noise) |
Accuracy preserved; the fix's real test is the Firefox postal-code modification-rate eval (needs export+ship).

### 16c. Class-selective hint (suppress noisy-class hints) + dropout
DEFAULT_HINT_SUPPRESS (build_dataset.py) blanks the hint to **hintnone for the classes the hint
hurt in 16b (cc-number/cc-exp*/cc-csc, tel-national/area/local*/extension, address-housenumber,
address-level3, address-extra-housesuffix, additional-name); keeps high-precision + address-
disambiguation hints. bb-intl g4 config, 2026-09-20:
| exp | run | data_variant |
|-----|-----|--------------|
| B3 selective       | argo-autofillflow-kfrlj | -relabelhintsel |
| B4 selective+drop  | argo-autofillflow-b56sn | -relabelhintseldrop |
Compare vs B1 (full hint ptjwh) / B0 (no hint prvt4): goal = keep B1's postal-code/address/--NONE--
gains, erase the cc-exp-year (-0.25) / tel-national (-0.145) regressions.

### 16d. RESULTS — class-selective hint (bb-intl) + triple promotion
bb-intl test (n=4979): B0 no-hint 0.8891/0.9162 (macroF1 .706) | B1 full-hint 0.8954/0.9205 (.703)
| B3 SELECTIVE 0.8974/0.9213 (macroF1 .726, BEST) | B4 sel+drop 0.8911/0.9171 (.688).
Override buckets vs B0: B3 correct +.015 / wrong -.039 (halves B1's -.063 tax) / silent +.024.
postal-code B3 P0.978/R0.973 (best both); name P0.738/R0.855 (best). WINNER = B3 (selective, no dropout).
Stragglers cc-exp-year(n23)/tel-national(n42) didn't recover from suppression = small-sample noise.
Promoted B3 to triple: triple-intl-hintB3sel argo-autofillflow-4sj68 (g8 config, -relabelhintsel).
Ship path (token hint): FF reorder regex-before-tokenize + ~15 lines in tokenizeAttributes + pref +
model reship; NO tokenizer/ONNX/arch/telemetry change.

### 16e. RESULTS — triple + selective hint (promotion)
triple-intl-hintB3sel argo-autofillflow-4sj68: test 0.9084/0.9263 (EN .954 nonEN .899).
vs triple no-hint postal-fix gsmrp 0.9069/0.9243; live kbns9 0.9103/0.9266. Per-class ~wash
(postal-code P.983/R.955 vs .986/.965; family-name up; cc-exp-year recovered R1.0). CAVEAT: not a
clean ablation (4sj68=Python -relabelhintsel vs gsmrp=FF -relabel). Clean bb ablation B0->B3 was +0.83pt.
FINDING: hint value is architecture-dependent — clearly helps bb (no explicit neighbor model), ~neutral
on triple (fusion head already models neighbors). SHIP REC: bb = selective-hint (B3, folds in over-fire
fix + hint); triple = postal-fix only (gsmrp), skip hint (not worth FF integration for ~0 gain).

### 16f. Regex-hint as one-hot concat at the triple head (arch variant)
Hypothesis: token hint was ~neutral on triple because it dilutes in the encoder mean-pool; a direct
class-wise one-hot concatenated to the fused vector just before the classifier gives an undiluted
categorical prior, complementary to triple's neighbor fusion. dotraining: TripleEncoder concatenates
per-section (cur/prev/next) hint one-hots (HINT_NUM_CLASSES=67 each) before classifier; readFile/
_triple_classify pop the '**hint' token from each section into a one-hot id and strip it from text
(hint reaches head only via categorical path). Config.hintEmbed + autofill_flow --hint_embed.
Uses -relabelhint (FULL hint, all classes) so the head learns class-wise trust itself (no whitelist).
Run: triple-intl-hintOneHot argo-autofillflow-nmf8b (g8 config, -relabelhint, hint_embed True).
Compare vs triple-B3 token-selective (4sj68 0.9084) and triple no-hint (gsmrp 0.9069). Deploy cost if it
wins: a 2nd ONNX input (hint vector) + FF plumbing — bigger than the token hint's mlData reorder.

### 16f RESULTS — one-hot-at-head triple (nmf8b)
triple one-hot 0.9092/0.9263 | token-selective B3 4sj68 0.9084/0.9263 | no-hint gsmrp 0.9069/0.9243.
macroF1 ~tie (.7965/.7976/.7970); weightedF1 one-hot best (.9082). One-hot WORKS AS DESIGNED: best on
the noisy CC classes the token had to suppress — cc-number .816->.878, cc-exp-month .857->.939 — because
it uses the FULL hint and the head learns per-class trust itself (no whitelist). BUT overall ~wash on
triple (~4 fields, close identical): triple's fusion already captures neighbor context, so the hint is
redundant regardless of representation. VERDICT: one-hot not worth shipping on triple (2nd ONNX input +
FF plumbing for +0.08pt). Hint value stays bb-specific. SHIP: bb token-selective (B3) + triple no-hint
(gsmrp). Keep one-hot code (validated; useful for a weaker/frozen encoder e.g. federated-head).

### 16g. CORRECTION — triple DOES benefit from the hint (esp. tail languages)
Earlier "triple ~neutral" was a DATA-BUILD CONFOUND: 4sj68 (Python -relabelhintsel) vs gsmrp
(FF -relabel) mixed hint-effect with the ~0.4pt Python-vs-FF penalty. Clean inference ablation on the
SAME trained triple model (4sj68), hint tokens present vs stripped: all 89.4->90.8 (+1.4), sparse-regex
tail (ru/ua/bg/id/tr) 83.8->84.7 (+0.9), English 94.9->96.0 (+1.1). Tail gains: es+3.8 gr+5.7 fi+4.8
tr+4.5 fr+2.9 ru+2.5 jp+1.6. Unlike bb, English IMPROVES on triple (fusion robust to noisy EN regex).
CAVEAT: inference-ablation on a hint-trained model = UPPER bound (OOD stripping); confounded cross-model
+0.15pt = LOWER bound. Truth between. Launched clean matched no-hint baseline: triple-intl-nohint-py
argo-autofillflow-whmj8 (-relabelpy, g8) -> vs 4sj68 = definitive same-build triple hint benefit.
REVISED SHIP LEAN: hint helps BOTH bb and triple; regex_hint is a global mlData feature so both served
models must be hint-trained anyway -> ship bb-B3 + triple token-selective (4sj68), one FF change, one pref.

### 17. Multilingual weak-labeling (ML+regex consensus) + LLM audit -> weakmulti dataset
Mined 5 CC WARCs (/Users/mozilla/crawl/2026-34) via 5 subagents -> 7500 forms, 15 non-EN regex langs
(extended gen/ccextract FORM_KEYWORDS with ru/ko/zh/hi/fa/ml/tr/pl/fr native address terms).
consensus_label.py: model(889f8 no-hint) INTERSECT regex -> 30191 fields: KEEP 13002 (agree 12354 +
model_only 648), AUDIT 4070, drop 13119. $0 LLM. data/cc_mined_2026/.
LLM audit (audit_llm.py, o4-mini, $1.79 for all 4070): on disagreements LLM sides MODEL 61% / REGEX 31%
/ NEITHER 8%. Per-lang: model wins most (it/es/nl/ja/ru/fr 66-78%) BUT regex wins zh (model 1%!), ko
(19%), hi -> model undertrained on CJK/Korean. 1002 skipped (LLM declined = drop). -> per-lang routing:
model-primary except zh/ko/hi (regex-primary).
build_weak_dataset.py: label priority gold(audited) > agree > model_only(conf>=0.9, dropped for zh/ko/hi);
cap 1000 fields/lang whole-forms; Python mlData (ff_preprocess) to match -relabelpy base. Verdict
re-association by deterministic collection order (asserted aligned). training-weakmulti.txt = 59256 gold
+ 9805 weak+gold = 69061 rows; val/test = -relabelpy gold (NL kept OFF test per user).
Run: bbintl-weakmulti argo-autofillflow-sjqpt (g4 config). Baseline = B0 prvt4 (-relabelpy, 0.8891).
Watch: tail-language accuracy (esp ru/zh/nl) vs baseline on the gold test set.
FF-vs-Python note: Firefox ml_driver is the more accurate converter (real DOM: visibility/label/grouping);
Python port approximates it (diverges on complex full pages, muted for form-fragments). FF unusable in this
env; do an authoritative FF rebuild if weakmulti proves out.
