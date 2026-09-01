# dataset v5_iaor

Input-attribute + option-range mlData, legacy ±1 bb/aa context. train+gen+cc.
- Generator: `data_tools/build_iaor.sh` → build_dataset_ff.py driving Firefox with
  prefs `useml.inputAttributes=true`, `useml.selectOptionRange=true`, `contextWindow=1`
  (branch select-option-range-mldata @ obj-25.6.0).
- New tokens vs the -supported baseline: `**maxlen<N>` (maxlength < 16),
  `**inputmode<mode>`, and `<first>...<last>` select option-range (blank leading/
  trailing options skipped). Labels 1-62 (getAdjustedFieldName).
- Split: frozen manifest seed 20260824 (same 562-file split as v1/v2/v4). Test = data/testing.
- `-iaor` = full new tokens. `-iaorctl` = identical fields with the new tokens
  stripped (`data_tools/strip_iaor_control.py`), a matched control isolating the
  token effect.

training-iaor=41050 validation-iaor=4550 testing-iaor=4758
training-iaorctl=41050 validation-iaorctl=4550 testing-iaorctl=4758

Re-pruned base for this data: `rolf-mozilla/minilm-pruned-notest-iaor` (vocab 16007,
pruned on -iaor train+val, test held out — vs v2's 12455).
