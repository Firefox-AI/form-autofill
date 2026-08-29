# dataset v2_optrange

v1 + select first...last option token in mlData (option-range ON).

- Sources: data/training + data/generated + data/common_crawl (train/val); test = data/testing
- Generator: data_tools/build_dataset_ff.py (drives Firefox, branch select-option-range-mldata @ obj-25.6.0)
- selectOptionRange pref: ON
- Labels: getAdjustedFieldName (SUPPORTED_FIELDS/NOMINAL_MAP, D305753); unsupported->other. Range 1-62.
- Split: file-level 90/10 train/val, seed 20260824. Test = data/testing.

| split | rows | sources | label range | sha256 |
|---|---|---|---|---|
| training | 40615 | {'real': 7551, 'gen': 25330, 'cc(crawl)': 7734} | (1, 62) | 9a3078dca010df86 |
| validation | 4537 | {'real': 658, 'gen': 2944, 'cc(crawl)': 935} | (1, 62) | a213040b2e226ce6 |
| testing | 4749 | {'real': 4749} | (1, 62) | 8af564cc23444b1c |
