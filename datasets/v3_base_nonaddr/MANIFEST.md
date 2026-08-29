# dataset v3_base_nonaddr

= datasets/v1_base + labeled non-address negatives (data/non_address_labeled, 3+ fields).
- selectOptionRange pref: OFF
- Non-address labels: data_tools/label_non_address.py (autocomplete/type/regex -> email/name/tel/given/family; else other).
- Non-address split: 80/10/10 file-level, seed 20260826, 176 files.
- Labels 1-62 (getAdjustedFieldName). Test now includes held-out non-address negatives.

| split | base rows | +non_addr | total |
|---|---|---|---|
| training | 40615 | 689 | 41304 |
| validation | 4537 | 78 | 4615 |
| testing | 4749 | 74 | 4823 |
