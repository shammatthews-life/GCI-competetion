# Phase 14 OOF integrity and independent artifact verification

- Status: **PASS**
- Training rows / unique IDs / positives: 171,202 / 171,202 / 13,821
- Phase 13 input row/ID/target alignment: PASS
- Phase 14 OOF row/ID/target alignment: PASS
- Phase 14 stratified folds reproduced with seed 20261113: PASS; fold sizes {"1": 34241, "2": 34241, "3": 34240, "4": 34240, "5": 34240}
- OOF prediction columns: 7; missing/non-finite values: 0
- Pooled AUCs recomputed from saved OOF predictions: 7/7 match `MODEL_RESULTS.csv` within `1e-12`
- Fold AUCs recomputed from saved predictions: 35/35 match `FOLD_LEVEL_METRICS.csv` within `1e-12`
- CatBoost fit-partition maximum category cardinality: 4,564; `one_hot_max_size`: 10,000; target-derived CTR threshold was not reached
- Paired subgroup outputs: 441 fold/group/model rows; 48 model/group summaries. Subgroups are descriptive, not independent validation.
- Training SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- Phase 13 OOF SHA-256: `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`
- Phase 14 OOF SHA-256: `4c3e31495dc9da5615aacb1fd87fbc529f51ee8128c500b924037866a8b5bfca`

No test data, external data, submissions, or champion files were used in this verification.
