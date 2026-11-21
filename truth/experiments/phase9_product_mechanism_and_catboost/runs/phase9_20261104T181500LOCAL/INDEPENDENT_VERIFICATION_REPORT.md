# Independent Phase 9 result verification

- Status: **PASS**
- Training rows: 171,202; train input SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- OOF rows and ID/TARGET order: exact match to the training input.
- Prediction columns: 45; all finite.
- Independently recalculated metrics: 45 pooled seed/model AUCs and 225 fold AUCs; all match saved results within `1e-10`.
- Nested target-map integrity: 135 rows; all outer/inner disjointness and no-outer-validation-label checks pass.
- Post-hoc subgroup AUCs/deltas, exact adjacent-annuity placebo counts, and Q4 residual tables were independently recalculated and match.
- CatBoost device metadata: `NVIDIA GeForce RTX 4050 Laptop GPU, 592.82, 6141 MiB`; run configuration is GPU device 0.
- Runner SHA-256: `7d07433e312e969e8b41fd0b31743f0963a6cbeee34a02d49451e896ab33deb8`.

These are repeated-seed OOF checks on historically examined training labels, not independent confirmation or a competition-eligibility ruling. LightGBM's installed build lacks GPU tree support; LightGBM therefore remained CPU while the CatBoost comparisons ran on GPU.
