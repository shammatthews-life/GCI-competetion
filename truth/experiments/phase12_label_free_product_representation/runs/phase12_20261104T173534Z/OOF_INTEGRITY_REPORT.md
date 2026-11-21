# OOF integrity report

- Training file: `D:\gci\inpuy\train.csv`
- Training SHA-256 before/after run: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1` / `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- Rows: 171,202; unique IDs: 171,202; target positives: 13,821.
- Primary folds: 5; seed `20261105`; each row receives exactly one outer validation prediction.
- Secondary seed: `20261106`; its OOF prediction file includes a distinct complete fold assignment.
- Main-model OOF predictions present: 7; all finite and within [0,1].
- Ablation-model OOF predictions present: 20; all finite and within [0,1].
- Negative-control OOF predictions present: 10; all finite and within [0,1], with IDs and targets aligned.
- Missing prediction count across primary, ablation, control, and secondary OOF outputs: 0.
- ID order and target alignment were checked against the exact training input row order.
- The runner did not open test.csv, sample_submission.csv, external files, or a holdout.
- Negative-control predictions were validated for completeness before pooled metrics; each control repetition and fold is recorded in NEGATIVE_CONTROL_RESULTS.csv.
- No target encoding, target-derived category statistics, CatBoost CTRs, or test-fitted transforms were used.
- Input file hash remained unchanged during execution.
- Status: PASS.
