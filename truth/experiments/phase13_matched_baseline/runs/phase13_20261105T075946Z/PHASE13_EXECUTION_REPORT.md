# Phase 13 matched-baseline execution report

- Run folder: `D:\gci\truth\experiments\phase13_matched_baseline\runs\phase13_20261105T075946Z`
- UTC interval: `2026-11-05T07:59:46.355550+00:00` to `2026-11-05T08:49:29.037820+00:00`
- Training file: `D:\gci\inpuy\train.csv`; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Rows: 171,202; primary seed: 20261113; 5 stratified outer folds; 3 inner folds.
- CPU LightGBM; all transformations/encoders fit in the relevant training partition. No test, external data, target encoding, target-derived category statistics, holdout, submission, or champion change.

## Matched primary benchmark

| Variant | Pooled OOF AUC | Delta vs A | Positive folds vs A |
|---|---:|---:|---:|
| A_raw_numeric | 0.744128 | +0.000000 | 0/5 |
| B_raw_numeric_categories | 0.752227 | +0.008100 | 5/5 |
| C_complete_raw_rowwise | 0.759619 | +0.015492 | 5/5 |
| D_C_plus_keys_ohe | 0.765270 | +0.021143 | 5/5 |
| E_C_plus_keys_frequency_ohe | 0.766038 | +0.021910 | 5/5 |
| F_C_plus_supported_numeric_decompositions | 0.760269 | +0.016141 | 5/5 |

## Product-key incremental value over complete raw baseline C

- D: C plus ordinary one-hot keys: 0.765270; delta vs C +0.005651.
- E: C plus fit-fold frequency-supported one-hot keys: 0.766038; delta vs C +0.006418.
- Highest primary feature set by incremental OOF AUC over C was `E_C_plus_keys_frequency_ohe` (+0.006418); primary fold and bootstrap results are also reported.
- Compare `MATCHED_BASELINE_RESULTS.csv`, `PRODUCT_KEY_INCREMENTAL_VALUE.csv`, and `PAIRED_BOOTSTRAP_REPORT.md`. These are selection-sensitive CV estimates on previously examined labels, not independent confirmation.

## Key ablation and controls

- Strongest single-key ablation was `single_P4` at pooled OOF AUC 0.763452. All ablations share the D configuration/rounds selected inside each outer fit and remain exploratory.
- Mean control pooled OOF AUCs: `{"frequency_preserving_shuffled_keys": 0.7580433962193344}`.
- Do not remove any key on pooled rank alone. Inspect every fold and paired contrast.

## Formula provenance and eligibility

The 130.5 annuity gap is a repeated numerical gap in the Phase 9 audit, not an established fee. The 0.0066 step has no located permitted-data derivation and is not an established financing rate/tier. Both remain unresolved; D/E reproduce these hypotheses only to measure prediction lift. F excludes both. See `FORMULA_PROVENANCE_AUDIT.md` and `COMPETITION_ELIGIBILITY_MATRIX.csv`.

## Secondary-seed robustness

- C_complete_raw_rowwise seed 20261114: AUC 0.759620, delta vs C +0.000000, 0/5 positive folds.
- E_C_plus_keys_frequency_ohe seed 20261114: AUC 0.765533, delta vs C +0.005913, 5/5 positive folds.
This is robustness replication using the same applicants/labels, not independent confirmation.

## Decisions

1. True incremental key contribution is the D/E paired contrast against complete raw+rowwise C; report the observed result without comparing it to the numeric-only Phase 12 delta.
2. Frequency support is preferred only if E reliably beats D across folds and the secondary seed; category counts use the fit partition only.
3. Key consistency is descriptive across the fixed-fold ablation, not a basis for a final key subset.
4. Genuine versus shuffled-control performance is a negative-control check, not an untouched validation set.
5. No track is organizer-certified. P4/P5/KQ3/KQ4 remain eligibility-pending; F excludes their undocumented constants.
6. Advance only the best matched feature representation after a separately predeclared robustness replication; do not promote the model from these exposed OOF rows.
