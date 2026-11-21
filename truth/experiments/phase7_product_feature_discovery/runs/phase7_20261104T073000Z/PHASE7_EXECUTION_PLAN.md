# Phase 7 execution plan (pre-run lock)

Run ID: `20261104T073000Z`. Primary model comparison and confirmation rule were fixed before this run's Phase 7 model results were computed.

## Phase 6 input verification

- Phase 6 report, audit, hidden-pattern report, eligibility register, validation framework, baseline OOF, and manifest were read.
- Baseline OOF SHA-256: `03a88fc40fa8071cce342bc26eda221bc0359993c8688311a43e8c36521e8de3`; matches Phase 6 manifest: yes. IDs are unique, all 171,202 rows align to current train TARGET, predictions are finite.
- Phase 3 product implementation and Phase 3B feature selection code were read. Feature formulas, fit-only parameter/maps, rounding, cash-loan masks, and random-control mechanisms are recorded in the formula register and reproduction report.
- Source/input SHA-256 values are in `PHASE7_SOURCE_PROVENANCE.csv`.

## Locked data protocol

- Confirmation partition: `StratifiedShuffleSplit(test_size=0.2, random_state=20261112)`.
- Development folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=20261113)`, applied only to development rows.
- Development/holdout ID manifest SHA-256: `5f61f347650639d47bda74779046e1f7ab9b5397ad4b314cb43a30588fa7f56b`. The manifest records IDs and roles, not labels.
- Important contamination limitation: all supplied training labels have already participated in earlier project OOF analyses. This partition is locked against *new Phase 7* feature/model decisions, but it is **not an untouched independent confirmation set**. Its score is reported once as a secondary holdout diagnostic only; it cannot satisfy promotion confirmation.

## Frozen model and feature comparisons

- Primary candidate: `raw_plus_all_product` (raw baseline plus the exact 16-column Phase 3B all-product block).
- Matched baseline: `raw_baseline`.
- Secondary discovery: each of the 16 individual product columns; K, Q, P, KQ, KP, QP blocks; existing `no_redundant` block; four negative controls.
- All variants use the same fold-specific boosting rounds selected on an inner split using only outer development-fit raw baseline features. No candidate-specific stopping or parameter tuning.
- Product parameters and key frequencies are refit from outer development-fit rows only. No TARGET is used in decoder fitting or feature maps.
- Primary holdout comparison, evaluated once after development CV is complete: raw baseline vs frozen `raw_plus_all_product`. No feature/weight/threshold changes after seeing holdout results.
- Main metric ROC-AUC. Report pooled development OOF, fold AUCs/deltas, paired bootstrap interval (descriptive; repeated-label exposure), rank correlation/disagreement, runtime, and one-time holdout AUC.

## Negative controls and promotion

- `random_permuted`, `random_keys`, `random_fee_P` (fixed false offset 177.1), `random_step_K` (fixed false step 0.0097).
- An apparent control gain is treated as noise/selection warning, not useful signal.
- No score target is promised. Promotion requires eligible methods, correct OOF, gains over raw and existing product baseline, stable folds, controls beaten, and genuinely unseen labeled confirmation. This data set cannot provide the last requirement.
