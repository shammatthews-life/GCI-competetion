# Phase 20 competition-eligibility audit

## Materials reviewed

- Local official-rules audit: `reports/final_competition_audit/OFFICIAL_RULES_AUDIT.md`.
- Master course/competition materials review: `reports/COMPETITION_MATERIALS_REVIEW.md`.
- Runner: `experiments/phase20_segmentation_expert_research/run_phase20_segmentation_study.py`.
- Preregistered definitions and results in this run directory.

The local rule review says to use only supplied train/test inputs and prohibits external data and hand-labeling. It permits deterministic features derived from provided columns, but does not certify any particular model or guarantee that repeated OOF evaluation is unbiased.

## Data and label flow

- The run loaded only the supplied `train.csv`; no test, sample-submission, external, pretrained, or cached prediction data was read. The run summary records the train SHA-256.
- Every route uses only supplied feature columns. Route generation does not inspect `TARGET`; the income median is computed from each applicable training fold.
- The expert learners use TARGET only as their supervised training outcome, with all validation and holdback outcomes excluded from their model fit.
- Learned F15 gates train against feature-defined route membership, not applicant TARGET, and their training-side gate scores are cross-fitted.
- All candidate scoring is automatic. No row receives a hand-entered prediction, manual override, or EDA-driven test adjustment.

## Features and model behavior

- Raw/rowwise transformations and ratios are deterministic functions of supplied columns.
- C08's key-frequency features use only the direct constant-free ratios annuity/credit and annuity/goods at fixed decimal precisions. No fee-offset `130.5` or financing-step `0.0066` key was used. Frequency maps are learned from the fit partition and applied to validation/holdback rows.
- The three financial decompositions and external-score polynomials are deterministic. No TARGET encoding or target-derived feature statistics were used.
- CatBoost training explicitly uses `task_type='GPU'`, `one_hot_max_size=256`, and `max_ctr_complexity=1`; fit categorical cardinality is checked. These settings prevent native target-statistic CTRs from silently entering these experiments. GPU-blocked cluster definitions were not substituted with CPU clustering.
- No model or prediction cache was persisted; results contain aggregate pooled and fold metrics only.

## Eligibility conclusion

**No obvious violation of the reviewed local data-source, hand-labeling, or reproducibility rules was found in the Phase 20 runner.** This is a code-level audit, not organizer approval. Constant-free derived keys/frequencies and all engineered features remain subject to the organizer's interpretation; do not present this audit as formal eligibility certification. Repeated evaluation on known labels is a validation limitation even where data use is permitted. No submission or champion was altered.
