# Phase 3B validation governance (pre-registered)

## Status and scope lock

This protocol is frozen before Phase 3B model ablations. The historical five folds are retained only for paired comparison with the saved baseline and previous Phase 3 results. They have already been repeatedly examined during prior project work; therefore neither these folds nor a new split of the same exposed training table can constitute an untouched independent confirmation.

The Phase 3B experiment is a controlled discovery/robustness study. No feature or mechanism will be promoted on pooled AUC alone. No test-set labels exist, and test rows will not be used for feature fitting, parameter estimation, validation, or AUC.

## Frozen model and split rules

1. Use the preserved raw-only baseline LightGBM builder, parameters, five saved stratified folds, and fold seed assignments. Do not tune hyperparameters or change the baseline.
2. Select the number of boosting rounds inside each outer training partition using one stratified inner 10% stopping split, on the raw baseline feature set only. Apply that same fold-specific number of rounds to every Phase 3B feature variant. This fixes model capacity across ablations rather than selecting rounds separately for each candidate.
3. Fit all unsupervised product parameters, key-frequency maps, rarity thresholds, and feature grouping boundaries only on the current outer-training partition. For the inner stopping split, any preprocessing needed for constructing inner-fit/inner-stop matrices is fit on inner-fit rows only.
4. Outer validation labels are used only once to calculate the held-out fold AUC. No validation label is used for feature construction, parameter selection, key grouping, feature selection, or early stopping.
5. Candidate families and negative controls are defined in the executable script before results are generated. No candidate is selected by inspecting outer-fold outcomes.
6. Report pooled OOF AUC, all five fold AUCs, mean and standard deviation of fold AUC, and paired fold deltas against the saved baseline. The saved baseline is a historical reference, not a contemporaneous independent baseline refit.

## Predeclared Phase 3B families

- Baseline reference (saved prediction; no retraining).
- K, Q, and P product components separately.
- K+Q, K+P, Q+P, and all 16 product-decoder columns.
- A fixed redundancy-pruned product subset selected from formula equivalence/redundancy, not target association.
- Stable-parameter subset: Q and K; the fee-estimated P family is excluded because its estimated fee varied across historical folds.
- Rare-key variants: key values below a fit-only count threshold of 25 are either dropped to missing or grouped to a sentinel.
- Negative controls: separately permuted product columns, random keys matched approximately by cardinality, an arbitrary fee constant, an arbitrary lattice step, alternate ratio rounding, and a small seeded annuity perturbation.
- One predeclared new-formula family, evaluated only after the product/control matrix, with no target-based selection.

## Interpretation limits

- The fixed folds are discovery evidence, not independent confirmation.
- One stratified 10% inner stopping split per outer fold estimates iteration count; it is not full nested cross-validation of the feature hypothesis.
- A positive result after extensive prior researcher exposure may reflect adaptive overfitting. A truly independent confirmation requires a newly collected, previously unseen labelled cohort or other genuinely new labels.
- Frequency counts are unsupervised descriptive transformations, not target encoding. No external data, inferred applicant identities, or hand-created labels are permitted.
- The official rules audit found pooled train/test statistics and target encoding unresolved; both remain disabled.
- Annuity periodicity is unspecified by the local data dictionary; no annual/monthly repayment or contractual-term interpretation is made.

## Decision rule

Keep a feature family as a research candidate only if it is mathematically defensible, fold-safe, stable across multiple folds, beats the predeclared random-key/permutation controls, and is not explained by redundancy. Even if it passes those checks, do not call it independently confirmed or replace the champion without new unseen labelled data and rule-owner approval where required.
