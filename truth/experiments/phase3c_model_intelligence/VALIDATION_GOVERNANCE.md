# Phase 3C initial benchmark governance

## Scope and evidence status

This initial milestone is a fixed-fold model and ranking-error discovery study.
The historical five folds have been examined in prior phases and are not an
independent confirmation set. No test labels exist; the competition test file
is not used for fitting, feature construction, selection, or scoring.

## Reproducible references

The corrected baseline, prior Phase 3 product decoder, and Phase 3B controlled
product OOF vectors are loaded by applicant ID. The script verifies exact ID
coverage, target alignment, fold alignment, and finite predictions before
recalculating pooled and per-fold AUC. The prior 0.762826 Phase 3 result and
the controlled 0.762444 Phase 3B result remain distinct references.

## Fold and model rules

1. Use the saved Phase 2 five-fold assignments unchanged.
2. Select boosting rounds or iterations from one stratified 10% inner stopping
   split in each outer training partition, using only raw baseline features.
   The selected capacity is fixed across all three feature variants for that
   model and fold.
3. Fit imputers, one-hot encoders, category vocabularies, and Phase 3B product
   parameters on training partitions only. Validation rows only receive
   transforms fitted on their corresponding training partition.
4. Score each outer validation fold once. Do not select configurations,
   variants, thresholds, or blend weights using pooled outer-fold labels.
5. Configurations are fixed in the executable runner: preserved LightGBM
   settings, one compact CatBoost and XGBoost configuration, a bounded
   HistGradientBoosting iteration selection, and a fixed 250-tree Random
   Forest diagnostic.
6. Report pooled AUC, fold AUCs, fold mean and sample standard deviation,
   same-model paired fold deltas, runtime, and available working-set memory.
7. Error analysis focuses on rank positions and ranking strata. A 0.5
   probability cutoff is included only as a separate probability diagnostic.

## Interpretation limits

These are development scores on reused folds. The configurations and feature
representations are exploratory, and selection among them adds researcher
degrees of freedom. A new random split of the same exposed labels cannot
remove that limitation. Do not promote a model, claim the 0.79 target, or
report a blend as unbiased without genuinely unseen labeled applicants.

Segmentation, nonlinear feature screening, and ensemble fitting are deferred
until the benchmark and OOF integrity checks complete. If continued, those
analyses must retain the same fold-local fitting and discovery-only labels.
