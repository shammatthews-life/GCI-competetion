# Phase 20 — segmentation and expert-model research

**Status:** GPU Stage 1 completed. All results are exploratory because the labels have been examined in prior phases; the 40% holdback is not an untouched confirmation set.

## Preregistered design

- Registry SHA-256: `361ea70d6fd9b522b08126b2b6c59f0f775dd870ea6abdb99d625c8fdb82732f`.
- Exactly 150 real experiments (15 feature-only segmentation families x 10 distinct GPU CatBoost configurations) and 30 placebo controls.
- Data: supplied train file only; 60% stratified discovery / 40% confirmation partition, seed `20261106`; five discovery folds, seed `20261113`.
- Stage 1 used R2: a fixed 50/50 blend of the global GPU model and hard-routed experts; R1 hard routing is also reported diagnostically.
- The learned-gate configurations use R5: a GPU gate trained on feature-defined membership with cross-fitted training-side gate predictions, mixing two GPU experts; these are not fixed R2 blends.
- Model training was CatBoost `task_type='GPU'`; one-hot categorical threshold 256 prevents target-statistic CTRs on the supplied categorical columns (maximum supported by the installed CatBoost). No early stopping was used.
- Fold-local preprocessing: raw rowwise transforms are deterministic; key-frequency features, income median, and any model gate are fit within the relevant training fold.
- Cluster families are explicitly marked GPU-blocked if no CUDA clustering implementation is available; no CPU clustering fallback is used.
- Placebos: 30 row-random or ID-parity splits, using three controls per model configuration to estimate a pooled null distribution.

## Stage 1 results

- Real experiments: 150; completed GPU: 130; GPU-blocked: 20; unsupported groups: 0.
- Placebo controls: 30; completed GPU: 30.
- Placebo 95th percentile of R2 pooled-AUC deltas: `-0.000074085`.
- Discovery candidates passing both the placebo threshold and >=4/5 positive folds: 31.
- Contract-type routing was the most consistent family: all 10 configurations had positive pooled deltas (mean `+0.000872`, median `+0.000922`). The largest single discovery delta was `+0.001204`; the highest discovery mixture AUC was `0.755217`. These results are from the 60% discovery sample and are not comparable to the full-data historical OOF scores.
- A production-path pilot exposed an exact-route output-shape bug before any candidate result was checkpointed. It was fixed, unit-tested, and a complete five-fold GPU family smoke test passed before the corrected grid started.
- The first corrected-grid attempt checkpointed 177 rows and stopped on an ID-parity placebo parsing error. The parser was fixed and regression-tested; the remaining three placebos completed from the durable checkpoint. No completed candidate experiment was rerun, although the C10 global OOF baseline was recomputed.
- Corrected-grid wall time was approximately 5,065 seconds, based on its first-launch and final-report timestamps (preflight and smoke tests excluded). The summary distinguishes this from the 226-second resume process. The 108 GPU telemetry samples (maximum 81% utilization and 4,706 MiB) cover the successful resume/finalization process only; they are not a full-grid utilization profile because first-attempt telemetry was not persisted after its exception.

The complete 180-row status, experiment definitions, pooled/fold AUCs, paired deltas, subgroup sizes/AUCs, log loss, runtimes, and failures are in `PHASE20_STAGE1_RESULTS.csv`. The exact pre-run definitions are in `PHASE20_EXPERIMENT_REGISTRY.csv`.
Fold metrics, configuration summaries, segmentation-family summaries, and detailed error analysis are in `PHASE20_FOLD_METRICS.csv`, `PHASE20_FEATURE_ABLATION.csv`, `PHASE20_SEGMENTATION_SUMMARY.csv`, and `PHASE20_ERROR_ANALYSIS.md`.

## Confirmation and interpretation

Only 10 discovery-qualified candidates were evaluated once on the predeclared 40% holdback.
Because all labels were previously examined in historical phases, this holdback cannot establish independent generalization. No champion or submission was changed.
The Phase 15 three-seed OOF score remains `0.768824708`; the retrospective Phase 9 mean-prediction score remains `0.769839127`; neither is an independent champion. No `0.780` claim is made unless it is supported by a valid, previously untouched evaluation.

## Eligibility

All routes use supplied feature values and never TARGET. No external data or test data was read. CatBoost target-statistic CTRs are disabled by one-hot encoding the supplied categorical features. Constant-free product-key frequency features are fold-local and unsupervised but remain subject to organizer interpretation. Cluster families are not run on CPU if CUDA clustering is unavailable.
The detailed code-level rules assessment is in `PHASE20_ELIGIBILITY_AUDIT.md`; it is not formal organizer approval. No row-level prediction cache was saved, so per-row residual analysis and prediction-correlation analysis are unavailable.
