# Phase 9 one-fold parity preflight

## Decision

**BLOCKED — exact Phase 9 reproduction cannot be claimed under the
zero-EXP85-dependency rule.** The allowed official dictionary documents the
raw application columns, not the complete historical row-wise feature
builder. The independent feature matrix is intentionally specified from raw
predictors and documented row-local mathematics; it is not an attempt to
recover the prohibited helper's undocumented transformations.

## Preflight performed

One outer fold (seed 42, fold 1) was fit on the GPU with the documented Phase
9 CatBoost hyperparameters, model seed, and Phase 9 selected iteration count.
The two fits were the independent-builder baseline and the same independent
features plus the eight row-local product keys. Their validation predictions
were compared with the historical Phase 9 saved OOF columns only for
diagnostic reporting. The Phase 9 outputs were not used to construct, choose,
or fit features.

Per-variant AUCs, prediction rank correlations, maximum prediction
differences, and feature counts are in `PHASE9_PARITY_PREFLIGHT.csv`. Because
the feature matrix differs by specification, score similarity would not prove
source parity; score differences do not indict either model. The exact
reproduction status remains **BLOCKED**, regardless of numerical proximity.

No Phase 9 helper was restored, imported, or read to improve the match. No
remaining Phase 10 ablation, robustness grid, or champion decision was made.
