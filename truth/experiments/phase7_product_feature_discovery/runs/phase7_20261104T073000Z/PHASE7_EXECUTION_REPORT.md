# Phase 7 execution report

Run ID `20261104T073000Z` completed. The Phase 6 baseline provenance and OOF rows were verified before experiments. All primary and secondary feature construction used training data only, with feature parameters and maps fit within the relevant development training fold.

## Results

- Primary predeclared raw-only development pooled OOF AUC: `0.756878879`.
- Primary `raw + all Phase 3 product` development pooled OOF AUC: `0.759320302` (delta `+0.002441424`).
- One-time holdout primary delta: `+0.000919365`; this holdout is **not independent confirmation** because its labels appeared in prior project OOF research.
- Detailed individual/block/control results: `PHASE7_MODEL_COMPARISON.csv`, `PRODUCT_INDIVIDUAL_ABLATION.csv`, `PRODUCT_BLOCK_ABLATION.csv`, `NEGATIVE_CONTROL_RESULTS.csv`.
- Decision: **RESEARCH CANDIDATE ONLY — independent confirmation unavailable**.

## Scope

Exact Phase 3 product formulas were reproduced, 16 individual columns tested, eight product blocks tested, and four negative controls run. Lattice stability was assessed across five development folds and fixed placebo steps. One primary raw-vs-product holdout comparison was scored once. No V16/EXP85 reconstruction, test prediction pipeline, submission, champion change, or broad model sweep was performed.

All labeled rows and historical folds have prior researcher exposure. Therefore results are development evidence only; no independent 0.77 claim or model promotion is justified.
