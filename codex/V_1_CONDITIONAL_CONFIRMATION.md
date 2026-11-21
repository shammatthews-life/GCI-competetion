# v_1 conditional-product confirmation

**Decision: reject for leaderboard use. No submission CSV was generated.**

## Question

Do product-by-applicant-category or product-by-formula/affordability target encodings add repeatable value beyond ordinary product Q4/Q3/contract encodings?

This is the next targeted question from the earlier v29 research. It was intentionally framed as an *incremental* comparison because the v15 pipeline already contains product keys and product target encodings.

## Protocol

- Input: supplied `train.csv` only for labels; `test.csv` predictors were included only when forming label-free product frequency and quantile representations.
- Three outer stratified folds on each of two fresh seeds: `20261041` and `20261042`.
- Inner 3-fold cross-fitting for every training-row target encoding. Each outer validation fold's encodings use only the corresponding outer-training labels.
- Same compact LightGBM model parameters for the three feature variants.
- Common reference: `product_only` (Q4, Q3, and Q4 × contract encodings).
- Candidates: `product_applicant` (adds Q4 × income type/occupation/education) and `product_formula_burden` (adds Q4 × formula-residual and affordability bins).
- Paired stratified bootstrap: 250 replicates per candidate/seed.

Reproduce with [run_v1_conditional_confirmation.py](./run_v1_conditional_confirmation.py). Exact OOF arrays and metrics are retained in [v_1_conditional_confirmation_oof.npz](./v_1_conditional_confirmation_oof.npz), [v_1_conditional_confirmation.csv](./v_1_conditional_confirmation.csv), and [v_1_conditional_confirmation.json](./v_1_conditional_confirmation.json). The run used LightGBM 4.7.0 and completed in about 315 seconds.

## Results

Delta and confidence intervals below compare each candidate against `product_only` on the same outer folds.

| Outer seed | Candidate | Pooled AUC delta | Fold deltas | Paired bootstrap 95% interval |
|---:|---|---:|---|---|
| 20261041 | Product × applicant | −0.000467 | −0.000963, −0.000627, +0.000201 | [−0.001072, +0.000254] |
| 20261041 | Product × formula/burden | +0.000623 | −0.000495, +0.000551, +0.001787 | [−0.000152, +0.001277] |
| 20261042 | Product × applicant | −0.000645 | −0.001806, −0.000354, +0.000283 | [−0.001217, −0.000013] |
| 20261042 | Product × formula/burden | −0.000895 | −0.001524, −0.000480, −0.000616 | [−0.001548, −0.000273] |

The applicant-conditioned block lost on both seeds; on seed 20261042 its paired interval was below zero. Formula/burden conditioning was inconsistent: it gained on one seed but lost on all three folds of the second. Neither clears a promotion gate.

## Interpretation and next action

- **Do not make a leaderboard submission from v_1.** These are compact-model experiments, not comparisons against the v15 full ensemble.
- Do not repeat these exact conditional blocks unchanged. Their apparent gains in the earlier v29 run did not reproduce on these fresh splits.
- The remaining credible feature signal is product identity itself, which v15 already represents. A new experiment must test genuinely incremental information against v15's row-aligned OOF/member predictions and not just a compact model.
- No submission has been created, so there is no submission path to test. The v15 file remains the incumbent.

## Required before an actual candidate can be judged

Provide the exact v15 submission CSV and the complete aligned `gci_cache_final` bundle (OOF/member predictions, fold indices, and any required view/residual caches). If that cache is unavailable, provide the exact v15 notebook/source plus the scored v15 CSV. Without aligned champion predictions, this workspace cannot establish whether an alternative is better than public AUC 0.77891, and should not emit a speculative leaderboard CSV.
