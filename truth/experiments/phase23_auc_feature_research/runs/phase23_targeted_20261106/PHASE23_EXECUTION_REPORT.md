# Phase 23 - targeted AUC feature research

## Protocol and evidence boundary

Train-only, five matched stratified folds (seed 20261113), CatBoost GPU, 700 iterations, depth 6, learning rate 0.05. Each feature family was evaluated separately, together, and as a fixed 50/50 OOF blend with the matched global model. Positive-class weights 2 and 4 and a cross-fitted low-score specialist were included as predeclared AUC-directed tests. No test or external data were read. Every label has prior exposure, so this is exploratory rather than independent confirmation.

- Rows: 171,202; positives: 13,821 (8.073%).
- Phase 15 reference AUC: 0.767510928.
- Refit matched baseline AUC: 0.767488294 (delta -0.000022633).
- GPU fits: 60; runtime: 1427.3 seconds.

## OOF results

| Candidate | Pooled AUC | Delta vs matched baseline | Positive folds |
|---|---:|---:|---:|
| application_process_fixed_50_50_blend | 0.768177708 | +0.000689414 | 5/5 |
| applicant_consistency_fixed_50_50_blend | 0.768020705 | +0.000532411 | 5/5 |
| external_disagreement_fixed_50_50_blend | 0.767995554 | +0.000507260 | 5/5 |
| financial_residuals_fixed_50_50_blend | 0.767977611 | +0.000489316 | 5/5 |
| application_process | 0.767949642 | +0.000461348 | 4/5 |
| scale_pos_weight_4 | 0.767792631 | +0.000304336 | 3/5 |
| scale_pos_weight_2 | 0.767686338 | +0.000198044 | 4/5 |
| financial_residuals | 0.767672508 | +0.000184213 | 3/5 |
| applicant_consistency | 0.767643520 | +0.000155226 | 3/5 |
| external_disagreement | 0.767615182 | +0.000126888 | 3/5 |
| all_four_feature_families | 0.767540195 | +0.000051901 | 3/5 |
| matched_baseline | 0.767488294 | +0.000000000 | 0/5 |
| low_score_fixed_50_50_expert | 0.767162972 | -0.000325322 | 0/5 |
| low_score_hard_routed_expert | 0.765031626 | -0.002456668 | 0/5 |

Fixed 50/50 blends and class weights are exploratory diagnostics; no weight or family was promoted. See `PHASE23_AUC_RESULTS.csv` for fold AUC/deltas and `PHASE23_PAIRED_BOOTSTRAP.csv` for conditional row-stratified intervals.

## Pairwise interpretation

Candidate ranking changes are decomposed over all positive-negative pairs, and by baseline positive-risk decile, in `PHASE23_PAIRWISE_RISK_BANDS.csv`. A subgroup-only AUC is not a promotion criterion.

## Train-only discovery profile

- Distinct missingness profiles: 26.
- Exact duplicate predictor rows: 0.
- Missingness and target-rate summaries are descriptive on previously examined labels; they do not establish incremental feature value.

## Interpretation limits

Financial residuals and applicant-pair frequencies are constructed without TARGET: their training-side values are inner-fold cross-fitted, and mappings are fit only on each outer training fold. The process-pattern vocabulary is fit on outer training rows. The low-score gate comes from inner OOF scores and is never selected on outer labels. Bootstrap intervals condition on the saved OOF predictions and do not account for model-training variation or accumulated research selection. Competition eligibility of inherited Phase 15 features remains unresolved.
