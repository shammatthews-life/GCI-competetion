# Final Audit Report

- The preserved raw baseline source was located and used as the model reference.
- Formula values were reproduced directly from the provided training CSV and saved to the CSV outputs.
- The earlier AUC claims are not valid cross-validation evidence and therefore are not treated as proven improvements.
- The actual 5-fold CV comparison was executed using the raw baseline and feature-augmented variants.

## Reproduced results
- ext_score_dispersion_ratio: raw_auc=0.6068, direction_adjusted_auc=0.6068, missing=38728, inf=0
- loan_to_goods_price: raw_auc=0.5681, direction_adjusted_auc=0.5681, missing=0, inf=0
- regional_risk_pressure: raw_auc=0.5417, direction_adjusted_auc=0.5417, missing=0, inf=0
- annual_payment_per_loan: raw_auc=0.5336, direction_adjusted_auc=0.5336, missing=0, inf=0
- debt_service_gap: raw_auc=0.5167, direction_adjusted_auc=0.5167, missing=0, inf=0
- monthly_payment_to_income: raw_auc=0.5167, direction_adjusted_auc=0.5167, missing=0, inf=0
- annuity_to_income: raw_auc=0.5167, direction_adjusted_auc=0.5167, missing=0, inf=0
- family_payment_to_income: raw_auc=0.5145, direction_adjusted_auc=0.5145, missing=2, inf=0
- credit_to_income_per_member: raw_auc=0.4998, direction_adjusted_auc=0.5002, missing=2, inf=0
- loan_to_income: raw_auc=0.4955, direction_adjusted_auc=0.5045, missing=0, inf=0
- bureau_recent_ratio: raw_auc=0.4915, direction_adjusted_auc=0.5085, missing=52271, inf=0
- income_per_member: raw_auc=0.4833, direction_adjusted_auc=0.5167, missing=2, inf=0
- employment_sentinel: raw_auc=0.4678, direction_adjusted_auc=0.5322, missing=0, inf=0

## 5-fold CV summary
- B_all_verified_formulas: mean_auc=0.7612, std_auc=0.0041, delta_vs_baseline=0.0079
- C_top3_formulas: mean_auc=0.7608, std_auc=0.0043, delta_vs_baseline=0.0076
- A_raw_baseline: mean_auc=0.7532, std_auc=0.0046, delta_vs_baseline=0.0000

## Remaining blockers
- A full competition-cleared submission still requires valid fold-safe feature selection and reproduction of the official baseline pipeline from the project.
- No feature improvement should be claimed without measured OOF evidence from the same folds and model configuration.