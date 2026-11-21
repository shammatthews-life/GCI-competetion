# Source code audit

## Files inspected
- D:\gci\truth\experiments\domain_formula_probe\domain_feature_formulas.py
- D:\gci\truth\research_archive\exp85_phase2_source_snapshot\project\source\run_independent_baseline.py
- D:\gci\inpuy\train.csv

## Existing script findings
- The existing script computes the candidate formulas directly and calls roc_auc_score on the full training data after filling missing values with the median.
- This is a raw single-dataset AUC check, not a true out-of-fold or 5-fold validation result.
- The preserved baseline source file defines the official raw rowwise features and uses a LightGBM training protocol with five stratified folds.
- The baseline pipeline is the correct starting point; the formula audit should be validated against the same fold assignments rather than on the full training set alone.

## Formula set implemented
- loan_to_goods_price: AMT_CREDIT / AMT_GOODS_PRICE
- annual_payment_per_loan: 12 * AMT_ANNUITY / AMT_CREDIT
- debt_service_gap: 12 * AMT_ANNUITY / AMT_INCOME_TOTAL - 0.25
- ext_score_dispersion_ratio: std(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) / mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
- family_payment_to_income: AMT_ANNUITY * CNT_FAM_MEMBERS / AMT_INCOME_TOTAL
- income_per_member: AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
- credit_to_income_per_member: AMT_CREDIT / income_per_member
- bureau_recent_ratio: (AMT_REQ_CREDIT_BUREAU_HOUR + AMT_REQ_CREDIT_BUREAU_MON) / (AMT_REQ_CREDIT_BUREAU_HOUR + AMT_REQ_CREDIT_BUREAU_MON + AMT_REQ_CREDIT_BUREAU_QRT + AMT_REQ_CREDIT_BUREAU_YEAR)
- employment_sentinel: 1[DAYS_EMPLOYED == 365243]
- regional_risk_pressure: REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE
- loan_to_income: AMT_CREDIT / AMT_INCOME_TOTAL
- annuity_to_income: AMT_ANNUITY / AMT_INCOME_TOTAL
- monthly_payment_to_income: 12 * AMT_ANNUITY / AMT_INCOME_TOTAL