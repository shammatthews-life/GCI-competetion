# Existing code and metric audit

## Baseline implementation

- Source: `D:\gci\truth\research_archive\exp85_phase2_source_snapshot\project\source\run_independent_baseline.py`
- Reuses `build_rowwise_features`, `make_lightgbm_frames`, `MODEL_CONFIG['lightgbm']`, `N_SPLITS=5`, and `FOLD_SEED=42` from the preserved baseline source.
- The baseline is the raw-only Phase 2 snapshot, not the stronger Phase 2 fee/target-encoding feature stack.
- Model: LightGBM binary GBDT, learning rate 0.03, 31 leaves, min child samples 80, feature fraction 0.80, bagging fraction 0.80, L1 0.10, L2 5.0, max bin 127, 3,000-round cap, 150-round early stopping.
- Categorical frame construction is performed by the existing helper from each relevant fit partition; missing numeric values use LightGBM native handling.

## Prior domain-formula execution audit

- `domain_formula_probe/domain_feature_formulas.py` computes formulas row-wise and calculates each univariate AUC on the full training labels after median imputation. Those AUCs are descriptive in-sample diagnostics, not CV performance.
- `domain_formula_execution_audit_01/run_domain_formula_audit.py` used the outer validation fold as LightGBM's early-stopping set. Its fold AUCs are contaminated by iteration selection and are not accepted as unbiased OOF results.
- The earlier “top 3” were selected from full-training TARGET AUC, which leaks labels into feature selection. That variant is not repeated.
- That previous runner also rebuilt categorical dummy columns using the whole training feature matrix rather than the baseline helper's fit-partition category handling, so its baseline was not an exact reproduction of the preserved baseline pipeline.
- Existing deliverables remain untouched as historical records; this corrected run is the new validation reference.

## Formula overlap / semantics

- `ANNUITY_CREDIT_ANNUALIZED` is a constant multiple of baseline `ANNUITY_TO_CREDIT`.
- `DEBT_SERVICE_GAP` is a linear scaling and shift of baseline `ANNUITY_TO_INCOME`.
- `INCOME_PER_MEMBER` approximates baseline `INCOME_PER_FAMILY`; their missing-family handling differs.
- `EMPLOYMENT_SENTINEL` duplicates baseline `DAYS_EMPLOYED_SENTINEL`.
- `LOAN_TO_GOODS_PRICE` reverses baseline `CREDIT_TO_GOODS` and is a monotone reciprocal for positive goods values; tree splits may not be identical under finite binning, but it does not add new raw information.
- `CREDIT_TO_INCOME_PER_MEMBER` algebraically equals `CREDIT_TO_INCOME * CNT_FAM_MEMBERS` when the family denominator is valid.
- Bureau recent/older sums combine distinct supplied windows and are exploratory counts, not standardized rates. Individual window fields are also retained.
- Annualizing `AMT_ANNUITY` is a mathematical rescaling. The data dictionary calls it “Loan annuity” but does not state its period; therefore interpreting it as an annual payment is not supported. The feature name here makes no annual-payment claim.
- Multiplying annuity by household size assumes equal financial responsibility and is retained only for an ablation test, not endorsed as a domain fact.

## Corrected metrics

                        variant  mean_fold_auc  std_fold_auc  pooled_oof_auc  delta_pooled_vs_baseline
                   raw_baseline       0.760729      0.003770        0.760658                  0.000000
baseline_plus_existing_formulas       0.760452      0.003521        0.760372                 -0.000287