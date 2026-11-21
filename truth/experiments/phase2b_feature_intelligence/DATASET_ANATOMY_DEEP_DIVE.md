# Dataset anatomy — deep dive (initial evidence pass)

- Training rows/columns: (171202, 34)
- Test rows/columns: (61500, 33)
- Predictors excluding ID: 32
- Positive target rate: 0.080729
- Source is limited to the supplied competition train/test CSVs and local official data dictionary.
- Descriptive target correlations are diagnostics only; they were not used for feature selection or to tune the outer-fold experiment.

## Domain structure

domain
Age and time since event                DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, ...
Applicant category                      CODE_GENDER, NAME_EDUCATION_TYPE, OCCUPATION_T...
Application contact / data indicator          FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE
Credit bureau inquiries                 AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUR...
External risk scores                             EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
Household and family                    CNT_CHILDREN, NAME_FAMILY_STATUS, CNT_FAM_MEMBERS
Housing / property / assets                                NAME_HOUSING_TYPE, OWN_CAR_AGE
Identifier                                                                     SK_ID_CURR
Income and financial capacity                          AMT_INCOME_TOTAL, NAME_INCOME_TYPE
Loan and contract structure               NAME_CONTRACT_TYPE, AMT_CREDIT, AMT_GOODS_PRICE
Outcome                                                                            TARGET
Regional                                 REGION_POPULATION_RELATIVE, REGION_RATING_CLIENT
Repayment / annuity                                                           AMT_ANNUITY

## Notable data-quality observations

- DAYS_EMPLOYED=365243 occurs in 30,898 training rows and is treated as a sentinel, never as employment tenure.
- Bureau inquiry fields share missingness in 23,116 rows.
- External score missing counts: {'EXT_SOURCE_1': 118928, 'EXT_SOURCE_2': 369, 'EXT_SOURCE_3': 54586}
- See COLUMN_STATISTICS.csv and DATA_QUALITY_AND_MISSINGNESS.csv for every field's distribution and missingness.
- Numeric Pearson relationships, including with TARGET for descriptive review, are in COLUMN_RELATIONSHIP_MATRIX.csv.

## Interpretation boundary

Statistics and target associations describe this training sample; they do not establish causal effects or justify hand-written prediction rules. The value 365243 is handled as an explicit sentinel. No test row was used for profiling or feature fitting in this file.

## Phase 2B experiment

                    experiment  mean_fold_auc  std_fold_auc  pooled_oof_auc  delta_pooled_vs_baseline  folds_improved_vs_baseline
                  raw_baseline       0.760729      0.003770        0.760658                  0.000000                           0
         external_interactions       0.760178      0.003202        0.760058                 -0.000600                           1
               peer_deviations       0.760800      0.003309        0.760693                  0.000034                           3
combined_peer_and_interactions       0.759994      0.003633        0.759909                 -0.000750                           1
