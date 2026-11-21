# Hidden loan-product and repayment structure — Batch 1

## Scope and guardrails

Only the supplied training predictors and training TARGET were used. TARGET is used solely for held-out ROC-AUC. The official data dictionary defines AMT_ANNUITY only as “Loan annuity”; it does not specify payment periodicity, so this report does not call it monthly or annual and does not infer a contractual term from CREDIT/ANNUITY. Test covariates were schema-checked but excluded from discovery, fitting, and validation.

The v17 source defines `K_DEC = round((CREDIT/GOODS - 1)/0.0132)` with a 0.1 lattice tolerance, and product ratios `Q3/Q4/Q5 = round(ANNUITY/CREDIT, 3/4/5)`. It estimates no documented derivation for constants 129.5 and 0.0132. This experiment re-estimates the candidate fee offset and lattice step using each training partition only; the fixed v17 constants are reported solely as hypotheses.

## Mathematical hypotheses

                       feature_name                                                       formula                                                                    meaning
             FIN_CREDIT_MINUS_GOODS                                  AMT_CREDIT - AMT_GOODS_PRICE              Financing amount above stated goods price; signed difference.
             FIN_GOODS_MINUS_CREDIT                                  AMT_GOODS_PRICE - AMT_CREDIT                           Reverse signed difference; exact sign duplicate.
              FIN_EXCESS_OVER_GOODS              (AMT_CREDIT - AMT_GOODS_PRICE) / AMT_GOODS_PRICE                                    Credit premium relative to goods price.
         FIN_EXCESS_SHARE_OF_CREDIT                   (AMT_CREDIT - AMT_GOODS_PRICE) / AMT_CREDIT                              Share of credit exceeding stated goods price.
                  FIN_SYMMETRIC_GAP                 (CREDIT - GOODS) / (abs(CREDIT) + abs(GOODS))                                   Bounded scale-free financing difference.
         FIN_CREDIT_GOODS_LOG_RATIO                                  log1p(CREDIT) - log1p(GOODS)                    Log-scale financing gap; guarded to nonnegative values.
                FIN_GOODS_TO_CREDIT                                                GOODS / CREDIT    Goods coverage relative to credit; reciprocal relation to credit/goods.
               FIN_ANNUITY_TO_GOODS                                               ANNUITY / GOODS                                 Payment intensity relative to asset price.
       FIN_ANNUITY_TO_FINANCING_GAP                                    ANNUITY / (CREDIT - GOODS) Payment amount relative to financed excess; undefined for nonpositive gap.
                 FIN_GAP_TO_ANNUITY                                    (CREDIT - GOODS) / ANNUITY                                         Financing excess per unit annuity.
                  FIN_GAP_TO_INCOME                                     (CREDIT - GOODS) / INCOME                                       Financing excess relative to income.
                  FIN_GAP_TO_CREDIT                                     (CREDIT - GOODS) / CREDIT                                     Same as excess share; duplicate alias.
               FIN_CREDIT_TO_INCOME                                               CREDIT / INCOME                 Loan size relative to income; existing baseline duplicate.
                FIN_GOODS_TO_INCOME                                                GOODS / INCOME                                            Asset price relative to income.
              FIN_ANNUITY_TO_INCOME                                              ANNUITY / INCOME                   Annuity relative to income; existing baseline duplicate.
              FIN_ANNUITY_TO_CREDIT                                              ANNUITY / CREDIT                            Annuity intensity; existing baseline duplicate.
              FIN_CREDIT_TO_ANNUITY                                              CREDIT / ANNUITY Payment-multiple proxy; existing baseline duplicate and not a stated term.
       FIN_ANNUITY_CREDIT_LOG_RATIO                                log1p(ANNUITY) - log1p(CREDIT)                                         Log-scale payment/credit relation.
              FIN_CREDIT_PER_FAMILY                                      CREDIT / CNT_FAM_MEMBERS     Mechanical per-member allocation; assumption not economically literal.
                 FIN_GAP_PER_FAMILY                            (CREDIT - GOODS) / CNT_FAM_MEMBERS                         Mechanical per-member allocation of financing gap.
             FIN_ANNUITY_PER_FAMILY                                     ANNUITY / CNT_FAM_MEMBERS              Mechanical equal-responsibility assumption; exploratory only.
            FIN_CREDIT_PER_AGE_YEAR                                 CREDIT / (-DAYS_BIRTH/365.25)         Age-scaled loan amount; unit has no direct lending interpretation.
           FIN_ANNUITY_PER_AGE_YEAR                                ANNUITY / (-DAYS_BIRTH/365.25)             Age-scaled annuity; unit has no direct lending interpretation.
                            PROD_Q3                                      round(ANNUITY/CREDIT, 3)                                      v17 Q3-style rounded repayment ratio.
                            PROD_Q4                                      round(ANNUITY/CREDIT, 4)                                      v17 Q4-style rounded repayment ratio.
                            PROD_Q5                                      round(ANNUITY/CREDIT, 5)                                      v17 Q5-style rounded repayment ratio.
                            PROD_P3                            round((ANNUITY - F_fit)/CREDIT, 3)                 Fee-adjusted ratio; F_fit estimated on training fold only.
                            PROD_P4                            round((ANNUITY - F_fit)/CREDIT, 4)                 Fee-adjusted ratio; F_fit estimated on training fold only.
                            PROD_P5                            round((ANNUITY - F_fit)/CREDIT, 5)                 Fee-adjusted ratio; F_fit estimated on training fold only.
                      PROD_PF4_FLAG fit-only repeat-frequency(P4) > fit-only repeat-frequency(Q5)            v17-style relative support flag, recomputed from fit rows only.
          PROD_FEE_ADJUSTED_ANNUITY                                               ANNUITY - F_fit                       Annuity after fold-estimated candidate fixed offset.
   PROD_FEE_RECONSTRUCTION_RESIDUAL                A - (F_fit + CREDIT*round((A-F_fit)/CREDIT,5))          Absolute reconstruction residual from rounded fee-adjusted ratio.
          PROD_Q5_FIT_FREQUENCY_LOG                                log1p(count_fit(round(A/C,5)))                                  Training-fold-only product-key frequency.
          PROD_P5_FIT_FREQUENCY_LOG                        log1p(count_fit(round((A-F_fit)/C,5)))                     Training-fold-only fee-adjusted product-key frequency.
                      PROD_FEE_FLAG fit-only repeat-frequency(P5) > fit-only repeat-frequency(Q5)                                      Candidate fixed-fee family indicator.
                     PROD_K_DECIMAL                                  (CREDIT/GOODS - 1)/delta_fit v17 K_DEC-style financing lattice coordinate; delta estimated on fit only.
             PROD_K_NEAREST_INTEGER      round(PROD_K_DECIMAL) if cash and lattice residual < 0.1                   Candidate integer financing tier; validity gate applied.
            PROD_K_LATTICE_RESIDUAL                   abs(PROD_K_DECIMAL - round(PROD_K_DECIMAL))                                    Distance to inferred financing lattice.
                       PROD_K_VALID                    cash & CREDIT>GOODS & lattice residual<0.1             Indicator for rows consistent with inferred financing lattice.
           PROD_K_FIT_FREQUENCY_LOG                       log1p(count_fit(round(PROD_K_DECIMAL)))                                 Training-fold-only integer-tier frequency.
      PROD_EXT_MEAN_X_FINANCE_SHARE                              EXT_MEAN * (CREDIT-GOODS)/CREDIT                    External risk score conditional on financing intensity.
PROD_EXT_DISPERSION_X_FINANCE_SHARE                               EXT_STD * (CREDIT-GOODS)/CREDIT            External-score disagreement conditional on financing intensity.
                  PROD_EXT_MEAN_X_K                                     EXT_MEAN * PROD_K_DECIMAL                External risk score conditional on inferred financing tier.
            PROD_EXT_DISPERSION_X_K                                      EXT_STD * PROD_K_DECIMAL                 Score disagreement conditional on inferred financing tier.
             PROD_CREDIT_INCOME_X_K                              (CREDIT/INCOME) * PROD_K_DECIMAL                      Income burden conditional on inferred financing tier.
            PROD_ANNUITY_INCOME_X_K                             (ANNUITY/INCOME) * PROD_K_DECIMAL                     Payment burden conditional on inferred financing tier.
       PROD_CREDIT_GOODS_X_EXT_MEAN                                     (CREDIT/GOODS) * EXT_MEAN                          External score conditional on credit/goods ratio.
     PROD_ANNUITY_CREDIT_X_EXT_MEAN                                   (ANNUITY/CREDIT) * EXT_MEAN                        External score conditional on annuity/credit ratio.
           PROD_FINANCE_SHARE_X_AGE                                     finance share * age years                                                Financing intensity by age.
        PROD_FINANCE_SHARE_X_FAMILY                               finance share * CNT_FAM_MEMBERS              Financing intensity by household size; not per-member income.
       PROD_ANNUITY_INCOME_X_FAMILY                            (ANNUITY/INCOME) * CNT_FAM_MEMBERS               Repayment burden interaction; not an equal-share assumption.

## CV family result

                                experiment  pooled_oof_auc  delta_vs_raw_baseline  mean_fold_auc  feature_count_added                                    status
           verified_raw_baseline_saved_oof        0.760658               0.000000       0.760729                    0           Historical reference reproduced
batch1_loan_product_and_repayment_features        0.761353               0.000694       0.761438                   38 Discovery CV; confirmation still required

The model family contains 38 distinct numeric columns after excluding exact aliases and baseline-duplicate ratios. It is one family-level ablation, not 39 individually tested ablations. The result uses the historical fixed five folds and is a discovery measurement, not an untouched confirmation.

## Fold-local decoder estimates and held-out key coverage

 fold  fee_estimate_fit_only  fee_fit_repeat_support  fee_constant_v17  fixed_fee_adjusted_q5_seen_in_train_pct  unadjusted_q5_seen_in_train_pct  step_estimate_fit_only  step_fit_lattice_support  step_constant_v17  validation_v17_step_lattice_support  validation_cash_credit_gt_goods_n  validation_k_within_0_1_of_integer_pct
    1                  127.5                0.968133             129.5                                 0.991288                         0.993902                  0.0066                  0.948325             0.0132                             0.897620                              22231                                0.948720
    2                  142.5                0.968267             129.5                                 0.991494                         0.992982                  0.0066                  0.948300             0.0132                             0.897194                              22061                                0.947690
    3                  130.0                0.969100             129.5                                 0.991247                         0.992856                  0.0066                  0.949500             0.0132                             0.895528                              22226                                0.948169
    4                  123.0                0.967200             129.5                                 0.990935                         0.992677                  0.0066                  0.948325             0.0132                             0.895111                              22252                                0.946567
    5                  124.0                0.968200             129.5                                 0.991678                         0.993614                  0.0066                  0.947375             0.0132                             0.901656                              22167                                0.950918

Fee estimation maximized within-fit duplicate support for rounded 5-decimal `(ANNUITY-F)/CREDIT` keys over F in [0, 250] at 0.5 increments. The financing increment maximized the within-fit proportion of cash-loan `(CREDIT/GOODS-1)/delta` values within 0.1 of an integer over a predeclared grid, with a local refinement. These are exploratory unsupervised fit objectives, not proof of an actual fee/tariff or financing mechanism.

## Interpretation limits

- A repeated rounded ratio may be caused by amount rounding, common standardized product terms, or finite precision; it is not itself evidence of a separate financial product.
- Annuity differences and ratios are not labeled with a time period in the official dictionary. No annualization or repayment-term claim is supported.
- `K_DEC` is only a lattice coordinate; no borrower-facing interpretation is established.
- A fixed fee/step estimate is accepted only if it is stable across folds and predicts repeated structure on held-out rows. See `PRODUCT_KEY_FOLD_STABILITY.csv`, `PRODUCT_DECODING_RESULTS.csv`, and `PRODUCT_FORMULA_RECONSTRUCTION_ERRORS.csv`.
- The family AUC is not a final claim of improvement; repeated discovery on these folds can overfit. All features remain on hold until a separately frozen confirmation.
