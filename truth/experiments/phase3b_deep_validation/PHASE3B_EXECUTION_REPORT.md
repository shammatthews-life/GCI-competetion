# Phase 3B execution report — deep product-structure validation

**Status:** completed for score reproduction, fold-local structural diagnostics, the predeclared product ablation/control matrix, and selected formula tests. This is not a claim that the entire competition research program is complete or that any feature is promoted.

## Reproduced OOF references

Train rows: 171,202; target prevalence: 8.073%; supplied test rows were not used for fitting or validation. All saved OOF sets were joined by applicant ID and checked for exact ID coverage, TARGET alignment, fold alignment, and finite predictions.

                       variant  pooled_oof_auc  mean_fold_auc  std_fold_auc
            corrected_baseline        0.760658       0.760729      0.003770
           product_decoder_all        0.762826       0.762902      0.003929
product_decoder_all_components        0.762826       0.762902      0.003929
                k_lattice_only        0.761012       0.761097      0.003699
     q_rounded_ratio_keys_only        0.760939       0.761049      0.003169
      p_fee_adjusted_keys_only        0.762184       0.762280      0.003591
           p_fee_fit_estimated        0.762184       0.762280      0.003591
             p_fee_fixed_129_5        0.761656       0.761768      0.003426
          q_unadjusted_control        0.760939       0.761049      0.003169

The corrected saved raw baseline reproduces at **0.760658450595**. The previously reported product-decoder OOF reproduces at **0.762826418435**, a difference of **+0.002168** from that saved baseline. The latter is a reused-fold discovery score, not a public leaderboard result or independent confirmation.

## Phase 3B controlled model comparison

The in-run raw-only baseline was refit on the same saved outer folds. Each feature family used the same fold-specific boosting-round count selected on one inner 10% split using raw baseline features only. The identical count was applied to every family in that fold. Candidate feature maps and unsupervised parameters were estimated on the current outer-training rows only.

                 variant  pooled_oof_auc  delta_vs_baseline  mean_fold_auc  std_fold_auc  folds_improved
             all_product        0.762444           0.001786       0.762508      0.004160               5
       annuity_perturbed        0.762145           0.001487       0.762216      0.004002               4
                       P        0.762075           0.001416       0.762139      0.003575               4
            no_redundant        0.762059           0.001401       0.762117      0.004138               5
               rare_drop        0.762026           0.001367       0.762091      0.003993               5
                     Q+P        0.761912           0.001253       0.761986      0.003933               5
              rare_group        0.761765           0.001107       0.761841      0.003953               4
                     K+P        0.761687           0.001028       0.761751      0.003520               5
            random_fee_P        0.761290           0.000632       0.761346      0.004092               4
                 q_floor        0.761188           0.000529       0.761249      0.003712               3
                       Q        0.761158           0.000499       0.761234      0.003737               4
                       K        0.761059           0.000400       0.761130      0.003634               4
               stable_QK        0.760985           0.000326       0.761043      0.003880               3
                     K+Q        0.760985           0.000326       0.761043      0.003880               3
           random_step_K        0.760938           0.000280       0.761004      0.003466               4
      raw_baseline_refit        0.760658           0.000000       0.760729      0.003770               0
saved_baseline_reference        0.760658           0.000000       0.760729      0.003770               0
            new_formulas        0.760656          -0.000003       0.760728      0.004022               2
         random_permuted        0.758833          -0.001825       0.758906      0.003446               0
             random_keys        0.758805          -0.001853       0.758855      0.003718               0

Best non-control Phase 3B candidate: **all_product: 0.762444 (delta vs in-run raw baseline +0.001786; 5 of 5 folds improved)**. In-run raw baseline AUC: **0.760658** (saved baseline reference: 0.760658). Product-only family AUC: **0.762444** (delta vs in-run baseline +0.001786).

### Negative controls

        variant  pooled_oof_auc  delta_vs_baseline  folds_improved
   random_fee_P        0.761290           0.000632               4
  random_step_K        0.760938           0.000280               4
random_permuted        0.758833          -0.001825               0
    random_keys        0.758805          -0.001853               0

The target-free reconstruction controls are recorded separately in `NEGATIVE_CONTROL_RESULTS.csv`. Arbitrary fee constants have near-baseline repeated-key coverage, so key repetition alone does not identify a literal fee. Permuted/random keys and arbitrary parameter controls are not all equivalent to a valid financial mechanism; their model results are a check against generic added-column effects, not a causal test.

## Structural interpretation

- **Fee hypothesis:** fit-mode fee mean 129.40, range 123.00–142.50; 60%-subsample median estimates range 125.50–127.00. Multiple contract/credit regimes and robust subsample estimates are documented in `FEE_HYPOTHESIS_ANALYSIS.csv`. Held-out repeated-key coverage differences are small relative to their high base level. Random fee constants often produce similar coverage. This is not credible evidence for a fixed contractual fee; at most P-family reparameterizations may be predictive.
- **K financing lattice:** 0.0066 fit/validation support 94.84%/94.84%, mean reconstruction MAE 114.47, mean p95 error 957.24, odd-coordinate share 5.40%; versus 0.0132 support 89.74%/89.74%. The coarser 0.0066 grid is a better numerical description than 0.0132 on the predeclared fit/validation diagnostics, but exact support is not universal and extreme reconstruction errors remain. Since 0.0132 is twice 0.0066, its coordinate grid admits only a subset of the finer-grid points. This does not establish a primitive contract increment, business meaning, or causal repayment mechanism.
- **Reconstruction:**                                valid  median_abs_error        mae  p95_abs_error  exact_pct  within_cent_pct
component                                                                                                   
K_fit_step_credit_over_goods  154988      1.164153e-10  81.937172       1.800000  78.757636        78.757636
P5_fit_fee_annuity            154988      1.200000e+00   1.541456       4.204062   0.000000         0.356043
Q5_unadjusted_annuity         154988      1.006312e+00   1.448463       4.500000  11.146112        11.770705 The lattice's high near-integer-coordinate share must not be confused with exact money reconstruction; the complete error quantiles and fold estimates are in `PRODUCT_RECONSTRUCTION_STABILITY.csv`.
- **Applicant strata:** group-conditioned sample sizes, key coverage, and errors are recorded in `PRODUCT_GROUP_STABILITY.csv`. Coverage is lowest in high-cardinality organization/occupation groups and remains high for low-cardinality K keys; raw key coverage is therefore not evidence of group-specific contracts.
- **Redundancy and interpretation:** product features largely re-express supplied credit, goods-price, and annuity fields. In particular, fee-adjusted annuity is a constant translation of the raw annuity, K-decimal is an affine transform of credit/goods, and Q/P precisions are quantized ratio variants. A tree model may use these representations differently, but they do not prove a new underlying loan product.
- **Univariate associations and mutual information in `PRODUCT_FEATURE_DIAGNOSTICS.csv` are descriptive only** and were not used to select formulas. They must not be interpreted as incremental feature value. Fold importance was collected only for the all-product model.

## Additional formula research

Candidate formula evidence, definitions, missingness, infinities, redundancy, and semantic decisions are in `NEW_FORMULA_HYPOTHESES.csv` and `NEW_FORMULA_VALIDATION.csv`. Exact duplicates of baseline features are not treated as new discoveries. Bureau inquiry windows overlap; summing them double-counts events, and month/year ratios are not rates. These are excluded from incremental modeling. Annuity frequency remains unspecified by the local dictionary.

                  feature_name  pooled_oof_auc  pooled_delta_vs_refit_baseline  folds_improved
NEW_CREDIT_GOODS_GAP_TO_INCOME        0.760468                       -0.000191               2
           NEW_GOODS_TO_INCOME        0.760643                       -0.000015               2
          NEW_ANNUITY_TO_GOODS        0.760633                       -0.000026               2
    NEW_CREDIT_GOODS_LOG_RATIO        0.760317                       -0.000341               2

The four individually re-tested ratios overlap with earlier Phase 3 financial-formula work; the refreshed fixed-round ablations are not independent replications. No feature is promoted.

## Compliance, validation limits, and decision

- The saved folds have been used repeatedly in earlier research. Nested outer/inner processing prevents outer-label use during transformations and stopping-round selection, but researcher exposure remains. A new random split of this same dataset would not erase that exposure.
- This is a fixed five-fold discovery evaluation with inner raw-baseline early stopping, not a fresh independent test. A genuinely independent confirmation requires previously unseen labelled applicants.
- The competition test set is unlabeled and was not used. No pooled train/test statistics, target encoding, external data, hand-created labels, inferred applicant identity, ensemble tuning, or submission generation was performed.
- The local rule audit says pooled test-covariate statistics and target encoding are unresolved; both remain disabled. No live organizer amendment was verified during this work.
- The requested **0.790 AUC target is not verified or reached**. The Phase 3B best candidate is all_product: 0.762444 (delta vs in-run raw baseline +0.001786; 5 of 5 folds improved); its gap to 0.790 is 0.027556 AUC points. The existing Phase 3 discovery remains 0.762826.

**Recommendation:** do not change the champion. Freeze the best predeclared candidate family and obtain genuinely new labelled confirmation (or organizer-approved unseen validation data) before making a promotion decision. Do not start ensemble/hyperparameter tuning based on these reused folds.

## Reproducibility and artifacts

Execution logs, commands, timestamps, software versions, source/data hashes, model outputs, fold-level results, and errors are recorded in `EXECUTION_LOG.txt`, `INPUT_FILES_SHA256.txt`, and the `environment.json` files. The manifest hashes every Phase 3B file except itself. The initial ablation attempt failed due to a boolean-index alignment bug in the floor-key transform; it produced no final ablation artifacts. The index handling was corrected and the complete matrix was rerun. No historical project output was modified.
