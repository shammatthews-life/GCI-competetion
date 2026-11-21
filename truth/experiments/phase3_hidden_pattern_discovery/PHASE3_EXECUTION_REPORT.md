# Phase 3 execution report — first milestone

Status: **completed for reference verification and Batch 1 discovery; not a complete Phase 3 program.**

Execution: 2026-11-03T13:57:44.871820+00:00 to 2026-11-03T13:59:27.328469+00:00 UTC; runtime 102.5 seconds. Code SHA-256: `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`. Inputs and environment are recorded in `INPUT_DATA_HASHES.txt` and `environment.json`.

## Historical baseline verification

                    variant  pooled_oof_auc  mean_fold_auc  std_fold_auc  fold_1_auc  fold_2_auc  fold_3_auc  fold_4_auc  fold_5_auc  claimed_historical_auc  absolute_difference_from_claim  reproduced
               raw_baseline        0.760658       0.760729      0.003770    0.757195    0.761577    0.766257    0.757183    0.761433                0.760658                             0.0        True
   existing_domain_formulas        0.760372       0.760452      0.003521    0.757389    0.762220    0.765294    0.756762    0.760597                0.760372                             0.0        True
     peer_relative_features        0.760693       0.760800      0.003309    0.758953    0.762216    0.764975    0.756288    0.761570                0.760693                             0.0        True
external_score_interactions        0.760058       0.760178      0.003202    0.757824    0.760804    0.764583    0.756380    0.761300                0.760058                             0.0        True
           combined_phase2b        0.759909       0.759994      0.003633    0.757609    0.760212    0.765481    0.755942    0.760727                0.759909                             0.0        True

Saved OOF IDs and labels matched the supplied training file; fold IDs matched the corrected five-fold assignment. This reproduces the saved metrics, not an independent rerun of the historical training script. Complete 23-row v17 reuse register was read and confirmed before running.

## Batch 1 result

                                experiment  pooled_oof_auc  delta_vs_raw_baseline  mean_fold_auc  feature_count_added                                    status
           verified_raw_baseline_saved_oof        0.760658               0.000000       0.760729                    0           Historical reference reproduced
batch1_loan_product_and_repayment_features        0.761353               0.000694       0.761438                   38 Discovery CV; confirmation still required

The added family uses 38 engineered numeric features, with unknown fee and financing-step parameters estimated independently on each applicable training split. Outer validation rows were not used for feature fitting, product-key counts, early stopping, or parameter estimation. Iterations were selected on an inner stratified split of each outer training fold. OOF predictions are saved in `PHASE3_PRODUCT_OOF_PREDICTIONS.csv`.

## Scientific interpretation

Batch 1 tests the v17 fixed-fee and financing-lattice ideas rather than copying the constants. The output files show decoder stability, held-out repeated-key coverage, and reconstruction errors. Even a high lattice support is not proof of the business meaning; evidence must be stable and out-of-sample. The exact product formula result is not promoted if its OOF family delta is negligible, unstable, or negative.

All 51 listed candidate formulas were implemented for diagnostics; duplicates and unsuitable aliases are marked in `DOMAIN_FORMULA_CATALOG_V4.csv`. Only the predeclared nonduplicate feature family was evaluated in the model. Individual formula AUCs are exploratory held-out univariate diagnostics, not individual model improvements. No individual feature is promoted.

## Compliance and limitations

No external datasets, hand-created labels, target encoding, pooled train/test statistics, repeat-applicant inference, or test labels were used. Frequency maps and decoder parameters were fitted only on the current training partition. See `PHASE3_RULE_COMPLIANCE.md`.

This is not an untouched confirmation because the historical folds have been used by prior discovery work. A frozen confirmation split or nested validation is required before promotion. No 0.79+ result is claimed, no test inference/submission was produced, and future batches are not run or marked complete.

## Produced outputs

See the Phase 3 folder for reference verification, rule matrix, full predictor statistics, relationship/catalog files, decoder diagnostics, formula registry, fold-level results, and reproducible runner. Runtime log is `EXPERIMENT_LOG.md`.
