# Phase 3 milestone update — hidden loan-product discovery

## What was executed

The corrected saved OOF predictions were checked against the supplied train IDs and TARGET, and each historical pooled AUC/fold AUC was recalculated. All five historical values reproduced. The input files, feature source, saved predictions, dictionary, and folds are hashed in `INPUT_DATA_HASHES.txt`; the exact corrected fold assignment is copied into this workspace.

The complete 23-entry v17 reuse register was inspected. It is not valid CSV for generic parsers because several fields contain unquoted commas; this did not affect any experiment. Only its documented ideas were used. All unknown decoder constants and product-key frequencies were fit on the current training partition. Outer validation data were not used for fitting or early stopping.

## Measured comparisons

| Candidate family | Pooled OOF AUC | Delta vs 0.760658 baseline | Improved folds |
|---|---:|---:|---:|
| Corrected raw baseline | 0.760658 | — | — |
| Product decoder only (16 features) | 0.762826 | +0.002168 | 4/5 |
| Financial formulas only (11 features) | 0.760724 | +0.000065 | 2/5 |
| Product/financial interactions only (11 features) | 0.760324 | -0.000334 | 2/5 |
| All Batch 1 candidate features (38 features) | 0.761353 | +0.000694 | 3/5 |

Component checks:

| Product component | Pooled OOF AUC | Delta vs baseline | Improved folds |
|---|---:|---:|---:|
| Fold-estimated financing lattice | 0.761012 | +0.000354 | 4/5 |
| Unadjusted rounded Q ratio keys | 0.760939 | +0.000281 | 3/5 |
| Fold-estimated fee-adjusted P keys | 0.762184 | +0.001526 | 4/5 |
| Fixed 129.5 P keys (v17 hypothesis) | 0.761656 | +0.000997 | 5/5 |

These are paired discovery-fold measurements, not an untouched confirmation. The largest product-only gain is promising but not sufficient for promotion; its fold-delta standard deviation is about 0.00147, and repeated use of these folds can overfit.

## What the product formulas actually show

1. **Q keys alone contribute little** in this model comparison (+0.000281 pooled).
2. **P fee-adjusted representations are the strongest component**. Using a fee estimated only on each outer-fit partition scores +0.001526; using v17's fixed 129.5 scores +0.000997. These results validate an AUC signal, not the claimed fee's business meaning.
3. **The fee constant is not independently identified by key repetition.** Fold-only fitted values are 123, 142.5, 130, 123, and 124, rather than a stable 129.5. In all five held-out folds, repeated-key coverage is slightly better for unadjusted Q5 than for estimated/fixed P5. Thus the feature's model value does not establish a literal fixed fee; the formula may be acting as a useful numerical reparameterization.
4. **The financing lattice is stable but has a half-step ambiguity.** The fit-only grid chose 0.0066 in all five folds. Its held-out integer-proximity support is 94.7–95.1%, versus 89.5–90.2% for 0.0132. The 0.0066 coordinate is exactly twice the 0.0132 coordinate; therefore the data support a finer lattice, but do not yet establish a primitive contract increment or explain its economics.
5. **No annual/monthly repayment term was inferred.** The official dictionary only says “Loan annuity”; it does not identify a period. `CREDIT/ANNUITY` remains a payment multiple, not a contractual term.
6. External/product interactions did not improve in isolation in this batch. The previous Phase 2 external-interaction family also scored below baseline.

Detailed evidence:

- [family and fold-level results](./batch1_family_ablation_02/BATCH1_FAMILY_ABLATION_RESULTS.csv)
- [Q/P/K component results](./batch1_product_component_03/PRODUCT_COMPONENT_ABLATION_RESULTS.csv)
- [fixed versus fit-only fee results](./batch1_fee_constant_04/FEE_CONSTANT_ABLATION_RESULTS.csv)
- [fee key-repeat tests](./batch1_family_ablation_02/FEE_HYPOTHESIS_FOLD_TESTS.csv)
- [financing lattice tests](./batch1_family_ablation_02/FINANCING_LATTICE_FOLD_TESTS.csv)
- [all Phase 3 OOF rows for the family ablation](./batch1_family_ablation_02/BATCH1_FAMILY_OOF_PREDICTIONS.csv)
- [formula-catalog metadata corrections](./DOMAIN_FORMULA_CATALOG_CORRECTIONS.csv)

## Decision and next gate

Keep the product-decoder family as a **research candidate**, not as a champion change. The 0.79 objective remains unmet; the best discovery score here is 0.762826, still about 0.027174 below target.

Before any promotion, freeze the selected formula family and run a genuinely nested/untouched confirmation with the feature-estimation procedure repeated inside every outer training fold. Then compare Q keys, P keys, and K lattice under a predeclared ablation; examine product/contract/income/occupation strata without target-derived features; and test whether the gain is stable by fold and not an artifact of repeated rounded values. Do not start batches 2–10 or tune models until that confirmation is complete.
