# Hidden-pattern discovery report

## Scope

All analyses below use the `171,202` supplied training rows only. `TARGET` is used for descriptive rates/associations, not for transforming features or fitting unsupervised clusters. The test file was used only for raw schema, ID uniqueness/count, and file hashing. These are exploratory findings on previously inspected labels and do not establish independent predictive lift or causal mechanisms.

## Observations

- The official dictionary defines the core financial fields as client income, loan credit amount, loan annuity, and consumer-loan goods price. It does **not** specify annuity payment period, actual payment duration, interest rate, or household allocation of debt.
- Raw numeric fingerprints, repeated-value shares, heaping, decimal precision, and outlier quantiles are in `NUMERICAL_FINGERPRINT_AUDIT.csv`. Integer-like shares for the four key financial amount fields are: AMT_INCOME_TOTAL=99.9883%; AMT_CREDIT=100.0000%; AMT_ANNUITY=58.0034%; AMT_GOODS_PRICE=100.0000%. This describes source encoding, not an actual contract price grid.
- The prior product decoder's actual lattice hypothesis is `(AMT_CREDIT / AMT_GOODS_PRICE - 1) / step` for cash-loan rows, not the unadjusted credit/goods ratio divided by step. This run's training-only diagnostic: PRIOR_LATTICE_HYPOTHESIS_0.0066: integer-within-0.01=0.9629, residual<0.1=0.9631, n=154988; PRIOR_LATTICE_HYPOTHESIS_0.0132: integer-within-0.01=0.9264, residual<0.1=0.9266, n=154988. The pattern is worth a carefully controlled ablation, but remains a numerical regularity, not proof of actual financing tiers.
- Ordinary ratio AUCs are descriptive and already represented in historical features. For example, `CREDIT_TO_GOODS` univariate AUC is 0.5681; it is not incremental gain. `FINANCIAL_RELATIONSHIP_AUDIT.csv` records the tested definitions and semantic caveats.
- Leading external-score raw univariate AUCs: EXT_MEAN_AVAILABLE AUC=0.2940; EXT_SOURCE_3 AUC=0.3204; EXT_SOURCE_1 AUC=0.3353; EXT_SOURCE_2 AUC=0.3418. Pairwise disagreements, range, dispersion, interactions, and missingness are in `EXTERNAL_SCORE_DISCOVERY.csv`; these numbers do not establish incremental model gain. Prior Phase 3C interaction/missingness ablations were near zero or negative.
- Strongest single-field missingness indicators from the train-only descriptive screen: AMT_REQ_CREDIT_BUREAU_MON missing=23116 (13.5%), default=10.378%, indicator AUC=0.5210; AMT_REQ_CREDIT_BUREAU_YEAR missing=23116 (13.5%), default=10.378%, indicator AUC=0.5210; AMT_REQ_CREDIT_BUREAU_HOUR missing=23116 (13.5%), default=10.378%, indicator AUC=0.5210; AMT_REQ_CREDIT_BUREAU_QRT missing=23116 (13.5%), default=10.378%, indicator AUC=0.5210. The four credit-bureau request fields share the same missing rows in this dataset; this is one missingness block rather than four independent findings. Sparse-pattern rates remain noisy.
- PCA/K-means/GMM/hierarchical numerical exploration is in `BORROWER_SEGMENT_DISCOVERY.md` and companion CSVs. Clusters are geometric partitions, not validated borrower classes. Cluster features must be learned inside each model fold.
- Exact duplicate signatures excluding ID and TARGET: 0 duplicate groups among 171,202 rows. Numeric nearest-neighbor results are sample-only and are not entity linkage. See `DUPLICATE_SIMILARITY_AUDIT.csv`.
- Leading per-feature nonlinear screens (raw / fold-local bins): EXT_SOURCE_3 raw=0.3204, fold-local bins=0.3235; EXT_SOURCE_1 raw=0.3353, fold-local bins=0.3386; EXT_SOURCE_2 raw=0.3418, fold-local bins=0.3452. Cutpoints are fit per outer training fold. These are univariate screens, not model-level improvement evidence; the monotone log transform cannot create new rank information by itself.
- Four-cluster training-only default rates span 5.311%–9.765%; K-means seed agreement and PCA profiles are exploratory, not proof of stable real borrower types.

## Pre-existing hidden-pattern evidence

Prior Phase 3 evidence found a product-decoder feature block at `0.762826` discovery OOF and a controlled Phase 3B product model at `0.762444` versus `0.760658` raw; the decoder's purported lattice/fee meaning remains unproven, and seed-42 folds are exposed. Phase 3C equal-weight blend was `0.763957` but selected after inspecting multiple candidates on reused folds. Phase 2B peer deviations gave only `+0.000034`. EXP02 ratio block showed `+0.006336` against its own raw CatBoost baseline on a different seed, but aliases existing Phase 3C ratios; it does not show incremental lift over that stronger baseline.

## Conclusions

1. The most credible signal already observed is combined supplied-field structure around loan/product amounts, but it requires a preregistered incremental ablation against the raw-plus-existing-product baseline.
2. External-score disagreement, missingness, clustering, and generic interactions have not shown convincing incremental lift in prior controlled reports.
3. No new hidden pattern found here is promoted as predictive until it beats an aligned baseline in fold-safe model ablation and survives a genuinely untouched labeled confirmation.
4. Numerical regularities do not prove loan terms, fees, causal risk mechanisms, or applicant identity.
