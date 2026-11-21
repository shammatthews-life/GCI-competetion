# Hidden-pattern discovery report

## Scope

All analyses below use the `171,202` supplied training rows only. `TARGET` is used for descriptive rates/associations, not for transforming features or fitting unsupervised clusters. The test file was used only for raw schema, ID uniqueness/count, and file hashing. These are exploratory findings on previously inspected labels and do not establish independent predictive lift or causal mechanisms.

## Observations

- The official dictionary defines the core financial fields as client income, loan credit amount, loan annuity, and consumer-loan goods price. It does **not** specify annuity payment period, actual payment duration, interest rate, or household allocation of debt.
- Raw numeric fingerprints, repeated-value shares, heaping, decimal precision, and outlier quantiles are in `NUMERICAL_FINGERPRINT_AUDIT.csv`. Integer/multiplicity structure is descriptive; no inferred contract property is asserted.
- Financial formula relationships and prior-grid hypotheses are in `FINANCIAL_RELATIONSHIP_AUDIT.csv`. The ordinary credit/income, annuity/income, credit/goods, and credit/annuity ratios have already been represented in historical model features; re-adding them is not novel. Lattice-like shares are diagnostics, not contract proof.
- Missingness indicators and common joint patterns with fold-specific default-rate ranges are in `MISSINGNESS_PATTERN_AUDIT.csv`. Sparse-pattern rates are particularly noisy.
- External-score levels, pairwise disagreement, range, dispersion, and missingness configurations are in `EXTERNAL_SCORE_DISCOVERY.csv`. Their univariate AUC and fold variation are not incremental model gains; prior Phase 3C missingness and interaction ablations were near zero or negative.
- PCA/K-means/GMM/hierarchical numerical exploration is in `BORROWER_SEGMENT_DISCOVERY.md` and companion CSVs. Clusters are geometric partitions, not validated borrower classes. Cluster features must be learned inside each model fold.
- Exact duplicate signatures exclude ID and TARGET. Numeric nearest neighbors are sample-only, train-only, and are not entity linkage. Results are in `DUPLICATE_SIMILARITY_AUDIT.csv`.
- Fold-local quantile cutpoints and univariate raw/log/bin AUC summaries are in `NONLINEAR_PATTERN_DISCOVERY.csv` and `NONLINEAR_VALIDATION_BIN_PROFILES.csv`. These are univariate screens, not model-level improvement evidence.

## Pre-existing hidden-pattern evidence

Prior Phase 3 evidence found a product-decoder feature block at `0.762826` discovery OOF and a controlled Phase 3B product model at `0.762444` versus `0.760658` raw; the decoder's purported lattice/fee meaning remains unproven, and seed-42 folds are exposed. Phase 3C equal-weight blend was `0.763957` but selected after inspecting multiple candidates on reused folds. Phase 2B peer deviations gave only `+0.000034`. EXP02 ratio block showed `+0.006336` against its own raw CatBoost baseline on a different seed, but aliases existing Phase 3C ratios; it does not show incremental lift over that stronger baseline.

## Conclusions

1. The most credible signal already observed is combined supplied-field structure around loan/product amounts, but it requires a preregistered incremental ablation against the raw-plus-existing-product baseline.
2. External-score disagreement, missingness, clustering, and generic interactions have not shown convincing incremental lift in prior controlled reports.
3. No new hidden pattern found here is promoted as predictive until it beats an aligned baseline in fold-safe model ablation and survives a genuinely untouched labeled confirmation.
4. Numerical regularities do not prove loan terms, fees, causal risk mechanisms, or applicant identity.
