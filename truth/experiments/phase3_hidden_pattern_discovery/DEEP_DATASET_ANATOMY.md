# Dataset anatomy — Phase 3 Batch 1

Training shape: (171202, 34); test shape: (61500, 33). Training predictors: 33; test predictors: 33. Positive rate: 0.08072920. Dictionary descriptions and numeric/categorical summary statistics for every training column are in `ALL_COLUMN_STATISTICS.csv`.

The dictionary describes credit, annuity, goods price, income, family size, external scores, and bureau inquiries. Importantly, AMT_ANNUITY has no declared frequency, and the bureau windows have distinct intervals; they are not summed into a rate in this batch.

Numeric pairwise Spearman correlations with |rho| >= 0.80 are listed in `COLUMN_DEPENDENCY_REPORT.csv`. Such correlations are screening signals, not proof that one field is mathematically derived from another. Sentinel, missingness, zeros, quantiles, IQR outlier fraction, and TARGET-default rate are recorded per feature in the statistics file.

## Structural themes examined

- Loan structure: annuity/credit, credit/goods, goods/credit, difference, financing share, and their rounded/repeated forms.
- Candidate fixed-fee adjustment: `(ANNUITY - F_fit) / CREDIT`, with fee chosen on outer-fit rows.
- Candidate financing increment: `(CREDIT/GOODS - 1) / delta_fit`, with delta chosen on outer-fit cash loans.
- Repayment / affordability: annuity-to-income, annuity-to-goods, financing gap relative to income/annuity, and household-size interactions.
- Product × creditworthiness: external score summary and dispersion crossed with financing gap or candidate lattice coordinate.

No causal interpretation or inferred applicant identity is attempted. `MATHEMATICAL_RELATIONSHIP_CATALOG.csv` records exact algebraic dependencies and caveats.
