# Borrower segment discovery

## Scope and safeguards

Training data only; ID and TARGET excluded from imputation, scaling, PCA, K-means, and GMM fitting. TARGET is used only after fitting to describe segment default rates. This is exploratory structure, not evidence that clusters are real borrower classes and not a validation feature. Any cluster feature for a predictive model must be refit within each outer training fold.

- Numeric input predictors: 24.
- PCA explained variance by first components: [0.139533, 0.126793, 0.070295, 0.067132, 0.05, 0.048725, 0.042211, 0.041727, 0.04162, 0.040987].
- K-means: k=4; fit on a fixed random sample of 30,000 training rows in PCA space; ARI across two initializations/seeds on that sample: 0.995160.
- GMM diagonal-covariance BIC diagnostics on 8,000 rows and first six PCs: see `GMM_BIC_DIAGNOSTICS.csv`.
- Hierarchical Ward linkage: computed on 1,500 sampled training rows; no population-level claim is made.
- UMAP unavailable in the current Python environment; not installed or run.
- Interpret clusters as geometric partitions of standardized numerical predictors; not validated real-world segments.

## K-means descriptive profile

```text
                                  method  cluster     n    share  default_rate_descriptive  median_AMT_INCOME_TOTAL  median_AMT_CREDIT  median_AMT_ANNUITY  median_AMT_GOODS_PRICE  median_EXT_SOURCE_1  median_EXT_SOURCE_2  median_EXT_SOURCE_3
KMeans_K4_fit_on_random_train_sample_PCA        0 29709 0.173532                  0.054394                 112500.0           450000.0            21010.50                423000.0             0.731493             0.556166             0.593718
KMeans_K4_fit_on_random_train_sample_PCA        1 37507 0.219080                  0.096435                 135000.0           450000.0            23625.00                445500.0             0.419637             0.540255             0.511892
KMeans_K4_fit_on_random_train_sample_PCA        2 68816 0.401958                  0.097652                 135000.0           386977.5            21192.75                337500.0             0.455598             0.540659             0.508287
KMeans_K4_fit_on_random_train_sample_PCA        3 35170 0.205430                  0.053113                 225000.0          1096020.0            42142.50                927000.0             0.607321             0.636901             0.558507
```

## Decision

Retain segment analysis as a hypothesis generator only. Segment default rates use already-exposed training labels and must not be used to select a segment-specific model without a prespecified fold-local experiment and independent confirmation.
