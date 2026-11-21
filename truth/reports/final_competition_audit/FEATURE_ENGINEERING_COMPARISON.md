# Feature-engineering and method comparison

This is a provenance and validation comparison, not a performance ranking.
“Compliant” means consistent with the local provided-data rule at the
feature-source level; it does not certify all competition rules or validate
an OOF estimate.

| Method / source | Inputs and label use | Fold-local requirement | Compliance and evidence | TRUTH evidence / decision |
|---|---|---|---|---|
| EXP02 financial ratios | `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, `CNT_FAM_MEMBERS`; no labels in formula. | No learned state; deterministic row-wise formulas, guarded denominators. | Allowed feature derivation from supplied columns. No external constants. | Tested once in EXP02: +0.00722884 pooled OOF, 5/5 folds. Retain only as provisional pending independent seed. |
| Friend financial and demographic ratios | Supplied financial amounts, family count, age/employment/registration/phone fields; no labels directly in formulas. | Any clipping/learned transforms must be fixed from permitted training data and fitted within each outer fold. | Source columns are permitted; hard-coded clipping and insurance `1.001` threshold lack derivation records. | Do not copy wholesale. Reconsider one formula at a time after source and validation controls are frozen. |
| EXT_SOURCE summary, missingness, interactions | Supplied `EXT_SOURCE_1/2/3` and supplied financial/demographic fields; no labels for row-wise formulas. | No fold fitting for arithmetic; learned transforms must be fold-local. | Provided columns are distinct from downloading underlying external data. Fixed weights `0.28/0.44/0.28` are unexplained. | HOLD pending predeclared rationale and isolated validation. |
| Category label/frequency features in friend notebook | Supplied categorical columns; no target for label/frequency codes. | Fit category vocabulary/counts on each outer training fold, then transform held-out rows. | Combined train/test vocabulary and frequencies are transductive; local rule does not decide permissibility. | HOLD; do not use pooled statistics without written clarification. |
| Friend OOF target encoding | Supplied category columns plus supplied training `TARGET`. | Must be nested: generate outer-training encodings without the outer validation labels; validation/test values only from outer-fit labels. | Target encoding is not explicitly prohibited, but this implementation's global OOF encoding is not nested in its outer model folds. | Reimplement only if needed; the current friend's OOF evidence is invalid. |
| Group mean / z-score features | Supplied organization and education/occupation categories with supplied `EXT_SOURCE`, credit/annuity, and income values; no target in means. | Compute group statistics from the outer training fold only, or obtain approval for test-covariate fitting. | Combined train/test group statistics are transductive and unresolved. | HOLD; not demonstrated in EXP02. |
| PCA / unsupervised UMAP | Derived supplied numerical columns; no labels for unsupervised transform. | Imputer, quantile transform, scaler, PCA, and UMAP must fit on outer training rows only. | Transductive train/test pooling is unresolved; current full-train fitting leaks validation covariates into OOF features. | Do not reuse as written. |
| Supervised UMAP / positive landmarks | Supplied numeric features and training labels. | Fit inside each outer training fold, excluding outer validation labels and rows. | Uses permitted labels in principle, but current embedding is fit on a full-training subset before outer CV. | Reject reported AUC as unbiased; reimplementation not prioritized. |
| Entity matching / peer graph | Friend archive's 226-feature matrix and ID-agnostic attribute combinations (birth/registration/ID publication, gender, education); risk stream predictions. | Reconstruct pairs from outer-fit data only and prove peer predictions do not carry held-out labels. | Feature source and cache lineage unknown. Selected-row test score coupling may be construed as a prohibited EDA-rule decision. | Do not use pending full provenance and written organizer clarification. |
| Probability/probit/rank blending and meta-stacking | OOF/test prediction arrays and training labels. | OOF model predictions and meta features must be regenerated nested within each outer fold; blend weights must not be tuned on the reported evaluation labels. | No blanket blend ban in local rules; friend code tunes final weights on the same full OOF labels it reports. | Do not trust the claimed OOF metrics or re-use these weights. |
| Historical Phase 2 fee/product hypotheses | Fee/product values involving `AMT_ANNUITY`, `AMT_CREDIT`, and fixed values `129.5`, `0.0132`. | Document derivation from permitted columns; fold-local if data-fitted. | Constants lack verified permitted-data derivation. | Exclude from competition candidates unless derivation is demonstrated. |
| External prediction caches / precomputed matrices | `.npy`/`.npz` model streams, `meta.json`, feature matrix, labels, test IDs; source and row provenance absent. | Require raw-data lineage, model code, folds, IDs, and hashes. | Could be compliant if derived solely from official inputs, but that is not proven. External-data rule makes this a blocking provenance issue. | Reject/hold; do not copy any arrays or final output. |

## Recommendations

1. Preserve EXP02's current artifact and do not claim a leaderboard result.
2. Do not reuse the friend's caches, submitted predictions, graph correction,
   or validation metrics.
3. If organizers confirm test-covariate fitting and score coupling, still
   repair the friend notebook's nested CV, early stopping, and stacking before
   evaluating any individual feature family.
4. Keep every future experiment's formula, constants, data hashes, fold
   assignments, and transformation-fit boundaries explicit.
