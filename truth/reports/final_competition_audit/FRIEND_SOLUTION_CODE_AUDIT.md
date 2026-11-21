# Friend solution code audit

## Scope and handling

Audited the user-provided `GCI_UMAP_Research_Notebook%20(1).ipynb` and the
source files in `final_submission.rar`. The notebook has eight cells, four
Python code cells, no executed outputs, and all code cells pass static Python
syntax parsing. The RAR contains `README.md`, `reproduce_final_submission.py`,
and `submission_final.csv`. Only the README and Python script were read in a
temporary directory; **the bundled submission CSV and its prediction values
were not opened**. Neither code artifact was executed.

`Competition Log.pdf` was not used as a rule source and its content was not
extracted. The README's leaderboard and OOF figures are unverified claims,
not independently reproduced evidence.

## Finding

**The friend solution does not pass this audit gate and is not cleared for
submission or reuse.** This is not a finding that the friend intentionally
used outside data. Rather, the code has material validation leakage, uses a
prediction-adjustment rule that may conflict with the hand-labeling rule, and
depends on artifacts whose origin cannot be traced from the supplied source
package.

### UMAP/PCA notebook

1. **Input selection is not restricted to the official files.** `find_csv`
   searches `/input`, `/content/data`, `/content`, the current directory and
   parent directory and returns the first sufficiently large CSV matching a
   name/header pattern. There is no official-file SHA-256 pin or unambiguous
   path check. No external-data read is proven, but the code cannot prove
   which files it actually selected.
   No explicit HTTP client or download call was found in the inspected source;
   this does not establish the provenance of the precomputed caches.
2. **Train and test are pooled during feature construction.** The notebook
   concatenates raw train/test rows, then computes category encodings,
   category frequencies, and group means/standard deviations on the combined
   frame. This is unlabeled test-covariate use, not proof of outside data.
   The provided rules do not explicitly resolve whether this transductive
   fitting is permitted.
3. **Global preprocessing leaks outer-fold covariates.** Imputation,
   quantile transformation, and scaling are fitted on all training rows
   before the later CV. PCA is also fit before the outer models. These
   validation covariates therefore affect the derived features used to claim
   OOF performance.
4. **Target encoding is not nested inside model folds.** The notebook creates
   OOF encodings with one global `StratifiedKFold`, then uses those columns
   in another CV. An outer validation row's encoding can use labels from
   other outer-validation rows; outer-training encodings may also use labels
   belonging to the outer validation fold. This is target leakage for the
   reported outer-fold AUC.
5. **Supervised UMAP is fit before outer CV.** It uses `y_full[sub_idx]` from
   the full training dataset, so labels from later outer validation folds
   influence their own representation and scores. The fallback's positive
   landmarks are likewise selected using all training labels.
6. **Early stopping uses the reported validation fold.** LightGBM, XGBoost,
   and CatBoost use `Xva,yva` for early stopping and then report AUC on those
   same rows. The reported outer-fold estimate is selected against its own
   labels.
7. **Stacking and blend scores are not an independent OOF estimate.** Base
   OOF predictions are made in a separate CV, then used to train/score meta
   models in another CV without regenerating base predictions inside each
   meta-training fold. Meta-validation labels can therefore have influenced
   meta-training base predictions. Additionally, final blend weights are
   grid-searched using `roc_auc_score(y_full, candidate_train)` and the
   selected blend's AUC is reported on those same labels. This is direct
   in-sample selection optimism.
8. **Some behavior is silently optional.** A broad `except Exception: pass`
   around supervised UMAP can omit those features without recording a failed
   run. `warnings.filterwarnings("ignore")` further reduces diagnostic
   visibility.
9. **Feature constants lack derivation records.** Clipping cutoffs,
   `DAYS_EMPLOYED` handling, the `1.001` insurance threshold, EXT_SOURCE
   weights, target-encoding smoothing, and other fixed coefficients are
   embedded in code or prose without a source or predeclared derivation in
   the supplied notebook. A constant is not automatically prohibited, but
   its permitted-data provenance and rationale are unverified.

### `final_submission` reproduction script

The README claims public ROC-AUC `0.80041` and OOF ROC-AUC `0.802330`. These
numbers were not verified. The reproduction script is **not a standalone
raw-data training pipeline**: it loads `y.npy`, `test_ids.npy`, a 226-feature
matrix, `meta.json`, relational streams, and numerous model OOF/test
prediction archives, then synthesizes a final output. It does not create
those model streams or feature matrices.

The required cache families include:

- `group_a_oof_10stream.npy`, `group_a_test_10stream.npy`, `meta.json`.
- `matrix_226_features_buhlmann_tier2.npz`, `y.npy`, `test_ids.npy`.
- `oof_and_test_*` archives for neural, TabM, ResNet, DAE, GBDT, Buhlmann,
  heterogeneous-zoo, and grand-synthesis streams.

The RAR does not include these upstream inputs or their training entry
points. The script resolves files from developer-specific absolute paths and
fallback locations, without a manifest binding IDs, folds, source CSV hashes,
and prediction arrays. Consequently, external-data use and prediction-cache
provenance cannot be ruled out.

The script also:

- Computes ensemble signal from full `y_true` OOF AUCs and uses it to set
  blend weights.
- Fits another layer of blend weights and searches final blend coefficients
  against the same full OOF labels later used to print the final OOF AUC.
- Applies score coupling to selected test rows paired by feature-derived
  entity keys, with a fixed coefficient. It is automated rather than a
  literal hand-entered label, but the rule selects a subset and changes its
  predictions. Because the official text explicitly bans manual EDA-rule
  decisions for part of the predictions, this is a serious unresolved
  eligibility risk; obtain written organizer guidance or omit it.
- Does not explicitly verify test IDs/order against the supplied sample
  submission before writing the CSV.

The script parses successfully but was intentionally not executed. The
provided output CSV was not inspected, so its schema, identity, score, and
hash remain unverified.

## Technique classification

| Method | Classification | Reason |
|---|---|---|
| Ratios and interactions from supplied application columns | **COMPLIANT AND USEFUL in principle** | Uses supplied columns; validate cutoffs and fit learned transforms within outer folds. |
| EXT_SOURCE composites/missingness features | **COMPLIANT AND USEFUL in principle** | The columns are in the supplied CSV; do not fetch underlying external-source data. |
| PCA/unsupervised UMAP | **RULES UNRESOLVED / validation defective as written** | Train/test pooling is not expressly resolved, and global preprocessing crosses outer folds. |
| Target encoding and supervised UMAP | **COMPLIANT BUT INVALIDLY EVALUATED as written** | Uses training labels, but encodings/embeddings are not nested in outer folds. |
| Group frequency/means over pooled train and test | **RULES UNRESOLVED** | Transductive fitting on supplied test covariates; permission absent from local text. |
| Feature-derived entity graph score coupling | **RULES UNRESOLVED — high risk** | Automated, but explicitly adjusts a selected subset using hard-coded EDA-like matching conditions. |
| Multi-model blend/stack | **COMPLIANT IN PRINCIPLE; invalid evidence as written** | No blanket ensemble ban found, but OOF feature construction and weights leak/are selected on claimed evaluation labels. |
| Precomputed prediction/model streams | **UNVERIFIED — do not use** | Their training code, data source, licenses, IDs, fold assignments, and labels are absent. |

## Verdict

**BLOCKED / NOT CLEARED.** Do not copy the final submission, prediction
caches, graph correction, or reported OOF metrics into TRUTH. If an individual
feature family is independently reimplemented, use official CSVs only,
record its exact formula and source, fit every learned transform inside the
outer training fold, and obtain written clarification for transductive
fitting and selected-row graph score changes. No method is endorsed from the
reported public score.
