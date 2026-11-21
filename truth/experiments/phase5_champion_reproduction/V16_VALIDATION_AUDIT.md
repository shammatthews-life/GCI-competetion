# V16 validation and leakage audit

## Positive controls

- The source removes `TARGET` before building the common feature frame.
- Outer model OOF uses `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Training encodings use nested `StratifiedKFold(5, shuffle=True, random_state=7)` inside the outer training rows. Outer validation target labels are not used in `enc(a,b)`.
- The partner-label encoding for CatBoost is nested within the outer training set; validation and test encodings use only the outer training labels.
- V16 uses fixed training iteration counts, not outer-validation early stopping.
- The optional regime-aware blending cell cross-fits its blend weights across two halves rather than fitting and scoring weights on the same half.

## Validation limitations

1. `al` concatenates train and test before category coding, value counts, group means, product frequency selection, entity fingerprints, and residual fitting. Outer validation covariates therefore contribute to some V16 features. These are transductive transforms, not fold-isolated OOF feature construction.
2. The residual learner uses KFold over the concatenated predictor rows, not the outer target folds; no outside labels are used, but the outer-validation covariates affect its fitted transforms.
3. V16 cache reads accept matching filenames without validating hashes, shapes, fold maps, feature order, or configuration. The configured caches were not present in the audited local locations.
4. The displayed `0.77649` OOF result and feature ablations are saved notebook output, not independently reproduced output. The code/source hash matches the unexecuted copy, but no saved OOF arrays or cache manifest tie results to the exact code/data.
5. Repeated historical inspection of these same labels and OOF scores increases selection exposure. Any future result should be labelled exploratory unless confirmed on a genuinely untouched evaluation set.

## Competition rules

The local official-rules audit says no external data, no hand-labeling, and ROC-AUC evaluation. It does not settle pooled unlabeled train/test fitting or target encoding. Thus V16's supervised encodings have good fold isolation in their implementation, but rule permission remains unresolved; pooled covariate transforms also require organizer clarification. The EXP85 test-to-test score coupling has additional high-risk ambiguity.

## Decision

V16 does not get a champion-improvement claim. The score and saved ablations remain unverified, the cache provenance is unavailable, and the exact EXP85 final OOF comparator is missing. No model was trained and no submission or champion file was modified.
