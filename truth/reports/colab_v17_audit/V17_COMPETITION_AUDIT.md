# GCI v17 Colab notebook — competition and reproducibility audit

## Decision

**Do not adopt v17 as the project baseline or run its submission path as-is.**
The notebook contains useful, testable feature hypotheses, but the complete
pipeline is not cleared for competition use or for accepting its reported
validation/public scores. No notebook code was copied into the champion, no
submission was generated, and the original notebook was not modified.

Audited file:

- `D:\exp colab\GCI_v17_optimised.ipynb`
- Size: 96,063 bytes
- SHA-256: `7103fecb0d2b894ab3f4e78c84ba52383ad9e289ce1aa6b9c59aab7921a3c457`
- 25 notebook cells, 12 code cells; every code cell has `execution_count=null`.

The folder also contains v16, v18 ExtraTrees, v19 neural-embedding, and v19
research-candidate notebooks. Their headers/embedded claims were inventoried
to understand the lineage, but they were not fully audited as independent
pipelines.

## Executive findings

| Area | Finding | Decision |
|---|---|---|
| Permitted input files | Notebook's finder searches `/content`, Drive, and the working directory for matching filenames and accepts files by row/column counts. It does not hash inputs or require the supplied competition package. | Reproducibility/provenance **not established**. Pin the exact input files and hashes. |
| External data | No explicit web or third-party dataset download was found in the reviewed v17 cells. However, the notebook can reuse v15 caches and search for precomputed artifacts without manifests that prove their data lineage. | No direct external-data use found; cache provenance **unverified**. |
| Transductive preprocessing | `al = concat(train_without_TARGET, test)` precedes category coding, value counts, group means, repeat-applicant linking, fee decoding and product keys. | Uses test covariates to construct training features. The official text does not clearly authorize this; **ambiguous—avoid until organizer clarification**. |
| Outer-fold isolation | Some features are correctly label-cross-fitted, but feature construction is performed before the outer CV and uses whole train/test covariates. Residual regressors also use continuous source columns as pseudo-targets and are cross-fitted across the pooled dataset, allowing held-out rows' source-feature values to shape training-row residual features. | OOF AUC is **not a clean inductive estimate**. Rebuild all learned transforms inside each outer fit fold. |
| Target encoding | `enc()` counts labels from `fit` only, and `train_enc()` nests encoding over inner folds. This is a promising pattern for the category encodings. | Preserve as a research idea only; fully test all inputs, partner encodings, cache behavior, and fold scopes independently. Competition rule text does not explicitly settle target encoding. |
| Repeat-applicant feature | Candidate identities are inferred from exact/loose combinations of applicant attributes and date differences across pooled train/test rows. `partner_enc()` uses fit labels only, but the entity grouping itself is global and potentially false-match-prone. | **High-risk / unresolved**. Do not use in submission absent organizer interpretation and an independently validated, fold-local implementation. |
| Hard-coded product constants | `129.5` and `0.0132` are embedded in fee/product decoding; no permitted-data derivation or provenance is established in the notebook. | **Unverified—hold**. Do not reproduce as fixed facts. |
| Cached artifacts | Code can reuse `gci_cache_final`/v6 feature files and per-member `.npz` predictions. No hash manifest ties those artifacts to this notebook, official input files, or exact feature code. | Cached predictions/features **not accepted as evidence** until all artifacts are provided, hashed, and traced. |
| Prediction adjustments | The final path generates model/rank blends, not hand-edited individual test labels. Partner matching and hand-specified regime logic are nevertheless custom rules whose compliance should be clarified before use. | No explicit row-level manual score edits found; entity/regime methods remain high-risk. |
| Execution and scores | Stored output/markdown claims include OOF `0.77646`, public `0.78237`, and final leaderboard `0.78235`. The two public-score claims differ. All code-cell execution counters are null; no input/cache hashes or run manifest bind those claims to this exact source. | **Not reproduced or verified**; neither OOF nor leaderboard score is accepted as evidence for this file. |
| Attribution | The notebook describes v15 and “community” ideas but does not provide complete source-level attribution/license details in the inspected notebook. | Add accurate citations before any reuse, as the course README requires. |

## Specific implementation evidence

1. **Pooled train/test features:** `al` concatenates train and test before
   category coding. `f_count()` maps values to counts from `al`; `f_group()`
   computes group means from `al`; product fee frequencies are computed from
   pooled `q`/`_qb`; and partner groups are constructed over all `al`.
2. **Potential validation contamination:** `compute_resid()` builds continuous
   predictors for raw fields such as external scores and financial fields.
   Its K-fold regressors are fit across the combined `al` rows. For an outer
   validation row, its raw feature value can therefore be used as a training
   target when constructing residual features for other rows in the same outer
   experiment. This is not target-label leakage, but it violates a strict
   train-fold-only feature-fitting protocol and can inflate OOF estimates.
3. **Target encodings:** `enc(fit, qi)` uses `y[fit]`; `train_enc(a)` makes
   inner out-of-fold encodings for training rows. These portions appear
   structurally leakage-aware, but their feature keys are factorized on all
   pooled rows, and partner keys have separate global matching concerns.
4. **Stacker claim:** cell 11 cross-fits logistic weights over stored first-level
   OOF predictions. This alone does not ensure strictly nested stacking:
   base-model predictions for meta-training rows may come from models whose
   training labels include meta-validation rows. A valid meta-level OOF
   evaluation requires base predictions generated without access to the
   corresponding meta-holdout labels, typically via nested/fold-aligned
   predictions.
5. **Distribution shift:** the notebook reports adversarial train/test AUC
   `0.6868`. This is a notable train/test distinguishability signal, not proof
   of leakage. It merits investigation with the pinned inputs and a
   fold-isolated diagnostic.
6. **Submission score discrepancy:** embedded text records `0.78237` in one
   location and `0.78235` in another. No verified leaderboard URL/entry or
   submission hash is included to resolve the difference.

## Reusable information — not copied code

The best ideas to investigate independently from the supplied data are listed
in `V17_FEATURE_REUSE_REGISTER.csv`. Lower-risk hypotheses include:

- external-score summaries/differences and their interactions with loan
  burden, subject to matched-fold testing;
- employment sentinel flag plus cleaned employment duration;
- financial ratios already represented in the preserved baseline, so test
only alternative representations if they bring information beyond the
existing features;
- exact financial product relationships only after their constants are
derived from permitted training-fold rows and the procedure is documented.

The current corrected experiment already tested a set of simple formulas and
external-score interactions; that run did not show a reliable pooled OOF
improvement. Repeating those same ideas under another name is not an
optimization.

## Required conditions before reconsidering the full notebook

1. Obtain and preserve every referenced v15/v6 cache and prediction artifact,
   with source provenance and hashes, or disable reuse and recompute.
2. Replace broad file discovery with exact train/test/sample paths and verify
   input hashes and schemas.
3. Fit category vocabularies, counts, group statistics, residual learners,
   product-frequency decoders, and entity-matching statistics only on each
   outer-fit partition; apply them to inner-stop/outer-validation/test rows.
4. Move early stopping and all model/feature decisions to inner training
   splits; keep outer folds untouched.
5. Rebuild any target encoding and stacking with genuinely nested OOF
   predictions.
6. Establish the provenance and data-only derivation for `129.5`, `0.0132`,
   fixed blend weights, and any hand-selected partner/regime rules.
7. Record source attribution, code hash, data hashes, environment, fold
   assignments, complete OOF predictions, and submission hash.
8. Re-run the approved candidate using a predeclared validation protocol and
   compare it with the corrected raw baseline. Do not use the reported
   leaderboard score to select features.

## Scope and limitations

This is a static notebook-source and embedded-output audit, not a full
execution/reproduction. The notebook was not executed: its main path may take
hours and depends on unavailable Colab Drive caches. No score, cache artifact,
external data use, or public leaderboard entry was independently verified.
The competition rule assessment follows the local course materials already
audited; ambiguity requires organizer clarification.
