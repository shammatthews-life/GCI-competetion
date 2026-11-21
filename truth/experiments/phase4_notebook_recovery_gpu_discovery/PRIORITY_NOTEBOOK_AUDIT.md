# Phase 4 priority notebook source audit

## Scope and provenance

This is a static source/output audit only. No notebook was executed and no model was trained. All notebook cells inspected here have `execution_count=null`; some retain text outputs. Saved outputs are source artifacts, not independently authenticated run logs.

| Notebook | Bytes | SHA-256 | Cell/output evidence |
|---|---:|---|---|
| `C:\Users\SHARON\Downloads\GCI_v5_dropPF4_colab.ipynb` | 36,653 | `91ced0b7595f83a4a2de99d2e8973c68f212871ea2f851669b1ce78088b04ce6` | 9 code cells, all execution counts null; saved text outputs |
| `D:\exp colab\GCI_v16_everything_merged.ipynb` | 87,934 | `a1b3aec64c6730d58ad2d22f1c3a49dd0d3a9b66f05ab22054514add61f9dec3` | 11 code cells, all execution counts null; saved text outputs |
| `D:\exp colab\GCI_v17_optimised.ipynb` | 96,063 | `7103fecb0d2b894ab3f4e78c84ba52383ad9e289ce1aa6b9c59aab7921a3c457` | 12 code cells, all execution counts null; saved text outputs |
| `D:\gci\truth\baselines\exp85_phase2_complete_submission_baseline\EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb` | 6,706,327 | `ab73f6f39a0fee803fc946de5bd64a4219f9b59a35663c8510b9c002570069f0` | 43 code cells, no saved outputs or execution counts |

The EXP85 copy in Downloads and the separate working-copy notebook are byte-identical to the preserved baseline (same 6,706,327 bytes and SHA-256). No notebook was changed.

## Reported score evidence: saved output only

| Notebook output claim | Score | Evidence status |
|---|---:|---|
| v5 `qf_shallow` | 0.77070 | Saved text output; not reproduced |
| v5 `qf_reg` | 0.77049 | Saved text output; not reproduced |
| v5 `v1_allfam_oh` | 0.77047 | Saved text output; not reproduced |
| v5 `v2_allfam_oh` | 0.77055 | Saved text output; not reproduced |
| v5 `v2_mono` | 0.77056 | Saved text output; not reproduced |
| v5 same-fold 50/50 old/new blend | 0.77213 | Saved output says old blend 0.77202, new blend 0.77198, mix 0.77213; not independently reproduced |
| v16 fixed champion formula | 0.77649 | Saved output; not independently reproduced |
| v17 fixed champion formula | 0.77646 | Saved output; not independently reproduced |
| v17 optional cross-fitted stacker | 0.77657 | Saved output, +0.00010 over fixed v17; below its own stated +0.00020 promotion rule |

These are not public leaderboard scores. The v17 audit elsewhere records two conflicting public-score claims (0.78237 and 0.78235), neither tied to a verified submission. No leaderboard claim is upgraded by this source inspection.

## Source-level findings

### V5

- The notebook concatenates train predictors and test rows before building features. Frequency counts, group means, product-frequency/fee decoding, and one-hot vocabulary therefore include validation and test covariates.
- Target encoding has nested inner folds for training rows and uses the outer-fold training labels for validation/test encodings. This is the strongest leakage-control component in the notebook.
- The main LightGBM fold routine uses fixed iteration counts and does not early-stop against the outer validation labels. However, the same seed-42 folds are reused across multiple feature/model variants and the final blend; this is exploratory selection on exposed folds.
- The code writes/reuses cache files. The embedded outputs do not provide data hashes or a run manifest binding the cached arrays and scores to the current source.

### V16 and V17

- Both construct `al = concat(train_without_TARGET, test)` before category coding, value counts, group means, fee/product keys, and applicant matching. This is label-free transduction, but it is not outer-fold-isolated. Local rule materials leave permission for test-covariate fitting unresolved (see `../../reports/final_competition_audit/OFFICIAL_RULES_AUDIT.md`).
- The product hypotheses include `AMT_ANNUITY / AMT_CREDIT`, a fixed 129.5 fee adjustment, and a 0.0132 credit/goods tier. The saved v5 output says 5,335 raw 5-decimal ratios collapse to 1,047 adjusted product keys and estimates a 0.75 cash-loan fee share. This is a plausible discrete contract-structure hypothesis, not proof: constants and frequencies are fit/selected using pooled covariates, and their permitted-data derivation must be independently established.
- `train_enc()` nests target encodings inside the outer training rows; `enc(a, validation)` uses only the outer training labels. Partner-label features likewise draw labels from fit rows, not validation/test labels. However, candidate identity groups are formed globally from demographic/date-gap fingerprints, not verified applicant IDs. There is no application timestamp to establish that another matched row's outcome would have been known at prediction time. False matches and temporal label availability remain risks.
- `compute_resid()` cross-fits regressors for raw non-target columns over the concatenated train+test matrix. This does not use `TARGET`, but outer-validation/test covariates influence learned residual transforms for other rows. It is not strict outer-fold isolation.
- The main LightGBM models use fixed iteration counts and do not early-stop on the scored outer fold; CatBoost uses fixed iterations without an outer validation `eval_set`. This differs from the candidate Python source and EXP85, and avoids that specific early-stopping reuse issue. It does not fix pooled transforms or repeated model/feature selection on the same folds.
- The fixed champion blends were scored on the same OOF rows used during the research. The optional regime-aware blend and v17 stacker split the existing OOF rows to fit meta-weights, but the base OOF predictions are not generated in a fully nested way relative to those meta-splits. The cross-fitted meta-scores are therefore not independent confirmation.
- The permutation-label check has a saved result near 0.50; it tests direct target-encoding leakage under that check, not transductive feature fitting, cache provenance, false entity matches, or model-selection bias.

### EXP85

- The baseline/Downloads/working-copy notebooks are identical, but contain source only: 0 saved outputs and 0 execution counts.
- Its embedded pipeline has input/schema/hash checks and nested target encoding. Prior static audit evidence identifies outer-fold validation reuse for early stopping and pooled frequency/category-vocabulary transforms. Treat its OOF value as a reproduction target, not an unbiased independent estimate.
- The local official rule review confirms that transductive statistics and target-encoding permission are not explicitly resolved; organizer clarification is required before promotion.

## Current interpretation

The most promising structural hypothesis to validate independently is that annuity/credit ratios cluster around financing-contract families, possibly with an additive periodic fee. The apparent pattern may be real, but its AUC value, fixed-fee interpretation, external validity, and competition eligibility remain unverified. Next experiments should derive candidate contract features from permitted training-fold data, use an untouched outer fold, and compare them against the preserved baseline with paired fold deltas. Do not copy caches or trust embedded score text as reproduction.
