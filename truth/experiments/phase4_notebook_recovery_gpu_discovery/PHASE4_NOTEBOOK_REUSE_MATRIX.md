# Phase 4 notebook and feature reuse matrix

## Scope and evidence standard

This is a static audit of the inventoried project notebooks plus an independent
recalculation of the existing EXP02 OOF metrics from its saved arrays. No
notebook was executed and no model was trained during this audit. The
competition-package input hashes are verified in
`DATA_DICTIONARY_AND_INPUT_AUDIT.md`.

Evidence labels:

- **Artifact-recomputed**: predictions and folds already existed; this audit
  independently recalculated the metrics and checked the recorded hashes.
  This is not a second training run.
- **Saved output only**: notebook output text exists, but it is not sufficiently
  bound to the exact executed code, data, folds, or caches for independent
  reproduction.
- **No execution evidence**: no saved outputs or execution counts.

## Candidate-source matrix

| Artifact | SHA-256 | Feature families / reuse | Validation and evidence | Disposition |
|---|---|---|---|---|
| EXP85 baseline, Downloads, preserved baseline, and working copy | `ab73f6f39a0fee803fc946de5bd64a4219f9b59a35663c8510b9c002570069f0` (all three identical) | Historical EXP85 pipeline and feature blocks; retain as a reference, not a feature specification. | Preserved baseline has no executed outputs. The approximately 0.77158 OOF value is a historical claim; early-stopping leakage was found in the prior audit. | Preserve unchanged. Do not treat its score as a valid comparator without a corrected reproduction. |
| `GCI_v5_dropPF4_colab.ipynb` | `91ced0b7595f83a4a2de99d2e8973c68f212871ea2f851669b1ce78088b04ce6` | Product-ratio/fee hypotheses, external-score combinations, and earlier blend components. | No execution counts; saved OOF text reports 0.77213 for an old/new blend. Pooled transformations and non-nested blend selection were found in the prior audit. | Research reference only; score remains unverified. |
| `GCI_v16_everything_merged.ipynb` | `a1b3aec64c6730d58ad2d22f1c3a49dd0d3a9b66f05ab22054514add61f9dec3` | Product keys, target-encoded categories, multi-view/residual features, and ensemble components. | Saved outputs report 0.77649; no execution counts or bound run manifest. Several fitted transforms use pooled train/test covariates. | Do not promote the embedded score. Rebuild under a frozen, fold-isolated protocol if needed. |
| `GCI_v17_optimised.ipynb` | `7103fecb0d2b894ab3f4e78c84ba52383ad9e289ce1aa6b9c59aab7921a3c457` | v16 feature/model lineage plus optional stack/blend changes. | Saved outputs report 0.77646; optional stacker reports 0.77657, only +0.00010 and below its own +0.00020 gate. Execution and data lineage remain unverified. | No demonstrated accepted improvement. Preserve as a candidate, not a champion. |
| `GCI_v18_ExtraTrees_diversity_colab.ipynb` | `7a2e17527ca1bf9dc2f4d62855f4b456a5629a53b8f0af60b6bfab019ff01e17` | ExtraTrees on residual, target-encoded, and multi-view features; fixed 10%/15% rank-blend candidates. | No execution counts. Requires unmanifested `gci_cache_final` and related cached OOF/test files. Saved outputs say ExtraTrees AUC 0.76758; additions to the saved 0.77404 comparator are approximately 0.00000 and -0.00006, positive on 2/5 folds, and both fail the stated gate. | The claimed blend gives no evidence of improvement. Cache provenance and source execution are unverified. |
| `GCI_v19_research_candidate.ipynb` | `3d235f11afe2967605b2fb32b5022b0b0d6ce861425f4ae0e0951f8bcbf147d3` | v15/v16 features plus credit/annuity keys, the 129.5/0.0132 product hypotheses, candidate implied-rate/term features, date differences, and inferred repeat-applicant label features. | All execution counts are null. Saved outputs claim a 0.77649 blend and positive feature ablations, but depend on unmanifested caches and pooled train/test transforms. A concrete output mismatch exists: current code writes `GCI_submission_v19_candidate.csv`, while its adjacent saved output says it wrote `GCI_submission_v16.csv`. | Output is not attributable to the current source; do not use its score or submission as verified evidence. |
| `GCI_v19_NN_entity_embeddings_colab_0.ipynb` (Downloads) and `GCI_v19_NN_entity_embeddings_colab.ipynb` (D: copy) | `a0b3fad94e7e94ce72b9895e24d71a14351e27f2f7656d823ee5d06efcd13688` and `4c9f9ce6891848fb6dc655665dc22bfd2241533ad5f52a45eaf2a0dff84fb558` | Neural entity embeddings for product keys plus application numerics and residual features. The two files have identical cell source and saved outputs; their execution-count metadata differs. | Downloads copy has execution counts and saved CUDA text; D: copy has null execution counts. Saved standalone OOF AUC is 0.76732. The 10%/15% blends report +0.00016/+0.00017 mean gain and 4/5 positive folds, so both fail the +0.00020 gate. No input/cache run manifest is supplied. Quantile and residual features are prepared globally before outer CV. | Saved diagnostic suggests no accepted champion lift. Not a verified local GPU run or independent reproduction. |
| `GCI_UMAP_Research_Notebook (1).ipynb` | `6470148ac0f24d37e8f6f7c5d24fb12ec31300b0ddd9d2081f254d85e030a13d` | Weighted external-score transform, financial ratios, category/group encodings, PCA, unsupervised and target-guided UMAP, model zoo, and meta-blending. | No execution counts or outputs. Target encodings are OOF across all training rows rather than nested within each model's outer fold. Supervised UMAP is fit using training labels before model CV; pooled imputation/quantile/scaling/PCA/UMAP also sees validation/test covariates. Meta blending is not fully nested with base-model OOF predictions. | OOF scores from this design would be contaminated. Do not use for model comparison or competition submission without a full fold-local rewrite and rule review. |
| `EXP02_FINANCIAL_BEHAVIOUR.ipynb` | `b3882dc3ed7a8b6a3370a2f65da7232e91ee7f0d65baf9da3362810f6d13cd7c` | Seven deterministic ratios: credit/income, annuity/income, goods/credit, credit/annuity, and income/credit/annuity per family member. No learned statistic, fee constant, or external data. | Prior GPU execution has source/input/artifact hashes and a 5-fold OOF record. This audit verified every manifest hash, test-ID order, prediction shape/bounds, all five fold AUCs, and pooled AUCs from saved arrays. Decision remains `NEEDS_INDEPENDENT_SEED_REPLICATION`. | Best-supported formula result so far. Reproduce with a new seed, then compare the block against the champion using identical folds/model settings. It is not a champion result. |

Competition README/tutorial and past-winner instructional notebooks are rule or
teaching references, not candidate model branches. The README remains the
authority for data and prediction rules; instructional examples do not override
its restrictions.

## Feature reuse summary

| Feature family | Earlier occurrences | Later reuse / distinction | Current assessment |
|---|---|---|---|
| Basic financial amount ratios | EXP85 / v5 lineage | EXP02 tests seven rowwise ratios against a matched raw-feature CatBoost baseline. v16-v19 contain overlapping ratios among many other changes. | EXP02 shows a real block-level gain in its saved OOF arrays, but not which individual formula caused it. No annualization is supported by the local dictionary. |
| External-score combinations | v5, v16/v17, UMAP, v19 research | Weighted means, products, differences, dispersion, and external-score interactions recur. | Repeated and highly correlated family; compare additions individually. Local dictionary gives no score direction or optimal weights. |
| Contract/product structure | v5 and v16/v17 | v18/v19 reuse rounded annuity/credit or credit/annuity keys, label encodings, and fixed 129.5/0.0132 fee/tier hypotheses. | Plausible structure, not confirmed contract metadata. Candidate constants were inferred from pooled rows and are not independently validated. |
| Credit-to-annuity term proxy / implied rates | Screenshots and v19 research | v19 uses credit/annuity as a proxy and computes candidate-term rate curves for 12-60 months. | Credit and annuity alone do not identify actual term/rate; payment frequency is unspecified. Keep as hypotheses, not observed loan terms. |
| Family-normalized affordability | Screenshots and EXP02/v19 | EXP02 tests three per-family-member amount ratios; v19 reuses related forms. | Empirically testable, but equal allocation across household members is an assumption. No individual ablation yet. |
| Repeat-applicant linkage | v16/v19 research | Fingerprints built from gender/education/region/mobile plus date-difference signatures; nested training-label aggregates use matched rows. | Potentially informative but identity precision/collision rate is unmeasured. Groups are constructed on pooled train/test covariates. The sign of `DAYS_BIRTH` differences can represent chronology only if a match is truly the same person. Do not accept as established repeated-borrower history. |
| Bureau-window transformations | Screenshots and v5/v19 | Recent-month/year count combinations recur. | YEAR interval wording in the supplied dictionary is ambiguous; `/4` and acceleration formulas are unsupported until clarified. |
| Learned residual / grouped statistics | v16-v19 and UMAP | Residuals, value counts, category/group means, and train/test category handling recur. | Several are fitted globally or pooled. They need fold-local refitting; transductive rule permission remains unresolved. |
| PCA/UMAP and learned representations | UMAP notebook; NN embeddings in v19 | UMAP notebook includes label-guided UMAP; NN notebook embeds product/category keys. | UMAP's global target-guided embedding is direct outer-fold label leakage. NN early stopping is inner-fold, but pooled quantile/residual preprocessing breaks strict outer-fold isolation. |
| Rank blends and stacking | v5-v19 | The same old model OOF arrays, CatBoost branch, new members, and rank transforms recur. | Repeated blend selection on shared OOF rows makes displayed pooled AUCs optimistic. Only use nested out-of-fold stacking or a separate untouched validation set. |

## EXP02 result: independently checked, not independently retrained

The existing EXP02 run's input hashes match the official-package copies. The
run manifest records the RTX 4050 GPU, five stratified outer folds (seed 42),
and inner-only early stopping. Its code makes rowwise ratios with non-finite
values and non-positive denominators mapped to missing; `SK_ID_CURR` and
`TARGET` are excluded from model features.

This audit independently recomputed the saved predictions:

| Metric | Raw baseline | Financial block | Delta |
|---|---:|---:|---:|
| Pooled 5-fold OOF ROC-AUC | 0.75126491 | 0.75849375 | +0.00722884 |
| Mean paired fold ROC-AUC | - | - | +0.00728454 |

Fold deltas were +0.00967347, +0.00754652, +0.00519816, +0.00656127, and
+0.00744330 (5/5 positive). All saved artifact hashes, fold metrics, test IDs,
and prediction bounds passed the independent checker. The experiment's own
predeclared status is still `NEEDS_INDEPENDENT_SEED_REPLICATION`: it is one
CatBoost setup with a combined seven-feature block, not a comparison against
the current champion. It establishes predictive signal for the block under
that setup, not a causal or uniquely financial mechanism.

## Conclusions and safe next actions

1. The source family repeatedly reuses a small set of ideas; many later
   notebooks add model members or blending around the same product keys,
   target encodings, residuals, and external scores. Avoid counting repeated
   implementations as independent evidence.
2. The EXP02 ratio block is the only result in this matrix whose stored OOF
   arrays and exact artifact hashes were independently checked here. Its
   meaningful next test is a frozen new-seed replication, followed by a
   matched champion ablation; do not port all seven features as a bundle and
   claim the full +0.0072 on the champion.
3. If the replication passes, test the ratio features in predeclared groups
   and individually with identical outer folds. Do not annualize annuity,
   label credit/goods as universal collateral LTV, or interpret
   credit/annuity as actual duration.
4. Keep UMAP, globally pooled residual/group features, product constants, and
   partner-label transfer blocked from score claims until their exact
   provenance, fold-local implementation, and competition-rule scope are
   settled.
5. No EXP85 notebook, champion, submission, or previous experiment was
   modified. No saved score in this matrix is a verified public leaderboard
   result.
