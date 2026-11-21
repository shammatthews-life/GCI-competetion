# Strictly inductive AUC research — findings

## Decision

The best-supported lower-ambiguity reference remains Phase 37's fixed equal
blend of a three-seed XGBoost ensemble and LightGBM: pooled OOF ROC-AUC
**0.765968** and mean-fold AUC **0.765982**. It uses the official training
data, constant-free ratio keys, training-partition-only category
frequencies, fixed weights, and no target encoding or test-covariate fitting.
The new V3 50/50 probability blend measured **0.766081** mean-fold AUC, but
its +0.000099 change is too small and inconsistent to promote over the Phase
37 reference. All these are development results on repeatedly examined
labels, not untouched estimates or leaderboard scores.

The new class-weight and application-process tests did not establish a
material improvement. Their small rank/probability blend gains are not
consistent enough to promote. A model-generated candidate submission was
created, but it was not uploaded.

## What I reviewed

The evidence pass covered the local official-rules audits and competition
materials, V22/V28 source audit, the 2026-11-13 compliant-model research
report, the Phase 2 formula audit, the Phase 22 AUC-objective diagnostic,
Phase 23's class-weight and process-feature results, and primary code/results
for Phases 33, 37, 38, and 44–52. The repository inventory contains 322
Python files, 78 notebooks, and 255 Markdown files. This is a focused,
evidence-ranked review of relevant sources, not a claim that every line of
every inventoried file was read.

The local competition README prohibits external data and hand-labeling,
requires reproducibility and citations, specifies ROC-AUC/submission format,
and requires reproducible code for honors consideration. It does not settle
test-covariate pooling or target encoding. This workspace follows the stricter
project boundary: no pooling, target encoding, peer-label transfer, old
predictions as pseudo-labels, or unverified empirical thresholds absent
written organizer clarification.

## Evidence from prior work

| Candidate | Evidence | Interpretation |
|---|---:|---|
| V22 historical blend | User-reported public leaderboard score 0.78239 | Not locally verifiable from model-generated OOF evidence; its source includes train/test pooling and `partner_enc` label transfer. Do not treat as a compliant model or compare its public score directly with OOF. |
| V28 | Reported nested OOF AUC 0.771679 | No row-level OOF vector was saved, script/notebook defaults differ, and the evaluation has historical exposure. Lead only, not a verified benchmark. |
| Phase 37 | Pooled OOF 0.765968; mean-fold 0.765982 | Best-supported lower-ambiguity path found in the reviewed records. Fixed multi-seed/cross-model blend; fold-local categories and no target encoding. |
| Phase 44 contract expert | Pooled OOF 0.765192 | Fold-safe and label-free gate; modest gain over its own global XGBoost baseline, but below Phase 37 and subject to repeat-selection bias. |
| Phase 49/50 | Phase 50 fixed blend OOF 0.767070 | Uses empirical pricing constants whose provenance is unresolved; hold out under the strict boundary. |
| Phase 51 | OOF 0.765235 with twin/group-label statistics | Uses cross-fitted training label statistics; organizer interpretation unresolved, so excluded. |
| Phase 52 | OOF 0.758640 for raw plus label-free v7-style feature families | Strictly inductive GPU feature experiment; below the stronger Phase 37 blend. |
| Pairwise ranking diagnostic | XGBoost pairwise ranker OOF AUC 0.500028 | The tested implementation collapsed to five score levels; do not assume an AUC-specific loss will automatically improve this dataset. |

These are not all evaluations on identical seeds, feature sets, or model
selection histories. Use each figure only within the stated protocol.

## New experiments

All three runs used XGBoost CUDA where training occurred. Training data SHA-256
was checked against the supplied `train.csv`; the new runners did not read
`test.csv`, external data, or prediction caches. Every OOF artifact contains
row-aligned IDs, targets, folds, and model-generated predictions.

### V1 — nested positive-class weighting

Tested `scale_pos_weight` values 1, 2, 4, and 8. Each outer fold selected its
weight using only one inner stratified split and refit on the complete outer
training fold. Selection chose weights `[1, 2, 8, 2, 8]`.

| Candidate | Pooled OOF AUC | Mean outer-fold AUC |
|---|---:|---:|
| Fixed weight 1 | 0.764835 | 0.764873 |
| Fixed weight 2 | 0.764924 | 0.764971 |
| Fixed weight 4 | 0.764536 | 0.764580 |
| Fixed weight 8 | 0.764190 | 0.764231 |
| Inner-selected weight | 0.711775* | 0.764820 |

`*` The adaptive version selects different weights across folds; raw scores
from those different models are not on a reliably shared scale. The pooled
OOF number is therefore explicitly scale-sensitive; mean outer-fold AUC is
the primary estimate of this nested selection procedure. It does not improve
on the Phase 37 benchmark. Weight 2 is only marginally above weight 1 and
does not justify promotion.

Run details: `runs/v1_class_weight_verified_20261114T214500Z/`.
The original and corrected-reporting V1 runs produced byte-identical OOF CSVs.

### V2 — rank-aware blend

Combined the Phase 37 OOF component with V1's fixed-weight-2 OOF predictions
at fixed 50/50 weights. The primary method ranked scores within each matching
outer fold before averaging; weights were not tuned on labels.

| Candidate | Mean outer-fold AUC |
|---|---:|
| Phase 37 reference | 0.765982 |
| V1 fixed weight 2 | 0.764971 |
| Fixed 50/50 within-fold rank blend | 0.766027 |
| Fixed 50/50 probability blend | 0.765859 |

The rank blend adds only **0.000046** to mean-fold AUC; the probability blend
is lower than the Phase 37 reference. The rank gain is too small to treat as
a reliable improvement given prior selection exposure.

Run details: `runs/v2_rank_blend_20261114T215300Z/`.

### V3 — row-wise application-process features

Added missingness flags and interactions with age and contract type. These are
deterministic transformations of each applicant's supplied columns; they do
not use labels or other applicants' rows. Compared a standalone fold-safe GPU
XGBoost model and fixed 50/50 blends with the same Phase 37 OOF reference.

| Candidate | Mean outer-fold AUC |
|---|---:|
| Phase 37 reference | 0.765982 |
| Application-process XGBoost alone | 0.764951 |
| Fixed 50/50 within-fold rank blend | 0.766062 |
| Fixed 50/50 probability blend | 0.766081 |

The standalone feature model lost to Phase 37 in all five folds. The blends
improved mean-fold AUC by only **0.000080** and **0.000099**, respectively,
with fold gains not consistent across all five folds. This is exploratory,
not strong evidence of a true lift.

Run details: `runs/v3_application_process_20261114T220000Z/`.

### V4 — reproducible test-inference candidate

`build_submission_v1.py` fits the Phase 37-style three-seed XGBoost plus
LightGBM probability ensemble and the V3 application-process XGBoost model
on all official training rows. It averages the two components at fixed
50/50 probability weights. The test data are used only for row-wise feature
calculation and inference; learned category vocabularies and key frequencies
are fit on training rows. XGBoost CUDA use is checked after every fit.

The generated candidate has **61,500 rows**, exact columns
`SK_ID_CURR,TARGET`, unique IDs in sample order, finite predictions in
**[0.0020841, 0.8010191]**, and SHA-256
`cb08aa3147c52c4ba992490ce9b6e3d2fe8e2aa0c58cc94682a26c87ce5e6ef5`.
Two runs of the current builder produced byte-identical CSVs. The public and
private leaderboard scores remain unknown because the file was not uploaded.

Run details:
`runs/submission_v1_verified_20261114T230000Z/` and
`runs/submission_v1_repeat_20261114T233000Z/`.

## Recommendation and next boundary

1. Keep Phase 37 as the current lower-ambiguity reference; do not claim the
   V1/V2/V3 changes have materially raised generalization AUC.
2. Do not use the historical V22 score as a target for tuning a compliant
   model. Its reported public score and the local OOF metrics answer
   different questions.
3. This candidate is reproducible and follows the project boundary, but its
   performance on the competition test set is unknown; do not describe the
   cross-validation estimate as its leaderboard score.
4. Future tests should be predeclared and grouped into a small number of
   orthogonal model/feature hypotheses, with a locked evaluation protocol.
   Repeatedly screening many blends or weights on the same OOF labels would
   make the apparent top score increasingly optimistic.

## Reproduction

```powershell
python D:\gci\truth\compliant_inductive_auc_20261114\run_v1_class_weight.py `
  --output-dir D:\gci\truth\compliant_inductive_auc_20261114\runs\new_v1_run

python D:\gci\truth\compliant_inductive_auc_20261114\run_v2_rank_blend.py `
  --output-dir D:\gci\truth\compliant_inductive_auc_20261114\runs\new_v2_run

python D:\gci\truth\compliant_inductive_auc_20261114\run_v3_application_process.py `
  --output-dir D:\gci\truth\compliant_inductive_auc_20261114\runs\new_v3_run
```

The output directory must not already exist. V2 requires the V1 OOF output;
V3 requires the verified Phase 37 OOF artifact. V1 and V3 fail if CUDA is not
active.

The submission builder is:

```powershell
python D:\gci\truth\compliant_inductive_auc_20261114\build_submission_v1.py `
  --output-dir D:\gci\truth\compliant_inductive_auc_20261114\runs\new_submission_run
```

The final candidate is a fixed 50/50 probability average of (a) three
full-training XGBoost seeds plus one LightGBM model and (b) a full-training
XGBoost model with the V3 row-wise application-process features. Each model's
iteration count is selected by an inner training-only holdout. XGBoost used
CUDA; LightGBM used CPU. Training, test, and sample input hashes are enforced
and recorded. This construction avoids rank-normalizing or fitting score
statistics on the test prediction batch.
