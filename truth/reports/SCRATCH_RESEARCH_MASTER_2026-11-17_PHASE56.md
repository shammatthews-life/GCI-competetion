# Scratch Research Master Log — Phase 56

New, append-only experiment record. The Phase 56 runner reads only the official
`train.csv` and `test.csv`; Phase 53-55 predictions, caches, reports, and
submissions are not runtime inputs. No earlier Python source or run output is
modified.

## Research evidence and hypothesis

- Phase 20's contract-type routing was positive in all ten tested discovery
  configurations (mean lift `+0.000872324`; largest `+0.001203792`), but those
  results used a reduced discovery subset and previously exposed folds.
- Phase 55 found a coarse `annuity / credit` key improved its matched
  LightGBM baseline by `+0.003038341` OOF AUC; adding fold-local target-free
  key support changed the result by only another `+0.000155771`.
- Phase 54 showed that adding Q3, Q4, and Q5 keys together hurt by
  `-0.005874651`. The Q4/Q5 resolutions created roughly 790/4,900 key levels,
  while Q3 had about 96 levels and roughly 3,000 rows per key.
- These findings motivate one focused interaction: allow the coarse Q3 risk
  profile to vary by the supplied `NAME_CONTRACT_TYPE`. This is a deterministic
  within-row key; no subgroup model or label propagation is used.

## What this run deliberately excludes

- The q-fingerprint archive's `129.5` fee correction and other empirical
  constants.
- Custom target encodings, fold/row label transfer, applicant matching,
  group means, and train/test feature pooling.
- External data, caches, previous OOF vectors, submissions, and leaderboard
  feedback.
- Rank averaging or blending predictions from separate CSVs.

The official competition materials allow reproducible feature engineering
from supplied columns, but they do not resolve every supervised-statistic or
transductive method. Phase 56 avoids those ambiguous paths; the result remains
research-only, not organizer-certified.

## Version register

| Version | New Python source | Question | Runner inputs | Results | Status |
|---|---|---|---|---|---|
| 53.1 (context) | `experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py` | Do row-local financial ratios help? | Official train/test | `0.759685` OOF; `+0.007380` vs matched raw control | Exploratory |
| 54.1 (context) | `experiments/phase54_qfingerprint_fold_local/phase54_qfingerprint_fold_local.py` | Do Q3/Q4/Q5 categories help without encodings? | Official train/test | `0.753231` vs `0.759106`; `-0.005875` | Rejected as a combined family |
| 55.1 (context) | `experiments/phase55_q3_support_ablation/phase55_q3_support_ablation.py` | Does Q3 and fold-local support help? | Official train/test | `0.762300` with Q3+support; `+0.003194` vs matched baseline | Coarse Q3 retained, support effect tiny |
| 56.1 | `experiments/phase56_q3_contract_interaction/phase56_q3_contract_interaction.py` | Does the Q3 risk profile vary by contract type? | Official train/test only | Q3: `0.762144`; Q3×contract: `0.761223` | Interaction rejected; exploratory |

## Phase 56 protocol

- **Run folder:**
  `experiments/phase56_q3_contract_interaction/runs/phase56_q3_contract_20261117/`.
- **Three fixed model variants:** rowwise-feature baseline; baseline plus
  Q3=`round(annuity/credit, 3)`; baseline plus Q3 and
  `NAME_CONTRACT_TYPE|Q3`.
- **Model:** LightGBM CPU, five seeded stratified folds; identical fixed model
  configuration for each arm; 900 rounds; no outer-fold early stopping.
- **Fold safety:** category vocabularies come only from the applicable model's
  training fold; unseen validation/test keys map to missing.
- **Submission:** arithmetic mean of the five Q3×contract model probabilities,
  in original test ID order; no rank transformation or CSV-to-CSV blend.
- **Audit:** syntax and synthetic interaction checks; complete OOF; independent
  pooled/fold AUC and log-loss recalculation; paired bootstrap; input/source
  hashes; output schema/order/range; SHA-256 manifest.
- **Evidence caveat:** labels and folds have prior exposure across project
  research. Even a positive result is exploratory, not untouched confirmation
  and not a leaderboard score.
- **Pre-run checks:** Python compilation and the script's synthetic self-test
  passed. The run output directory did not previously exist.
- **SHA-256:** runner
  `4712cdf54048965457a5cbf711e447a8996f7b43f08a10c5c4f78d21b40c04c0`;
  official train
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  official test
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.

## Phase 56 results

Full five-fold run completed in `419.66` seconds. Independent verification
recalculated pooled and fold AUCs and log loss from the saved OOF CSV to within
`1e-12`; row count, ID/target/fold alignment, test ID order, uniqueness, and
prediction bounds all passed.

| Model | Pooled AUC | Mean fold AUC | Log loss | Fold AUCs |
|---|---:|---:|---:|---|
| Rowwise baseline | `0.759105515` | `0.759141612` | `0.245785542` | `0.763026, 0.764707, 0.756379, 0.756045, 0.755550` |
| Q3 | `0.762143856` | `0.762192819` | `0.245029565` | `0.765564, 0.769350, 0.758306, 0.758863, 0.758880` |
| Q3 + contract | `0.761223134` | `0.761250079` | `0.245397576` | `0.764857, 0.767741, 0.756261, 0.758369, 0.759023` |

### Phase 56 decision

- Q3 vs matched baseline: `+0.003038341` pooled AUC, positive in `5/5`
  folds; log loss improved by `0.000755977`.
- The specific question was whether contract type modifies Q3. The joint key
  underperformed Q3 alone by `-0.000920722` AUC, with only `1/5` folds
  positive and a conditional paired-bootstrap 95% interval
  `[-0.001656860, -0.000353467]`. Its log loss was worse by `0.000368010`.
  Reject this interaction; do not claim the Phase 20 routing result transfers
  to this row-local key.
- Q3+contract still exceeds the raw baseline by `+0.002117619`, but that does
  not demonstrate interaction value; the Q3-only ablation is stronger.
- Bootstrap intervals are conditional on fixed saved OOF vectors and do not
  account for model selection, training variation, or prior exposure of these
  labels/folds. This is exploratory evidence, not untouched confirmation, a
  0.775 result, or a new champion.

Artifacts:
- OOF: `experiments/phase56_q3_contract_interaction/runs/phase56_q3_contract_20261117/PHASE56_OOF_PREDICTIONS.csv`
- Submission-shaped experiment output (not a recommended champion):
  `experiments/phase56_q3_contract_interaction/runs/phase56_q3_contract_20261117/submission.csv`
- Full details and conditional bootstrap results are in
  `experiments/phase56_q3_contract_interaction/runs/phase56_q3_contract_20261117/PHASE56_REPORT.md`
- The pre-run runner, input, and output hashes are recorded in the protocol
  and `PHASE56_SHA256_MANIFEST.json`.
