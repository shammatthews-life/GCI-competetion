# Scratch Research Master Log — Phase 55

New append-only record for the Phase 55 experiment. No previous Python source,
submission, OOF output, or verification ledger is modified. The Phase 54 log
was updated only to replace its in-progress placeholder with its completed
negative result. The standalone runner reads only the official `train.csv`
and `test.csv`; files from Phase 53, Phase 54, and the archived q-fingerprint
experiments were inspected for research context but are not runtime inputs.

## What we learned before Phase 55

- The archived notebook
  `experiments/extracted_from_notepad/notepad_nb2_qfingerprint_0.777.ipynb`
  reports a five-fold pooled held-out AUC of `0.77070` for its best member,
  `0.77035` for its monotone member, and `0.77158` for a three-member rank
  average. Its single fold 3 result of `0.77639` is not its pooled score.
- That notebook's `q=AMT_ANNUITY/AMT_CREDIT` keys are not the only differences:
  it also uses explicit smoothed label encodings, pooled train/test category
  codes/value counts/group means, and a pooled-frequency-derived `129.5` fee
  correction. It combines several candidates with rank averaging. These
  methods are not part of this conservative track.
- Phase 54 tested the q keys as three categorical features, rounded to 3, 4,
  and 5 decimals, using only training-fold vocabularies and no explicit target
  encoding or pooled statistics. The result was negative:
  `0.753231` versus `0.759106` for its matched LightGBM baseline
  (`-0.005875`), negative in all five folds. It also worsened log loss.
- The Phase 54 keys had about 96, 790, and 4,900 distinct levels per fold.
  This suggests the 4/5-decimal high-cardinality keys may be noisy without
  the archive's explicit smoothing. It does not show that the key alone
  generalizes.
- Phase 53 demonstrated that row-local financial geometry can improve a
  matched CatBoost raw control by `+0.007380`, but that comparison is from a
  different learner and is not directly comparable to Phases 54/55.
- The organizer's supplied text encourages feature engineering from provided
  columns, but does not resolve pooled test-covariate fitting. It contains no
  explicit target-encoding ban. This run avoids both pooled test statistics
  and custom label encodings; it is still not an organizer eligibility ruling.

## Version register

| Version | New Python source | Research question | Runtime inputs | Result | Decision |
|---|---|---|---|---|---|
| 53.1 (context) | `experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py` | Do row-local financial formulas help? | Official train/test | `0.759685` OOF; `+0.007380` matched | Exploratory |
| 54.1 (context) | `experiments/phase54_qfingerprint_fold_local/phase54_qfingerprint_fold_local.py` | Do Q3/Q4/Q5 categories help without encodings? | Official train/test | `0.753231` vs `0.759106`; `-0.005875` | Reject this representation in the tested setup |
| 55.1 | `experiments/phase55_q3_support_ablation/phase55_q3_support_ablation.py` | Does the low-cardinality Q3 category work, and can label-free fit-fold support explain the Q3/Q4/Q5 failure? | Official train/test only | Q3+support OOF AUC `0.762300`; `+0.003194` vs matched baseline | Retain Q3 as exploratory feature evidence; support adds only `+0.000156` over Q3 alone |

## Phase 55 protocol

- **Run folder:**
  `experiments/phase55_q3_support_ablation/runs/phase55_q3_support_20261117/`.
- **Three paired model variants:** baseline; baseline plus Q3 as a categorical
  feature; baseline plus Q3 and its fit-fold-only support count.
- **Q3 definition:** `AMT_ANNUITY / AMT_CREDIT`, rounded to exactly three
  decimals; no fee correction or fitted threshold.
- **Support definition:** for each model fit fold, compute the count of each Q3
  key in that fit partition. Fit rows receive `log1p(count - 1)` (leave-one-out);
  validation/test rows receive `log1p(count)` from fit rows only. Counts use
  no labels; no validation or test rows contribute.
- **Model:** LightGBM CPU, same fixed settings, five stratified folds, 900 fixed
  rounds, no early stopping, no parameter selection on outer-fold labels.
- **No-go list:** custom target encoding, label propagation, test pooling,
  test-to-test interactions, external data, empirical fee constants, cached
  model/OOF/submission inputs, and rank blending.
- **Submission:** one file from the predeclared Q3+support model, arithmetic
  mean of the five fold-model probabilities, test ID order preserved.
- **Validation:** compile and synthetic category/support-count tests; full
  execution; independently recompute all three OOF scores, fold deltas,
  confidence intervals, input/output hashes, and submission structure.
- **Evidence status:** the dataset and folds have extensive prior exposure.
  Any metric here is exploratory, not untouched confirmation or a public
  leaderboard score.
- **Pre-run checks:** Python compilation and the synthetic category, rowwise
  feature, fit-row leave-one-out support, query support, and LightGBM fitting
  tests passed.
- **Train SHA-256:**
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- **Test SHA-256:**
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- **Pre-run source SHA-256:**
  `f54df005504e6a7be83705577749ee5b526cb4b7f009b3af6730dd92e60e2d73`.

## Phase 55 execution record

### Results

| Model | Pooled OOF AUC | Mean fold AUC | Log loss |
|---|---:|---:|---:|
| Baseline | `0.759105515` | `0.759141612` | `0.245785542` |
| Q3 category only | `0.762143856` | `0.762192819` | `0.245029565` |
| Q3 category + support | `0.762299627` | `0.762329932` | `0.244968423` |

- Q3-only improvement over baseline: `+0.003038341` pooled AUC, positive in
  all five folds; conditional paired-bootstrap 95% interval
  `[+0.001713780, +0.004426084]`.
- Q3+support improvement over baseline: `+0.003194112`, positive in all five
  folds; conditional interval `[+0.002065771, +0.004438304]`.
- Adding support over Q3 alone changes pooled AUC by only `+0.000155771`;
  its fold deltas are mixed (`3/5` positive). The present evidence does not
  establish a separate support-count gain.
- Q3 had only 94-96 levels per fit fold, with median support about 3,000 rows
  per key. Phase 54's Q4/Q5 features had roughly 790/4,900 levels and lost
  `0.005875` versus baseline. This supports a resolution/sparsity explanation:
  the broad high-cardinality key set overfits in this setup, while a coarse,
  well-supported Q3 product key improves rankings.
- This is a real matched feature gain, not the reported `0.775`. Absolute
  pooled AUC is `0.762300`; it does not reach the user's reported `0.775` or
  establish `0.78`. The q-fingerprint notebook's best archived pooled result
  was `0.77070`, not its `0.77639` peak fold.
- **Decision:** retain the coarse Q3 key for further conservative research;
  treat support count as optional and unproven incremental value. Do not copy
  the archive's explicit label encoding, pooled statistics, fee constant, or
  rank ensemble. Do not promote the submission as a champion.

### Validation and output

- Runtime `654.55` seconds, CPU, LightGBM `4.7.0`, five fixed stratified folds,
  900 fixed rounds; no outer-fold early stopping.
- Compilation and synthetic tests passed. Independent verification recalculated
  all three pooled AUCs and log losses, each fold AUC, both bootstrap intervals,
  OOF row/target/fold alignment, submission order/range, and every manifest
  hash.
- Submission:
  `experiments/phase55_q3_support_ablation/runs/phase55_q3_support_20261117/submission.csv`
  SHA-256
  `e7390643ef69a6e98056cab321605ae6e831e2fa8b3c955c62e17b4b4e260cad`.
- OOF predictions:
  `experiments/phase55_q3_support_ablation/runs/phase55_q3_support_20261117/PHASE55_OOF_PREDICTIONS.csv`
  SHA-256
  `048b4a38d6db801a1e8fdf131bf512c946f3d0203fb76b90bc1707550f482f94`.
- Detailed report, JSON summary, and full hashes are in the same run folder:
  `PHASE55_REPORT.md`, `PHASE55_SUMMARY.json`, and
  `PHASE55_SHA256_MANIFEST.json`.
- Final source SHA-256:
  `f54df005504e6a7be83705577749ee5b526cb4b7f009b3af6730dd92e60e2d73`.

All results are exploratory because these labels and folds have prior project
exposure. The bootstrap intervals are conditional on OOF predictions and do
not include training or research-selection uncertainty.
