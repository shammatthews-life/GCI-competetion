# Scratch Research Master Log — Phase 54

This is a new log for the new Phase 54 scratch code. It supplements but does
not modify `SCRATCH_RESEARCH_MASTER_2026-11-17.md`, the Phase 53 source, or any
previous experiment. The executable Phase 54 code reads only the official
`train.csv` and `test.csv`; prior source/report files were inspected for
research context only and are not runtime inputs.

## What the higher-score files actually show

### Likely `loc aug` reference: q-fingerprint notebook

The closest local source match is
`experiments/extracted_from_notepad/notepad_nb2_qfingerprint_0.777.ipynb`.
Its saved outputs report:

- The best individual model (`qf_allfam_oh`) at pooled held-out OOF AUC
  `0.77070`, compared with its reported no-product-key reference `0.76678`.
- The monotone member at `0.77035`, and the three-member rank average at
  `0.77158`.
- One fold reached `0.77639`. That is a single fold, not the pooled OOF result;
  it should not be quoted as the model's five-fold score.

The notebook motivates q-fingerprints with
`q = AMT_ANNUITY / AMT_CREDIT`, rounded to 3/4/5 decimal places, and a
fee-corrected product key using a `129.5` constant. Its pipeline also:

- concatenates train and test to construct category codes, value counts, and
  group means;
- uses explicit smoothed label encodings for product keys;
- estimates the fee rule from pooled train-plus-test key frequencies;
- uses several preselected model members and a rank average; and
- predicts test with fold models plus full-data models, whereas its reported
  held-out vector comes from the fold models.

Those details mean the source's score does not isolate a purely row-local,
test-independent q-key gain. The source's pooled metrics and final-test
procedure are not directly comparable to a conservative, fold-local run.
The competition audit finds train/test pooling unresolved and requires
organizer clarification; it also identifies the source notebook's validation
and data-scope risks. Phase 54 excludes the fee correction, custom target
encoding, value counts, group means, row matching, and rank blending.

### `autonomous_loc_auc_optimization_cycle.py`

The similarly named local optimizer is not a standalone modeling experiment:
it loads many precomputed OOF/test matrices and an existing champion
submission, ranks them, optimizes weights on the complete OOF label vector,
and builds a train/test entity graph with train labels. Its output cannot be
used to establish a clean independent `0.775` result. Phase 54 does not import
or read any of its cache, model, or submission files.

### Phase 53 knowledge

The new Phase 53 scratch code
`experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py`
used only row-local formulas and reported a matched CatBoost OOF AUC increase
from `0.752306` to `0.759685` (`+0.007380`), positive in all five folds.
That paired result motivated continued study of amount ratios, but its AUC is
not comparable to Phase 54's LightGBM result because the learner/configuration
differs. Phase 53 added rounded term values as numeric columns; it did not test
them as fold-local categorical features.

## Version register

| Version | New Python source | Hypothesis | Inputs read by runner | Result | Decision |
|---|---|---|---|---|---|
| 53.1 (context) | `experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py` | Financial geometry helps a raw CatBoost control | Official train/test | `0.759685` OOF; `+0.007380` vs matched raw control | Exploratory, not promoted |
| 54.1 | `experiments/phase54_qfingerprint_fold_local/phase54_qfingerprint_fold_local.py` | Q=annuity/credit rounded categories add ranking signal beyond numeric geometry | Official train/test only | `0.753231` OOF; `-0.005875` vs matched baseline | Reject Q3/Q4/Q5 categorical bundle in this setup |

## Phase 54 protocol and governance

- **Run path:**
  `experiments/phase54_qfingerprint_fold_local/runs/phase54_qfingerprint_20261117/`.
- **Model:** LightGBM CPU, five seeded stratified folds; same frozen config and
  fold IDs for baseline and candidate; 900 fixed rounds; no outer-fold early
  stopping or OOF-driven tuning.
- **Baseline:** original supplied predictors plus deterministic, row-local
  arithmetic/log/age/external-score features.
- **Candidate-only change:** three applicant-local categories formed from
  `AMT_ANNUITY / AMT_CREDIT` rounded to 3, 4, and 5 decimals.
- **Fold safety:** category vocabularies/codes are constructed from that
  model's training partition only. Unknown validation/test categories map to
  missing. No cross-row counts or label-derived feature construction.
- **Excluded methods:** fee constants, pooled train/test feature statistics,
  group means, explicit target encoding, repeat-applicant matching, prior OOF
  or submission files, test-to-test interactions, and rank blending.
- **Output:** the one candidate's five fold-model probabilities are averaged
  arithmetically in original test ID order. OOF rows include ID, target, fold,
  baseline prediction, and candidate prediction.
- **Audit gates:** syntax and synthetic LightGBM categorical self-test; full
  five-fold completion; independent AUC/log-loss recalculation; row/fold
  alignment; submission order/range/schema; paired bootstrap; input/source
  hashes; output manifest. Record any failure in a later append-only entry.
- **Evidence status:** all project labels have prior research exposure, so
  even a clean fold implementation is exploratory, not untouched confirmation.
- **Pre-run checks:** Python compilation and the synthetic row-independence,
  feature-exclusion, training-only category-map, and LightGBM categorical-fit
  smoke tests passed.
- **Input SHA-256:** train
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  test
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- **Pre-run source SHA-256:**
  `64e74b91f5110bb722d4f6237a29f206b8d8d1e9601b134cdc3561641de8a6e1`.

## Phase 54 results

- Baseline pooled OOF AUC `0.759105515`; q-key candidate `0.753230864`,
  delta `-0.005874651`; all five fold deltas were negative.
- Candidate log loss increased from `0.245785542` to `0.248769847`.
- Conditional paired-bootstrap 95% interval:
  `[-0.007954601, -0.003712842]`.
- The fold-local q3/q4/q5 categorical bundle is rejected for this setup. The
  q4/q5 keys have hundreds/thousands of unique levels per fold and likely add
  sparse, unstable categories without the archived notebook's smoothing.
  Phase 55 therefore tests only the lower-cardinality q3 key and a
  target-free fit-fold support statistic.
- Run completed in `410.40` seconds on CPU with LightGBM 4.7.0. Compilation,
  self-test, independent pooled/fold metric recalculation, paired bootstrap,
  ID/schema/range checks, and manifest hashes passed.
- Submission SHA-256:
  `c66c8c31df73c85e190f7dd642c1c1fa7b3b8fa1b837af118ebbee2f782fe0f8`.
- OOF SHA-256:
  `460579368672e8ef046dc8e8068de72e6dd36a23c96fb3ff41fccb46252d70ab`.
- Source SHA-256:
  `64e74b91f5110bb722d4f6237a29f206b8d8d1e9601b134cdc3561641de8a6e1`.
- Detailed output and full hashes:
  `experiments/phase54_qfingerprint_fold_local/runs/phase54_qfingerprint_20261117/`.
