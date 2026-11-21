# Phase 11: independent product-key research

Results and decisions are summarized in
[`PHASE11_RESULTS_REPORT.md`](./PHASE11_RESULTS_REPORT.md).

## Scope and evidence boundary

This study reconstructs row-wise predictors and the eight mathematical product
keys using the independent Phase 10 builder. It does not import the archived
EXP85 helper or consume Phase 9 cached matrices, predictions, or model code.
Only the permitted `train.csv` is read. The identifier is excluded from model
features; `TARGET` is passed only to fitting and evaluation. No test file,
locked holdout, external data, or submission is used.

The primary comparison is a fixed-fold GPU CatBoost raw-feature model against
the same model plus the eight product keys. Two different model seeds are
averaged within each outer fold; every model producing an OOF prediction
excludes that row's label. The bootstrap interval is a paired, stratified
row-level interval conditional on these folds and fitted models. It is not
independent confirmation because applicants and labels are reused across folds.

CatBoost's native categorical processing can use target statistics internally.
The key values themselves are label-free constructions, but this does not
settle the competition eligibility of CatBoost's categorical statistics.
Likewise, the separate target-rate scan is explicitly label-derived and remains
exploratory and ineligible for promotion until the organizer rules are clarified.

## Run

From this directory:

```powershell
python -m unittest discover -s . -p 'test_phase11*.py' -v
python phase11_independent_runner.py --folds 5 --iterations 700 --replicas 2 --output-dir ..\phase11_independent_research\runs\phase11_gpu_cv_<run-id>
```

The runner requests CatBoost GPU device 0 and fails rather than silently
falling back to CPU. `--smoke-rows`, `--folds 2`, and a small iteration count
are for plumbing checks only; their scores are not model evidence.

The CV output contains:

- `RUN_MANIFEST.json`: data/source/OOF hashes, software versions, feature
  lineage, fixed configuration, explicit exclusions, and summary metrics.
- `oof_predictions.csv`: row index, fold, target, each seed prediction, and
  within-fold bagged predictions for both feature sets.
- `fold_metrics.csv`: paired fold scores and per-seed scores.

## Candidate-key search

After the full CV run completes, use its raw-model bagged OOF predictions:

```powershell
python run_phase11_candidate_scan.py --train D:\gci\inpuy\train.csv --predictions D:\gci\truth\experiments\phase11_independent_research\runs\<run-id>\oof_predictions.csv --output-dir D:\gci\truth\experiments\phase11_independent_research\runs\<scan-id> --top-blends 100 --null-repetitions 10
```

This screens numeric singles and pairwise sums, differences, and ratios at
predeclared significant-digit resolutions; smooth-bin controls; low-cardinality
pairs/triples; selected numeric-bin/category crosses; and the eight explicit
product keys. Encodings are fitted inside the fixed OOF folds with smoothing
30. Label-free fold-frequency encodings and frequency-preserving shuffled-key
null controls are written separately. Frequency scores are reported as
standalone AUC/correlation diagnostics; they are not blended directly with
model probabilities because their raw scales are not calibrated.

The broad search adapts to labels and evaluates many candidates on the same
OOF labels. Its top result is a hypothesis, not a confirmation or unbiased
estimate of a final model. Fixed-weight blends (0.05, 0.10, 0.20) are diagnostic
only; no weights are selected for a final model. The scan checkpoints candidate
scores every 50 candidates. It does not currently resume an interrupted scan.

To test whether fold-local target-rate features add anything beyond the
eight-key CatBoost OOF model (rather than only the raw baseline), run:

```powershell
python phase11_product_baseline_te.py --predictions D:\gci\truth\experiments\phase11_independent_research\runs\phase11_gpu_cv_20261105\oof_predictions.csv --output-dir D:\gci\truth\experiments\phase11_independent_research\runs\<target-rate-id>
```

This compares the predeclared eight keys and selected amount ratios against the
same product-key OOF predictions, with frequency-preserving shuffled-key
controls. It is still an adaptive, label-derived diagnostic and remains
research-only.

## Model-side branch

The bounded model-side experiment is separately run as:

```powershell
python phase11_nested_ensemble.py --output-dir D:\gci\truth\experiments\phase11_independent_research\runs\<ensemble-id> --outer-folds 5 --inner-folds 3 --iterations 400
```

For each outer fold, the CatBoost depth/learning-rate choice is made from
inner-fold OOF predictions only. LightGBM receives numeric features plus
fold-fitted, label-free one-hot representations of raw categories and product
keys. A logistic stacker is fit on inner OOF logits; outer-fold predictions
remain untouched until evaluation. The LightGBM branch runs on CPU because the
installed build does not provide the required GPU learner. `--smoke-rows`
enables a small plumbing-only run; its scores are not evidence.

Single-key and leave-one-key-out CatBoost experiments are available separately:

```powershell
python phase11_key_ablation.py --output-dir D:\gci\truth\experiments\phase11_independent_research\runs\<ablation-id> --folds 5 --iterations 350
```

These are a fixed 5-fold diagnostic grid, not a separately held-out
confirmation. `--smoke-rows` enables a plumbing-only run. Review per-fold
changes and category support, not only pooled OOF ranking.

## Candidate-search correctness

Missing and non-finite numeric values are represented by a real category before
cross-fitting. Negative factorization codes are rejected to prevent NumPy's
`-1` indexing behavior from accidentally selecting the last category. Target
encoding requires strictly positive smoothing, and a unit test verifies that
held-out labels cannot change their own cross-fit encodings.

Do not compare any result to historical Phase 9 numbers as though those were
reconstructed on the same independent feature pipeline.
