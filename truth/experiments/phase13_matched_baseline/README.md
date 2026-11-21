# Phase 13 matched-baseline research

Run the matched benchmark from the repository root:

```powershell
python .\experiments\phase13_matched_baseline\run_phase13.py
```

The runner reads only `D:\gci\inpuy\train.csv` by default and creates a new
timestamped directory under `experiments\phase13_matched_baseline\runs`.
`--train`, `--output-root`, and `--seed` can override those defaults.

The fixed five-fold stratified outer split compares:

- A: raw numeric predictors.
- B: raw numeric plus fold-fitted one-hot raw categories.
- C: B plus deterministic rowwise missingness, sentinel, age, and financial
  ratio features.
- D: C plus eight ordinary one-hot product keys.
- E: C plus eight one-hot product keys with fit-fold frequency support.
- F: C plus direct numeric formula components that omit the unresolved 130.5
  and 0.0066 constants.

Each outer training partition selects one of two preregistered LightGBM
configurations through three inner folds, including inner-fold early stopping.
The runner also saves individual-key and leave-one-out ablations, five
frequency-preserving shuffled controls, a secondary-seed robustness run,
paired bootstrap intervals, fold metrics, category support, OOF integrity,
formula provenance, eligibility, execution logs, and SHA-256 hashes.
Secondary-seed results are not independent confirmation.

After the primary run completes, the separate hidden-key screen can be run
against its OOF file:

```powershell
python .\experiments\phase13_matched_baseline\run_hidden_key_screen.py `
  --phase13-oof .\experiments\phase13_matched_baseline\runs\<phase13-run>\OOF_PREDICTIONS.csv
```

That screen uses fold-cross-fitted target rates and is exploratory, adaptive,
and eligibility-unresolved. It does not belong to the label-free Phase 13
benchmark and must not be used to certify a competition-ready model.
Neither runner creates a submission or changes a champion.

To recompute all three blend weights for the eight product-key positive
controls from the saved cross-fit rates:

```powershell
python .\experiments\phase13_matched_baseline\run_hidden_key_weight_supplement.py `
  --hidden-run .\experiments\phase13_matched_baseline\runs\<hidden-screen-run>
```

The initial execution's ablation outputs are retained. The corrected,
same-seed ablation-only rerun is launched with:

```powershell
python .\experiments\phase13_matched_baseline\rerun_phase13_ablations.py `
  --original-run .\experiments\phase13_matched_baseline\runs\<phase13-run>
```

It reuses the saved D configuration and iteration count per fold, applies the
same model seed to every key subset within each fold, and writes a new
timestamped correction directory. Use that directory's ablation report for
key-level consistency conclusions.

The direct frequency-support versus ordinary-one-hot bootstrap supplement is
run with:

```powershell
python .\experiments\phase13_matched_baseline\run_phase13_comparison_supplement.py `
  --run-dir .\experiments\phase13_matched_baseline\runs\<phase13-run>
```
