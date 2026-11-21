# Phase 14 label-free model development

Run from the repository root:

```powershell
python .\experiments\phase14_label_free_model_development\run_phase14.py
```

The runner reads only `D:\gci\inpuy\train.csv` and the verified Phase 13 OOF
file. It reproduces Phase 13's fixed five-fold assignments, checks row/ID/
target alignment, and does not open test, submission, or external-data files.
Each candidate uses fold-fitted one-hot/frequency support and inner-fold
configuration selection and early stopping. Its outer-fold model is refit on
the full outer-fit rows.

The predeclared comparison includes a deterministic numeric-enhancement block,
LightGBM and CatBoost, an equal-weight LightGBM/CatBoost blend, and a separate
four-key track excluding the unresolved 130.5 and 0.0066 constants. Phase 13
E is the fixed reference and is not needlessly refit. LightGBM's two configs
are selected in inner folds. CatBoost uses raw categories with a fixed one-hot
threshold above every fold-fitted category count, a fixed parameter set, and
inner-fold early stopping; the runner aborts if the threshold would allow a
target-derived CTR. All eight keys remain in the main research models.
Hidden-screen target-rate candidates are excluded.

The repository's competition labels have already been exposed in prior
experiments. Therefore, nested OOF estimates are model-development evidence,
not untouched confirmation. No split over these same labels can restore
independence. Formula eligibility remains pending local-rules interpretation
and organizer certification.

The timestamped run saves OOF and fold metrics, inner selection, prediction
correlations, subgroup diagnostics, paired bootstrap intervals, an
eligibility matrix, execution log, and SHA-256 manifest. Bootstrap intervals
are conditional on fitted OOF predictions and do not account for earlier
label exposure or candidate selection.

After completion, independently verify row/ID/target/fold alignment, recompute
pooled and fold AUCs, and create paired subgroup deltas with:

```powershell
python .\experiments\phase14_label_free_model_development\verify_phase14_run.py
```

The verifier is pinned to the recorded run and its outputs. Rebuild the
run-folder `SHA256_MANIFEST.csv` after verification/report supplements have
been generated.
