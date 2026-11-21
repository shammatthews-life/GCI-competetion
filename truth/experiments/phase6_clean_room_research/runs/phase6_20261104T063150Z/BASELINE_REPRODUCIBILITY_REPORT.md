# Baseline reproducibility report

## Result

- Fresh raw-only five-fold OOF ROC-AUC: **0.760658451**.
- Historical corrected Phase 2 reported OOF: `0.760658` (existing reproducibility evidence).
- Fresh minus historical rounded claim: `+0.000000451`.
- Fold AUCs and validation counts: `BASELINE_FOLD_METRICS.csv`.
- OOF file: `RAW_BASELINE_OOF.csv`; SHA-256 `03a88fc40fa8071cce342bc26eda221bc0359993c8688311a43e8c36521e8de3`; exactly one finite prediction per training row, keyed by ID and joined back to TARGET/fold.
- Fold assignments SHA-256: `a8455332203da5234e3eb180cc1852b95b5dccdc627ceb7f6d22e8d0f5cd5cfb`; match to preserved Phase 2 fold mapping by ID: `True`.
- Training runtime: `80.4` seconds. No feature-family search or large model sweep was run.

## Protocol and restrictions

The raw-only runner is the preserved corrected Phase 2 implementation: five stratified folds, seed 42, nested early stopping on a stratified inner split of each outer training partition, fit-partition feature framing, and validation prediction only after refitting on all outer-fit rows. The model uses raw/rowwise fields from the supplied training data. No test predictor values were read; test header/IDs were used only for schema and ID checks. No external data, target encoding, pooled train/test statistics, manually curated prediction rule, or submission was used.

This confirms the baseline under the current available code and data hashes; it does not make the reused five folds an untouched estimate. The model/configuration was inherited from prior research and the training labels have been repeatedly examined. The reported historical Phase 2 score is numerically reproduced only if the fresh value agrees within numerical tolerance; otherwise the discrepancy remains explicit and must be traced, not rounded away.

## Recommendation

Use this as the Phase 6 clean-room development reference, not as a champion claim. Require identical folds for initial paired discovery, and acquire genuinely unseen labeled data for independent confirmation.
