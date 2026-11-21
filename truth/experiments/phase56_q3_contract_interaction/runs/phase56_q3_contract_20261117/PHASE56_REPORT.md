# Phase 56 — Q3 × contract-type interaction

## Hypothesis

A coarse annuity/credit key may behave differently for Cash versus Revolving loans. A row-local joint category tests that interaction without specialist label transfer or an inferred applicant identity.

## Protocol

- Five stratified folds, seed `20261113`; fixed `900` rounds.
- Same LightGBM settings/folds for baseline, Q3, and Q3-plus-contract interaction.
- Fold-local category dictionaries; unseen validation/test keys map to missing.
- No target encoding, cross-row label transfer, test pooling, fee constants, cached predictions, or rank blending.

## Results

| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |
|---|---:|---:|---|---:|
| baseline | 0.759105515 | 0.759141612 | 0.763026, 0.764707, 0.756379, 0.756045, 0.755550 | 0.245785542 |
| q3 | 0.762143856 | 0.762192819 | 0.765564, 0.769350, 0.758306, 0.758863, 0.758880 | 0.245029565 |
| q3_contract | 0.761223134 | 0.761250079 | 0.764857, 0.767741, 0.756261, 0.758369, 0.759023 | 0.245397576 |
- Q3 vs baseline: `+0.003038341` AUC; positive folds `5/5`; conditional bootstrap CI `[+0.001663771, +0.004534587]`.
- Q3×contract incremental vs Q3: `-0.000920722` AUC; positive folds `1/5`; conditional bootstrap CI `[-0.001656860, -0.000353467]`.
- Q3×contract vs baseline: `+0.002117619` AUC; positive folds `4/5`; conditional bootstrap CI `[+0.000763874, +0.003526736]`.

## Interpretation

The primary question was whether adding the contract-type interaction improves the Q3-only model. It did not: pooled AUC fell by `0.000920722`, the conditional bootstrap interval was entirely negative, and only one of five folds improved. Reject this interaction for now. Q3 alone improves its matched raw-feature baseline by `0.003038341`, but this remains exploratory and does not establish a generalizable gain or a 0.775 result. The Phase 20 routing result does not automatically transfer to this row-local interaction. All labels have prior project exposure.

- Runtime seconds: `419.66`.
- Source SHA-256: `4712cdf54048965457a5cbf711e447a8996f7b43f08a10c5c4f78d21b40c04c0`.
- Full hashes: `PHASE56_SHA256_MANIFEST.json`.
