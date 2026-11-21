# Phase 55 — q3 support-count ablation

## Protocol

- 171202 train / 61500 test rows; five stratified folds, seed `20261113`.
- Same fixed LightGBM configuration and folds for the raw-rowwise baseline, q3 categorical key, and q3 plus support count.
- The q3 feature is `AMT_ANNUITY / AMT_CREDIT`, rounded to three decimals.
- Support counts use only each model's fit partition; fit rows use leave-one-out counts, while validation/test rows use fit-partition counts.
- No target encoding, train/test pooling, fee correction, external data, or CSV blending.

## Results

| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |
|---|---:|---:|---|---:|
| baseline | 0.759105515 | 0.759141612 | 0.763026, 0.764707, 0.756379, 0.756045, 0.755550 | 0.245785542 |
| q3_only | 0.762143856 | 0.762192819 | 0.765564, 0.769350, 0.758306, 0.758863, 0.758880 | 0.245029565 |
| q3_support | 0.762299627 | 0.762329932 | 0.766117, 0.768520, 0.757585, 0.759312, 0.760116 | 0.244968423 |

- q3-only delta: `+0.003038341`; positive folds `5/5`; conditional bootstrap CI `[+0.001713780, +0.004426084]`.
- q3+support delta: `+0.003194112`; positive folds `5/5`; conditional bootstrap CI `[+0.002065771, +0.004438304]`.

## Interpretation

This separates a lower-cardinality product-key category from its unsupervised training-fold support. It tests a legal conservative alternative to explicit target encoding and pooled train/test counts, not the source notebook's exact pipeline. All labels have prior project exposure; treat the result as exploratory.

- Runtime seconds: `654.55`.
- Source SHA-256: `f54df005504e6a7be83705577749ee5b526cb4b7f009b3af6730dd92e60e2d73`.
- Full hashes: `PHASE55_SHA256_MANIFEST.json`.
