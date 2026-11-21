# Corrected Phase 2 experiment results

## Decision

**Research result: retain for further investigation; not a submission candidate.**
The experiment passed its local validation and artifact-integrity checks, but
its official score was not evaluated and competition-rule compliance has not
been certified. No submission was created.

## Matched OOF results

Both variants used the same five seed-42 stratified outer folds, inner stopping
protocol, model settings, and fixed 50:50 probability blend.

| Variant | LightGBM OOF AUC | CatBoost OOF AUC | Fixed blend pooled OOF AUC |
|---|---:|---:|---:|
| Raw-only matched baseline | 0.76065845 | 0.75880847 | 0.76136207 |
| Corrected Phase 2 features | 0.76972349 | 0.76993248 | 0.77110265 |
| Phase 2 minus baseline | — | — | **+0.00974058** |

| Outer fold | Baseline blend AUC | Phase 2 blend AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.75869926 | 0.76802278 | +0.00932352 |
| 2 | 0.76194799 | 0.77402299 | +0.01207500 |
| 3 | 0.76534742 | 0.77538780 | +0.01004039 |
| 4 | 0.75777287 | 0.76861939 | +0.01084652 |
| 5 | 0.76341747 | 0.76973876 | +0.00632129 |

All five paired fold deltas were positive. These figures are local OOF
measurements, not leaderboard evidence and not directly comparable to a
public score from another validation or submission protocol. The previously
stated 0.79000 objective was not reached.

## Validation and integrity

- 171,202 training rows and 61,500 test rows; 32 supplied predictor columns.
- Saved fold assignments independently reproduced from the labels and seed.
- Iteration selection used a deterministic 10% split inside each outer
  training fold; the outer validation fold did not control early stopping.
- Target encodings and fee-frequency statistics were fitted within the
  corresponding training partitions.
- Raw input and source snapshot hashes matched the run manifest.
- OOF/test arrays, fixed 50:50 blends, metric calculations, and sample/test ID
  order passed independent verification.
- No external dataset, prediction cache, or model checkpoint was used.
- No submission CSV was produced.

The machine had an NVIDIA GeForce RTX 4050 Laptop GPU. CatBoost was configured
for `GPU:0`; LightGBM was configured for CPU. `nvidia-smi` snapshots before
and after training showed 306 MiB / 0% and 549 MiB / 5% utilization,
respectively, from 6,141 MiB total. During-run spot samples observed about
5.1–5.4 GiB memory use and 36–50% utilization; these were not continuous
measurements and should not be treated as true peak utilization.

## Compliance boundary and next step

This run used only supplied train/test CSV fields and did not use outside
datasets. However, the available evidence does not establish organizer
approval. In particular, the provenance of the archived fee/product constants
(`129.5` and `0.0132`), supervised target encoding, and any transductive
interpretation still require review against the official rules or organizer
clarification. Do not describe this experiment as certified compliant.

The authoritative machine-readable metrics and independent checks are in
[`VERIFIED_METRICS.json`](./VERIFIED_METRICS.json). The exact executed runner
snapshot and all run artifacts are hash-recorded in
[`SHA256_MANIFEST.json`](./SHA256_MANIFEST.json).
