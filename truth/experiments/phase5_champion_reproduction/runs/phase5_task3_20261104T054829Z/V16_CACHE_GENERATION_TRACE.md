# V16 cache generation trace

Investigation run: `20261104T054829Z`; UTC start `2026-11-04T05:48:32.038531+00:00`.

## What was recovered

- The V16 executed notebook is `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb` (file SHA-256 `ca77c6a6dbf6f5d1ea80b4f0f23de1f0f0c73ea7b4ac6fab937fd32ed8881306`; canonical code-cell SHA-256 `abcd2b75559f51e5744ec97763dc83cfd5fbecffb13a73d008b2de318149a630`). Its setup explicitly writes model caches to Colab Drive `gci_cache_v16`; the saved notebook output identifies `/content/drive/MyDrive/gci_cache_v16`.
- The local `gci_cache_final` archive is `D:\gci\codex\gci_cache_final-20261021T075423Z-1-001.zip` (SHA-256 `a6261b5ca5a90dceb793bf2f50ca16a4437114375c718e0a89313295afd15999`, 190200921 bytes; archive entry times 2026-10-15T21:23:08 through 2026-10-20T22:22:18). The contemporaneous research note identifies it as the V15 Google Drive export. It has 49 members.
- Every archive member that exists in the extracted `D:\gci\codex\v15_cache\gci_cache_final` directory is inventoried in `V16_CACHE_PROVENANCE_MATRIX.csv`; the matrix records bytes, SHA-256, timestamps, member hash match, NPZ keys/shapes/dtypes, and absent row/fold/source/data sidecars. Candidate V15 NPZ files listed in the archive are byte-identical to their archived members. This verifies extraction integrity, **not** the original model training run.
- Candidate model arrays are in the V15 `gci_cache_final` archive, not in a located `gci_cache_v16` directory. A recursive local search found configured V16 cache directory present = `False`. The V16 notebook says it creates its own cache folder and only intentionally reuses selected label-free V15 residual/view helpers. Thus the recovered V15 model arrays are the wrong generation for reproducing the V16 model sets.
- The V16 source implements 5-fold `StratifiedKFold(shuffle=True, random_state=42)`, fixed LightGBM rounds, five fixed member weights, and CatBoost folds. Its feature/label encoding code is present, but the missing V16 cache manifests do not record actual fold indices, input hashes, model versions, or feature schemas.

## Source candidates and provenance confidence

| Candidate | Evidence | Confidence |
|---|---|---|
| `D:\gci\pre info\GCI_FINAL_v15_end_to_end_colab.ipynb` | Source hash `421d223754c75fce1fff7347ca7ac2b91f8b171929c59d185fb07314369f4c1d`; documented as the formula/cache lineage for V15; cache ZIP dated Oct 21 and exact extracted member hashes. | V15 lineage plausible; exact run/input binding absent. |
| `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb` | Source hash `abcd2b75559f51e5744ec97763dc83cfd5fbecffb13a73d008b2de318149a630`; executed outputs claim V16; configured cache root is `gci_cache_v16`. | Historical V16 source and saved output identified; original model caches absent. |
| `D:\gci\gemini\train_v16_and_optimized.py` | Hash `d8e1eebc5656a1f1fcfa557ac0838eaeb2d370c6d8b28efa67ae1cd16bc18cb8`; a separate local optimized pipeline that fits different member code and searches blend weights on the same OOF labels. | Not the executed V16 notebook pipeline; cannot explain its reported fixed-weight metric without run evidence. |
| `D:\gci\gemini\scratch\generate_exp85_solo_notebook.py` | Hash `77607c0fd838b31828b154131c95de7a4557b7f85cc389b95944e119fce4e374`; contains code writing V6/V8/V16 bundles from a shared `v6_oof`. | Strong match to anomalous bundle structure; execution provenance not stored. |
| `D:\gci\gemini\scratch\build_clean_exp85_solo_notebook.py` | Hash `5504f009d42e809c4aa00e5728a04d1c1dafee29e8e3a63216644536d99e4dd2`; contains the same shared-array placeholder and fixed AUC fields. | Same scaffold pattern; current file's filesystem time is later than bundle mtime. |

Filesystem timestamps are retained as supporting clues only; copying, extraction, and clock settings can change or preserve timestamps. The V16 cache provenance is **not fully recovered**.

## Why the candidate score is low

The `0.77404329` candidate fixed blend uses V15 archive members and the fixed V16 blend formula; it is in the V15 reference band (`0.77404`). The historical V16 output says it used new V16 set predictions, saved to `gci_cache_v16`, and reports `0.77649`. The candidate cache component scores do not match those outputs. The principal explanation is now identified: the previous comparison substituted V15 prediction caches for missing V16 model caches. Therefore `0.77404329` is not evidence that V16 scored lower; it is a V15-cache diagnostic and does not reproduce V16.

The archive and prediction arrays do not embed row IDs or fold IDs. Source code implies row-order use and seed-42 folds, but neither can be verified from these arrays. The CatBoost reconstruction additionally infers shard-to-fold assignment from shard order and expected validation lengths. See the discrepancy and provenance CSVs for exact metrics.
