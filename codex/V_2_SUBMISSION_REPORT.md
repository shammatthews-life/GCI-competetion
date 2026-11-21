# v_2 standalone product-model submission

## Recommendation

**The CSV is valid, but I do not recommend spending a leaderboard submission on it.** The local OOF result is well below the reported v15 OOF reference. Use it only if you explicitly want one diagnostic leaderboard probe and have spare submissions.

## Leaderboard result

The user-submitted screenshot shows that this v_2 candidate scored **0.77198**. It is therefore rejected and should not be used again. The same screenshot's score history/chart shows the earlier **0.77891** result still exists; v_2 did not reproduce or improve it.

## Files

- Candidate CSV: [v_2_submission_hierarchical_product.csv](./v_2_submission_hierarchical_product.csv)
- Runner: [run_v2_submission.py](./run_v2_submission.py)
- Metrics: [v_2_submission_metrics.json](./v_2_submission_metrics.json)
- Paired OOF and test scores: [v_2_submission_oof.npz](./v_2_submission_oof.npz)

The candidate was built entirely in `D:\gci\codex`, using the isolated environment at `D:\gci\codex\.venv`. It is a two-seed, five-fold LightGBM ensemble with formula-aware application features and leakage-safe Q4/Q3/Q4×contract target encodings. The final submission scores are rank-normalized across the test set and preserve sample ID order.

## Local evaluation

| Measure | Result |
|---|---:|
| Five-fold OOF AUC, seed 42 | 0.764543 |
| Five-fold OOF AUC, seed 117 | 0.765182 |
| Two-seed averaged OOF AUC | **0.766138** |
| Lowest / highest fold AUC across both seeds | 0.760509 / 0.769258 |
| Model fits | 10 |
| Submission rows / columns | 61,500 / `SK_ID_CURR,TARGET` |
| Candidate SHA-256 | `6a2ed527d4889f0f6a54b4e0b403c98ca5217188c2ae01ace61fae65b73747be` |

The archive reports v15 OOF AUC of 0.774104 on its five-fold seed-42 recipe and a public leaderboard score of 0.77891. Although those are not an exact paired comparison with the v_2 two-seed model, this candidate's standalone OOF is about 0.008 lower. There is no evidence that it will beat v15; it is likely to score lower on the leaderboard as well.

## If you decide to submit it anyway

Upload exactly:

`D:\gci\codex\v_2_submission_hierarchical_product.csv`

Keep v15 as the fallback and record the returned score. Do not replace the champion based on this local OOF result.

## Reproduction and environment

From PowerShell:

```powershell
& 'D:\gci\codex\.venv\Scripts\python.exe' 'D:\gci\codex\run_v2_submission.py'
```

Validated with Python 3.12 and LightGBM 4.7.0. The venv lives inside `codex`; all generated result files are saved there. The only inputs read are the supplied competition train, test, and sample-submission CSVs.
