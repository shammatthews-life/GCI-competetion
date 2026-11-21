# GCI follow-up: rescuing weak hypotheses through targeted optimization

Date: 2026-10-20. Metric: ROC-AUC. Data scope: supplied `train.csv` and `test.csv` only.

## Validation protocol

- Compact formula-feature baseline, same fold assignments within each experiment.
- Fresh confirmation seed: `20261027`, three stratified folds.
- All target-encoded training features were inner-cross-fitted inside the outer training partition. Outer validation encodings used only outer-training labels.
- The compact baseline on the fresh split scored **0.75704260 OOF AUC**.
- These results are **not the v15 full ensemble** and must not be interpreted as a measured gain over public leaderboard 0.77891.

## Five distinct optimized mechanisms with positive fresh-split evidence

| Mechanism | Fresh-split pooled OOF AUC | Delta vs compact baseline | Fold deltas | How to interpret |
|---|---:|---:|---|---|
| Hierarchical Q4/Q3/contract product encoding | 0.76314147 | **+0.00609887** | +0.005698, +0.006238, +0.006372 | Best stable standalone branch in this run; fine product identity is shrunk toward coarser product and contract context. |
| Product × applicant-category backoff | 0.76159879 | **+0.00455619** | +0.005237, +0.003656, +0.004644 | Product risk changes by income type, occupation and education; smoothing prevents rare groups from dominating. |
| Product × formula-residual / affordability | 0.76235639 | **+0.00531379** | +0.005238, +0.004825, +0.005758 | Product-specific deviations from the annuity mechanism and affordability burden provide useful conditional signal. |
| Product × external-score profile | 0.75844409 | **+0.00140150** | +0.001264, +0.002071, +0.000819 | Smaller, but positive in all three folds. |
| Local-neighbour risk correction, 30% rank blend | 0.75756924 | **+0.00052665** | +0.000589, +0.000649, +0.000306 | Standalone local-risk predictions are weaker; their errors contain complementary ranking information when blended conservatively. |

Two additional blend-only mechanisms also repeated on the fresh split:

- Quantile-regime target encoding, 30% rank blend: **+0.00043719**, all three folds positive.
- Multiscale unsupervised prototype risk, 30% rank blend: **+0.00053788**, all three folds positive.

The category-conditioned Q4 encoding separately scored **+0.00499885** versus its compact baseline on the fresh split, with all three folds positive. Product Q4 target encoding scored **+0.00581192** on the same fresh split, all three folds positive.

## Complementarity

On the fresh split, blending the hierarchical Q4/Q3/contract branch with the product-formula/burden branch improved on the hierarchical branch itself. At a 40% weight for the formula/burden branch, pooled OOF AUC increased by **+0.00060883**, with fold gains **[+0.0007565, +0.0002912, +0.0007182]**. The blend weight was explored on validation predictions, so treat the exact weight as exploratory until another frozen confirmation.

## Why this is a real advance—but not yet a leaderboard claim

The earlier negative results were not all dead ends. Changing simple category-pair frequencies into smoothed category-conditioned product encodings, and changing arbitrary risk interactions into hierarchical product backoff, yielded repeatable gains. Prototype and local-neighbour branches did not win alone, but conservative rank blending recovered small positive gains across all three folds.

However, the current public champion's OOF/member cache is not available in the working directory. The incumbent v15 notebook already contains Q3/Q4/Q5/P3/P4/P5/PF4 product target encodings, so product-only gains against a compact baseline cannot establish a v15 improvement. The new product×category and product×formula-quality blocks are the highest-priority additions to test inside the exact v15 pipeline.

**No submission file was created.** Next promotion gate: add these feature blocks to the exact v15 member pipeline, compare against matched v15 OOF predictions, and keep a candidate only if the full ensemble improves with fold stability.
