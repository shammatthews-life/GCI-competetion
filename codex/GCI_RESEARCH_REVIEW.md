# GCI competition research review

**Prepared:** 20 October 2026  
**Scope:** Local files under `D:\gci\inpuy`, `D:\gci\pre info`, and `D:\gci\gemini`.  
**Metric:** ROC-AUC.  
**Purpose:** Preserve what has been tried, separate real evidence from leaderboard claims, and identify the next experiment worth running.

## Executive summary

- The current reported champion is **v15**, with public score **0.77891** and reported rank **17**. Keep it unchanged as the fallback.
- The best validated progression came from decoded loan-product features, a complementary product-native CatBoost branch, LightGBM ExtraTrees members, consistent multi-view predictions, and rank-space blending. The reported public progression is 0.77865 → 0.77884 → 0.77891.
- The archive's strongest *new research hypothesis* was smoothed product identity and product-conditional target encoding. Q4 encoding beat its own compact LightGBM baseline by **+0.005812** on the reported second split, positive on all 3 folds. This remains a compact-model finding, not a v15 result.
- A new, leakage-safe **v_1 confirmation** tested product × applicant and product × formula/affordability encodings against product-only features on two fresh split seeds. Applicant conditioning lost on both; formula/burden conditioning was mixed and lost on all folds of the second seed. Neither is promoted.
- The subsequent `v_2` standalone submission scored **0.77198** on the user's leaderboard screenshot, well below the earlier **0.77891** result. It is rejected; do not submit it again.
- **No replacement submission or validated v15 improvement was produced.** The exact v15 OOF/member cache and champion submission are not in this workspace. A standalone compact model would be a known weaker reference, not a responsible leaderboard candidate.
- The isolated ML environment and reproducible v_1 experiment are in `codex`; do not treat installing the ML packages as establishing a leaderboard gain.

## What I inspected

### Supplied data

The input folder is named `inpuy` (spelling as present on disk).

| File | Rows × columns | Checks |
|---|---:|---|
| `inpuy\train.csv` | 171,202 × 34 | 171,202 unique IDs; target counts 157,381 zero / 13,821 one (8.073% positive) |
| `inpuy\test.csv` | 61,500 × 33 | 61,500 unique IDs |
| `inpuy\sample_submission.csv` | 61,500 × 2 | IDs match `test.csv` exactly and in order |
| `inpuy\WhatsApp Image 2026-10-20 at 5.20.00 PM.jpeg` | image | Read; supports the dated research context, not a new validation artifact |

Data hashes, headers, representative rows, last rows, target counts, and ID alignment were checked with the Python standard library. IDs are sequential (`train` 0–171201, `test` 171202–232701); prior analysis found no useful ID signal, so the ID should be used only for submission alignment.

### Research files

I read the master research archive, standalone v17/v18/v20/v21/residual reports, the v17 code dump and notebook, the v18 notebook, and the v33 experiment script. I also inspected text and metrics in all five ZIP archives:

- `GCI_CA_EXT_promising_research.zip`
- `GCI_v24_feature_family_results.zip`
- `GCI_v26_optimized_five_research (1).zip`
- `GCI_v29_five_data_mechanisms_research.zip`
- `GCI_v17_catboost_E1_stack_colab.ipynb` (the two extensionless copies are byte-identical to this notebook)

The ZIP result arrays are OOF/cache data; their available array headers and shapes were inspected where relevant. The original leaderboard submission files and full `gci_cache_final` are **not** present under `D:\gci`.

## How the current best model was built

The feature work is not just generic ratios:

1. **Clean and represent the original application fields.** Preserve missingness, clean the `DAYS_EMPLOYED == 365243` sentinel, and add age/employment, financial burden, and external-score summaries.
2. **Decode the loan pricing mechanism.** Annuity is approximately `AMT_ANNUITY = Q × AMT_CREDIT + F`; a fixed monthly fee around 129.5 explains many cash loans. Fee-adjusted Q keys at multiple precisions (`Q3/Q4/Q5`, `P3/P4/P5`, `PF4`) expose repeated product identities. The research reports roughly 1,047 decoded products versus 5,335 rounded raw ratios. A separate `K_DEC` / credit-to-goods tariff representation is also used.
3. **Encode product risk without label leakage.** Product target encodings are smoothed and nested/out-of-fold for training rows. This prevents a row's own target from entering its feature.
4. **Build diverse learners and views.** The E1 stack adds learned-expectation residual features and EXT/LOAN/PERSON view scores. ExtraTrees LightGBM replacements improved all five members in the earlier direct check; a product-native CatBoost branch supplied different ranking signal.
5. **Align stacking and blend ranks.** v15 averages inner-view models for validation/test to better match how training-row view features were generated, then uses a fixed rank-space blend.

The v15 recipe reported in the research archive is:

`0.90 × rank(0.75 × rank(B_v14G_base) + 0.25 × E1Aet_blend) + 0.10 × rank(CatBoost_product)`

Its matched OOF gain over v14G was +0.00016669, positive on 5/5 folds, with a reported paired-bootstrap interval [+0.00007838, +0.00026092]. The corresponding public score reported by the user is 0.77891. The five folds were reused during a long search, so this interval does not remove model-selection bias.

## What the experiment bundles teach us

| Finding | Evidence in the supplied research | How to use it |
|---|---|---|
| Discrete loan-product identity matters | v29 Q4 target encoding: +0.005812 vs its compact baseline on the reported second split, 3/3 positive folds. The report describes the hierarchical Q4/Q3/contract version as strong on its first split too. | Do not simply add another Q4 encoding to v15: v15 already has product encodings. Test *incremental conditional/backoff* structure against the exact champion. |
| Product risk may vary by applicant/product context | v29 compact-model results were positive, but the fresh v_1 follow-up lost for product × applicant on both seeds; formula/burden was inconsistent and lost across all folds of the second seed. | Do not repeat these exact blocks or treat the earlier screen as stable evidence. Revisit only with a genuinely different hypothesis and matched v15 OOF. |
| Product formula features can help a compact model | v24 fee-adjusted annuity formula +0.007256; affordability scales +0.006220; tariff tiers +0.001187, all vs that bundle's compact baseline. v29 supports formula/burden conditional encodings. | Keep the domain hypothesis, but measure the incremental value after v15's existing ratios and product keys. |
| A product × EXT signal exists but is smaller | v29 product × external-score profile: +0.001034, all 3 folds positive. A separate CA/EXT bundle reports +0.004862 vs its own compact baseline using rounded credit/annuity fingerprints and EXT quantiles. | Lower priority; use only if it adds value to v15 after product × applicant and product × affordability tests. |
| Standalone AUC is not ensemble value | v17 CatBoost-E1 standalone AUC 0.77444, but replacing/averaging it into the champion changed AUC by about 0.00000/+0.00002. Its rank correlation with the old CatBoost was 0.9758. | Evaluate new components by paired full-ensemble OOF gain and error diversity, not their standalone score. |
| Cheap screens are only filters | Bureau bursts and contract-affordability passed 80%/3-fold screens but were neutral in full v15-member tests. | Do not turn a quick-screen delta into a submission weight. |
| A little rank-space perturbation can hurt the public score | v20/v21 tariff probes were compact-model positive, but user-reported leaderboard results were about 0.7789 / 0.77889 / 0.77888, not better than 0.77891. | Keep v15 and don't infer leaderboard improvement from the compact branch. |

### Specific v29 result caveat

The file `metrics_seed20261027_confirmation.json` records `"seed": 20261032`, but its runner `run_gci_v29_fresh_confirmation.py` explicitly uses `StratifiedKFold(..., random_state=20261027)` for the outer folds. The report and result-file name also identify the run as 20261027. This looks like a stale metadata value; use the runner's split code as the evidence for the seed and fix the metadata in any reproduction. The result remains a compact-model comparison, not a v15 comparison.

## Important failures to avoid repeating

- **Wrong reference model:** residual probe's +0.000095 was against a compact model, not v15; its submitted score was 0.77869. It also trained its correction on long-iteration OOF scores but supplied a 60-tree inference score, creating train/inference mismatch.
- **Feature blocks that failed in the full pipeline:** bureau enquiry bursts (full delta −0.00000107), contract-decomposition affordability (+0.0000022), product × segment statistics in the earlier implementation (−0.000063), and repeated view-cross-fitting. Their quick screens do not override the full results.
- **Other weak directions:** ID-only trend, generic KNN, DAE, scratch MLP, pairwise ranker, ordinary XGBoost, class reweighting, GOSS, linear trees, extra CatBoost seeds/depths/losses, extra ExtraTrees seeds, and repeated blend-weight tuning. See the archive for the exact run-specific numbers.
- **Negative v25 screen:** category-pair frequency, quantile interactions, prototypes, IsolationForest, and risk-threshold hinges all lost against the compact baseline; the risk-threshold block was strongly negative.
- **v17 writes rejected candidates:** notebook cell 6 prints PASS/fail, but writes each CSV unconditionally after the `ok` calculation. A file being present is not proof that it passed. The standalone review recommends fixing the write to occur only inside the passing branch.
- **v33 is exploratory, not ready to submit:** its script writes 0.5%-blend and model-only CSVs without computing OOF AUC or testing the formula against aligned v15 OOF. It also expects an existing v15 CSV and the ML packages, neither present here.

## Recommended next experiment

Do not run another broad sweep. First recover the exact v15 notebook/source recipe, its row-aligned five-fold OOF predictions and fold indices, member OOF predictions, and the v15 test submission. Then:

1. Freeze the v15 predictions, fold membership, and model parameters as the baseline.
2. Add one predeclared, leakage-safe feature family at a time to the existing pipeline: prioritize **hierarchical/smoothed Q4/Q3 product backoff conditioned on applicant category**, then **Q4 × formula-residual/affordability backoff**. Do not repeat raw pair target means or product-only Q4 encoding already in v15.
3. Cross-fit every target encoding inside each outer training fold; compute validation encodings from only that outer-training fold; for the test set, fit encoders on all training labels. Keep the feature construction and model scoring recipe identical between OOF and test.
4. Compare the *whole fixed v15 ensemble* to the candidate on identical held-out rows. Report pooled AUC delta, each fold delta, paired bootstrap interval, rank correlation, and whether errors improve specifically on pairs v15 misranks.
5. Use the established promotion gate: at least **+0.00015** pooled gain, **4/5** positive fold gains, and paired-bootstrap 95% lower bound above zero. Because the same folds have been extensively reused, confirm a frozen winner on a fresh split or an untouched confirmation set before generating one candidate submission.
6. Preserve v15 as fallback; make no public leaderboard submission until the exact comparison passes.

## Workspace limitation and next handoff

This review verified the supplied files and archived evidence and ran the fresh v_1 compact-model confirmation using an isolated environment in `codex`. The critical v15 OOF/cache plus scored submission are still missing from `D:\gci`. The next required input is the `gci_cache_final` OOF/member cache (or a complete equivalent export), the v15 notebook/source, and the exact champion submission. Without those, a claimed score increase would be unsupported.

## Local file map

- Data: `..\inpuy\train.csv`, `..\inpuy\test.csv`, `..\inpuy\sample_submission.csv`
- Research archive and reports: `..\pre info\GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md`, `..\pre info\REPORT.md`, and the other v17–v33 reports listed above
- Current detailed branch script: `..\pre info\GCI_v33_product_conditional_experiment.py`
- v17 experiment code: `..\pre info\GCI_v17_catboost_E1_stack_colab.ipynb` and `..\gemini\v17_cells_dump.py`
