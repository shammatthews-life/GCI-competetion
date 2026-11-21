# GCI World 2026 — Complete Cross-Chat Research Archive & Improvement Master Plan

**Consolidation date:** 20 October 2026  
**Purpose:** Preserve the research and decisions across the GCI project conversations and available project files so future work does not repeat failed experiments or lose context.  
**Metric:** ROC-AUC  
**Dataset:** supplied `train.csv` (171,202 rows), `test.csv` (61,500 rows), `sample_submission.csv`; training target prevalence approximately 8.073%.  
**Current champion:** `GCI_submission_v15_rankspace_consistent_view.csv`; leaderboard screenshot shows `SHAM_MATTHEW_S` at rank 17 with **0.77891**. User has also reported 0.77892 in chat; preserve both statements and treat screenshot-visible 0.77891 as the latest directly visible figure.  
**Most recent rejected candidate:** v18 alt-direction submission, public score **0.77882**.  

> **Scope and completeness note:** This master combines the full research log, master handoff, v15 report, local feature-experiment reports, residual forensic diagnosis, v17/v18 follow-up results, OLD/NEW member experiment, and the 20 October leaderboard screenshot details available in the current GCI project context. It cannot include a chat or attachment that is not indexed/available in this project context; where a source is missing, that gap is stated instead of invented. The source reports are retained verbatim in appendices below to preserve detailed terminology and results.

## Quick navigation
1. [Current state and decisions](#current-state-and-decisions)
2. [Leaderboard snapshot and target gaps](#leaderboard-snapshot-and-target-gaps)
3. [Champion recipe and timeline](#champion-recipe-and-timeline)
4. [Cross-chat research findings](#cross-chat-research-findings)
5. [v17 and v18 findings](#v17-and-v18-findings)
6. [OLD/NEW member direction](#oldnew-member-direction)
7. [Next research plan](#next-research-plan)
8. [Artifact inventory](#artifact-inventory)
9. [Appendix A: full research log](#appendix-a-full-research-log)
10. [Appendix B: master handoff](#appendix-b-master-handoff)
11. [Appendix C: v15 candidate report](#appendix-c-v15-candidate-report)
12. [Appendix D: feature experiment reports](#appendix-d-feature-experiment-reports)
13. [Appendix E: residual forensic diagnosis](#appendix-e-residual-forensic-diagnosis)

---

## Current state and decisions

- **Keep v15 unchanged as the fallback/champion.** Public leaderboard evidence is stronger than every later candidate currently discussed.
- **Do not submit v18 again.** User reports public score 0.77882, below v15's 0.77891/0.77892.
- **Do not submit the residual-ranking probe again.** It scored 0.77869 and had train/inference iteration mismatch plus an invalid comparison against a weaker compact baseline.
- **Do not promote v17's view-feature CatBoost candidate.** A quick 30k-row/3-fold screen showed consistent-view features reduced AUC by about 0.00159; this was not the full v17 pipeline, so it is a failed screen rather than a universal rejection of CatBoost stacking.
- **Treat compact OLD/NEW local blends as hypotheses only.** They have not been compared to the exact v15 OOF predictions.
- **Critical missing artifacts:** `gci_cache_final` / aligned champion OOF and fold metadata. Without those, we cannot honestly claim a local candidate improves the exact v15 ranker.
- **Rule:** never compare raw OOF AUC values from different fold recipes, sample sizes, feature pipelines, or training lengths as if they were the same benchmark.

## Leaderboard snapshot and target gaps

The 20 Oct screenshot visibly showed:

| Rank | Account shown | Score |
|---:|---|---:|
| 1 | ankit_chaubey | 1.00000 |
| 2 | ParamveerSingh | 0.87342 |
| 3 | Sourav_Pandey | 0.85454 |
| 4 | im_jayanthh | 0.80372 |
| 5 | diwanparker | 0.80182 |
| 6 | WULRICHNYENDEZA3 | 0.79438 |
| 7 | NestoV_2207 | 0.79362 |
| 8 | ypdv2608 | 0.79362 |
| 9 | hasan_mazi11 | 0.79106 |
| 10 | LucasGastiaburu | 0.78616 |
| 11 | DragunDaniel | 0.78616 |
| 12 | Jamesty16 | 0.77963 |
| 13 | ThanaphumS | 0.77939 |
| 14 | nawatpim | 0.77920 |
| 15 | ahmadmutawa | 0.77899 |
| 16 | Shehraz_S_67 | 0.77899 |
| **17** | **SHAM_MATTHEW_S** | **0.77891** |
| 18 | Joanna_Carrion | 0.77878 |
| 19 | nShah1dx | 0.77878 |
| 20 | Alexander_L | 0.77875 |
| 21 | kaikubez | 0.77874 |

Displayed score gaps from 0.77891: +0.00008 to tie/pass ranks 15–16; +0.00029 to rank 14; +0.00048 to rank 13; +0.00072 to rank 12; +0.00725 to ranks 10–11. The screenshot alone cannot explain the 1.00000 score at rank 1; do not infer leakage or legitimacy from score alone.

## Champion recipe and timeline

| Version | Public score reported/visible | Local OOF anchor where recorded | Status |
|---|---:|---:|---|
| v9 reproduction | 0.77865 | ~0.77330055 | Historical baseline |
| Product-aware CatBoost blend | displayed ~0.7787 | ~0.77352213 | Historical step; rounding ambiguity |
| v13_B OLD + E1 + E1 ExtraTrees | no independently verified current public score | ~0.77376822 | Useful model-training step |
| v14G + 10% CatBoost | 0.77884 | 0.77393726 | Former champion |
| v15 rank-space consistent-view | 0.77891 (screenshot); user also said 0.77892 | 0.77410395 | **Current champion** |
| Residual-ranking probe | 0.77869 | invalid comparison vs champion | Reject |
| v17 CatBoost + consistent-view probe | no accepted public score | quick screen negative by ~0.00159 | Do not promote |
| v18 alt-direction probe | 0.77882 | local simple blend only | Reject as submission |
| Compact OLD/NEW blend | not submitted | exploratory local AUC ~0.75675 | Research only |

**v15 formula (rank-space, as recorded in the handoff):**

`candidate = 0.90 * rank(0.75 * rank(B_v14G_base) + 0.25 * E1Aet_blend) + 0.10 * rank(CatBoost_product)`

Here `B_v14G_base` is the branch mixture underlying v14G before its 10% CatBoost combination, and `E1Aet_blend` is the fixed-weight blend of five consistent-view ExtraTrees members. This is not identical to the raw arithmetic G3 formula in the original notebook.

v15 matched local OOF comparison against v14G: +0.0001666930; fold gains +0.00026809, +0.00020539, +0.00005971, +0.00002809, +0.00026588 (5/5 positive); paired bootstrap 95% interval [+0.00007838, +0.00026092]. Note extensive reuse of folds means this interval does not eliminate selection bias.

## Cross-chat research findings

### Repeated successful patterns
- Decoded product structure was important: approximately `AMT_ANNUITY = Q0 * AMT_CREDIT + F`, with a monthly fee near 129.5 for about 75% of cash loans; roughly 1,047 decoded product values versus 5,335 rounded raw annuity/credit values.
- Product-aware CatBoost added complementary ranking information in the v14G blend.
- LightGBM `extra_trees=True` on E1 multi-view features helped the v13/v15 lineage; do not assume it helps on every simpler feature matrix.
- Consistent generation of view predictions (averaging inner view models for validation/test) was motivated by train/inference alignment and led to the v15 successor.
- Rank-space blending of diverse models has repeatedly been more promising than adding arbitrary feature blocks.

### Repeated failure pattern
Several features improved a fast or sampled screen but did not improve the full ensemble: bureau-enquiry bursts and contract/product affordability are explicit examples. A single member's AUC gain is not evidence of a gain after blending with the champion.

### High-level rejected/weak directions (details in appendix)
ID order trend; generic KNN neighbour default rate; DAE; scratch MLP/pairwise AUC; ordinary XGBoost; sampled pairwise LightGBM; reweighting; GOSS; linear trees; `max_bin=127`; additional CatBoost seeds/depth/loss variants; product×external-score deciles; external-score×affordability and age/employment interactions; product×segment encodings; repeated cross-fit views; some bureau/contract feature blocks; small blend weight changes. Some individual tests were underpowered or model-specific; “reject” means do not repeat unchanged without a new hypothesis.

## v17 and v18 findings

### v17 CatBoost-E1 consistent-view probe
The uploaded notebook `GCI_v17_catboost_E1_stack_colab.ipynb` was considered as a possible CatBoost replacement/augmentation. A quick local screen used 30,000 rows and 3 held-out folds:
- base CatBoost variant pooled AUC about 0.75354;
- base + consistent-view predictions about 0.75195;
- difference about -0.00159, negative on all three folds.
This is a negative screen only; it did not run the full residual-feature pipeline with the missing `gci_cache_final` artifacts. Do not describe it as a full v17 validation.

### v18 alt-direction probe
- Concept: rank blend two simpler LightGBM variants, then combine 5% of v18 rank with 95% of v15 rank.
- The local simple model branch had encouraging complementarity: the frozen 50:50 rank blend improved the better individual variant by about +0.000536 on one independent 3-fold screen; another feature-variant confirmation recorded +0.000599 on a separate split. These are local model results only.
- Public submission scored **0.77882**, below v15 by about 0.00009–0.00010 depending on whether the v15 score is read as 0.77891 or 0.77892.
- Decision: reject this exact 5% v18 submission formula. Local component diversity did not transfer to a positive leaderboard result; do not repeat same candidate.

## OLD/NEW member direction
“OLD” and “NEW/E1” in the historical project refer to model/feature families, not an actual supplied customer-tenure label. The source application table has registration/document/phone recency features but no explicit customer membership start date. Do not claim `DAYS_REGISTRATION` equals customer tenure.

- OLD family: core application predictors and established engineered financial/age/sentinel/missingness features, with product/target-encoding details depending on member.
- NEW/E1 family: learned-expectation residual representations, cross-fitted view scores for EXT/loan/person groups, view logits and pairwise differences, product-decoded keys and fold-aware encodings.
- Compact exploratory OLD/NEW experiment used a simpler feature recipe and new three-fold seed. The pooled local numbers reported in the chat were OLD ~0.755212, NEW ~0.755694, 50:50 rank blend ~0.756750 (gain +0.001056 vs the stronger single model); all three fold-level blend comparisons were reported positive. These numbers are not champion-matched, and the experiment should not be submitted directly.
- A separate feature ablation study showed that in its local compact model, removing the interaction block improved AUC; removing category-frequency features also improved it; the frozen blend of those variants reached 0.758005 on fresh confirmation versus 0.757406 for the better single variant (+0.000599). Again, this is not v15-aligned evidence.

## Next research plan

### Priority 0 — recover the exact cache bundle
Find/upload `gci_cache_final` ZIP or directory with aligned OLD/E1/E1ET/E1Aet OOF/test predictions, exact outer fold indices, `e1A_views.pkl`, `resid16.pkl`, and `catboost_product_fold_*.npz`. The existing champion submission CSV is not enough to perform leakage-safe champion-conditioned validation.

### Priority 1 — matched error-diversity test
Use the same five outer folds and exact row order as v15. For every candidate report pooled AUC, per-fold deltas, rank correlation to v15, paired bootstrap interval and candidate gain when blended into v15 at a small predeclared weight. Do not choose weights on the same validation labels and then report that as unbiased.

### Priority 2 — test OLD/NEW components against v15
Diagnostic probes only: `rank(v15)`; 1%/2% OLD; 1%/2% NEW; 1%/2% rank-blended OLD+NEW. These are test grid candidates, not automatic submission weights. Freeze a recipe before final confirmation.

### Priority 3 — genuinely different signals
Prefer candidates that answer a new hypothesis: error-focused champion-conditioned correction; uncertainty/distance to known product tariff families if not duplicating existing Q/P/PF/K_DEC features; a fresh model family only if it adds rank diversity. Avoid re-running rejected feature blocks unchanged.

### Promotion gate
Suggested working gate: mean OOF gain at least +0.00015, at least 4/5 folds positive, paired 95% interval lower bound above zero, and preferably a fresh-split/nested confirmation. This is a project heuristic, not a guarantee of leaderboard gain. Preserve v15 and submit only one candidate after a locked matched validation.

## Artifact inventory

### Core data and current submissions
- `train.csv`, `test.csv`, `sample_submission.csv`
- `GCI_submission_v15_rankspace_consistent_view.csv` and uploaded duplicate `(1).csv`
- `GCI_submission_v18_alt_direction_probe.csv`
- `GCI_v17_catboost_E1_stack_colab.ipynb`
- `GCI_v18_alt_direction_probe.ipynb`
- `create_v18_submission.py`
- `run_old_new_member_research.py`

### Main notebooks / pipeline history (from research logs)
- `GCI_final_reproduction_v9.ipynb`
- `GCI_catboost_product_step_colab.ipynb`
- `GCI_v13_reweight_and_ET_colab_cc.ipynb`
- `GCI_v13_product_view_test.ipynb`
- `GCI_v14_ET_deepening_colab.ipynb`
- `GCI_v15_consistent_views_ET_colab.ipynb`
- `GCI_v16_regulariser_screen_colab.ipynb`
- `GCI_v17_catboost_E1_stack_colab.ipynb`
- `GCI_v18_alt_direction_probe.ipynb`

### Research logs/reports
- `GCI_FULL_CHAT_RESEARCH_LOG_2026-10-18.md`
- `GCI_MASTER_HANDOFF_2026-10-18.md`
- `GCI_v15_rankspace_candidate_report_2026-10-17.md`
- `GCI_three_experiment_report.md`
- `GCI_repair_and_feature_experiment_report.md`
- `WHY_RESIDUAL_SUBMISSION_DROPPED.md`
- Prior reports referenced in logs: `GCI_RESEARCH_CONTINUATION_REPORT_2026-10-17.md`, `GCI_RESEARCH_STATUS_2026-10-17_evening.md`, `GCI_ExtraTrees_Direct_Check_Report_2026-10-17.md`, `GCI_FAST_KFOLD_SCREEN_REPORT_2026-10-17.md`, `GCI_v14G_submission_report_2026-10-17.md`, `one_more_experiment_2026-10-17.md`.

### Historical experiment result artifacts
`full_e1et_oof_compare_summary.csv`, `catboost_product_experiment_folds.csv`, `v9_full_oof_catboost_comparison_folds.csv`, `dae_experiment_folds.csv`, `dae_full_champion_comparison_folds.csv`, `pairwise_auc_folds.csv`, `catboost_compact_product_progress.csv`, `v13_product_stack_folds.csv`, `catboost_lossguide_folds.csv`, `catboost_ordered_folds.csv`, `catboost_sqrtbalanced_folds.csv`, `extra_seed17_qfreg_folds.csv`, `extra_seed17_qfshallow_folds.csv`, `e1_extra_seed17_full_branch_folds.csv`, `e1_extra_seed17_full_branch_member_folds.csv`, `old_qfreg_extra_seed17_folds.csv`, `fast_sampled_kfold_product_extbins_folds.csv`, `fast_sampled_kfold_ext_affordability_interactions_folds.csv`, `fast_sampled_kfold_ext_age_employment_folds.csv`, `GCI_three_experiment_report.md`, `GCI_repair_and_feature_experiment_report.md`, `confirmed_feature_blend_results.csv`, `fresh_confirmation_summary.csv`, `repair_feature_sweep_summary.csv`.

### Packages listed in historical logs
- `GCI_CatBoost_Product_Experiment_Package.zip`
- `GCI_Research_Continuation_Package_2026-10-17.zip`
- `GCI_ExtraTrees_Direct_Check_Artifacts_2026-10-17.zip`
- `GCI_v14G_submission_and_research_2026-10-17.zip`
- `GCI_v18_alt_direction_package.zip`

## External methodology references

These are general methodological references, not evidence that any particular change improves this leaderboard:
- Scikit-learn stacking: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- Scikit-learn ensemble guide: https://scikit-learn.org/stable/modules/ensemble.html
- Scikit-learn cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html
- LightGBM docs: https://lightgbm.readthedocs.io/
- CatBoost docs: https://catboost.ai/en/docs/
- Home Credit Default Risk competition (inspiration only; do not import disallowed related tables): https://www.kaggle.com/competitions/home-credit-default-risk

---

# Appendix A: Full research log

# GCI World 2026 — Full Chat Research & Experiment Log

**Prepared:** 18 October 2026  
**Coverage:** Research, experiments, decisions, submissions, leaderboard feedback, and current status discussed in this chat through the latest message.  
**Target metric:** ROC-AUC  
**Data size:** 171,202 training rows; 61,500 test rows; target prevalence approximately 8.07%.  
**Competition constraints followed in this work:** supplied `train.csv`, `test.csv`, and `sample_submission.csv` only; no external labels, hidden test labels, pseudo-labeling, manually edited predictions, or pretrained model weights. The feature set contains 32 original predictors, with extensive engineered representations.

> **How to read this report:** “Public LB” means a score the user reported from the competition platform. “OOF AUC” means an out-of-fold validation metric computed from training labels. These are different evaluations and must not be equated. Results reported from earlier Colab runs are labeled as supplied notebook output; results described as rebuilt/recomputed are based on the research-run artifacts available in this chat. Bootstrap intervals are conditional on the tested predictions and do not correct for all the repeated model selection that happened across this chat.

---

## 1. Executive summary

### Current leaderboard state

| Candidate | Public leaderboard score reported by user | Status |
|---|---:|---|
| `GCI_submission_reproduced_final.csv` / v9 E1 multiview swap | **0.77865** | Historical baseline |
| `GCI_submission_catboost_product_blend.csv` | **0.7787** | Small rounded display; not necessarily meaningfully above 0.77865 because 0.77865 rounds to 0.7787 at four decimal places |
| `GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` | **0.77884** | Public-score champion after the user reported the result |
| `GCI_submission_v15_rankspace_consistent_view.csv` | **0.77891** | Latest public-score champion reported by the user |

The latest confirmed user-reported score is **0.77891**. This is a genuine reported rise from 0.77884 (+0.00007 displayed) and a rise of +0.00026 from 0.77865. The exact public score for each uploaded file is based on the user’s report; this assistant cannot independently access the competition platform.

### What appears to work

1. **Decoded loan-product structure** provides meaningful representation signal. The data has a repeated relationship approximately `AMT_ANNUITY = Q0 * AMT_CREDIT + F`, with a fee around 129.5 on approximately 75% of cash loans. This creates 1,047 decoded product values compared with 5,335 distinct rounded raw `annuity/credit` values. A related `K_DEC` / `CG_RATIO` feature decodes credit-to-goods-price tariff tiers.
2. **Product-aware CatBoost** provides complementary rankings to LightGBM and was the basis of v14G. A nominal fixed 10% rank blend was evaluated against v9, and the user subsequently reported public score 0.77884 for the resulting v14G candidate.
3. **LightGBM `extra_trees=True` on the E1 multiview members** is the strongest model-training change discovered in the chat. The `old + E1 + E1 ExtraTrees` blend (`v13_B`) reached OOF AUC 0.77376822 versus 0.77330055 for original v9. It was independently reproduced close to the uploaded notebook results.
4. **Consistent view prediction generation (v15)** modified how EXT/LOAN/PERSON view predictions are created so validation/test predictions are averaged across the inner models rather than refitted on all available rows. This was followed by an E1Aet branch and a rank-space blend candidate; the user reported public score **0.77891** for that candidate.
5. **Isolated bureau-enquiry burst features** recently passed a cheap 80%-sample / 3-fold screen with pooled OOF AUC +0.000565 and all three folds positive. Its full-model confirmation was started but had not finished by the time the user requested the complete log. This is the most promising *new feature block* still awaiting a full gate in the latest state.

### What has not worked

The chat also tested and rejected ID trend, DAE features, a scratch MLP with pairwise AUC fine-tuning, generic KNN neighbour-rate features, smaller product-key subsets, a product specialist stack, several CatBoost growth/weighting variants, ordinary XGBoost, sampled pairwise LightGBM, `scale_pos_weight`, extra seed variants, new blend weights, `max_bin=127`, product-by-external-score deciles, external-score affordability interactions, external-score age/employment interactions, contract-affordability decomposition in the full stack, repeated view cross-fitting, and linear trees. Detailed results appear below.

### Current status at end of chat

- **Current champion:** v15 rank-space consistent-view candidate, user-reported public LB 0.77891.
- **Best next active research lead:** bureau-enquiry burst block. Quick screen passed; the full v15 E1Aet `qf_reg` confirmation was in progress and looked mixed at the first several folds. No pass/fail decision and no submission CSV from that feature block had been produced yet.
- **Do not treat a feature-screen win as a submission win.** The bureau block must first pass full-fold and full-ensemble validation.

---

## 2. Dataset, constraints, and initial pipeline

### Data and required submission format

- `train.csv`: 171,202 rows; binary `TARGET`; 32 original predictor columns plus target and `SK_ID_CURR` (34 total columns according to the notebook file checks).
- `test.csv`: 61,500 rows.
- `sample_submission.csv`: submission template.
- Submission must have exactly `SK_ID_CURR,TARGET`, 61,500 rows, unique IDs in sample-submission order, finite continuous scores, and no extra index column.
- The test IDs are sequential row indices after the training IDs in this dataset (`train` IDs 0–171201; `test` IDs 171202–232701). ID-order tests found no usable target relationship.
- Only competition files are permitted in this work. External Home Credit bureau/previous-loan/installment/POS/cash/credit-card tables were not used, nor were pretrained weights.

### Original v9 reproduction pipeline

The v9 reproduction blend uses a fixed weighted ensemble of five LightGBM members, blended in rank space. The weights for each OLD and E1/new ensemble were:

- `qf_shallow`: 0.24
- `qf_reg`: 0.12
- `v1_allfam_oh`: 0.32
- `v2_allfam_oh`: 0.16
- `v2_mono`: 0.16

The final v9 prediction is 50% OLD ensemble + 50% NEW/E1 ensemble. The NEW E1 members add:

- 16 label-free learned-expectation residual model features: expected value and actual-minus-expected residuals for selected columns;
- multi-view stacking: small models on EXT-source, loan/financial, and person/demographic views, including view logits and pairwise differences;
- fold-aware product target encodings and the underlying decoded product structure.

The final reproduction OOF AUC is approximately **0.77330055**. This is a local validation number; it is not the public leaderboard score.

### Feature engineering foundations

The pipeline builds 67 core features and 130 all-family features before the v6-style appended representations; per model family the full v6 recipe reaches about 99 core and 162 all-family features. Main feature groups:

- cleaned employment duration and sentinel indicators (`DAYS_EMPLOYED == 365243`), missing indicators and missing-value counts;
- financial ratios: credit/income, annuity/income, annuity/credit, credit/annuity term, goods/credit, credit-minus-goods, goods/income, and per-family measures;
- age/employment/registration/ID-age/phone-age derived ratios;
- bureau enquiry sums, recency, and monthly/yearly ratios;
- external score combinations, differences, extrema, means, spread and products;
- one-hot categoricals, frequency/count features, and group-mean deviations;
- decoded product keys Q/P/PF variants and `K_DEC`/`CG_RATIO` where appropriate;
- fold-aware target encodings with smoothing and counts. Training rows receive inner out-of-fold target encodings; validation/test rows use only the relevant training fold.

Important raw signals noted in historical project handoff: `EXT_SOURCE_1/2/3`; occupation, organization, income type, education and gender categorical associations; predictive missingness; and the special `DAYS_EMPLOYED=365243` sentinel.

---

## 3. Detailed chronology and experiment ledger

### Phase A — ID-order and neighbour-risk hypothesis

**Hypothesis:** the sequential application ID might encode a trend or split artifact.

**Test:** examine ID distributions, ID-target rank correlation, default rates over 20 ID bins, and an ID-only forward-validation predictor.

**Observed result:**
- training IDs: 0–171201; test: 171202–232701;
- ID-target Spearman correlation: approximately **0.002733**;
- p-value approximately **0.2581**;
- ID-only forward AUC approximately 0.50.

**Decision:** reject; IDs behave as assigned row indices, not predictive risk information.

**KNN neighbour default rate:** historical experiment notes show this was tested earlier and failed (one prior note reports 0.75759 against 0.75958 baseline). It must not be presented as untested or repeated as a novel lead.

### Phase B — Native-product CatBoost (initial promising candidate)

**Hypothesis:** CatBoost’s ordered categorical statistics on decoded loan-product keys can recover risk patterns differently from LightGBM’s numerical/smoothed encoding treatment.

**Implementation:** CatBoost with native categorical product features including Q/P/PF precision keys, `K_DEC`, fee indicator and the original categoricals; 5 folds, depth 7, 900 iterations, fixed seed/settings. Candidate blends 90% v9 rank predictions + 10% product-CatBoost rank predictions.

**Fold AUCs supplied by user’s Colab log:**
- 0.76639
- 0.77180
- 0.77448
- 0.76710
- 0.76776

The Colab log reported CatBoost standalone pooled AUC 0.76943 (older paper trail in the notebook referenced 0.76969 in another run). This variation reflects differing cached/rebuilt runs and should not be silently merged into a single exact value.

**Full-v9 comparison in an initial run:** approximately 0.7733005 → 0.7735291; +0.0002286 OOF; 5/5 fold gains positive; paired bootstrap 95% CI reported around [+0.0001244,+0.0003317]. Later aligned artifacts gave current champion OOF around **0.77352213**.

**Gate details:** the CatBoost notebook sometimes printed “gate failed: keep v9” because the unrounded gain was below its strict +0.000200 threshold in that particular cached run. This is an important nuance: do not claim every rebuild passed that exact original gate. Despite this, the user submitted the generated CatBoost candidate and reported public score **0.7787** (rounded). Later the stronger v14G candidate, which includes the ExtraTrees blend, scored 0.77884.

**Artifact:** `GCI_submission_catboost_product_blend.csv`; later a reproduced candidate named `GCI_submission_catboost_product_blend_reproduced.csv` was generated by the notebook. The first file was structurally validated; public score originally provided as 0.7787.

### Phase C — Denoising autoencoder (DAE) representation

**Hypothesis:** learn latent features from input reconstruction, add them to LightGBM.

**Test:** supervised evaluation of 64 latent dimensions plus reconstruction error as an additional feature block; fixed folds.

**Observation:** reconstruction loss fell from about 1.99 to 0.79 during early training, but that did not translate into AUC.

- DAE-augmented standalone lightgbm OOF around 0.76931 vs single-member reference 0.77070 (approximately −0.00139).
- Full v9+CatBoost reference approximately 0.7735221; DAE addition approximately 0.7734956 (−0.0000266).
- Paired bootstrap CI for full champion gain: approximately [−0.0001106,+0.0000653], only 2/5 folds positive.
- One 10% blend looked positive against a *single* LightGBM member, but failed against the full incumbent.

**Decision:** reject; no submission created. The lesson was that good reconstruction loss does not imply better ranking signal.

**Artifact:** `dae_experiment_folds.csv`, `dae_full_champion_comparison_folds.csv`.

### Phase D — Scratch MLP with AUC-oriented fine-tuning

**Hypothesis:** binary cross-entropy pretraining followed by sampled pairwise/AUC-margin fine-tuning could improve ranking.

**Run:** MLP trained from scratch because TabM/LibAUC dependencies could not be installed from the network-restricted runtime. No pretrained weights or outside data.

**Fold outcomes:** early fold AUCs 0.7429 and 0.7469; later full OOF result approximately **0.7444464**, vs champion OOF 0.7735221 (−0.0290757). 10% blend hurt approximately 0.0001619; 2.5% blend gain approximately +0.0000194 only.

**Decision:** reject. Too far below the incumbent and no useful blend component identified.

### Phase E — Compact product-key CatBoost ablation

**Hypothesis:** remove redundant Q3/Q4/Q5/P3/P4/P5 keys and retain a smaller canonical set such as P3, PF4, K_DEC and FEE.

**Result:** compact version fixed 10% blend vs current incumbent gained about +0.0000605 OOF in one summary, but paired CI crossed zero and only 1/5 fold was positive. The fold-level progress also reported several small negative deltas.

**Decision:** reject. Reduced precision loses useful distinctions; retain the richer native product key bundle for CatBoost.

**Artifact:** `catboost_compact_product_folds.csv`, `catboost_compact_product_progress.csv`.

### Phase F — v13 product-specialist stacking

**Hypothesis:** product-specific specialist view might add information to the main model even if its standalone AUC is low.

**Result:** modified v13 product-stack full OOF approximately 0.7735782 vs 0.7735221 champion, gain +0.0000561; 3/5 folds positive; paired CI about [−0.0000577,+0.0001801].

**Decision:** reject; below practical gain threshold and unstable.

**Artifact:** `v13_product_stack_folds.csv`; notebook `GCI_v13_product_view_test.ipynb`.

### Phase G — Class weighting and CatBoost training-mode variations

1. **CatBoost `SqrtBalanced`:** model folds [0.76558, 0.77090, 0.77334, 0.76393, 0.76713], standalone pooled AUC about 0.76808; blend approx 0.773471 vs 0.773522 champion (−0.000051), 2/5 folds positive, CI [−0.000148,+0.000053]. Reject.
2. **CatBoost Ordered boosting:** same product features/folds/rounds, different boosting mechanism. Blend gain only +0.0000049; 3/5 positive; CI [−0.0000707,+0.0000734]. Reject.
3. **CatBoost Lossguide growth:** `grow_policy=Lossguide`, `max_leaves=31`, `min_data_in_leaf=150`, 900 rounds. Standalone OOF around 0.7683353; fixed blend gain +0.0000740, 4/5 folds positive but CI [−0.0000119,+0.0001720], below practical gate. Reject.
4. **CatBoost depth 8:** same feature set, loss, learning rate and 900 rounds, split CatBoost’s 10% contribution as 5% depth 7 + 5% depth 8. Depth-8 fold AUCs [0.766636, 0.771426, 0.774851, 0.767402, final fold around 0.767–0.768 per progress log]. Blend gain +0.000043, 4/5 folds positive, 300-replicate CI [+0.0000006,+0.0000845]; too small for +0.00015 practical gate. Reject.
5. **CatBoost second random seed 17:** standalone fold AUCs [0.768368, 0.772202, 0.774019, 0.76636, 0.76788]. Splitting existing CatBoost contribution between original seed and seed 17 gave gain only +0.0000114, 3/5 folds positive, interval crossed zero. Reject.
6. **Nested blend-weight optimization:** in-sample all-row optimized weights gave pooled OOF 0.77338 vs fixed 0.77330, but nested mean gain was −0.00014, positive on only 1/5 folds. Reject reweighted blend.

Artifacts include `catboost_sqrtbalanced_folds.csv`, `catboost_ordered_folds.csv`, `catboost_lossguide_folds.csv`, `catboost_blend_weight_grid.csv`, `catboost_blend_weight_nested_folds.csv`.

### Phase H — Sampled pairwise ranking objective

**Hypothesis:** optimize positive-negative ordering explicitly using a sampled logistic pairwise loss.

**Test:** LightGBM custom objective trained on four sampled negatives per positive, pairs drawn only from each outer training fold; 600 rounds. A fixed 10% rank blend with the incumbent.

**Result:** ranker OOF AUC 0.7707169; blended OOF 0.7735107 vs 0.7735221 (−0.0000114); paired CI [−0.0000747,+0.0000615]; 3/5 folds positive; score Spearman correlation with champion 0.9860.

**Decision:** reject.

**Artifact:** `pairwise_auc_folds.csv`.

### Phase I — Ordinary XGBoost classifier

**Hypothesis:** a different GBDT implementation might produce complementary ranking errors to LightGBM/CatBoost.

**Run:** XGBoost binary classifier on engineered features and nested product encodings; fixed 1,200 rounds; 5 folds; fixed 5% blend with v14G.

**Standalone fold AUCs:** 0.76818, 0.77269, 0.77637, 0.76903, 0.76929.

**Result:** blend delta −0.0000268, only 1/5 folds positive, CI crossed zero.

**Decision:** reject; no CSV.

### Phase J — ExtraTrees LightGBM (key improvement)

**Hypothesis:** stochastic/randomized threshold selection might change generalization and add diversity to the E1 stacked members.

**Method:** same features, folds, rounds and seeds, add `extra_trees=True` to the five E1 members. This randomizes threshold candidates per feature. Compare OLD + E1 + E1ET; do not tune weights after inspecting the OOF predictions.

**Original E1 member vs ExtraTrees E1ET results:**

| Member | Original E1 OOF | E1ET OOF | Gain |
|---|---:|---:|---:|
| `qf_shallow` | 0.77156 | 0.77253 | +0.00097 |
| `qf_reg` | 0.77158 | 0.77306 | +0.00148 |
| `v1_allfam_oh` | 0.77131 | 0.77248 | +0.00116 |
| `v2_allfam_oh` | 0.77122 | 0.77267 | +0.00145 |
| `v2_mono` | 0.77176 | 0.77294 | +0.00119 |

E1 blend rank correlation before/after ExtraTrees: approximately 0.9925.

**v13_B = OLD + E1 + E1ET, equal thirds:**
- Original v9 OOF 0.77330055.
- v13_B OOF 0.77376822: +0.00046768 vs v9; five folds positive in original direct comparison; CI [ +0.00029455, +0.00063715 ].
- Compared against then-current v9+CatBoost OOF champion 0.77352213: +0.00024610; 4/5 folds positive; CI [+0.00005891,+0.00043075].
- This was the strongest newly validated candidate in that phase.

**E1ET only:** blend OOF 0.77356771.
**v13_C (OLD + E1ET swap):** 0.77372276, +0.00020064 vs CatBoost champion but only 3/5 folds positive and interval [−0.00005099,+0.00046096]; reject C.

**Decision:** B was a pass vs the current CatBoost-enhanced reference. The user later submitted v14G, an additional CatBoost blend of B, and reported public score 0.77884.

Artifacts: `full_e1et_oof_compare_summary.csv`, `full_e1et_oof_compare_predictions.npz` within package/archives, `GCI_ExtraTrees_Direct_Check_Report_2026-10-17.md`.

### Phase K — Further ExtraTrees seed tests

1. **`qf_reg`, ExtraTrees seed 17:** OOF 0.77298680; 5% blend with champion gained +0.00009586; 5/5 folds positive; CI [+0.00005639,+0.00012582]. Reject for below +0.00015 practical gain.
2. **`qf_shallow`, ExtraTrees seed 17:** OOF 0.77277429; 5% blend gain +0.00009352; 5/5 folds positive; CI [+0.00005531,+0.00012528]. Reject for below threshold.
3. **Whole E1 ExtraTrees seed 17 branch:** the second seed branch OOF addition changed AUC by only +0.0000012, 3/5 folds positive; reject.
4. **Old-branch ExtraTrees seed 17 `qf_reg`:** fixed 5% blend gained only +0.000045; below gate.
5. **All old-branch ExtraTrees members:** multiplier screen: `qf_shallow` AUC results 1.0 multiplier 0.77114, 1.5 multiplier 0.77098, 2.0 multiplier 0.77040; use multiplier 1.0. Full oldET blend AUC 0.77233. Combinations D/E/H fail; a G combination B+10% CatBoost showed +0.00014 vs B, 5/5 folds, CI [+0.00004,+0.00024], but missed its +0.0002 incremental gate. No promotion from these oldET combinations.

**Decision:** multiple seeds helped small individual blends but were below practical gain or failed full-branch stability; do not keep adding threshold seeds without new evidence.

### Phase L — Public v14G candidate

**Formula:** B = one-third each OLD, E1, E1ET. G/v14G = 90% rank(B) + 10% rank(product-CatBoost).

**Recomputed OOF:**
- v9 baseline 0.77330055;
- v9 + 10% CatBoost 0.77352213;
- B 0.77376822;
- G 0.77393726.

**G versus CatBoost champion:** +0.00041513, fold gains [+0.00057279,+0.00060611,+0.00012743,+0.00023739,+0.00052303] (5/5 positive), 1,000-replicate paired stratified-bootstrap CI [+0.00025035,+0.00056812]. The notebook originally printed a fail against B because G’s +0.00014 delta vs B was below a stricter rule, but direct comparison to the then-current champion was a pass.

**Submission:** `GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` (61,500 rows, ID order verified). The user later reported public LB **0.77884**.

### Phase M — Add CatBoost back to v13_B after v14G

The 90/10 mix of B and current CatBoost champion did not improve B: OOF delta −0.0000037131, only 2/5 folds positive. This was rejected. Do not assume adding ensemble components monotonically improves AUC.

### Phase N — v15 consistent-view feature-generation experiment

**Hypothesis:** the previous multi-view stacking creates train/validation mismatch. Training rows receive view scores from inner-fold models trained on 2/3 of the data; validation/test scores were produced by a refit on all available training data. Variant A makes validation/test view features the average of inner models instead, aligning the scoring process more closely to training rows.

**Test:** rebuild view predictions; train five E1Aet members (`extra_trees=True`) with same feature definitions, rounds, original outer folds and production seeds. The supplied Colab output reported:

| Member | E1ET OOF AUC in supplied log |
|---|---:|
| `e1Aet_qf_shallow` | 0.77272 |
| `e1Aet_qf_reg` | 0.77322 |
| `e1Aet_v1_allfam_oh` | 0.77261 |
| `e1Aet_v2_allfam_oh` | 0.77267 |
| `e1Aet_v2_mono` | 0.77301 |

**G1 = OLD + E1 + E1Aet + 10% CatBoost:** pooled 0.77395, gain +0.00004 vs v14G; fold gains [0.00008,0,0.00008,−0.00005,0.00011], 4/5 positive; CI [−0.00002,+0.00010]; reject.

**G2 = OLD + E1ET + E1Aet + 10% CatBoost:** pooled 0.77403, gain +0.00011; fold gains [0.00031,0.00031,−0.00020,−0.00011,0.00028], 3/5 positive; CI [−0.00008,+0.00028]; reject.

**G3 = OLD + E1 + E1ET + E1Aet + 10% CatBoost:** pooled 0.77404, gain +0.00013; fold gains [0.00023,0.00019,0.00001,−0.00002,0.00025], 4/5 positive; CI [+0.00004,+0.00021]; rejected under original +0.00020 gain gate.

**Independent local rebuild / alternate rank-space candidate:** the actual local E1Aet branch was rebuilt with float32 frame downcasting for the 4 GB memory limit. The original raw arithmetic G3 recipe missed its original +0.00020 gain cutoff, but a separately specified rank-space candidate using the v14G submission and E1Aet blend passed the later practical gate:
- v14G OOF AUC 0.77393726;
- rank-space candidate OOF AUC 0.77410395;
- gain +0.00016669;
- fold gains [+0.00026809,+0.00020539,+0.00005971,+0.00002809,+0.00026588] (5/5 positive);
- 300-replicate paired-bootstrap CI [+0.00007838,+0.00026092].

**Submission:** `GCI_submission_v15_rankspace_consistent_view.csv`, SHA-256 `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7`; exact two required columns, 61,500 rows, unique IDs in sample order, finite continuous predictions.

**User-reported public score:** **0.77891**. This is the current public champion. Public improvement from 0.77884 to 0.77891 is +0.00007 at displayed precision.

### Phase O — Blend and model perturbation after v15

Following the user-reported 0.77891 score, the following controlled attempts were made:

1. **CatBoost seed 17:** second product-aware CatBoost seed with the same features/depth/rounds, fixed at 5% contribution replacing half the existing CatBoost allocation. Fold AUCs [0.768368,0.772202,0.774019,0.76636,0.76788]. Replacing half the CatBoost blend weight changed OOF AUC by about +0.0000114, 3/5 positive, CI across zero; reject.
2. **Ordinary XGBoost classifier:** 1,200 rounds; fold AUCs [0.76818,0.77269,0.77637,0.76903,0.76929]. Fixed 5% blend gain −0.0000268; 1/5 folds positive, CI across zero; reject.
3. **Whole E1ET seed 17 branch:** AUC change +0.0000012, 3/5 positive, interval crosses zero; reject.
4. **CatBoost native segment combinations / depth 8:** prior manual product×segment target encoding failed. Depth-8 CatBoost changed the fixed blend by +0.000043, 4/5 folds positive, CI barely above 0 but below +0.00015 practical threshold; reject.
5. **CatBoost blend weight nested test:** best in-sample reoptimized weight gave only +0.000038 and 3/5 folds; nested result was not reliable; keep 10%.
6. **LightGBM `max_bin=127` ExtraTrees branch:** final timeout artifacts showed a pooled improvement around +0.000024 with CI spanning zero; reject.

### Phase P — Three fast 80%-sample feature screens

These were designed to reduce wasted compute before full model runs. Method: 80% stratified sample (~136,961 rows), three-fold stratified CV, 500 LightGBM rounds, identical baseline/candidate settings, existing all-family features, one-hot categorical variables and nested smoothed product encodings. Use within-screen paired comparisons, not comparison of their absolute AUC to the whole v14G/v15 ensemble.

| Block | Baseline AUC | Candidate AUC | Difference | Positive folds | 95% paired bootstrap CI | Decision |
|---|---:|---:|---:|---:|---:|---|
| Decoded product PF4 × external-score deciles | 0.766914 | 0.766601 | −0.000313 | 1/3 | [−0.000913,+0.000314] | Reject |
| External score × affordability (EXT × log annuity/income and log credit/income) | 0.769836 | 0.769812 | −0.000023 | 1/3 | [−0.000606,+0.000571] | Reject/inconclusive |
| External score × age/employment | 0.768612 | 0.768632 | +0.000021 | 1/3 | [−0.000535,+0.000620] | Reject/inconclusive |

No CSV created from these screens.

Artifacts: `GCI_FAST_KFOLD_SCREEN_REPORT_2026-10-17.md`, `fast_sampled_kfold_product_extbins_folds.csv`, `fast_sampled_kfold_ext_affordability_interactions_folds.csv`, `fast_sampled_kfold_ext_age_employment_folds.csv`.

### Phase Q — v15 exploration with fast model/feature screens

1. **Inner view fold count 3 → 5:** on a fixed 80% sample/3 outer folds, qf_reg baseline pooled AUC 0.769606; inner-3 views 0.769529; inner-5 views 0.769574. Inner-5 improved inner-3 by only +0.000045 but remained below baseline; reject.
2. **Contract-decomposition affordability block:** uses decoded principal-linked monthly payment and fixed-fee burden, isolated from bureau-burst features. Quick screen pooled AUC 0.769606 → 0.769920 (+0.000314), all 3 folds positive. Adding generic residual-income features weakened result, so excluded.
3. **Full qf_reg confirmation of contract block:** member fold AUCs were mixed; replacing qf_reg in full v15 candidate changed OOF by only +0.0000022, 3/5 folds positive and interval crossed zero; reject. Quick-screen lift did not transfer to full multiview ensemble.
4. **Repeated inner-3 cross-fitting for view predictions:** first fold 0.768710 (single inner-3) vs 0.768679 (repeated inner-3); full quick screen ended at −0.000454 pooled OOF, 0/3 folds positive; reject.
5. **LightGBM linear trees:** fixed 80%-sample/3-fold/600-round screen; pooled AUC approximately 0.75282 vs 0.76892 for normal trees, one fold collapsed to 0.72742; reject.
6. **Bureau-enquiry burst block (latest active lead):** isolated hourly/monthly/quarterly/yearly enquiry-burst rates based on supplied AMT_REQ_CREDIT_BUREAU_* counts. This avoids the previously failed combined contract-affordability + enquiry-burst block.
   - Quick 80%-sample/3-fold screen: pooled AUC 0.769606 → 0.770171, delta **+0.000565**.
   - Fold gains +0.001098, +0.000408, +0.000152 (all 3 positive).
   - The quick screen passed the screen-level gate and full confirmation was started on the v15 E1Aet qf_reg branch.
   - Full confirmation fold 1: AUC 0.770744, approximately tied/slightly below the existing E1Aet qf_reg fold-1 reference around 0.77079. Progress notes said later folds were mixed: a slight improvement on fold 2, losses on folds 1, 3 and 4, fold 5 essentially unchanged. The full-ensemble evaluation was not completed before the user requested this consolidated chat report.
   - **Status: not confirmed, not rejected finally, no submission CSV.** Next action if research resumes: recover `gci_research` log/artifacts for this block, finish 5-fold OOF and evaluate in full candidate before deciding. Keep v15 at public 0.77891 as champion meanwhile.

---

## 4. Important generated files and artifacts

All listed paths refer to files created/copied in the local session at `/mnt/data` and were verified to exist at the time of creation. They are useful to preserve the full trail.

### Submission CSVs

- `/mnt/data/gci_research/GCI_submission_reproduced_final.csv` — reproduction of v9/E1 original, public score reported 0.77865.
- `/mnt/data/gci_research/GCI_submission_catboost_product_blend.csv` — product-aware CatBoost blend, public score later reported as 0.7787.
- `/mnt/data/GCI_v14G_submission_and_research_2026-10-17.zip` includes the v14G submission and report; standalone file is `/mnt/data/gci_research/GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` — public score reported 0.77884.
- `/mnt/data/gci_research/GCI_submission_v15_rankspace_consistent_view.csv` — rank-space consistent-view candidate, public score reported 0.77891; current best.
- The v13_B CSV was written in the user's Colab output folder according to notebook log (`/content/drive/MyDrive/gci_output_final/v13_B_old_e1_e1et.csv`), but its standalone test CSV was not present in the local project tree when the v15 work began. It should not be claimed as locally generated in this session.

### Notebooks

- `GCI_final_reproduction_v9.ipynb` — original v9 reproduction.
- `GCI_catboost_product_step_colab.ipynb` — native product-category CatBoost experiment.
- `GCI_v13_reweight_and_ET_colab_cc.ipynb` — nested reweighting and ExtraTrees E1 members.
- `GCI_v13_product_view_test.ipynb` — product specialist view experiment.
- `GCI_v14_ET_deepening_colab.ipynb` — deeper ExtraTrees investigation.
- `GCI_v15_consistent_views_ET_colab.ipynb` — consistent view generation and E1Aet.
- User-uploaded `GCI_v15_consistent_views_ET_colab.ipynb` is the source of the v15 G1/G2/G3 numbers above.

### Reports and data artifacts

- `GCI_RESEARCH_CONTINUATION_REPORT_2026-10-17.md`
- `GCI_RESEARCH_STATUS_2026-10-17_evening.md`
- `GCI_ExtraTrees_Direct_Check_Report_2026-10-17.md`
- `GCI_FAST_KFOLD_SCREEN_REPORT_2026-10-17.md`
- `GCI_v14G_submission_report_2026-10-17.md`
- `GCI_v15_rankspace_candidate_report_2026-10-17.md`
- `one_more_experiment_2026-10-17.md`
- `full_e1et_oof_compare_summary.csv`
- `catboost_product_experiment_folds.csv`
- `v9_full_oof_catboost_comparison_folds.csv`
- `dae_experiment_folds.csv`
- `dae_full_champion_comparison_folds.csv`
- `pairwise_auc_folds.csv`
- `catboost_compact_product_progress.csv`
- `v13_product_stack_folds.csv`
- `catboost_lossguide_folds.csv`
- `catboost_ordered_folds.csv`
- `catboost_sqrtbalanced_folds.csv`
- `extra_seed17_qfreg_folds.csv`, `extra_seed17_qfshallow_folds.csv`
- `e1_extra_seed17_full_branch_folds.csv`, `e1_extra_seed17_full_branch_member_folds.csv`
- `old_qfreg_extra_seed17_folds.csv`
- `fast_sampled_kfold_product_extbins_folds.csv`
- `fast_sampled_kfold_ext_affordability_interactions_folds.csv`
- `fast_sampled_kfold_ext_age_employment_folds.csv`
- `GCI_submission_v15_rankspace_consistent_view.csv` report SHA-256: `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7`

Archives assembled during the chat include:
- `GCI_CatBoost_Product_Experiment_Package.zip`
- `GCI_Research_Continuation_Package_2026-10-17.zip`
- `GCI_ExtraTrees_Direct_Check_Artifacts_2026-10-17.zip`
- `GCI_v14G_submission_and_research_2026-10-17.zip`
- `GCI_v14_ET_deepening_colab.ipynb`
- `GCI_v15_rankspace_candidate_report_2026-10-17.md`

---

## 5. Decision gates used during research

Common practical promotion rules, with some changes noted above:

- Compare the candidate to the *current champion*, not a weaker earlier baseline.
- Measure paired pooled OOF AUC gain on the same row-aligned predictions.
- Require stable gains across folds (usually 4/5 or 5/5).
- Use a paired stratified bootstrap interval; where the gate required it, require the 95% lower bound above zero.
- Some older notebook gates required gain >= +0.00020; later practical gates used +0.00015 against the current champion. The exact gate in force for each experiment is called out in the relevant section.
- Do not create a submission for a candidate that fails the gate.
- If the gate passes, test one candidate on the public leaderboard, preserve the prior champion as fallback, and record the exact displayed score.

**Critical validation caveat:** most full-model experiments reused `StratifiedKFold(5, shuffle=True, random_state=42)`. Bootstrap intervals are conditional on those predictions; they do not correct the winner’s curse from choosing among many candidates on the same folds. Use a fresh confirmation split or nested evaluation for the bureau block if possible.

---

## 6. What is known, inferred, and not yet established

### User-reported public leaderboard facts

- v9 baseline/submitted file: 0.77865.
- Product-CatBoost candidate: 0.7787.
- v14G: 0.77884.
- v15 rank-space consistent-view candidate: 0.77891.

These were supplied by the user in chat and are recorded as reported, not independently verified against the platform.

### Measured local-validation facts

- v9 OOF about 0.77330055.
- v9 + product-CatBoost OOF about 0.77352213.
- v13_B OOF about 0.77376822.
- v14G OOF about 0.77393726.
- v15 rank-space candidate OOF about 0.77410395.
- New isolated bureau-burst quick-screen candidate baseline 0.769606 → 0.770171, +0.000565 on an 80% sample/3-fold single-member screen; not full confirmation.

### Not established

- No experiment has produced a confirmed 0.79 public score or an OOF increase remotely close to +0.01.
- The public leaderboard gain from a tiny OOF gain is not guaranteed.
- The new bureau-burst feature has not yet passed full v15 ensemble confirmation.
- No external relational data is permitted to be added under the current rules; studies using full Home Credit auxiliary tables are not directly comparable.
- A new CSV was **not** generated from the bureau-burst block in the last active state.

---

## 7. Immediate next steps if research continues

1. Keep **v15 / public score 0.77891** as the champion.
2. Recover the bureau-burst full-confirmation script, logs and fold predictions from the latest run. Complete the full paired OOF calculation, not just the screen AUC.
3. If the bureau-burst block improves the actual v15 E1Aet `qf_reg` member across folds, replace that member only once and compare the full ensemble to v15. Given the quick-screen improvement of +0.000565 but weak/mixed full-fold logs, treat success as possible but unproven.
4. Use a fresh seed/split or nested outer validation if compute allows, because the original folds have been reused throughout the research.
5. Only after a candidate passes full validation, produce one structurally checked CSV, submit it once, and record the returned public score. Do not continue broad tuning based solely on a small leaderboard change.

---

## 8. Final summary

This chat produced one clear leaderboard progression: 0.77865 → 0.77884 → 0.77891 reported public scores. The main successful ideas were product-native CatBoost as a complementary model, LightGBM ExtraTrees diversity, and a more consistent multiview stacking design. Many further ideas were tested and rejected; their results are preserved above so they are not repeated accidentally. The only fresh feature hypothesis still active at the end of the conversation is isolated bureau-enquiry burst engineering, whose cheap screen passed but full-ensemble confirmation remained unfinished.


---

# Appendix B: Master handoff

# GCI World 2026 — Master Research Handoff (for a new ChatGPT chat)

**Handoff date:** 18 October 2026 (India time)  
**Purpose:** Continue the existing competition project without repeating completed experiments.  
**Metric:** ROC-AUC.  
**Rules/data scope:** use only the competition's supplied `train.csv`, `test.csv`, `sample_submission.csv`; no outside labels, pretrained weights, or prohibited external tables.

## 0. Current state — read this first

### Public leaderboard history (user-reported)

| Artifact | User-reported public score | Status |
|---|---:|---|
| `GCI_submission_reproduced_final.csv` / v9 reproduction | **0.77865** | Historical baseline |
| `GCI_submission_catboost_product_blend.csv` | **0.7787** | CatBoost product candidate; first displayed score is ambiguous because 0.77865 itself rounds to 0.7787 at four decimals |
| `GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` | **0.77884** | Confirmed former champion (v14G) |
| `GCI_submission_v15_rankspace_consistent_view.csv` | **0.77891** | **CURRENT PUBLIC-SCORE CHAMPION** per user, as of 17 Oct 2026 |

**Important naming correction:** v14G public score is 0.77884. The current 0.77891 result belongs to the v15 rank-space consistent-view candidate. Do not call 0.77891 “v14G”.

### Current local validation anchor

- v14G OOF AUC from matched saved arrays: **0.7739372585**.
- v15 rank-space candidate OOF AUC: **0.7741039515**.
- Paired gain vs v14G: **+0.0001666930**.
- Fold gains: **[+0.00026809, +0.00020539, +0.00005971, +0.00002809, +0.00026588]**; 5/5 positive.
- Paired stratified bootstrap 95% interval (300 resamples): **[+0.00007838, +0.00026092]**.
- The user submitted that candidate and reported public score **0.77891**. It is the champion unless a newer actual leaderboard result is supplied.

The OOF AUC and public leaderboard AUC are different evaluation samples; do not directly compare their numeric levels.

### Latest attempted experiment — complete and rejected

The last run was a full 5-fold confirmation of bureau-enquiry burst features in the v15 `e1Aet_qf_reg` member. It completed; **there is no training job still running and no CSV was created from it**.

- Fast 80%-sample, 3-fold screen: baseline **0.76960562**, burst features **0.77017053**, delta **+0.00056492**, all 3 folds positive.
- Full v15-branch confirmation: reference **0.7741039515**, candidate **0.7741028863**, delta **−0.0000010652**.
- Fold deltas: **[+0.00000782, +0.00000268, −0.00000699, +0.00001246, −0.00002445]**, 3/5 positive.
- Paired bootstrap 95% interval: **[−0.00001361, +0.00001056]**.
- Decision: **reject; do not create/submit a bureau-burst CSV**. The attractive quick-screen gain did not transfer into the full multiview branch.

The exact script, log, and JSON are in `04_SCRIPTS/full_bureau_burst_qfreg.py`, `05_RESULTS/full_bureau_burst_qfreg.log`, and `05_RESULTS/full_bureau_burst_qfreg_results.json`.

## 1. Competition data and pipeline

- Train: **171,202 rows**, 34 columns including `TARGET` and ID.
- Test: **61,500 rows**, 33 columns.
- Input features available: approximately **32 actual predictors**, excluding ID and target. No external Home Credit relational history is allowed/available in this competition setup.
- Target prevalence reported in project logs: about **8.0729%**.
- Validation: repeated use of `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; nested product target encodings use inner folds (`random_state=7`), and view models use inner 3-fold splits with `random_state=11` unless the experiment explicitly changes them.
- Because the same five outer folds were reused in many model-selection rounds, bootstrap intervals are conditional on selected models and do **not** correct for repeated experimentation / selection bias.

### Core feature system

- 67 core features / 130 all-family features before v6-recipe additions; v6-recipe sizes about **99 core / 162 all-family**.
- Feature groups: cleaned sentinel and missingness features; EXT_SOURCE transforms; financial ratios; age/employment ratios; credit-enquiry features; count/group features; decoded tier `K_DEC` and `CG_RATIO` for selected members.
- Product decoder: cash-loan annuity relation roughly `AMT_ANNUITY = Q0 * AMT_CREDIT + F`, with fixed monthly fee around **129.5** for about 75% of cash loans. It yields about **1,047 decoded products** versus **5,335 distinct raw q rounded to 5 decimals**.
- Product keys used include `Q3`, `Q4`, `Q5`, `P3`, `P4`, `P5`, `PF4`; smoothed target means and counts are fold-aware / nested so a row's own target is excluded.
- E1 multi-view stack uses `EXT`, `LOAN`, `PERSON` views (in v15: 31, 28, 53 columns respectively) with their logits and pairwise differences.

## 2. Exact submission recipes

### v9 reproduction (public 0.77865)

Each OLD and E1 blend uses these fixed weights after rank transforms:

- `qf_shallow`: 0.24
- `qf_reg`: 0.12
- `v1_allfam_oh`: 0.32
- `v2_allfam_oh`: 0.16
- `v2_mono`: 0.16

`v9 = 0.5 * OLD_blend + 0.5 * E1_blend`.

OLD members use original engineered features and nested product encodings. E1 adds (a) 16 label-free learned-expectation residual representations, and (b) the EXT/LOAN/PERSON view-stack features.

Reference OOF summaries (matched prior saved arrays): OLD blend **0.77202**, E1 blend **0.77258**, final v9 **0.77330055**.

### Product-CatBoost candidate (public shown as 0.7787)

- CatBoost model uses native categorical columns for the 8 source categoricals plus decoded product keys `Q3`, `Q4`, `Q5`, `P3`, `P4`, `P5`, `PF4`, `K_DEC`, and `FEE`.
- Main setup: depth 7, 900 iterations, learning rate 0.05, `Logloss`, `eval_metric=AUC`, `l2_leaf_reg=8`, seed 42, CatBoost Bernoulli bootstrap and subsample 0.8.
- Blend: **90% v9 rank + 10% CatBoost product rank**.
- An early matched check found roughly +0.000229 OOF gain vs v9, positive 5/5, bootstrap CI approximately [+0.000124,+0.000332]. Some refits printed a gain just below the strict +0.000200 cutoff (about +0.00020 after rounding); numerical differences between saved/refit arrays were documented. Do not overstate this as a large lift.

### v14G (public 0.77884)

- `B = (OLD_blend + E1_blend + E1_ExtraTrees_blend) / 3`.
- `G = 0.90 * rank(B_test) + 0.10 * rank(CatBoost_product_test)`.
- Matched local OOF AUC: **0.7739372585**. In a paired comparison vs the v9+CatBoost champion, gain **+0.00041513**, 5/5 folds positive, 1,000-resample paired stratified bootstrap 95% CI **[+0.00025035,+0.00056812]**.
- The user submitted v14G and reported public score 0.77884.

### v15 consistent-view rank-space candidate (current public 0.77891)

The v15 notebook explores “variant A” view features: for validation/test rows it averages predictions from the inner view models rather than refitting one model on all outer-training data, making validation/test prediction generation more like the OOF features used for training. It then fits the five E1 members with `extra_trees=True`.

The notebook's original G1/G2/G3 variants all missed its strict `mean gain >= 0.0002` rule. A separately reconstructed matched **rank-space** candidate passed a later practical rule and was submitted successfully:

`candidate = 0.90 * rank(0.75 * rank(B_v14G_base) + 0.25 * E1Aet_blend) + 0.10 * rank(CatBoost_product)`

where `B_v14G_base` is the branch mixture underlying v14G before its 10% CatBoost combination, and `E1Aet_blend` is the fixed-weight blend of the five consistent-view ExtraTrees members. This is explicitly **not identical to the raw arithmetic formula** from the notebook's G3; use the candidate CSV/report as source of truth.

- Local OOF AUC **0.7741039515**; gain **+0.0001666930** vs v14G OOF.
- Fold gains 5/5 positive; paired bootstrap 95% interval [+0.00007838,+0.00026092].
- Candidate file: `02_SUBMISSIONS/GCI_submission_v15_rankspace_consistent_view.csv`.
- SHA-256: `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7`.
- User-reported public score: **0.77891**.

## 3. Full experiment ledger (chronological summary)

Scores from older-stage experiments may use an earlier model/data recipe; compare candidates only within the same experiment and same folds. “Reject” means do not submit based on that experiment.

| Experiment / combination | Result | Decision / note |
|---|---|---|
| Historical baseline application-only models | Previous versions around 0.75–0.76 OOF/public at older checkpoint | Superseded by v9 pipeline; avoid comparing absolute values across revisions |
| Extra EXT_SOURCE mean/min/max/std/products | 0.75050 vs 0.75185 baseline (older checkpoint) | Reject; slightly worse |
| Log-only transformations | Trees largely unchanged | Reject as standalone block |
| Count encodings / group mean differences | About 0.0000 to −0.0003 | No useful stable lift alone |
| Generic KNN neighbour default-rate | 0.75759 vs 0.75958 | Reject |
| Generic target encoding (pre-decoder) | Public about 0.76292, worse than then-current XGBoost | Reject; do not confuse with later decoded-product TE |
| LightGBM native categorical (older test) | About 0.75824–0.75908 vs 0.75972 | Reject that approach in the earlier feature/model recipe |
| Same-person “twin” features | About +0.00012, only 0.3% of rows | Not promoted |
| Additional monotone constraints (older model) | About +0.0003 in a prior checkpoint | Too small/local; not a general promotion |
| ID order as predictor | Spearman rho 0.002733, p=0.2581; ID-only AUC about 0.50 | Reject; IDs are sequential indexing |
| Decoded product structure | 1,047 products; fee share ~0.75 | **Important existing feature-engineering discovery** |
| Native product CatBoost | Product-free CatBoost about 0.75917 pooled; product-aware about 0.76969 in original run | Strong feature representation within CatBoost; 10% rank blending passed an early OOF gate |
| v13 blend weight reoptimization | In-sample optimized OOF 0.77338 vs fixed 0.77330, but nested mean gain −0.00014 and only 1/5 folds positive | Reject; in-sample weights overfit |
| E1 `extra_trees=True` member replacements | Member gains over E1: qf_shallow +0.00097; qf_reg +0.00148; v1 all-family +0.00116; v2 all-family +0.00145; v2 mono +0.00119 | **Strongest recent model/ensemble change** |
| E1 blend vs E1-ExtraTrees blend | 0.77258 → 0.77357; paired fold gains roughly [+0.00063,+0.00068,+0.00016,+0.00027,+0.00059], 5/5 positive | Accepted component of v14G/v15 |
| v13_B: (OLD + E1 + E1ET)/3 | OOF 0.77376822; +0.00046768 vs v9, 5/5, CI [+0.00029455,+0.00063715]; +0.00024610 vs v9+CatBoost, 4/5, CI [+0.00005891,+0.00043075] | Strong candidate; became the basis of v14G |
| CatBoost new seed 17 | 5% new seed blend gain about +0.0000114, 3/5 positive; CI crosses zero | Reject |
| ExtraTrees seed 17 on qf_reg | +0.00009586 at fixed 5%, 5/5; CI [+0.00005639,+0.00012582] | Consistent but below practical +0.00015 threshold |
| ExtraTrees seed 17 on qf_shallow | +0.00009352 at fixed 5%, 5/5; CI [+0.00005531,+0.00012528] | Same, reject |
| Full second E1 ExtraTrees seed 17 branch | About +0.0000012 vs v14G, 3/5, CI crosses zero | Reject |
| Denoising autoencoder features | Full champion 0.7735221 → 0.7734956, −0.0000266; CI [−0.0001106,+0.0000653], 2/5 | Reject. Reconstruction loss was not predictive utility |
| Scratch MLP, BCE + pairwise-AUC fine-tune | OOF 0.744446 vs ~0.773522 champion; small blends failed | Reject |
| Compact CatBoost product keys | Blend gain +0.0000605; 1/5 positive; CI crosses zero | Reject; some finer product distinctions help |
| v13 product specialist stack | +0.0000561 vs CatBoost champion; 3/5; CI crosses zero | Reject |
| CatBoost `SqrtBalanced` | −0.0000512 blend change; 2/5; CI crosses zero | Reject |
| CatBoost Ordered boosting | +0.0000049; 3/5; CI crosses zero | Reject |
| Sampled pairwise logistic LightGBM ranker | Standalone OOF 0.7707169; fixed 10% blend delta −0.0000114; 3/5; CI [−0.0000747,+0.0000615] | Reject |
| CatBoost Lossguide trees | Fixed 10% blend +0.0000740; 4/5, but CI [−0.0000119,+0.0001720] and under practical threshold | Reject |
| LightGBM `scale_pos_weight=2` | Blend change −0.0000710; 0/5; CI [−0.0001323,−0.0000070] | Reject |
| XGBoost binary classifier, fixed 5% blend | −0.0000268, 1/5 positive; CI crosses zero | Reject |
| CatBoost depth 8, fixed total 10% CatBoost split 5%/5% | +0.000043; 4/5 positive; bootstrap [about +0.0000006,+0.0000845], under +0.00015 practical threshold | Reject |
| LightGBM `max_bin=127` ExtraTrees | +0.000024, CI crossed zero | Reject |
| GOSS LightGBM | About −0.000019 | Reject |
| v15 original G1: old+E1+E1Aet + CatBoost | OOF about 0.77395, +0.00004 vs v14G; 4/5, CI [−0.00002,+0.00010] | Reject |
| v15 original G2: old+E1ET+E1Aet + CatBoost | OOF about 0.77403, +0.00011; 3/5, CI [−0.00008,+0.00028] | Reject |
| v15 original G3: old+E1+E1ET+E1Aet + CatBoost | OOF about 0.77404, +0.00013; 4/5, CI [+0.00004,+0.00021], but gain under +0.0002 rule | Reject under original rule |
| v15 matched rank-space candidate | OOF 0.77410395 vs v14G 0.77393726, gain +0.00016669; 5/5 positive, CI [+0.00007838,+0.00026092]; public 0.77891 | **Current champion**; later gate of +0.00015, 4/5 and lower CI >0 passed |
| Fast sample: inner view folds 3 vs 5 | baseline 0.76960562; inner-3 0.76952932; inner-5 0.76957431 | Reject; changing fold count did not improve vs no-view baseline |
| Quick decoded product affordability block | 0.76960562 → 0.76991969 (+0.000314), 3/3 positive; adding income residual reduced candidate to 0.76974003 | Quick screen promising, then full test required |
| Full contract-decomposition qf_reg member replacement | Current OOF 0.77410395 → 0.77410615 (+0.00000220); 3/5 positive; CI [−0.00001090,+0.00001494] | Reject; quick gain did not survive full ensemble |
| Repeated cross-fit views, inner seeds 11/42/2024 | 0.77000339 → 0.76954952, −0.00045387; 0/3 positive | Reject |
| LightGBM `linear_tree=True` | 0.76891783 → 0.75282037, −0.01609746; 0/3 positive, one fold 0.72742 | Reject sharply |
| Product × external-score decile target encoding (quick sample) | 0.766913? → 0.766601; −0.000313; 1/3 positive | Reject |
| External score × affordability interactions | 0.769836 → 0.769812, −0.000023; 1/3 positive | Reject |
| External score × age/employment | 0.768612 → 0.768632, +0.000021; 1/3 positive | Inconclusive/no promotion |
| Product × segment target statistics (occupation/income/education/region) | Full reference blend change −0.000063; 1/5 positive | Reject |
| Bureau enquiry bursts — quick screen | 0.76960562 → 0.77017053, +0.00056492; all 3 folds positive | Looks promising only at screen stage |
| Bureau bursts — full v15 qf_reg replacement | 0.77410395 → 0.77410289, −0.00000107; 3/5; CI [−0.00001361,+0.00001056] | **Reject; no CSV**. Latest completed test |

## 4. Why quick screens can fail on the complete ensemble

Single-member gains can disappear once multiview/model branches are included because they already explain similar risk information. This occurred with both (a) contract/product affordability and (b) bureau-enquiry-burst features. All three fast screens used 80% of rows and 3 folds, so their AUCs are **not directly comparable** with the 171,202-row full OOF results. Only the full v15 confirmation should decide promotion.

## 5. Submission files and checksums

| File | User-reported public score | SHA-256 |
|---|---:|---|
| `GCI_submission_reproduced_final.csv` | 0.77865 | `8fbc351fbef1206ad5a8180e5526afc3d6f0ceea9249871a0cdfc9bf12f86691` |
| `GCI_submission_catboost_product_blend.csv` | 0.7787 (rounded/display ambiguous) | `c3fc40999a7fd90fc2c660e48cad2f3b138126aafdf2b4d6d1ca7053df6a9cc4` |
| `GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` | 0.77884 | `ac43cb2e9e73c98405356c452948cb0b58da1450c494d17d4d82ea2edf7800bf` |
| `GCI_submission_v15_rankspace_consistent_view.csv` | **0.77891** | `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7` |

All four have been checked locally for 61,500 rows, columns exactly `SK_ID_CURR,TARGET`, unique IDs, finite continuous predictions. Always re-check against the current sample file after moving between environments.

## 6. File map

- `03_NOTEBOOKS/GCI_final_reproduction_v9.ipynb` — baseline model definition and reproduction.
- `03_NOTEBOOKS/GCI_catboost_product_step_colab.ipynb` — CatBoost product-native step.
- `03_NOTEBOOKS/GCI_v13_reweight_and_ET_colab_cc.ipynb` — weight reoptimization and E1 ExtraTrees experiments.
- `03_NOTEBOOKS/GCI_v14_ET_deepening_colab.ipynb` — v14G creation.
- `03_NOTEBOOKS/GCI_v15_consistent_views_ET_colab.ipynb` — current branch code and original G1/G2/G3 variants.
- `04_SCRIPTS/run_e1Aet_branch.py` and `evaluate_and_build.py` — local v15 branch reproduction / rank-space CSV recipe.
- `04_SCRIPTS/full_bureau_burst_qfreg.py` — latest full-feature attempt (complete and rejected).
- `05_RESULTS/` — reports, JSON metrics, fold CSVs, logs.
- `06_CACHES/` — OOF caches and view/residual/model prediction caches for continuing without retraining from scratch.
- `01_DATA/` — supplied train/test/sample files used by local scripts.
- `02_SUBMISSIONS/` — the four scored candidate CSVs.
- `historical_archive/` — previous research package with detailed older scripts, predictions and logs.

## 7. Important methodological cautions

1. The same `random_state=42` five outer folds were reused extensively. A reported bootstrap interval is conditional on the predictions and does not adjust for trying many candidate ideas. Do not keep sweeping small variants on these same OOF labels.
2. A public score changing from 0.77884 to 0.77891 (or 0.77891 after v15) is user-reported leaderboard evidence, not something inferable from OOF alone. New candidate selection should require a locked recipe and one actual platform check.
3. Do not import external Home Credit bureau, previous application, installment, POS, or card history tables. Those may explain why public full-table solutions reach ~0.80, but are outside this competition's allowed input scope if the rules restrict the supplied file only.
4. The supplied application table does not contain `DEF_30_CNT_SOCIAL_CIRCLE` / `OBS_30_CNT_SOCIAL_CIRCLE` or corresponding 60-day social-circle fields. Do not write code assuming they exist.
5. Earlier logs have slightly varying CatBoost OOF values due to different saved/refitted artifacts; use the exact values and formulas from the corresponding report/notebook run, and never mix OOF arrays from different fold orders.
6. The v15 candidate uses the separate rank-space formula documented above; don't label it as exactly the notebook G3 arithmetic blend.

## 8. Recommended next-chat protocol

- Read this handoff and the latest `GCI_FAST_RESEARCH_EXTENSION_2026-10-17.md` first.
- Do not rerun the bureau-burst, contract-decomposition, product×segment, external-affordability, repeated-view, LinearTree, CatBoost seed/depth, second ExtraTrees seed, DAE, MLP, pairwise ranker, XGBoost, GOSS, `max_bin=127`, reweighting, or inner-fold-count tests listed above.
- There is no active training process. The latest bureau burst test completed and failed.
- If proposing a genuinely new experiment, write the hypothesis, why not already tested, features/model, time cost, leakage controls, folds, acceptance rule, expected failure modes *before* running it. Prefer low-cost pre-screen, but no submission until full OOF confirmation.
- The current champion is the user's submitted v15 CSV (public 0.77891). Keep it unchanged as fallback.
- Do not promise 0.79. The new result must be measured, not merely plausible.


---

# Appendix C: v15 candidate report

# GCI v15 rank-space consistent-view candidate — experiment report

**Status:** Candidate generated for one controlled leaderboard test. This is not a claimed leaderboard result.

## Candidate file

`GCI_submission_v15_rankspace_consistent_view.csv`

- 61,500 rows
- Columns exactly `SK_ID_CURR,TARGET`
- IDs unique and in `sample_submission.csv` order
- All predictions finite and continuous
- SHA-256: `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7`

## Experiment and formula

The uploaded v15 notebook's consistent-view / ExtraTrees branch (`e1Aet`) was executed against the uploaded training and test files. The original 5-fold seed (`random_state=42`), existing decoded-product target encodings, original member features, member iteration counts, production seeds, and ExtraTrees setting were preserved. Numeric feature frames were downcast to float32 between feature-engineering steps to fit the available 4 GB runtime memory limit; model features and training parameters were otherwise unchanged.

The stored v14G submission and its associated CatBoost test predictions were available, but the original raw OLD/E1/E1ET test-branch predictions were not separately present in the supplied artifacts. Therefore, the deliverable uses an explicitly matched rank-space formula rather than pretending to reproduce the notebook's exact raw four-branch arithmetic mixture:

`candidate = 0.90 * rank(0.75 * rank(B_v14G_base) + 0.25 * E1Aet_blend) + 0.10 * rank(CatBoost_product)`

Here `rank(B_v14G_base)` is recovered algebraically from the v14G submission, `E1Aet_blend` is the fixed member-weighted blend of the five new test predictions, and `CatBoost_product` is the saved CatBoost test score. The same rank-space formula was evaluated on OOF predictions before writing the CSV.

## Out-of-fold result

Five StratifiedKFold splits (`shuffle=True`, `random_state=42`) were used.

| Measure | Result |
|---|---:|
| v14G OOF AUC | 0.77393726 |
| Candidate OOF AUC | 0.77410395 |
| Gain | +0.00016669 |
| Fold gains | +0.00026809, +0.00020539, +0.00005971, +0.00002809, +0.00026588 |
| Positive folds | 5/5 |
| Paired stratified bootstrap 95% interval (300 resamples) | +0.00007838 to +0.00026092 |

**Decision:** passes the working promotion rule used for this candidate: gain at least +0.00015, at least 4/5 positive folds, and bootstrap lower bound above zero. The notebook's original v15 G3 rule used a stricter +0.00020 threshold; its exact raw arithmetic OOF reconstruction scored 0.77410411 (+0.00016685 vs v14G), so it would miss that original cutoff. The rank-space alternative passes the project’s later +0.00015 gate.

## Limitations

- The five folds were reused during extensive model selection, so the interval does not correct for selection bias.
- Small local differences versus the supplied v15 notebook outputs are expected due to runtime/library and numerical differences.
- The candidate has not been submitted. Its public leaderboard score is unknown until evaluated by the competition platform.
- Keep v14G (public score 0.77884) as the fallback. Submit this new CSV only as one controlled attempt; do not infer public performance directly from OOF AUC.


---

# Appendix D1: Three experiment report

# GCI World 2026 — Three Experiment Report

**Run date:** 2026-10-18  
**Dataset:** supplied `train.csv` (171,202 rows; target prevalence 8.073%), `test.csv` (61,500 rows), and submission template.  
**Metric:** ROC-AUC.  
**Status:** completed local 3-fold experiment; **no submission CSV generated** because no candidate passed the predeclared local gate.

## Executive summary

The current working directory did not contain the exact v15 notebook, trained models, or row-aligned v15 OOF predictions. Therefore these experiments use a reproducible, compact LightGBM baseline with standard financial/external-score feature engineering. Results below are valid for this local proxy only and **must not be described as gains over v15** (the historical v15 OOF AUC is 0.77410395 and public LB score 0.77891, as reported in the supplied research log).

| Candidate | Pooled local OOF AUC | Difference vs local baseline | Decision |
|---|---:|---:|---|
| Local LightGBM baseline | 0.753331 | — | Reference |
| Bureau-enquiry burst features | 0.753183 | -0.000149 | Reject |
| Three categorical-pair target encodings | 0.753318 | -0.000013 | Reject / no evidence of gain |
| 5% cross-fitted residual-rank blend | 0.753427 | +0.000095 | Inconclusive; below gate |

The baseline here uses only 180 LightGBM estimators (or early stopping), so its absolute AUC is not directly comparable to the much larger v15 ensemble. Candidate differences are within this experimental setup only.

## Fold-level results

| Fold | Baseline AUC | Bureau AUC | Bureau delta | Pair-TE AUC | Pair-TE delta | Residual corrected AUC | Residual branch base AUC | Residual delta within branch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.750025 | 0.749800 | -0.000224 | 0.749045 | -0.000979 | 0.749461 | 0.749393 | +0.000068 |
| 2 | 0.752145 | 0.751964 | -0.000181 | 0.752620 | +0.000475 | 0.752236 | 0.752156 | +0.000080 |
| 3 | 0.758109 | 0.757990 | -0.000119 | 0.758534 | +0.000424 | 0.758589 | 0.758487 | +0.000101 |

**Interpretation:** bureau features lost in all three folds. Pair target encodings helped folds 2–3 but the loss in fold 1 outweighed those gains. The residual correction improved its own outer-model score in all three folds by a small amount, but its pooled gain against the separate local reference was only +0.000095. This is below the predeclared +0.0001 pooled-gain threshold, and its cross-model comparison is not enough to justify a submission after extensive prior model selection.

## Experiment details

### 01 — Cross-fitted residual ranking

- Three outer stratified folds.
- Within each outer training partition, a 3-fold inner cross-fit generates baseline scores for the correction model's training rows.
- The correction model sees the inner OOF score plus a compact set of raw/engineered financial and external-score features.
- Fixed blend: 95% outer-model rank + 5% correction-model rank; blend weight was not tuned on validation labels.
- Per-fold improvement against the outer model's own score was +0.000068, +0.000080, and +0.000101 AUC. However, the pooled corrected predictions exceeded the separate local reference by only +0.000095, below the promotion threshold. **Do not promote.**

### 02 — Bureau-enquiry burst features

Tested hourly/monthly, monthly/yearly, quarterly/yearly and hourly/yearly ratios, nonzero-count, log total, all-zero indicator, and missingness indicators for the four supplied `AMT_REQ_CREDIT_BUREAU_*` fields. Pooled delta was -0.000149 and all three folds were negative. **Reject this implementation**; the previous 80%-sample screen's lift did not reproduce here.

### 03 — Categorical-pair target encodings

Tested smoothed, fold-safe pair encodings for:

- `ORGANIZATION_TYPE × OCCUPATION_TYPE`
- `NAME_INCOME_TYPE × NAME_EDUCATION_TYPE`
- `NAME_INCOME_TYPE × OCCUPATION_TYPE`

Training encodings are generated out-of-fold within each outer training partition; validation/test encodings use mappings learned only from that outer training partition. Pooled delta was -0.000013, with fold deltas -0.000979, +0.000475, +0.000424. **Reject as a package; no stable pooled improvement.**

## Submission decision

**No `submission.csv` was created.** The selection rule required pooled local OOF gain greater than +0.0001 and positive gain on all three folds. None met both conditions. This is intentional: generating a submission from a candidate that failed the gate would give a false impression of evidence.

## Files produced

- `GCI_three_experiments.ipynb` — runnable notebook for all three tests.
- `run_three_experiments.py` — standalone runner.
- `summary.csv` — pooled OOF metrics.
- `fold_results.csv` — fold-level AUCs and deltas.
- `oof_predictions.npz` — labels and OOF prediction arrays.
- `run.log` — console log from this run.

## Limitations and next action

1. These results do not directly compare candidates with v15's row-aligned OOF predictions or production test predictions.
2. The compact baseline uses fewer boosting rounds than the v15 pipeline; absolute AUCs are expected to differ.
3. The residual specialist's within-branch gains are small; after many experiments, a fresh/nested confirmation and a paired bootstrap are warranted before any promotion.
4. To test a real v15 improvement, provide the v15 notebook/model recipe and aligned OOF predictions. Re-run all candidates on the same held-out rows and compare against v15, not this proxy.


---

# Appendix D2: Repair and feature experiments

# GCI repair + feature experiments — 18 Oct 2026

## Decision first
- Keep `GCI_submission_v15_rankspace_consistent_view.csv` as champion (user-reported public ROC-AUC **0.77891**).
- **Do not submit** the failed residual probe again.
- A new, reproducible local LightGBM feature ensemble showed a consistent gain over its own single-model baseline on a fresh 3-fold split. This is a real *local model* improvement, but it is **not yet validated against v15 OOF predictions**, so it is not represented as a new champion submission.

## Why the residual probe scored lower
The submitted residual candidate scored **0.77869**, below v15's **0.77891**. Forensic inspection found two concrete implementation/validation faults:

1. **Train/test iteration mismatch:** OOF baseline predictions were generated from models whose best iterations were about **168–180**, but the test-side baseline feature was generated with `n_estimators=60`. The correction model therefore saw a different score-generating process at inference time than during meta-training.
2. **Wrong comparison target:** the reported +0.000095 OOF gain was measured against a compact standalone baseline (pooled OOF AUC 0.753331), not against the v15 champion OOF AUC 0.77410395. It did not establish that residual correction improves v15.
3. **Blend formula did not mean a 5% correction:** `0.95*rank(v15) + 0.05*rank(0.95*rank(compact)+0.05*rank(correction))` gives the compact score roughly 4.75% of the outer mixture and the correction roughly 0.25%, before ranking effects.

The submitted CSV was structurally valid. The rank changes were small but meaningful (average absolute rank-percentile change ~0.005; top-1% overlap ~96.9%). These faults explain why the experiment was not a valid champion improvement; they cannot prove that they account for every point of the public leaderboard difference.

## Experiments run in this repair pass
All local models use the supplied train labels only, leakage-safe fold validation, LightGBM 4.6.0, and 3-fold StratifiedKFold. These absolute AUCs are from a simpler local model and are **not comparable in level** to the v15 multiview ensemble OOF AUC.

### Development split (`random_state=20261018`)
| Variant | Features | Pooled OOF AUC | Fold AUCs |
|---|---:|---:|---|
| Full engineered feature set | 114 | 0.755769 | 0.754256, 0.758549, 0.754574 |
| Remove decoded-product block | 103 | 0.754667 | 0.752943, 0.757986, 0.753297 |
| Remove bureau block | 104 | 0.755794 | 0.754884, 0.757355, 0.755402 |
| Remove age/EXT/affordability interactions | 111 | 0.756835 | 0.756064, 0.758457, 0.756221 |
| Remove category-frequency features | 108 | 0.756516 | 0.755035, 0.759003, 0.755772 |
| Fixed 50:50 rank blend: no-interactions + no-frequency | — | **0.757390** | 0.756360, 0.759374, 0.756676 |

### Fresh confirmation split (`random_state=20261019`), with the 50:50 blend frozen
| Variant | Pooled OOF AUC | Fold AUCs |
|---|---:|---|
| Remove interactions | 0.757406 | 0.754557, 0.760306, 0.757397 |
| Remove category frequencies | 0.757147 | 0.754613, 0.760481, 0.756412 |
| Fixed 50:50 rank blend | **0.758005** | **0.755335, 0.761187, 0.757553** |

The frozen 50:50 rank blend beat the best single feature variant by **+0.000555** on the development split and **+0.000599** on the fresh confirmation split; it was positive on all 3 confirmation folds. This supports model complementarity in the local branch. The feature ablation also suggests product decoding helps this local model (~+0.00110 pooled AUC versus removing product features), while the tested age/EXT/affordability interactions and category frequencies hurt this particular configuration. Bureau ratios were neutral in this ablation. These are model-specific conclusions, not universal truths.

An ExtraTrees variant of this simpler model was tested and rejected: pooled OOF AUC 0.753263 vs 0.756653 for its non-ET counterpart. Do not transfer the earlier successful ExtraTrees result on the champion's stacked feature family to this simpler feature family.

## What happens next
The new local ensemble is a promising challenger component, but no safe submission CSV was created. To test it against v15 correctly, its predictions must be aligned with the **exact v15 outer-fold OOF rows** (or the full v15 cache bundle must be available). The current local directory has the v15 test submission and source notebook, but not the `gci_cache_final` OOF/model caches. The original notebook rebuilds several branches and is documented to take roughly 1.7–2 hours on CPU. Without aligned v15 OOF predictions, making a 1–5% rank blend into a submission would repeat the same validation mistake.

## Artifacts
- `run_repair_and_feature_sweep.py` — first feature ablation (development split).
- `run_feature_sweep2.py` — combined feature ablation and ExtraTrees negative control.
- `run_fresh_confirmation.py` — frozen-feature comparison on fresh split.
- `repair_feature_sweep_summary.csv`, `fresh_confirmation_summary.csv`, `confirmed_feature_blend_results.csv` — results.
- `WHY_RESIDUAL_SUBMISSION_DROPPED.md` — detailed residual forensic analysis.

## Reproduction
Set `GCI_DATA_DIR` to the directory containing `train.csv`, `test.csv`, and `sample_submission.csv`, then run the three scripts in order. Each writes outputs to `GCI_three_experiment_results` beneath that directory. All feature variants are label-free apart from the model training target; no external data or pseudo-labels are used.


---

# Appendix E: Residual forensic diagnosis

# GCI residual-ranking submission: forensic diagnosis

Date: 2026-10-18

## Leaderboard observation

- v15 rank-space consistent-view submission: user-reported public AUC 0.77891.
- Residual-ranking probe: user-reported public AUC 0.77869.
- Difference: -0.00022.

A public score alone cannot establish whether all of the difference is model degradation or partly public-split variability. The implementation audit nevertheless identifies concrete train/inference and evaluation-comparison flaws.

## Root cause 1 — wrong baseline for the claimed +0.000095

The +0.000095 OOF AUC was measured against a compact, newly built LightGBM baseline (pooled OOF AUC 0.753331), not against the v15 champion's aligned OOF predictions (historical OOF AUC 0.77410395). It therefore never demonstrated improvement over v15. The leaderboard candidate's top-level blend used the residual branch to perturb v15 despite no direct paired OOF test of that formula.

## Root cause 2 — training/inference iteration mismatch

The OOF baseline recorded in `oof_predictions.npz` came from fold models with early-stopped best iterations around 168–180 (see `fold_results.csv` and `run.log`). But `create_residual_leaderboard_probe.py` trains the full-data test-inference baseline with `n_estimators=60`. The correction model is trained using OOF baseline scores from the longer models, then receives test scores from the much shorter 60-tree model. This is a train/test feature-distribution mismatch in the correction model's `BASE_OOF_SCORE` input.

## Root cause 3 — most of the added 5% was not the correction model

The submitted formula is:

`0.95 * rank(v15) + 0.05 * rank(0.95 * rank(compact_baseline) + 0.05 * rank(correction))`

Thus the overall candidate gives the compact baseline an effective 4.75% contribution and the correction model only an effective 0.25% contribution before rank nonlinearities. This is not a clean 5% residual-correction blend. The compact baseline is substantially weaker than v15, and the correction's tiny weight cannot be assumed to repair that dilution.

## What changed in the submitted ranking

The files have matching IDs and the candidate is structurally valid. Comparing prediction ranks:

- Rank correlation between v15 and candidate: 0.9997344.
- Mean absolute percentile-rank change: 0.004999.
- Maximum absolute percentile-rank change: 0.03935.
- Top 1% overlap: 96.92%.
- Top 5% overlap: 98.02%.
- Top 10% overlap: 98.33%.

So the file made a small but material reordering of cases; the AUC loss is consistent with the added ranking signal being unhelpful/noisy on the public evaluation set. These rank statistics do not by themselves identify which individual pairwise inversions caused the loss.

## New plan: champion-conditioned, cross-fitted rank correction

Do not blend a weak standalone model into v15 and call it residual ranking. Instead:

1. Rebuild v15 OOF predictions using the exact v15 feature and multiview pipeline, with every training row scored only by models that did not train on that row.
2. Fit a small correction learner on the v15 OOF score plus a small, pre-specified set of features that could explain local ranking errors (EXT_SOURCE values/spread, credit/income burden, employment sentinel, bureau enquiry recency/bursts, and selected categorical groups). Use only cross-fitted score features for the meta-training rows.
3. On an untouched outer fold, generate v15 predictions and correction predictions using exactly the same training procedure as meta-training. Compare the correction directly to v15 on the same held-out rows.
4. Test a pre-registered fixed rank-space blend (e.g. 1%, 2%, 3%; choose on inner validation only, then freeze it before outer confirmation). Also test a gated correction that only applies where the correction is confident and disagrees with v15; gating thresholds must be chosen inside inner folds.
5. Promote only if paired AUC improves on a fresh outer split, fold gains are stable, and the confidence interval supports a gain. Then train the matched pipeline for test predictions and generate one CSV.

## Current decision

- Keep v15 (public 0.77891) as champion.
- Do not resubmit the current residual probe.
- Do not claim the residual idea failed in general; this submission did not validly test it against v15.
- The v15 notebook/source pipeline and aligned v15 OOF predictions are needed to complete the decisive champion-conditioned test. The v15 test CSV alone is not enough to train and validate a correction model without leakage or distribution mismatch.


---

# Appendix F: Conversation-only updates not fully represented in earlier logs

## v17 probe
The 30k-row three-fold CatBoost screen compared a base model with a variant augmented by consistent-view predictions. Pooled AUC moved from approximately 0.75354 to 0.75195, a -0.00159 change, negative in all three folds. This is only a screen and not a full reproduction of v17 because `gci_cache_final` was unavailable.

## v18 public result
The v18 submission CSV was created using 95% v15 rank plus 5% v18 rank, where v18 is a 50:50 rank blend of two compact LightGBM variants. The user reported a public score of 0.77882, so it is rejected relative to v15's 0.77891/0.77892. The local compact feature-blend improvements are retained as hypotheses but are not submission evidence.

## OLD/NEW compact experiment
A separate new-seed three-fold experiment on 171,202 rows compared compact OLD and NEW feature families. Reported pooled AUCs: OLD 0.755212, NEW 0.755694, 50:50 rank blend 0.756750, a +0.001056 improvement over the better single member. The values are from a simple compact recipe and not comparable to v15. Its generated runner is `run_old_new_member_research.py`; associated output files were recorded under `GCI_old_new_member_research/` when available.

## 20 October leaderboard image
Screenshot path in the session: `image(20261020-093631).png`. It shows the user account at rank 17 with score 0.77891. This confirms the position/score shown in the image, not the hidden evaluation method or competitors' methodology.

## Important integrity note
The consolidation preserves claims with their source-level caveats. It does not claim that all old chat content is present if a file or conversation is not accessible in the current project context. If the project has additional chats whose reports or attachments are not indexed, those sources need to be added to the project context for literal completeness.
