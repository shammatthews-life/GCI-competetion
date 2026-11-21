# GCI Home Credit Default-Risk Research — Master Verification & Progress Ledger

> **Purpose:** One append-only record of every source audit, rule check, experiment, score, decision, and next action.
>
> **Project status as of:** 2026-11-03  
> **Competition metric:** ROC-AUC  
> **Target:** Improve the competition score through reproducible, competition-compliant modelling. A target of 0.79+ is aspirational, not a guaranteed outcome.
>
> **Critical rule:** Never delete or overwrite historical experiments, reports, notebooks, OOF files, submissions, or previous ledger entries. New work gets a new experiment folder and a dated entry below. Correct mistakes by adding a correction entry; do not silently rewrite history.

---

## 0. How to maintain this file

### Append-only update procedure

After **every** source inspection, rule audit, code audit, experiment, validation, or submission:

1. Add a new dated entry under `Daily verification log` at the end of this file.
2. Record the exact source files and SHA-256 hashes used.
3. Record the code/script/notebook version and its hash.
4. Record the dataset row counts, columns, target availability, fold IDs, random seeds, library versions, and preprocessing rules.
5. Record the metric, pooled OOF AUC, fold-wise AUCs, mean/std, delta against the named baseline, and fold wins.
6. State whether the result is:
   - `REPRODUCED` — rerun from available source and inputs;
   - `INDEPENDENTLY RECALCULATED` — metric recomputed from saved OOF predictions;
   - `REPORTED / UNVERIFIED` — claimed in a notebook, message, screenshot, or report but not independently reproduced;
   - `EXPLORATORY` — discovered on already-examined validation folds;
   - `CONFIRMED` — tested under a genuinely untouched/nested evaluation protocol;
   - `REJECTED` — invalid, non-compliant, broken, or not useful.
7. Record all failed tests and negative results. Do not omit them.
8. Update the `Current verified position` section only by adding a dated status note; preserve prior status notes.
9. Update the `Experiment index` by adding a row. Never remove an older row.
10. Verify output hashes and file existence; record any missing artifact explicitly.

### Suggested folder structure

```text
D:\gci\truth\
  GCI_MASTER_VERIFICATION_LEDGER.md
  experiments\
    phase2_advanced_feature_engineering\
    phase2b_feature_intelligence\
    phase3_hidden_pattern_discovery\
    phase3b_deep_validation\
    phase3c_model_intelligence\
    phase3d_competition_compliance_and_domain_research\
    phase3d_claude_hypothesis_probes\
    phase4_unified_reproduction\        # create only when ready
  reports\
    final_competition_audit\
```

Do not create a replacement "final" pipeline until the source models, feature definitions, fold protocol, and rules have been audited. The eventual unified pipeline must be a new, versioned implementation and must retain individual model outputs and component-level ablations.

---

## 1. Current verified position — 2026-11-03

### Best scores and what they mean

| Result | Pooled OOF ROC-AUC | Evidence status | Interpretation |
|---|---:|---|---|
| Corrected Phase 2 raw LightGBM baseline | 0.760658 | Reproduced in prior controlled run | Historical reference; five fixed folds |
| Phase 3B controlled product-feature model | 0.762444 | Reproduced in Phase 3B transcript | +0.001786 vs 0.760658; product features improved all five folds |
| Phase 3C LightGBM + product feature model | 0.762491 | Saved OOF/results inspected in current audit | Different run/reference; do not treat as the same exact fold/model run as Phase 3B |
| Phase 3C fixed LightGBM/CatBoost probability blend | 0.763957 | Saved OOF blend independently inspected/recalculated in follow-up transcript | Best pre-follow-up fixed blend in that pipeline |
| Phase 3D six-month-snapped payment/credit residual model | 0.763049 | Exploratory OOF result; recalculated from saved OOF | +0.000557 vs 0.762491 reference; 3/5 folds improved |
| Phase 3D rounded payment/credit residual model | 0.762983 | Exploratory OOF result; recalculated from saved OOF | +0.000492 vs 0.762491 reference; 4/5 folds improved |
| Phase 3D snapped-residual + CatBoost 50/50 blend | 0.764338 | Exploratory blend; adaptive discovery | +0.000381 vs prior 0.763957 blend; not independent confirmation |
| Claude seven-model rank blend | 0.7647 | Reported by Claude; source and OOF vectors not found locally | Not reproduced; not a verified score |
| V5 notebook blend | 0.77213 (reported) | Saved notebook output inspected; full source/cache/OOF reproduction unavailable | Unverified; do not use as current champion |
| V5 `v3_A` | 0.77746 (reported) | Not independently reproduced | Unverified |
| Earlier public leaderboard claim | 0.8013 (reported) | Not independently tied to compliant reproducible source | Historical claim only; do not compare directly with OOF |

**Do not collapse these scores into one leaderboard.** Different experiments can have different folds, model configurations, training rows, preprocessing, and validation histories. A valid comparison requires identical row IDs, target values, folds, metric calculation, and evaluation protocol.

### Current practical score statement

- Latest inspected exploratory result: **0.764338** (Phase 3D adaptive blend).
- Latest reproducible controlled product-model result in the earlier Phase 3B pipeline: **0.762444**.
- Best claim not verified: **0.77213** from V5.
- Claude's **0.7647** is unverified until its exact source and predictions are supplied and reproduced.
- No new champion promotion has been justified by the already-examined folds.
- 2026-11-04 Phase 9 adds a best exploratory CatBoost-GPU mean seed-pooled OOF AUC of **0.768240** from native product categories (+0.006213 against that model family's baseline); it is not an independent/untouched confirmation, does not resolve competition eligibility, and was not promoted.

### 2026-11-06 status addendum (append-only)

- Phase 20's corrected 150-experiment GPU grid is complete: 130 real GPU
  experiments completed, 20 clustering experiments were GPU-blocked, and all
  30 placebo controls completed. Its strongest contract-type discovery lift
  was `+0.001203792` on the 60% discovery subset; this is exploratory, not a
  full-data score or new champion.
- The best completed full-data reference remains the Phase 15 three-seed OOF
  mean-prediction AUC `0.768824708`; Phase 9's `0.769839127` is retrospective
  and not an independent confirmation. No 0.780 result has been demonstrated.
- Phase 21 focused full-data validation is not yet complete. Its first
  execution stopped before saving fold results; a checkpointed, method-correct
  rerun is waiting for the shared GPU to become available.

---

## 2. Data and competition rules

### Known dataset facts

| Item | Known value |
|---|---|
| Training rows | 171,202 |
| Test rows | 61,500 |
| Target | `TARGET` exists in train only |
| Identifier | `SK_ID_CURR` |
| Original predictors | 32 |
| Train columns | 34 including ID and TARGET |
| Test columns | 33 including ID |
| Metric | ROC-AUC |
| Primary data | Competition `train.csv` and `test.csv` |

Known paths previously used:

```text
D:\gci\inpuy\train.csv
D:\gci\inpuy\test.csv
D:\gci\rules and info\04. Competition (due Nov 20)\Competition\HomeCredit_columns_description.xlsx
D:\gci\truth\reports\final_competition_audit\OFFICIAL_RULES_AUDIT.md
D:\gci\truth\COMPETITION_COMPLIANCE_MATRIX.csv
```

**Path caveat:** The original competition input folder was also inspected at:
`D:\gci\rules and info\04. Competition (due Nov 20)\Competition\input\train.csv` and `test.csv`. Hashes must be compared before assuming these files are identical to `D:\gci\inpuy`.

### Rule status and governance

Earlier rule audit found unresolved/ambiguous permission concerning:
- pooled train/test statistics (transductive feature construction);
- target encoding and any encoding using labels;
- any pseudo-identity or repeated-person target aggregation;
- any use of external datasets or unapproved information.

Until the official competition text clearly permits a technique, default to **not using it**. No external data, test labels, hand labels, or manual applicant-specific prediction edits are permitted in the compliant path.

Every candidate must be:
- deterministic with recorded seeds;
- reproducible from source and competition inputs;
- trained with fold-local preprocessing;
- evaluated with held-out outer-fold labels excluded from feature construction, early stopping, feature selection, threshold/weight tuning, and model selection;
- saved with row IDs, fold IDs, targets (OOF only), predictions, code hashes, and a manifest.

The official dictionary must be checked before assigning units or financial meaning to `AMT_ANNUITY`, `AMT_CREDIT`, or any derived ratio. Do not assume annuity is monthly or annual unless the source explicitly establishes the period.

---

## 3. Experiment index — append new rows, never delete

| Date | Experiment | Main question | Result | Status |
|---|---|---|---|---|
| Earlier | Corrected Phase 2 baseline | Establish leakage-controlled raw LightGBM reference | 0.760658 | REPRODUCED |
| Earlier | Phase 2B feature intelligence | Peer-relative and external-score interactions | Best 0.760693; negligible lift | REJECTED for promotion |
| Earlier | Phase 3 hidden-pattern discovery | Product-decoder hypotheses | 0.762826 discovery result | EXPLORATORY |
| Earlier | Phase 3B deep validation | Re-test product feature family under controlled five-fold protocol | 0.762444 | REPRODUCED |
| Earlier | Phase 3C model intelligence | Compare model families and fixed blends | LGB product 0.762491; fixed LGB/CatBoost blend 0.763957 | REPRODUCED from saved OOF; validation exposure remains |
| 2026-11-03 | Phase 3D domain formula audit | Audit Claude/V5 formulas and test formula families | All broad formulas neutral/negative | REPRODUCED / EXPLORATORY |
| 2026-11-03 | Phase 3D Claude hypothesis probes | Test term proxies, payment/credit residuals, external-score weighting, monotonic constraints and fixed blends | Best single candidate 0.763049; exploratory blend 0.764338 | EXPLORATORY |
| Pending | Claude `solution.py` exact reproduction | Reproduce seven models and rank blend | Source/OOF unavailable | BLOCKED |
| Pending | V5 notebook exact reproduction | Recover caches, features, folds, predictions, and code provenance | Incomplete | BLOCKED |
| Pending | Unified pipeline | Assemble only audited/reproducible components | Not started | NOT STARTED |
| Pending | Untouched/nested confirmation | Confirm selected pipeline without adaptive fold reuse | Not available yet | NOT STARTED |
| 2026-11-04 | Phase 9 ChatGPT/Claude product-key and CatBoost tests | Discriminate ratio keys from amount/K keys; assess subgroup and 130.5-gap evidence; compare LightGBM nested means with CatBoost native categories/crosses | LightGBM Q4 +0.003074; CatBoost GPU native eight keys +0.006213; all core results independently recalculated | EXPLORATORY; rule scope unresolved |
| 2026-11-06 | Phase 20 segmentation grid (completion correction) | Test 150 feature-only segmentation/model configurations with placebo controls | 130 real GPU runs and 30 placebo controls completed; 20 clustering runs GPU-blocked; best discovery lift +0.001204 | EXPLORATORY; no champion; holdback labels previously exposed |
| 2026-11-06 | Phase 22 pairwise AUC error decomposition | Locate which baseline score bands account for pairwise ranking errors and where a nested blend corrects them | Phase 15 `0.767510928`; Phase 18 nested blend `0.768668866` (+0.001157938); exact 2.175B pair decomposition | EXPLORATORY; diagnostic only; no model selection |
| 2026-11-06 | Phase 21 focused full-data validation (completion) | Measure learning-curve and seed-scaling gains; ablate feature groups; test domain flags and specialist routing on matched five-fold OOF | Reproduced single-seed `0.767472835`; 20-seed mean `0.768650650`; calibrated contract mixture `0.767663401` | COMPLETED; exploratory; no champion or independent confirmation |

---

## 4. Findings: hidden patterns and feature hypotheses

### 4.1 Product / loan-contract structure

Phase 3 product-decoder experiments found a potentially useful signal:
- Phase 3 discovery pooled OOF AUC: 0.762826 versus 0.760658 baseline (+0.002168).
- Phase 3B controlled all-product pooled OOF AUC: 0.762444 versus 0.760658 (+0.001786), improving all five folds.
- Product components included K-lattice, Q-ratio keys, fee-adjusted P keys, and financing lattice variants.
- Fee estimate varied roughly 123–142.5; fixed fee and financing step assumptions were not established as literal contract mechanics.
- The v17 financing increment 0.0132 was not consistently supported; 0.0066 appeared more stable in one discovery.
- These are predictive encoding hypotheses, not verified descriptions of actual loan contracts.

Do not use arbitrary fee constants as facts. Re-derive candidate grids from training data only, report stability across folds, and compare with negative-control grids.

### 4.2 Phase 3D formula screening

Reference in this follow-up run: Phase 3C LightGBM + product, pooled OOF AUC 0.762491.

| Candidate | Pooled OOF AUC | Delta vs reference | Fold wins |
|---|---:|---:|---:|
| Risk-adjusted LTV | 0.762373 | -0.000118 | Not promoted |
| External-score CV | 0.762277 | -0.000215 | Not promoted |
| LTV × external-score uncertainty | 0.762255 | -0.000236 | Not promoted |
| Financial-stress proxy | 0.762227 | -0.000265 | Not promoted |
| Bureau-window ratio | 0.762367 | -0.000124 | Not promoted |
| All screened formulas | 0.762317 | -0.000174 | Not promoted |

### 4.3 Payment-to-credit residual hypotheses

Let `C = AMT_CREDIT`, `A = AMT_ANNUITY`.

| Candidate | Pooled OOF AUC | Delta vs 0.762491 | Fold wins |
|---|---:|---:|---:|
| `round(C/A)` term proxy | 0.762104 | -0.000387 | 1/5 |
| Six-month-snapped term proxy | 0.762199 | -0.000293 | 1/5 |
| `A * round(C/A) / C` | 0.762983 | +0.000492 | 4/5 |
| Rounded `A*n-C` gap | 0.762527 | +0.000036 | 3/5 |
| Six-month-snapped `A*n/C` | 0.763049 | +0.000557 | 3/5 |
| Six-month-snapped `A*n-C` gap | 0.762451 | -0.000040 | 2/5 |

Interpretation:
- The rounded residual can be written approximately as `round(C/A)/(C/A)`. It creates a sawtooth/quantization transform of the existing ratio and may expose fractional-ratio patterns to tree models.
- Its reported Spearman correlation with `C/A` was about 0.027.
- The snapped candidate's six-month grid is an unsupported assumption.
- The residuals do **not** prove actual loan term, total repayment, interest, or a real six-month schedule.
- A subgroup gain for revolving loans was exploratory and subject to multiple-comparison risk.

### 4.4 Loan annuity inversion: mathematical limitation

Standard amortization equation:

`A = P*r / (1 - (1+r)^(-n))`

where `P` is principal, `A` payment per period, `r` periodic rate, and `n` number of payments.

Important:
- Given only `P` and `A`, both `r` and `n` are unknown. There is not a unique solution.
- `round(P/A)` is not the true term when interest is positive.
- Six-month snapping, 6–72 clipping, and the 1.12 multiplier are assumptions, not observed contract facts.
- Solving for rate requires a specified duration and payment interval. If the duration/period is guessed, the resulting rate is conditional on that guess, not an observed implied rate.
- Do not call an inferred value an actual interest rate or loan duration.
- Do not use `A*n-P` as actual total interest unless `n` and payment periodicity are known.

### 4.5 External-score hypotheses

Claude proposed:
- mean, min, max, standard deviation, product;
- weighted score `0.20*E1 + 0.35*E2 + 0.45*E3`;
- missing-value replacement with 0.5;
- monotonic constraints.

Results in the tested Phase 3D setup:
- fixed 0.20/0.35/0.45 weighting: 0.762025 (negative);
- fold-local cross-fitted logistic external score: 0.762418 (slightly negative);
- monotonic external-score decreasing: 0.762368 (negative).

Do not assume 0.5 is a neutral missing value or that the proposed weights are optimal. Compare missingness flags, fold-fitted imputation, individual scores and interactions. Validate constraints individually.

### 4.6 Monotonic constraints

Tested:
- `DAYS_BIRTH` increasing: 0.761803;
- `DAYS_BIRTH` decreasing: 0.762075;
- external scores decreasing: 0.762368.

All were below the 0.762491 reference. Do not impose these constraints globally without evidence.

### 4.7 Blending

- Existing fixed LightGBM/CatBoost probability blend: 0.763957.
- Exploratory 50/50 snapped-residual model + CatBoost: 0.764338.
- Exploratory 50/50 rounded-residual model + CatBoost: 0.764328.
- All-five-model equal rank blend: 0.762585.
- Existing model rank correlations were high (about 0.925–0.984).

The 0.764338 and 0.764328 blends were examined after formula screening on the same exposed folds. They are adaptive discovery, not independent confirmation. Do not tune blend weights on the same OOF labels and report the resulting AUC as unbiased.

---

## 5. Claude and V5 source audit status

### Claude from-scratch solution

Claude's supplied summary claims:
- seven models;
- individual OOF AUC range 0.7585–0.7640;
- `lgb_onehot` about 0.7640;
- rank blend 0.7647;
- term proxy `round(P/A)` clipped 6–72;
- implied rate solved by bisection;
- a six-month snapped variant with a 1.12 multiplier;
- external-score weighted combination;
- monotonic constraints.

Audit conclusion:
- Exact `solution.py`, full seven-model source, and matching OOF prediction vectors were not found in the local search described in the 2026-11-03 transcript.
- Claude's 0.7647 cannot currently be reproduced or attributed to exact model code.
- The written summary is not a substitute for source code.
- Request the exact source, dependency versions, fold assignments, model parameters, and per-model OOF predictions.
- Do not merge Claude's formulas into the champion until exact feature and model code are audited.

### V5 notebook

Notebook inspected:
`C:\Users\SHARON\Downloads\GCI_v5_dropPF4_colab.ipynb`

Audit observations:
- Notebook cells and saved outputs were inspected.
- Some cached arrays/files used by the notebook were not available for source-level reproduction.
- The saved output showed a V5 blend and a gate that required a positive mean gain above +0.0002 and positive folds in at least 4/5 cases.
- In the displayed results, new-minus-old mean was about -0.00005 with 2/5 folds positive; 50/50 old/new mean was about +0.00010 with 4/5 positive. Neither met the stated +0.0002 mean-gain gate.
- `v3_A` score 0.77746 and V5 blend score 0.77213 remain unverified as fully reproducible results.
- The exact caches and source provenance must be recovered before accepting these claims.
- Do not copy saved arrays into a new solution without regenerating them from auditable source.

---

## 6. Unified code: requirements before implementation

The eventual unified code should be created as a **new** versioned project, not by overwriting the champion or any historical experiment.

Required design:

1. **Single entry point:** `python solution.py input_dir output_dir`.
2. **Reproducibility:** fixed seeds; record Python and package versions; deterministic preprocessing where supported.
3. **Input audit:** verify exact expected files, columns, dtypes, row counts, unique IDs, target distribution, and train/test schema.
4. **Feature registry:** each feature has formula, inputs, units/assumptions, source, missing-value behavior, fit scope, and compliance status.
5. **Separate feature families:** raw, financial ratios, external scores, product decoder, missingness, temporal/bureau, and approved interactions.
6. **Fold-local fitting:** all learned imputers, encoders, quantile bins, peer statistics, residual models and frequency statistics fit on training partitions only.
7. **Nested validation:** outer fold remains untouched. Early stopping, feature selection, model choice and ensemble weights must be determined without outer validation labels.
8. **Model registry:** each model trained and evaluated separately; save OOF and test predictions per model.
9. **Ablations:** raw-only, each family individually, approved combinations, and negative controls.
10. **Blend governance:** fixed weights or properly nested training-only optimization; save the weight-selection procedure.
11. **Prediction integrity:** output one prediction per test ID, preserve input order or explicitly align by ID, finite predictions in [0,1], no manual edits.
12. **Audit artifacts:** code hashes, input hashes, environment, fold assignments, OOF predictions, fold metrics, feature manifest, rule checklist, execution log, and output manifest.
13. **Submission safety:** never overwrite prior submissions; create timestamped/versioned outputs.
14. **No hidden dependencies:** no undocumented caches, old submission arrays, internet data, or files outside the declared inputs.

Do not put every candidate into one giant model immediately. First establish which source code and features are valid, then integrate them in controlled stages with ablations.

---

## 7. Current blockers

| Blocker | Why it matters | Required resolution |
|---|---|---|
| Claude `solution.py` missing | Cannot audit exact models or reproduce 0.7647 | Obtain exact source and OOF vectors |
| V5 caches/OOF vectors missing or incomplete | Cannot verify 0.77213 / 0.77746 | Recover original cache-generation code and outputs |
| Different fold/baseline references | Scores may not be comparable | Re-run candidates on one declared fold manifest |
| Validation folds repeatedly examined | Results are adaptive discovery | Use nested evaluation and/or genuinely new labelled data |
| Annuity periodicity unclear | Rate/term/DTI interpretations may be wrong | Confirm official dictionary or avoid period claims |
| Competition permission ambiguity | Transductive/target encoding could violate rules | Resolve exact official wording before use |
| Unified pipeline not yet assembled | No single reproducible end-to-end implementation | Build new version only after source audit |

---

## 8. Next execution sequence

### Step 1 — Freeze and inventory
- Hash all original inputs, rules, notebooks, scripts, experiment outputs and submissions.
- Inventory every file recursively under the research workspace.
- Classify each artifact: source, data, cache, OOF, submission, report, temporary.
- Identify duplicate files and compare hashes; do not delete duplicates.

### Step 2 — Recover and audit source
- Obtain Claude's exact `solution.py`, full notebook, OOF files and environment.
- Inspect every cell and every source line.
- Trace every prediction back to input rows and model outputs.
- Audit all reads/writes, caches, joins, target usage, train/test pooling, group statistics, encodings and blending.
- Reproduce one model at a time.

### Step 3 — Establish a common evaluation contract
- One frozen row-ID/fold assignment file.
- One exact baseline implementation.
- Same outer folds, metric, feature input rows, early-stopping rules and model parameters for all comparisons.
- Record fold-wise AUC, pooled OOF AUC, mean fold AUC, standard deviation, paired fold deltas and fold wins.
- State clearly that repeated fold exposure makes this discovery, not untouched confirmation.

### Step 4 — Feature and hidden-pattern research
- Re-test product decoding with negative controls.
- Examine distributions, repeated values, quantization, rounding lattices and conditional relationships.
- Test payment/credit residuals without claiming true loan term.
- Test external-score disagreement and missingness.
- Test feature-family complementarity and error overlap.
- Test applicant subgroup differences only as exploratory diagnostics, with sample sizes and multiple-testing caution.
- Do not promote a feature on univariate AUC alone.

### Step 5 — Model and blend audit
- Reproduce each model separately.
- Compare prediction correlations and error overlap.
- Test fixed and nested blends.
- Do not optimise and evaluate weights on the same OOF labels.
- Preserve per-model predictions.

### Step 6 — Unified solution and final reproducibility
- Assemble only audited components.
- Run from clean inputs with caches disabled.
- Verify exact output schema and test IDs.
- Re-run the same command and compare output hashes where deterministic.
- Only then create a candidate submission; never overwrite a historical submission.

---

## 9. Daily verification log

### 2026-11-03 — Phase 3D source/formula follow-up

**Sources inspected**
- Claude's written solution summary supplied in chat.
- `GCI_v5_dropPF4_colab.ipynb` (saved notebook source and outputs).
- Existing Phase 3B and Phase 3C experiment reports/scripts/OOF artifacts.
- Official rules audit and competition dictionary.
- Phase 3D formula experiment outputs.
- Phase 3D Claude hypothesis follow-up outputs.

**Work performed (as recorded in the agent transcript)**
- Searched local research, downloads and Colab folders for Claude's `solution.py`, model OOF predictions, and blend files.
- Inspected Phase 3C model benchmark results, ensemble ablations, fold results, correlation outputs and runtime records.
- Executed formula/constraint probes on the preserved five-fold setup.
- Analysed term-proxy distributions, default-rate patterns, correlations, and subgroup AUC diagnostics.
- Recalculated pooled OOF AUCs from saved predictions.
- Verified follow-up output manifest hashes and file sizes.
- Compared fixed blend variants against the previous fixed LGB/CatBoost blend.
- Left earlier champion and historical experiments unchanged.

**Key results**
- Phase 3D broad formula screening: all tested candidates were below the 0.762491 reference.
- Best single residual candidate: six-month-snapped `A*n/C`, 0.763049 (+0.000557), 3/5 folds improved.
- Rounded `A*round(C/A)/C`: 0.762983 (+0.000492), 4/5 folds improved.
- Exploratory snapped-residual + CatBoost blend: 0.764338.
- Exploratory rounded-residual + CatBoost blend: 0.764328.
- Claude 0.7647 and V5 0.77213 remain unreproduced.
- No candidate promoted to champion.

**Interpretation**
- The residual may expose quantization/fractional-ratio structure.
- It does not establish actual payment duration, interest rate, or total paid.
- Six-month snapping is unsupported by the dictionary.
- Blend gains are adaptive discovery because formulas and blends were examined on exposed folds.

**Artifacts recorded**
```text
D:\gci\truth\experiments\phase3d_competition_compliance_and_domain_research\
D:\gci\truth\experiments\phase3d_claude_hypothesis_probes\
```

**Outstanding**
- Obtain Claude's exact source and OOF predictions.
- Recover V5 cache-generation code and missing OOF artifacts.
- Resolve official rule ambiguity.
- Design nested/untouched validation.
- Build the unified pipeline only after source and feature audits.

---

## 10. New entry template — copy for every future test

### YYYY-MM-DD — [experiment / verification name]

**Question / hypothesis:**  
[What exactly are we checking?]

**Sources and hashes:**  
[List exact data, rules, notebook, script and hashes.]

**Code version:**  
[Script path, Git commit if available, SHA-256.]

**Dataset and protocol:**  
[Rows, columns, target, fold manifest, seed, preprocessing, fit scope.]

**Rule audit:**  
[Allowed / disallowed / unresolved, with exact rule citation.]

**Execution:**  
[Exact command, runtime, environment, completed/failed.]

**Results:**

| Candidate | Pooled OOF AUC | Mean fold AUC | Fold SD | Delta vs named baseline | Fold wins |
|---|---:|---:|---:|---:|---:|
| Baseline | | | | — | |
| Candidate | | | | | |

**Integrity checks:**  
[ID alignment, target alignment, fold alignment, finite predictions, hashes, manifest.]

**Findings:**  
[What the data supports; distinguish observation from hypothesis.]

**Limitations / leakage risks:**  
[What remains uncertain.]

**Decision:**  
[Keep / reject / repeat / needs nested validation / promote only after confirmation.]

**Next action:**  
[One concrete next task.]

---

## 11. Non-negotiable research principles

- Preserve every historical result, including bad results.
- Never describe a proxy as a true financial quantity without evidence.
- Never claim a score is reproduced unless the source, inputs, folds and predictions support it.
- Never compare AUCs from different folds/pipelines as if they were a controlled experiment.
- Never tune on outer-fold labels.
- Never use test labels or unauthorized external data.
- Never infer permission from silence in the competition rules.
- Never promote a feature because it has high univariate AUC.
- Never blend every model blindly; measure complementarity.
- Never overwrite the champion, a prior notebook, a submission, or an earlier ledger entry.
- Every new result must be appended to this ledger with date, evidence status and artifact paths.

**End of current ledger. Add future work below or in the dated log section; preserve all previous entries.**


---

### 2026-11-04 — Notebook and screenshot recovery inventory; GPU-first next phase

**Library inventory recovered**
- `GCI_v5_dropPF4_colab.ipynb`
- `GCI_v17_optimised.ipynb`
- `GCI_UMAP_Research_Notebook (1).ipynb`
- `EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb`
- `exp85_recovery.ipynb` and versioned recovery notebooks
- `exp85_solo` notebook variants
- `origin.ipynb`, `origin_exp1.ipynb`, `origin_exp2.ipynb`, `origin_exp3.ipynb`
- Multiple screenshot/chart images from 2026-10-29 through 2026-11-03.
- Research reports dated 2026-10-23 and 2026-10-24, including the complete handoff and progress reports.

**Source-derived historical evidence (not freshly re-executed)**
- v16 notebook `GCI_v16_everything_merged_o.ipynb`: recorded five-fold OOF ROC-AUC 0.77649; user-reported public leaderboard score 0.78222. These are separate metrics.
- v16 architecture includes raw/application features, missingness, employment-sentinel handling, financial and temporal ratios, EXT_SOURCE interactions, frequency and peer-group features, fixed-fee product decoding, nested target encodings, learned expected-value/residual features, multiple feature views, LightGBM-family members and CatBoost.
- The documented v16 fee decoder used 129.50. One report records 5,335 raw five-decimal annuity/credit ratios collapsing to about 1,047 decoded product keys; another audit reports 1,041/1,044 depending on scope/version. Preserve this discrepancy until source and row scope are reconciled.
- The v5 research report documents a matched seed-42 fee-decoder lift from 0.765532 to 0.768644 (+0.003113; all five folds improved), plus positive contributions from EXT expected-value/residuals, polynomial interactions, loan segments, peer-group EXT context, Extra Trees-style LightGBM, MultiFee, and CatBoost integration. These are historical report claims, not fresh reproduction.
- A fixed 50/50 rank blend of CatBoost and a six-member ensemble averaged 0.774194 across seeds 42/7/2026 in a separate runner; it was not a same-fold head-to-head comparison against v16.
- Applicant UID target history showed a small random-CV gain (+0.00039) but essentially no gain under applicant-grouped CV. Treat it as split-sensitive.
- v17 had prior static-audit concerns: pooled train/test feature statistics, residual features not isolated from outer folds, cache provenance, fixed product-decoding assumptions, and unresolved target-encoding permission.

**GPU decision**
- Do not mechanically change every model to GPU. First detect CUDA GPU, driver/runtime, free VRAM, and installed library builds.
- LightGBM GPU support depends on package build; XGBoost can use `tree_method="hist", device="cuda"` where supported; CatBoost can use `task_type="GPU"` where CUDA is available. Many ExtraTrees workflows remain CPU-bound unless replaced with a validated GPU alternative.
- Benchmark one representative fold before full execution. Record VRAM, wall time, utilization, warnings, CPU fallback, and package versions.
- If local Windows has no usable GPU, use a GPU-enabled Colab runtime. Never claim GPU execution without logs proving it.

**Decision**
- Next phase: `PHASE 4 — NOTEBOOK RECOVERY, GPU BENCHMARK & FEATURE-FAMILY REPRODUCTION`.
- Reproduce the strongest existing v16/v5/EXP85 components first under a common fold contract. Then test new domain formulas and interactions where source evidence identifies an untested mechanism.
- No new feature/model promoted. Historical 0.77649 v16 OOF and reported 0.78222 public score remain references, not proof that every branch is clean or reproducible.

**Next action**
- Inspect all notebooks and screenshot evidence; build a feature/code provenance matrix; classify leakage and rule risks; verify GPU; reproduce candidate components in priority order; append every result to this ledger. Do not overwrite notebooks, caches, experiments, champions, or submissions.

---

### 2026-11-04 — Append-only ledger verification and Phase 4 task addendum

**Ledger integrity and latest-entry check**
- Verified the requested versioned ledger exists at `D:\gci\truth\GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md`.
- The unversioned path named in the Phase 4 prompt, `D:\gci\truth\GCI_MASTER_VERIFICATION_LEDGER.md`, does not exist. The existing versioned file was used; no file was overwritten or renamed.
- The versioned ledger in `D:\gci\truth` is byte-identical to the downloaded copy in `C:\Users\SHARON\Downloads`.
- Pre-append ledger: 530 lines, 30,516 bytes, SHA-256 `257befc97aad163dd7b2fe76eaf74cfc297eb6d87a43d559620e2a2ed19d6e32`.
- Latest pre-existing dated entry: `2026-11-04 — Notebook and screenshot recovery inventory; GPU-first next phase` (began at line 496).
- All existing lines were preserved. This entry is appended after that entry; the older “End of current ledger” marker is retained as historical text.

**Phase 3D result record completed before Phase 4 training**
- This is a verification addendum to the preceding Phase 3D record; it does not replace or silently revise that entry.
- Claude `solution.py`, its seven-model definition, notebook, and OOF vectors were not found in the inspected `D:\gci\truth`, `C:\Users\SHARON\Downloads`, or `D:\exp colab` locations. Claude's reported 0.7647 therefore remains `REPORTED / UNVERIFIED`.
- Source/data: `D:\gci\inpuy\train.csv` (171,202 x 34; target rate 0.08072920), preserved fold map `D:\gci\truth\experiments\phase2_advanced_feature_engineering\FOLD_ASSIGNMENTS.csv`, and Phase 3C aligned OOF comparator `D:\gci\truth\experiments\phase3c_model_intelligence\MODEL_BENCHMARK_OOF_PREDICTIONS.csv`.
- Protocol: the same five existing folds; fold sizes 34,241 / 34,241 / 34,240 / 34,240 / 34,240. Product transforms, category handling, and supervised external-score model fitting were confined to each outer training partition. External-score model training rows were internally cross-fitted. No test labels or outside datasets were used. All findings remain `EXPLORATORY` because these folds were previously studied and multiple hypotheses were compared.
- Environment recorded by the run: Python 3.12.10, Windows 11, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0. Main run command: `python D:\gci\truth\experiments\phase3d_claude_hypothesis_probes\phase3d_followup_experiments.py`; completed successfully in 410.84 seconds. Term diagnostic and blend evaluation scripts then independently recalculated the saved OOF results.

| Candidate | Pooled OOF AUC | Delta vs. 0.762491 refit control | Fold wins | Status |
|---|---:|---:|---:|---|
| Refit Phase 3C product control | 0.762491 | +0.000000 | — | REPRODUCED |
| `round(C/A)` proxy | 0.762104 | -0.000387 | 1/5 | EXPLORATORY / negative |
| six-month-snapped proxy | 0.762199 | -0.000293 | 1/5 | EXPLORATORY / negative |
| `A*round(C/A)/C` rounded residual | 0.762983 | +0.000492 | 4/5 | EXPLORATORY |
| rounded `A*n-C` gap | 0.762527 | +0.000036 | 3/5 | EXPLORATORY / near neutral |
| six-month-snapped `A*n/C` residual | 0.763049 | +0.000557 | 3/5 | EXPLORATORY |
| six-month-snapped `A*n-C` gap | 0.762451 | -0.000040 | 2/5 | EXPLORATORY / negative |
| fixed 0.20/0.35/0.45 external summary | 0.762025 | -0.000467 | 1/5 | EXPLORATORY / negative |
| nested cross-fitted logistic external score | 0.762418 | -0.000074 | 2/5 | EXPLORATORY / negative |
| monotone `DAYS_BIRTH` increasing | 0.761803 | -0.000688 | 1/5 | EXPLORATORY / negative |
| monotone `DAYS_BIRTH` decreasing | 0.762075 | -0.000417 | 1/5 | EXPLORATORY / negative |
| monotone external scores decreasing | 0.762368 | -0.000123 | 2/5 | EXPLORATORY / negative |

Paired fold AUCs for reference / rounded residual / snapped residual:

| Fold | Reference | Rounded residual | Snapped residual |
|---:|---:|---:|---:|
| 1 | 0.758608 | 0.759520 | 0.760139 |
| 2 | 0.764193 | 0.763786 | 0.763898 |
| 3 | 0.769399 | 0.769569 | 0.769064 |
| 4 | 0.758656 | 0.758922 | 0.759159 |
| 5 | 0.761697 | 0.763240 | 0.763115 |

Additional fixed blend evidence, all adaptive/exploratory because the feature hypothesis was screened on these same exposed labels:
- Existing fixed LGB/CatBoost probability blend: 0.763957.
- Rounded-residual model + CatBoost, 50/50 probability blend: 0.764328 (+0.000371 vs. existing blend), 5/5 fold wins.
- Snapped-residual model + CatBoost, 50/50 probability blend: 0.764338 (+0.000381), 3/5 fold wins.
- Five-member equal rank blend: 0.762585. Model rank correlations range approximately 0.924–0.984; no evidence supports indiscriminately adding all models.
- Interpretation is limited to a possible tree-model quantization/sawtooth effect in `C/A`. `round(C/A)` and six-month snapping are not observed contract terms; the apparent subgroup behavior is exploratory and multiple-comparison-sensitive. No champion promotion.

**Source/code/input hashes**
- Train CSV SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Fold assignments SHA-256: `c19920148ae021e3f2070dffa7d43f2b030f2e19b29e24f0aceec21e22ebe1db`.
- Phase 3C model OOF SHA-256: `b64718bc94770ec19d00c94b4686b0da7129fd0c1364ab292f75b697701bfe57`.
- `phase3d_followup_experiments.py`: `9f895d58a9b2e3e2b258969aa91a90c1d0fe8ba95b07b59cd2384a4ee50b8745`.
- `analyze_term_pattern.py`: `4578cb8396685f1defc7c243dda328f8b308b0adf24fbd6f72d5e34610bc29e1`.
- `evaluate_term_candidate_blends.py`: `18ed3c493a3b433c1ae22968268bba6754a78a2924ecb415146777aaaed739f1`.
- `FOLLOWUP_OOF_PREDICTIONS.csv`: `96c0c027ba503808359abb a6faa53694b4c1b4f09fe2c991987c1cfb8047cded` (see exact digest in the output manifest; this line preserves the recorded digest with a visual grouping space only).
- `FOLLOWUP_SUMMARY.csv`: `171fa5685487f6dbb61c4548f5b9eb113f84cb6d8be2528948a94a60da043df6`.
- `FOLLOWUP_FOLD_RESULTS.csv`: `5d52d26ef16324da1b9ca8904592ed31fb0e38c81525a05e0d36fc84915b7634`.
- `TERM_PATTERN_ANALYSIS.md`: `db74270d2fe65c4176193ee72b63a10e4a34e412660ef6c9936cab753e64cebc`.
- `TERM_CANDIDATE_FIXED_BLEND_RESULTS.csv`: `87e4952c53dc5f0c30e40bf9a6e1180eb45a59b67278310566016ba66b617f38`.
- The follow-up output manifest had 23 entries; all output sizes and SHA-256 values were checked successfully. Full per-file hashes are preserved in `D:\gci\truth\experiments\phase3d_claude_hypothesis_probes\OUTPUT_MANIFEST.csv`.
- Artifacts: `D:\gci\truth\experiments\phase3d_claude_hypothesis_probes\FOLLOWUP_REPORT.md`, `TERM_PATTERN_ANALYSIS.md`, `FOLLOWUP_SUMMARY.csv`, `FOLLOWUP_FOLD_RESULTS.csv`, `FOLLOWUP_OOF_PREDICTIONS.csv`, `TERM_CANDIDATE_FIXED_BLEND_RESULTS.csv`, `CLAUDE_FORMULA_MATHEMATICAL_AUDIT.csv`, `OUTPUT_MANIFEST.csv`, and `EXECUTION_LOG.txt`.

**Correction / integrity note**
- The OOF output hash is recorded authoritatively in `OUTPUT_MANIFEST.csv`; the inline digest above is transcribed from the PowerShell console and must be treated as malformed if it does not exactly match the manifest. Do not use the inline value as a checksum. The output manifest remains the integrity source of truth.

**Phase 4 task accepted as an append-only execution contract**
- Target: investigate a controlled, competition-compliant path to 0.77000 pooled OOF AUC; no promise of reaching it.
- New workspace: `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery`.
- Preserve this versioned ledger and append after each stage. The canonical unversioned ledger path stated in the task is currently absent; do not create or overwrite it without an explicit decision. Continue using this verified versioned ledger.
- Stage order: (1) inventory all local notebooks, scripts, reports, caches, predictions, screenshots and provenance; (2) extract screenshot claims without inventing unreadable content; (3) build feature provenance/reuse matrix; (4) audit GPU, drivers, memory, framework builds, representative compatible model execution and fallback; (5) reproduce prioritized v16/V5/EXP85/V17 components individually; (6) enforce outer-fold isolation and inner-only stopping/selection; (7) test only genuinely new data-derived features with single/family ablations and negative controls; (8) compare models and fixed/nested blends without tuning weights on reported OOF labels; (9) create the required reports, matrices, OOF/results, logs and hash manifests; (10) protect all prior notebooks, inputs, experiments, submissions and champions.
- Required final deliverables remain those listed in the user-provided Phase 4 task: execution report; notebook and screenshot inventories/OCR; feature provenance and reuse matrices; GPU audit and benchmark; model/feature/fold results; OOF predictions; prediction correlations; leakage/rule audit; input/output manifests; execution log; next-phase recommendation.
- Full training is gated on completion of source inventory and a representative GPU benchmark. Existing fixed folds are exposed; any result on them must be labeled `EXPLORATORY`, not independent confirmation.
- Claude implementation and exact OOF remain blocked pending the actual source and vectors. V5 exact score reproduction remains blocked pending its old cache and OOF predictions.

**Phase 4 initial GPU detection (audit only; no model benchmark run)**
- Windows 11; Python 3.12.10; LightGBM 4.7.0; XGBoost 3.4.1; CatBoost 1.2.10; scikit-learn 1.9.1; PyTorch 2.15.0.
- `nvidia-smi` reports NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, 6,141 MiB total and 5,710 MiB free at inspection. This is a point-in-time device reading, not a Phase 4 utilization measurement.
- PyTorch reports `torch.cuda.is_available() == False`, no CUDA build, and zero devices; `nvcc` is absent.
- XGBoost build metadata reports `USE_CUDA=True`, CUDA version `[13, 3]`; GPU fitting is not yet demonstrated.
- LightGBM CUDA/GPU capability and CatBoost GPU fit capability remain unverified. No GPU benchmark or training has been claimed.
- Next: complete source inventory and GPU build/representative-fit checks; if GPU execution fails, report it explicitly and do not silently launch an expensive CPU fallback.

**Next action**
- Start Phase 4 Stage 1 inventory and complete Stage 4 capability checks in the new workspace. Do not start full model training before both gates are documented. Request Claude source/OOF and any missing V5 cache artifacts when the inventory confirms they are absent.

#### 2026-11-04 — Checksum transcription correction (append-only)

- The inline `FOLLOWUP_OOF_PREDICTIONS.csv` digest in the immediately preceding entry contains a malformed transcription and is not a valid SHA-256 string. Do not use it.
- Correct SHA-256, copied from the verified `OUTPUT_MANIFEST.csv`: `96c0c027ba503808359abba6faa53694b4c1b4f09fe2c991987c1cfb8047cded`.
- This correction is appended; no historical line was deleted or altered.

#### 2026-11-04 — Phase 4 inventory refresh and candidate-source audit (append-only)

**Ledger status**
- Before this work, checked the versioned ledger's latest entry: the append-only checksum transcription correction immediately above. The requested unversioned `GCI_MASTER_VERIFICATION_LEDGER.md` remains absent; the existing versioned ledger remains authoritative.
- The earlier statement that Claude's source had not been found is retained as a historical statement about the earlier scoped search. This entry corrects its current status without deleting it.

**Inventory refresh**
- Command: `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\phase4_inventory_gpu_audit.py`
- Result: success; 223 inventoried files, 15 notebooks, 14 images, 0 notebook parse failures. No notebook was executed and no model was trained.
- The inventory found `C:\Users\SHARON\Downloads\solution_v5_gci_research.py` (140,761 bytes; SHA-256 `4f0f1e9208f99ef2350af05f9c3da98eaae922de729295f2e0c410186924ae74`). The inventory did not find associated OOF vectors, checkpoints, or a run manifest.
- This is a related candidate source, not verified evidence of Claude's seven-model implementation: its registry contains 20 model names and the file does not establish author/source identity. The 0.7647 claim remains `REPORTED / UNVERIFIED`.

**Static audit finding**
- Created `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\CANDIDATE_SOLUTION_STATIC_AUDIT.md`; candidate source was not run.
- Material validation defect: `run_model()` supplies the outer validation labels to LightGBM/XGBoost/CatBoost early stopping and then reports AUC on those same rows. Thus the saved fold scores are not independent outer-fold estimates.
- Material transform-scope concern: `build()` concatenates train and test before winsorization, external-score imputation, frequency/group statistics, percentiles and product features. This is label-free transductive processing, but still violates strict outer-fold isolation; competition-rule permission is unresolved.
- Target encoding does perform inner training-row cross-fitting, and outer-validation/test target encodings use only outer-training labels. This favorable detail does not repair the other issues.
- Blend `auc_oof` is evaluated on the same OOF rows used to fit blend rules; its secondary row-level CV is not fully nested with the base learners. These blend scores are not suitable as independent confirmation.
- Cache identity does not include dataset-content or fold hashes. The script suppresses warnings and is not configured to use the detected GPU. Implied term/interest features remain assumptions, not observed contract facts.
- Existing source comments and accepted-feature flags are not execution evidence. No reported score was recalculated.

**GPU snapshot update (detection only)**
- At inventory refresh: NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, 6,141 MiB total / 5,835 MiB free, 0% utilization. PyTorch 2.15.0 reports CUDA unavailable; XGBoost build metadata says CUDA is compiled in, but this candidate config uses CPU-only estimator settings. No model-specific GPU fit has been performed. The reading is an instantaneous snapshot, not a benchmark.

**Disposition**
- Phase 4 notebook/source audit is still in progress. Do not compare this candidate's logged/comments-only claims with the validated pipeline, and do not start full training until notebook/source provenance, allowed feature scope, fold-safe iteration selection, and GPU execution behavior are verified.
- Next: inspect priority notebook sources and screenshot evidence, reconcile feature provenance, and only then run bounded GPU capability checks. Append new evidence; retain all prior history and artifacts.

#### 2026-11-04 — Phase 4 priority notebook and screenshot triage (append-only)

**Ledger status**
- Verified the latest ledger entry before this stage: `Phase 4 inventory refresh and candidate-source audit` immediately above. The versioned ledger remains the active append-only record.

**Priority notebook provenance**
- Created `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\PRIORITY_NOTEBOOK_AUDIT.md`.
- Hashed and inspected v5, v16, v17, and the preserved EXP85 baseline. All inspected notebook code cells have null execution counts. v5/v16/v17 contain saved text outputs; EXP85 contains no saved outputs. Those saved outputs are not bound to exact runtime/input manifests and were not reproduced.
- EXP85 copies in Downloads, the preserved baseline, and the working-copy directory are byte-identical: 6,706,327 bytes; SHA-256 `ab73f6f39a0fee803fc946de5bd64a4219f9b59a35663c8510b9c002570069f0`. None was modified.
- v5 output records model AUCs 0.77047-0.77070 and old/new 50:50 blend AUC 0.77213. This is a saved exploratory OOF claim, not a fresh run or public score.
- v16/v17 saved outputs record fixed-formula OOF claims 0.77649 and 0.77646; v17's optional stacker reports 0.77657 (+0.00010), below its own +0.00020 acceptance threshold. These values remain unverified and exposed to repeated selection on the same folds.
- v5/v16/v17 pool train and test covariates for several learned transforms. V16/v17 also use fixed 129.5 and 0.0132 product hypotheses, and globally constructed applicant fingerprints. Nested target encoding is present; v16/v17 main LightGBM folds use fixed iteration counts rather than outer-fold early stopping. The optional blends are not fully nested with the original base OOF model generation.
- Existing official-rule audit says test-covariate fitting and target-encoding policy are unresolved; no model training was authorized or started.

**Screenshot triage**
- Created `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\SCREENSHOT_FORMULA_TRIAGE.md` and updated `SCREENSHOT_INVENTORY.csv`.
- Visually inspected all 14 inventoried images: 10 formula/research screenshots, 4 unrelated images; `download.jpg` and `wallpaper.jpg` have identical SHA-256. `tesseract` is not installed, so formula text is marked as a manual visual transcription, not automated OCR.
- Transcribed candidate ideas include bureau-window acceleration, repayment-ratio-by-EXT interactions, down-payment proxy, LTV/PTI/external-score composites, external-score dispersion, an annuity/goods payment stress ratio, DSCR proxy, risk-adjusted LTV, and exact/approximate implied-rate curves.
- Screenshot AUC claims include about 0.568, 0.648, and 0.67067; the last is explicitly described as being on training data. Evaluation method, score direction, and incremental model effect are absent or unreadable. None is accepted as evidence or added to a model.
- The exact amortization equation is valid only after fixing term, rate periodicity, and fee treatment; credit and annuity alone do not identify both rate and term. `Income/(12*Annuity)` is a scaled inverse of PTI under annual-income/monthly-payment assumptions, not an independent signal by itself.

**Current scientific interpretation**
- A discrete product/contract structure around annuity-to-credit ratios and a possible additive periodic fee is the strongest candidate hidden pattern in these artifacts. It is not confirmed: its constants and cluster frequencies were pooled across train/test, there is no independent OOF reproduction, and organizer policy remains unresolved.
- No score improvement is claimed from this stage. No notebook, source pipeline, champion, submission, or previous experiment was changed.
- Next: complete the feature/source reuse matrix and review the supplied data dictionary and official-input provenance. Then run only bounded GPU capability probes after establishing safe fold/data scope; do not begin full model reproduction until all source and rule gates are recorded.

#### 2026-11-04 — Phase 4 input provenance and dictionary audit (append-only)

**Ledger status**
- Before this stage, verified the latest entry was `Phase 4 priority notebook and screenshot triage` immediately above. The versioned ledger remains authoritative; no historical entry was deleted or rewritten.

**Reproducible audit**
- Added and executed `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\audit_phase4_inputs.py`.
- Exact command: `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\audit_phase4_inputs.py`
- Execution: PASS at `2026-11-04T03:59:24.781640+00:00`; no errors or warnings were printed. No model training, feature generation, or AUC evaluation was run.
- Source SHA-256: `99fe96d979e439129cf0f0961f1bba4d86f1a06e251b7b5cebaa6575cac5060b`.
- Verified dimensions: train 171,202 x 34; test 61,500 x 33; sample submission 61,500 x 2. Train positive rate: 0.0807291971. TARGET is absent from test; IDs are unique within each file; test and sample-submission IDs/order match; train predictors and test columns match after excluding TARGET.
- Project input copies match the competition-package copies by SHA-256:
  - `train.csv`: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
  - `test.csv`: `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfcd525f7c08de15c6d2d42c65`
  - `sample_submission.csv`: `6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db`

**Dictionary implications and formula correction**
- The local dictionary defines `AMT_ANNUITY` only as “Loan annuity”; it supplies no payment frequency. Annualizing it is not supported by this dictionary.
- `AMT_GOODS_PRICE` is described as the price of goods “for consumer loans”; credit/goods must not be called a universal collateral LTV without verifying product coverage and meaning.
- The local `AMT_REQ_CREDIT_BUREAU_YEAR` definition literally reads “one day year (excluding last 3 months before application).” This ambiguity is retained verbatim; it is not silently repaired or treated as a confirmed 9-month interval. The screenshot formula dividing YEAR by four is unsubstantiated. Bureau-window normalization/acceleration features remain blocked pending exact interval definitions.
- Updated `SCREENSHOT_FORMULA_TRIAGE.md` to reflect these data-dictionary constraints. This is a correction to formula interpretation, not a feature result.

**Artifacts**
- `DATA_DICTIONARY_AND_INPUT_AUDIT.md` SHA-256: `03a949b2a59a1678f5ffbe33444775bf499333754b05ccd6038c7cda12c1670f`.
- `INPUT_HASH_MANIFEST.csv` SHA-256: `2f2ac87ab68e9e35d570027801d1e3a88bbfb0bb663fd65909f88e89ea9a8c6f`.
- `INPUT_AUDIT_LOG.txt` SHA-256: `3739b3bde5973c3cde1d39c9034c905eb4bcf0b1c82fa4c54b29804e71cac09f`.
- Updated `SCREENSHOT_FORMULA_TRIAGE.md` SHA-256: `758ce02fa2ec1f85cb23ea33fbd97d5eed097d34cb90a53aea127bb71c2b551a`.
- All four output/report files were verified to exist and be non-empty. Full input paths and file sizes are preserved in `INPUT_HASH_MANIFEST.csv`.

**Disposition**
- Input provenance and structural integrity are now confirmed for the local copies; no model claim is validated by this audit.
- Remaining blockers include organizer clarification on pooled unlabeled train/test fitting and target-encoding policy, confirmation of bureau interval semantics, and bounded verification of GPU estimator execution. Continue source/reuse review and rule clarification before comparative training.

#### 2026-11-04 — Phase 4 notebook reuse matrix and EXP02 artifact recheck (append-only)

**Ledger status**
- Before this stage, verified the latest entry was `Phase 4 input provenance and dictionary audit` immediately above. This entry adds evidence; it does not revise or remove prior history.

**Notebook/source reuse audit**
- Created `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\PHASE4_NOTEBOOK_REUSE_MATRIX.md`, covering candidate notebooks, duplicated copies, formula-family reuse, validation evidence, score claims, and dispositions.
- The matrix records the following as unverified saved-output claims: v5 blend 0.77213; v16 0.77649; v17 0.77646 and optional stacker 0.77657; v18 ExtraTrees 0.76758 with 10%/15% additions near 0.00000/-0.00006 and 2/5 positive folds; v19 research candidate 0.77649; and v19 NN 0.76732 standalone with +0.00016/+0.00017 mean blend deltas. None is a verified public score.
- v18/v19 research notebooks depend on unmanifested caches and pooled transforms. The v19 research notebook's current code writes `GCI_submission_v19_candidate.csv`, whereas its adjacent saved output says `GCI_submission_v16.csv`; its displayed results are not attributable to the current source.
- The v19 NN Downloads and D: copies have identical cell source and saved outputs, but different execution-count metadata. Its inner-split early stopping is separate from outer validation, but global quantile/residual preprocessing breaks strict outer-fold isolation; no input/cache run manifest is supplied.
- The UMAP notebook has no execution outputs. Its target encoding is generated across all training rows rather than nested within model outer folds; supervised UMAP is fit on training labels before outer CV. Its resulting OOF evidence would be contaminated.
- The complete feature/source reuse and dispositions are in the matrix; no candidate notebook, EXP85 copy, champion, or submission was modified or executed.

**EXP02 existing-run metric verification**
- Added persistent verifier `D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\verify_exp02_artifacts.py` and ran:
  `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\verify_exp02_artifacts.py`
- Execution: PASS at `2026-11-04T04:10:31.170153+00:00`. Training was not run; existing OOF arrays were checked against raw input hashes, notebook canonical source hash, all root/run artifact hashes, test/sample ID order, prediction dimensions/ranges, and stored results.
- Independent recomputation: pooled raw CatBoost AUC `0.751264908883`; financial-feature block `0.758493753314`; delta `+0.007228844431`. Mean paired fold delta `+0.007284542451`; fold deltas `+0.009673465665`, `+0.007546517258`, `+0.005198158007`, `+0.006561272467`, `+0.007443298861` (5/5 positive).
- This verifies the saved metric calculation, not a second training run, independent seed replication, champion lift, causal mechanism, or public leaderboard score. The source run's recorded status remains `NEEDS_INDEPENDENT_SEED_REPLICATION`.
- A first ad hoc metric-check command printed the recomputed AUCs but then exited with `KeyError: 'raw'` because it assumed the wrong JSON key. The persistent verifier uses the actual `raw_baseline` / `financial_features` keys and passed; no prediction or metric was changed.

**Artifacts and hashes**
- `PHASE4_NOTEBOOK_REUSE_MATRIX.md` SHA-256: `9aebe7ed9307c9e7af0d2f196a66648c2eb8f263e4e5dbb4e00dc7c91b85b67c`.
- `verify_exp02_artifacts.py` SHA-256: `a3fff8aca7c90723f36262758c76d41c05675cd4a7bbd74a0412bf95a587f143`.
- `EXP02_ARTIFACT_RECHECK_LOG.txt` SHA-256: `4cec0f5b613e518753c70c050a1c86bbe53e152eabf25540d97c5d32b4b5bafc`.
- EXP02 notebook file SHA-256: `b3882dc3ed7a8b6a3370a2f65da7232e91ee7f0d65baf9da3362810f6d13cd7c`; its canonical source SHA-256 in the run manifest is `dd4bcb77f05ee1820118d03e0dd965fd451e928c0fe45fb8a8847ebfae53bfa6`.

**Disposition**
- EXP02 is the most substantiated financial-feature result, but only for its matched raw-feature CatBoost setup and combined seven-feature block. The sizable gain does not imply the same gain on the current champion or identify an individual causal formula.
- Next: if advancing this block, run the predeclared frozen new-seed replication, then a matched champion ablation; preserve separate output paths and do not compare against unverified notebook claims. Continue to hold globally pooled features and UMAP target-guided features pending fold-scope/rule resolution.

#### 2026-11-04 — EXP02 independent-seed replication preregistration (append-only; pre-run)

**Ledger status**
- Verified the latest completed entry before preparation: `Phase 4 notebook reuse matrix and EXP02 artifact recheck` immediately above. The existing EXP02 seed-42 results and all prior experiments are recorded; no historical entry was altered.

**Frozen protocol**
- New isolated experiment directory: `D:\gci\truth\experiments\exp02_financial_behaviour_replication_seed20261105`.
- Predeclared outcome: replicate the seven rowwise ratio block against the matched raw-feature CatBoost baseline. This is not a champion comparison and cannot establish individual formula attribution.
- Outer folds: five stratified folds, shuffle enabled, seed `20261105`.
- Inner stopping seed base: `20262105` (+ 1-based fold number). Model seed base: `20263105` (+ 1-based fold number).
- CatBoost parameters, feature definitions, preprocessing, inner-only early stopping, refit policy, and acceptance threshold are copied unchanged from EXP02. Pass requires positive pooled delta, mean paired fold delta >= `+0.0002`, and positive improvement in at least 4/5 folds.
- Same three input hashes as the preceding EXP02 audit. Only rowwise formulas are used; no pooled train/test fit, feature selection, new constants, leaderboard feedback, or submission output is permitted.
- The original EXP02 notebook, reports, arrays, champion, and submission remain untouched.

**Pre-run environment and artifacts**
- A synthetic CatBoost GPU smoke test passed on the NVIDIA GeForce RTX 4050 Laptop GPU with CatBoost `1.2.10`; it used 512 synthetic rows and 8 iterations, not competition data. GPU snapshot before replication: 5,826 MiB free, 0% utilization.
- `jupyter`/`nbclient` are unavailable locally. The source notebook has no Jupyter magics, `get_ipython`, `display`, or exit calls; the prepared runner executes its seven Python code cells sequentially in one namespace, captures outputs, and saves an executed notebook. No notebook-specific display behavior is used by these cells.
- `run_exp02_seed_replication.py --prepare-only` completed at `2026-11-04T04:16:00.409317+00:00`: status `PREPARED`, training `NOT_RUN`. Only setup paths/ID/seeds differ from the original code cell; all prior cell outputs were cleared. The pre-run manifest confirms all seed values.
- Runner SHA-256: `f90aff036e7458bc0d4b175942508ee0dc96d9255f974b8f7c355f7c45efc5b6`.
- Protocol SHA-256: `6f8cd06020e574bdf3278bb60a118451cac7e2eee38c59f855cd5eb987ec2313`.
- Prepared notebook SHA-256: `539b7ac2e75c668439ad286eae2aeae38911e0cbe21ebeba1466038f2013a460`; canonical cell-source SHA-256: `d4f7c502d88e90f71b5875a46d3a1767b5c73083eec1ecee67b5f469b9cec9a3`.
- Clone manifest SHA-256: `fe4beb6b8f1d9b1d98d9e99419921178267a74f65b06d8ee481d77594aa38406`.

**Run gate**
- No competition-model training has begun in this replication. Before execution, recheck the prepared notebook/protocol hashes, exact inputs, and empty output paths. Record command output, fold metrics, artifact hashes, and any error before considering the replication complete.

#### 2026-11-04 — EXP02 replication pre-run hash correction (append-only)

- Preflight stopped before training because the recorded `test.csv` SHA-256 did not match the file. Investigation confirmed the prior preflight literal and the initial protocol entry had a transcription error (one extra `c`); no dataset bytes changed.
- Verified `D:\gci\inpuy\test.csv` and the official-package copy are both 15,818,118 bytes and hash to `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. This matches the already-existing EXP02 `SHA256_MANIFEST.json` and the Phase 4 `INPUT_HASH_MANIFEST.csv`. Train and sample-submission hashes remain as previously verified.
- Appended a dated correction to `EXPERIMENT_PROTOCOL.md`, retaining its original line and documenting that only the transcription was corrected before training; model, features, folds, and acceptance rule are unchanged.
- Updated the prepared clone manifest with the corrected current protocol hash while retaining the initial protocol hash and an explicit correction note. No predictions or model outputs exist.
- Corrected protocol SHA-256: `c33c8006bf3628f80484c2ec0c520ba65ca9559177639851b74480464b908bb6`; initial protocol SHA-256 retained: `6f8cd06020e574bdf3278bb60a118451cac7e2eee38c59f855cd5eb987ec2313`.
- Updated clone-manifest SHA-256: `bc4c79a521ad87f463686d3b155b6fa91cbfb034515f6394d4cd8f4c25d37693`. Prepared notebook SHA-256 remains `539b7ac2e75c668439ad286eae2aeae38911e0cbe21ebeba1466038f2013a460`.
- Correction recorded at `2026-11-04T04:18:08.4343642Z`. Next gate: rerun all input/artifact checks against the corrected hash, confirm all output locations are absent, then and only then launch training.
- Corrected preflight passed at `2026-11-04` after the correction: all six project/official input files matched the Phase 4 manifest, project inputs also matched the original EXP02 run manifest, source notebook/protocol/prepared clone hashes matched, all expected result paths were absent, and clone status remained `PREPARED` with training not started.

#### 2026-11-04 — EXP02 independent-seed replication executed and independently verified

**Execution evidence**
- Exact command: `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\run_exp02_seed_replication.py`
- Execution status: runner `PASS`; start `2026-11-04T04:18:35.651195+00:00`, completion `2026-11-04T04:30:43.198365+00:00`; model training runtime `714.6` seconds.
- Environment: Python `3.12.10`, NumPy `2.5.4`, pandas `3.1.0`, scikit-learn `1.9.1`, CatBoost `1.2.10`, NVIDIA GeForce RTX 4050 Laptop GPU. Training sampled GPU telemetry 668 times.
- Data: train `171,202 x 34`, test `61,500 x 33`, target positive rate `0.0807291971`. All train/test/sample hashes matched the package and prior manifest.
- Frozen seeds actually used by executed code: outer `20261105`; inner stopping `20262106`–`20262110`; model `20263106`–`20263110`. The code passes these constants into the inner splitters and CatBoost `random_seed`; the saved run metadata includes stale static seed text, documented below.

**Independently recomputed results**
- Pooled 5-fold OOF AUC: raw-feature baseline `0.751707723784`; seven-feature financial block `0.758043865380`; paired pooled delta `+0.006336141595`.
- Mean paired fold delta `+0.006296821542`; sample SD of fold deltas `0.001138139530`; all `5/5` folds improved. The preregistered threshold passed (positive pooled delta, mean fold delta >= `+0.0002`, at least 4/5 positive folds).
- Fold results (baseline -> feature block; delta):
  1. `0.749795120660 -> 0.756709536888; +0.006914416228`
  2. `0.754456830050 -> 0.760877485921; +0.006420655871`
  3. `0.755614786053 -> 0.760142786299; +0.004528000246`
  4. `0.750401777414 -> 0.756461231850; +0.006059454437`
  5. `0.748664408635 -> 0.756225989562; +0.007561580927`
- Relative to the earlier independently rechecked seed-42 run (pooled delta `+0.007228844431`; mean fold delta `+0.007284542451`), this independent seed also yields a positive block-level effect. Fold sets differ, so these deltas are not paired against each other.
- Interpretation: evidence supports a repeatable joint predictive signal among the seven ratios against this matched raw-feature CatBoost baseline. It does not show which individual feature matters, prove a causal/financial mechanism, exceed the champion, or establish a leaderboard lift. An identical-fold matched champion ablation remains required.

**Verification and artifacts**
- Ran `python D:\gci\truth\experiments\exp02_financial_behaviour_replication_seed20261105\verify_exp02_seed_replication.py`; independent verification `PASS`. It checked input/source/result/array/log hashes, regenerated the stratified folds, recomputed pooled/fold AUCs, checked ID order and prediction bounds, and enforced the predeclared screen.
- Preserved primary outputs: `EXECUTION_LOG.txt`, executed `EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb`, `RESULTS.json`, `FOLD_METRICS.csv`, `FEATURE_MANIFEST.json`, `SHA256_MANIFEST.json`, run-level OOF arrays/fold IDs/test predictions, `REPLICATION_AUDIT_REPORT.md`, and `REPLICATION_AUDIT_RESULTS.json`. No submission CSV was created; champion and original EXP02 artifacts were not modified.
- Independent audit report SHA-256: `3cac53c820e656dc509cd2b22b17f84a23534b2e23f13ba0a367ec70fab3b1a2`.
- Machine-readable independent results SHA-256: `4c41aafca83dfc0f45ce50faa36e3fdbf05ffeaba4d7b15c45ac64e2e761991b`.
- Verifier SHA-256: `891c98c020a0943424eb06fb07b89359a2148c8a879fdc494b8e39a4aa5e673f`.
- Execution log SHA-256: `ff8de281df406d7fc85859b3d6f1637b95519784776d5958c9d9533368157ea5`.
- Executed notebook SHA-256: `a68ed9771d35d0e3f3c221c512edd8c0cc5352d0ea18b5722b0d80690e156523`.
- Final clone-manifest SHA-256: `66aea6df5c9b2dfff2dbbfd02b25d6fd87ffb54b5be076999cd30c7a3f0a84cf`.

**Corrections and warnings**
- Preserved generated EXP02 outputs are not fully self-consistent: `EXP02_REPORT.md` still says outer seed 42, `RESULTS.json` still says independent replication is incomplete, and `run_manifest.json`/notebook output retain old inner/model seed-rule strings. Executed source uses the corrected preregistered constants. The supplemental independent audit is the authoritative corrected record; no generated source output was silently rewritten.
- Execution logged four pandas `DataFrame.reset_index(inplace=...)` deprecation warnings and CatBoost GPU AUC metric-period notices. Training completed and the independent metric/hash checks passed.
- Semantic caveats remain: `AMT_ANNUITY` frequency is unspecified in the available dictionary, and per-family-member ratios assume equal allocation. The block result is empirical, not a financial identity.

**Next status**
- EXP02 financial-ratio block: `INDEPENDENT_SEED_REPLICATION_PASS` for the matched raw-feature CatBoost setup only.
- Next experiment: freeze a matched, identical-fold champion ablation before training. Do not promote the ratio block to champion based on this replication alone.

#### 2026-11-04 — Selecting the next matched benchmark for ratio ablation (inspection only)

- Inspected the preserved Phase 3C model-intelligence pipeline and reports after recording the seed replication. `phase3c_initial_benchmark.py` SHA-256: `b560a8b593fc0d7dbe770fbf548a1b3fdbb5e9490f652ac098aab92ea9c701b2`; its raw baseline source is `run_independent_baseline.py` SHA-256 `c49aab451d1500e1bd7e0d618a764efeb1815c172f74e1bc5174e38a793ff3fe`; product family source `phase3b_ablation.py` SHA-256 `1470a412ab42aa6015f94af01859615a96fb127b7fc124d045acf8c9a81cf8ae`; shared product logic `run_phase3_hidden_pattern_discovery.py` SHA-256 `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`.
- Confirmed Phase 3C's best fixed-weight benchmark is the equal 50:50 probability blend of `lightgbm__raw_plus_product` and `catboost__raw_plus_product` (reported OOF AUC `0.763957` on seed-42 folds). Phase 3D's `0.764338` blend is explicitly adaptive discovery on those reused folds; it is not the safer primary benchmark for the next feature test.
- The Phase 3C runner selects rounds on inner splits using only raw features, then reuses those rounds across feature variants; product parameters are fit on each outer-training partition. Seven EXP02 ratios are not direct columns in the preserved raw baseline/product feature list. Existing Phase 3C OOF output provides a reference, but not OOF values for the proposed new feature block.
- The actual `D:\gci\champion` directory contains the preserved EXP85 artifact; its prior audit reports leakage concerns and does not provide a clean matched OOF comparator for this ratio block. Therefore the next test will be named a **Phase 3C fixed-blend benchmark ablation**, not a champion promotion or official champion score.
- Planned, not yet trained: add the same seven deterministic EXP02 ratios to both Phase 3C raw-plus-product component models, compare against matched baseline components/blend using identical fresh stratified folds, fixed 50:50 blend, inner-only round selection, and no submission. This new random split remains exploratory because all training labels have already been examined; it does not erase prior researcher exposure.
- This was source/report inspection and experiment selection only. No Phase 3C or champion files were modified, and no training was started.

#### 2026-11-04 — Correction: EXP02 ratios already exist in the Phase 3C baseline

- Deeper source inspection of `run_independent_baseline.py` found that its `build_rowwise_features` already creates `CREDIT_TO_INCOME`, `ANNUITY_TO_INCOME`, `CREDIT_TO_GOODS`, `CREDIT_TERM`, `INCOME_PER_FAMILY`, `CREDIT_PER_FAMILY`, and `ANNUITY_PER_FAMILY`. Phase 3C calls this builder for its `raw_plus_product` LightGBM and CatBoost models.
- Therefore the previous inspection note that the seven ratios were “not direct columns” was incomplete and is superseded. Six of seven have exact mathematical counterparts for ordinary positive denominators; `FIN_GOODS_TO_CREDIT` is the reciprocal of existing `CREDIT_TO_GOODS` where goods price is observed and positive.
- Missing-value policies differ: the baseline fills missing goods price with credit before `CREDIT_TO_GOODS`, while EXP02 yields missing for `GOODS_PRICE / CREDIT`; the baseline fills missing family count with 1, while EXP02 maps non-positive/missing denominators to missing. These are representation/missingness differences, not new source information.
- Since this model would already contain the same ratio signal, adding all seven again to Phase 3C is redundant and would not answer whether the formulas are new. The planned model-training ablation is paused before execution; no Phase 3C or champion files were touched.
- Next: run a lightweight, label-free alias/missingness audit to quantify exact equality, reciprocal equivalence, and edge-case differences on the supplied data. Only consider a model experiment afterward if a specific, justified representation change remains.

#### 2026-11-04 — EXP02/Phase 3C feature-alias audit completed

- Ran `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\audit_exp02_phase3c_ratio_aliases.py`. Status: `PASS`; 171,202 training rows; `TARGET` was removed before feature generation; no model training or AUC calculation.
- The six exact aliases (`CREDIT_TO_INCOME`, `ANNUITY_TO_INCOME`, `CREDIT_TERM`, and the three per-family ratios) match the corresponding EXP02 formulas on all 171,202 rows where both definitions are finite. The EXP02 per-family variants are missing on 2 rows because family size is missing; the Phase 3C builder fills missing family size with 1.
- `FIN_GOODS_TO_CREDIT` is mathematically reciprocal to `CREDIT_TO_GOODS` on all 171,202 rows (maximum product error `2.22e-16`; Spearman `-1.0`). No training row has missing `AMT_GOODS_PRICE`, so the baseline's goods-price fallback was not exercised.
- This confirms the EXP02 block is not novel relative to the Phase 3C rowwise feature builder. The seed-20261105 CatBoost experiment's `+0.00633614` lift is valid against its raw-only baseline, but cannot be carried over as expected incremental lift over Phase 3C, where the ratio concepts already exist. Phase 3C's reported 0.763957 fixed blend already uses those concepts.
- No extra ratio-addition model run was performed because it would duplicate existing features and answer no new financial hypothesis. The actual EXP85 submission/champion and all prior experiment artifacts remain untouched.
- Artifacts:
  - `audit_exp02_phase3c_ratio_aliases.py` SHA-256: `28c5e7544b1d3c00f97b1bad5d52ab7d3d107457625e6016553247101addaf0d`.
  - `EXP02_PHASE3C_RATIO_ALIAS_AUDIT.csv` SHA-256: `ada620d376d18e5d15c72c6f665acebef024ec761043fe312ffd935e6c09da1c`.
  - `EXP02_PHASE3C_RATIO_ALIAS_AUDIT.md` SHA-256: `347e5db94315bab1b06fc110673c45f41959f79bf06c8a666e1b1f58277463e5`.
  - `EXP02_PHASE3C_RATIO_ALIAS_AUDIT_LOG.txt` SHA-256: `295d6b967c21ed78a3088bbb21fe613fa0d922e603e9feb587ae06e82905d083`.
- Next: focus formula discovery on dimensions not already encoded by the Phase 3C baseline; audit all proposed features against the actual rowwise builder before training, then use a predeclared fold-safe ablation for any genuinely new representation.

#### 2026-11-04 - Phase 5 V16 and EXP85 source audit; champion comparison blocked

- Verified this ledger's latest entry before work: the EXP02/Phase 3C feature-alias audit immediately above. This entry is appended; prior history is preserved.
- Audited the executed V16 notebook `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb` and source-only copy `D:\exp colab\GCI_v16_everything_merged.ipynb`. Their canonical code-cell source SHA-256 is `0d6a0cc7be62873b66a759ffcd05d3d64e0b8e6b4dc8550990dbd829f9d09ef1`; executed notebook file SHA-256 is `ca77c6a6dbf6f5d1ea80b4f0f23de1f0f0c73ea7b4ac6fab937fd32ed8881306`, with 11/11 code cells marked executed. The source-only notebook file SHA-256 is `a1b3aec64c6730d58ad2d22f1c3a49dd0d3a9b66f05ab22054514add61f9dec3`.
- Independently verified raw inputs: train 171,202 x 34, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test 61,500 x 33, SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- V16 source implements pooled train+test frequency/group/product/entity/residual transformations, loan-product keys using hypotheses `129.5` and `0.0132`, nested target encodings, four LightGBM sets, a CatBoost product model, and a fixed rank blend. Source inspection confirms TARGET is dropped from the shared feature frame; target labels are later used in nested supervised encodings. Pooled covariates are not fold-isolated, and local rules do not resolve permission for pooled fitting or target encoding.
- Saved notebook outputs claim final V16 AUC `0.77649`, block-ablation AUCs `0.76125` baseline, `0.77185` product, `0.77206` community, `0.77307` community+partner, target-permutation AUC `0.5014`, adversarial train/test AUC `0.6856`, and regime blend `0.77648` vs fixed `0.77649`. These are **saved-output claims only**, not reproduced; V16 caches and provenance manifest were not found in audited local paths.
- Verified the frozen EXP85 submission `D:\gci\champion\submission_exp85_solo_champion.csv` is byte-identical to `00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`; both SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`. The available runner composes a test-only Anchor74 and eight-engine test suite; it does not create an OOF vector for that final composition. V17/V55/XGB component OOF arrays are not an exact final champion OOF. The expected Anchor74 OOF array was not present at the audited run path. The local log's official `0.80041` record is associated with a different submission hash (`8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`), so no score is transferred to the `78c31...` artifact.
- Pre-run gate: model training and CV comparison were **not run** because the exact EXP85 final OOF vector and fold map are missing; substituting Phase 2/3C or component-level OOF would change the requested comparator. No submission/champion file or notebook was modified.
- Execution: `python -B D:\gci\truth\experiments\phase5_champion_reproduction\audit_phase5_static.py`; corrected static audit completed at `2026-11-04T05:03:32.491745+00:00`, exit code 0. The first audit log is preserved at the experiment root; a second, non-overwriting run corrected the V55 path and recorded the bundle as present.
- Corrected run output directory: `D:\gci\truth\experiments\phase5_champion_reproduction\runs\static_audit_20261104_0501\`. Its `EXECUTION_LOG.txt` records the actual command/results; all 16 run outputs were verified non-empty. Static audit script SHA-256: `9d2ba8086179894a8fab372c6cf27b8c551530bea61a4b69958305e8a47d1c86`.
- Next gate: recover the exact EXP85 final OOF plus fold assignments and Anchor74 out-of-fold counterpart; recover V16 caches with provenance; obtain written organizer clarification for pooled covariates and test-to-test coupling. Do not claim a V16 improvement against EXP85 until these are resolved.

#### 2026-11-04 - Phase 5 recovery addendum: candidate caches found, exact comparison still blocked

- Verified the latest ledger entry before this addendum: the Phase 5 V16/EXP85 source audit immediately above. This entry corrects and extends that audit without rewriting it.
- Expanded the workspace recovery found local candidate caches under `D:\gci\codex\v15_cache\gci_cache_final\` and `D:\gci\codex\v_3_cache\`. These contain plausible OOF arrays and correctly sized feature caches, but no manifest binds them to the executed V16 notebook, current input hashes, row IDs, or fold assignments. No configured `gci_cache_v16` directory was found.
- Recomputed the fixed V16 blend formula from the available candidate arrays without fitting models. Candidate set AUCs were old `0.77202437`, e1 `0.77257636`, e1et `0.77356771`, and e1Aet `0.77368572`; the candidate fixed-formula diagnostic was `0.77404329`, `-0.00244671` versus the saved-output claim `0.77649`. These are **not reproduced V16 results** because cache identity and row/fold alignment are unverified. The CatBoost diagnostic (`0.76942847`) additionally assumes seed-42 split order not stored in the shards; the local filenames are `catboost_product_fold_*.npz`, not the notebook's expected `catboost_v16_fold_*.npz`.
- Audited the separate `D:\gci\gemini\cache\v_16_blend_bundle.npz`: recomputed AUC is `0.79467683`, while its stored `final_auc` is `0.782000`; `oof_e1Aet`, `oof_cb`, and `final_oof` are byte-for-byte identical. The file is unbound to the executed notebook and internally inconsistent, so it is rejected as reproduction evidence. No cause (for example corruption, leakage, or wrong row order) is asserted without provenance.
- Confirmed `rebuilt_eight_engine_oof.npy` is a finite `(171202, 8)` upstream component matrix, not the final Anchor74-plus-suite champion OOF. The frozen EXP85 file is test-only. The available EXP02 Anchor74 runner has source and README only, explicitly describes a different grouped-fold surrogate, and has no run outputs.
- Exact final EXP85 OOF, row-to-fold assignments, and Anchor74 OOF counterpart remain unrecovered. **Champion comparison remains blocked.** No training, AUC improvement claim, submission creation, or champion replacement was performed.
- Execution: `python -B D:\gci\truth\experiments\phase5_champion_reproduction\recover_phase5_evidence.py --run-id 20261104T053622Z`; completed at `2026-11-04T05:36:27.051785+00:00`, exit code 0. Inputs were train `(171202, 34)`, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test `(61500, 33)`, SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. No errors or warnings were reported.
- Recovery script SHA-256: `79062401132627273F7D79261FC782F9450B72139152176196653CF42FE8177C`.
- Deliverables are preserved in `D:\gci\truth\experiments\phase5_champion_reproduction\runs\recovery_20261104T053622Z\`: artifact inventory, V16 cache score comparison, OOF readiness matrix, source audit, final report, environment, executable script, execution log, and captured terminal output. All nine files were verified non-empty.

#### 2026-11-04 — Phase 5 Task 3: V16 cache provenance and anomalous OOF investigation

- Verified the latest ledger entry before this addendum: the Phase 5 recovery addendum immediately above. This entry is append-only and clarifies the V15/V16 cache mismatch without replacing prior findings.
- Final audit run: `D:\gci\truth\experiments\phase5_champion_reproduction\runs\phase5_task3_20261104T054829Z\`; command `python -B D:\gci\truth\experiments\phase5_champion_reproduction\investigate_phase5_task3.py --run-id 20261104T054829Z`; completed 2026-11-04 05:48:37 UTC, exit code 0. Audit script SHA-256: `BB3C1CBB10146FCEA0874CA1CA9BB1072ADFC1EC51F298E832567899028FBE15`. All 11 requested deliverables are present and non-empty; supplemental CSVs, script, execution log, and terminal capture are preserved in the same run folder.
- Inputs rechecked: train `(171202, 34)`, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test `(61500, 33)`, SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`; V15 cache archive SHA-256 `a6261b5ca5a90dceb793bf2f50ca16a4437114375c718e0a89313295afd15999`. No local `gci_cache_v16` directory was found.
- **Correction/clarification to the prior candidate score comparison:** the `0.77404329` fixed-blend calculation uses V15 candidate/archive prediction caches, not V16 model caches. It is not evidence that V16 scored lower and is not a V16 reproduction. The cached V15 archive members match their extracted files byte-for-byte, but this proves extraction integrity only; the arrays lack row IDs/fold sidecars and do not bind to a producing run or exact input hashes. The V16 notebook's source/output identify a distinct `gci_cache_v16` location, whose original caches remain missing.
- The anomalous `D:\gci\gemini\cache\v_16_blend_bundle.npz` SHA-256 is `9b1626016beb7a4a43bbfd1b38643dd20de64d6de97fb71094176d8a7ced19ac`. Its `oof_e1Aet`, `oof_cb`, and `final_oof` arrays are identical (array SHA-256 `2febb38d37455ee7875a8889a1cb6f4204c822510656904d0380602774649c1e`) and score `0.7946768339` against the current train target, while the stored `final_auc` is `0.782`. The same vector occurs in sibling V6/V8 bundles. Both inspected scaffold generators show the same vector being made by fitting on all training labels and predicting those training rows, then reusing it under multiple OOF keys and writing fixed AUC constants. This strongly explains the artifact as in-sample, invalid OOF evidence; an execution log proving the exact file-write invocation was not found.
- A separate local V16 optimizer tunes and reports blend weights on the same OOF labels; its selected score is therefore selection-biased. It is not the executed V16 notebook pipeline and does not authenticate the anomalous bundle.
- **V16 status: PARTIALLY RECOVERED — DISCREPANCIES REMAIN.** Source and saved-output claims are identified, but exact V16 predictions, run/cache provenance, row/fold alignment, and source-to-cache binding are not recovered. Recovering the historical result may be technically possible if the original Drive cache and metadata are obtained. Rerunning source would be a reproduction attempt, not historical recovery.
- **EXP85 final champion OOF status: NOT RECOVERED.** Exact final blend OOF, Anchor74 OOF, and fold assignments remain absent; the frozen submission is test-only, and the available raw grouped-fold Anchor74 path is a distinct surrogate, not historical EXP85.
- Competition eligibility is **not confirmed**: local rule materials do not settle pooled unlabeled train/test statistics, target encoding, or test-to-test relational score coupling used by historical methods. No organizer clarification was obtained. This audit used no external data, performed no model training, did not read test TARGET, generated no submission, and changed no champion.
- Execution record: train/test dimensions and hashes above; V15 candidate fixed-blend diagnostic `0.77404329`; anomalous bundle AUC `0.7946768339` versus stored scalar `0.782000`; training `NOT RUN`; errors `none`. All 11 requested reports/matrices and SHA-256 manifest are in the timestamped run folder. Key report hashes: `V16_CACHE_GENERATION_TRACE.md` `2DE2D117EFC15117EB49D4856E23BBBD24BE7459654F3E0B632E98A0BF082747`; `V16_SCORE_DISCREPANCY_REPORT.md` `034AEAE565350189B206B2DCC241E9CAA858EAF6B7E5CB050DFE8A92CDFEF308`; `EXP85_CHAMPION_RECOVERY_UPDATE.md` `7AD5DEB5A0E04AE73BFE11CB5ADE228562A3D5A4C4ED9E00752D86F2516F5E71`.
- Next: obtain the original V16 Drive cache and its row/fold/source/input manifest; recover exact EXP85 and Anchor74 OOF plus folds; request written organizer guidance before attempting any reproduction or submission that relies on unresolved transductive/supervised representations. Do not train or promote a champion before these gates.

#### 2026-11-04 — Phase 6 clean-room audit, baseline, and training-only discovery

- Verified the latest ledger entry before this work: Phase 5 Task 3 V16 cache provenance and anomalous OOF investigation immediately above. This Phase 6 entry is append-only. EXP85/V16 remain historical forensic evidence and were not used as a model baseline, feature source, or submission pipeline.
- Authoritative final run: `D:\gci\truth\experiments\phase6_clean_room_research\runs\phase6_20261104T063447Z\`. Runner: `D:\gci\truth\experiments\phase6_clean_room_research\run_phase6_clean_room.py`; source SHA-256 `d55efff7fe0860ced08e86b16410e497d6ede34b4bd6a614132db22d1496b668`. Completed 2026-11-04 06:36:33 UTC, exit code 0; execution runtime 103.936 seconds. All 21 requested deliverables are non-empty; the run manifest contains 29 artifacts and all recorded artifact hashes were independently verified.
- Inputs: train `(171202, 34)`, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test `(61500, 33)`, SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`; official local rules audit SHA-256 `620f0e3076c4585b180518ae3ff2e42cb077fd5692071513399abe30a2b90b91`; dictionary SHA-256 `4dcec9310a82dbd7d65a8290bad69bbac34e632494568a96281e8a779a88c25b`. Test predictors were not read: only CSV header/ID integrity and file hashing were performed. No test TARGET, external data, pooled train/test fitting, target encoding, test-to-test coupling, submission, or champion modification.
- Reproduced the raw-only Phase 2 five-fold seed-42 baseline with the preserved corrected runner and nested inner stopping: pooled OOF ROC-AUC `0.760658451`, fold AUCs `0.75719517`, `0.76157736`, `0.76625734`, `0.75718258`, `0.76143301`; historical rounded result `0.760658`, difference `+0.000000451`. Recreated fold assignment exactly matches preserved Phase 2 folds by applicant ID; fold CSV SHA-256 `a8455332203da5234e3eb180cc1852b95b5dccdc627ceb7f6d22e8d0f5cd5cfb`. Independently rechecked OOF row count, unique IDs, TARGET/fold alignment, finite predictions, and AUC. OOF CSV SHA-256 `03a88fc40fa8071cce342bc26eda221bc0359993c8688311a43e8c36521e8de3`. This is a reproduced development baseline on repeatedly examined labels, not independent confirmation.
- Static Phase 2/3 source and evidence audit: corrected raw Phase 2, Phase 2B peer, Phase 3 product decoder, Phase 3B product ablations, Phase 3C model/advanced discovery, Phase 3D rowwise probes, and EXP02 seed-20261105 verifier are catalogued in `HISTORICAL_EXPERIMENT_ELIGIBILITY.csv`. Rowwise/fold-local versions are eligible for continued training-only research under clean-room restrictions, but their results remain exploratory on exposed folds. Confirmed historical defects: an earlier domain-formula probe used outer-fold labels for early stopping and full-training TARGET AUC for feature selection; an earlier formula runner fit dummy/category columns globally. Reject those OOF/selection claims. Adaptive selection of multiple models/blends on reused folds also limits reported maxima.
- Code/evidence findings: Phase 2B peer-only delta was only `+0.000034`; Phase 3 product block had discovery OOF `0.761353`; controlled Phase 3B product model was `0.762444` versus `0.760658` raw, 5/5 fold wins; Phase 3C fixed equal-weight blend was `0.763957` but chosen after inspecting reused folds. EXP02 seed-20261105 joint ratio block improved its own raw comparator by `+0.006336142` over 5/5 folds, but its seven ratios alias Phase 3C inputs and it is not incremental lift over that stronger baseline. These are not champion promotions.
- Training-only hidden-pattern findings: the product decoder's actual prior cash-loan lattice hypothesis is `(AMT_CREDIT / AMT_GOODS_PRICE - 1) / step`; support within 0.01 of an integer was `96.2939%` at step `0.0066` and `92.6388%` at `0.0132` over 154,988 cash-loan rows. This is a numerical regularity, not proof of actual loan terms or financing tiers. Exact predictor duplicate groups excluding ID/TARGET: zero. Four-cluster PCA/K-means descriptive default rates spanned `5.311%–9.765%`, with seed agreement ARI `0.995160`; this does not establish real borrower classes. External score and missingness measurements are in `HIDDEN_PATTERN_REPORT.md`; their univariate associations are not incremental model gain. Annuity frequency, contract duration, interest rate, fee, and household liability are not given by the dictionary and were not inferred.
- **Historical V16 OOF remains NOT VERIFIED.** The executed notebook's original `gci_cache_v16`, source/input-bound predictions, fold map, and row sidecars remain unavailable. `0.77404329` is a V15-cache diagnostic, not V16. The saved `0.77649` remains an unverified claim. The `0.79467683` duplicated bundle is rejected as OOF evidence because its vector traces to full-fit in-sample predictions. No V16 reproduction was performed; the pooled transforms/target-encoding rule status remains unresolved.
- Scope not executed: no large model sweep, ensemble optimization, UMAP (not installed), EBM/GAM model training, or independent confirmation. The task's control gate was respected: after audit and baseline verification, the next product ablation is recommended as a preregistered exploratory test; an unseen labeled cohort is still required for independent confirmation. Local rule clarifications for pooled statistics, target encoding, and test-to-test coupling remain outstanding.
- **Correction record:** the preliminary Phase 6 run `phase6_20261104T062743Z` used an incorrect lattice formula `(credit/goods)/step`; its lattice percentages are invalid and must not be cited. Its original files remain unchanged, with the additive `CORRECTION_ADDENDUM.md`. The corrected run `phase6_20261104T063150Z` and final authoritative run `phase6_20261104T063447Z` use the source-matched `(credit/goods - 1)/step` equation. Do not silently substitute the first run's measurements.
- Key final outputs: `PHASE6_EXECUTION_REPORT.md` SHA-256 `c2cf37a5e4e8d700e802fd8f5956feb71b10b5fa800c4c44da491877c87caa4b`; `PHASE2_PHASE3_CODE_AUDIT.md` SHA-256 `f3413b105b1a32fbbbbdc1f1ce23399a0d17b432b0d91076e935f069ef3de204`; `HIDDEN_PATTERN_REPORT.md` SHA-256 `db1dfe44e95727d0b818d8a787f0dd73ecb744eca5a2d5307793e3c244917741`; `SHA256_MANIFEST.csv` SHA-256 `f6c9dd86f86902348f31a8e754501f666a1bfaa322810166c84489e33f4f7c82`.
- Next experiment: freeze a nonredundant Phase 3 product-feature incremental ablation versus the Phase 6 raw baseline on common folds, include random/permuted negative controls, fit every decoder parameter inside each outer training fold, and do not tune on the final comparison labels. Follow with a small external-score disagreement ablation. Keep OOF score reporting exploratory until new labeled confirmation exists; maintain all rule-ambiguous methods disabled.

#### 2026-11-04 — Phase 7 fold-safe product-feature ablations and lattice stability

- Verified the latest ledger entry before execution: the authoritative Phase 6 clean-room audit and baseline entry immediately above. This record is append-only; earlier findings and the historical end marker are preserved.
- Authoritative run: `D:\gci\truth\experiments\phase7_product_feature_discovery\runs\phase7_20261104T073000Z\`; runner `D:\gci\truth\experiments\phase7_product_feature_discovery\run_phase7_product_discovery.py`; runner SHA-256 `887f762901eb823d692e08db153fb17f1bdc9c1af47f38474717b099c38d2105`. Preserved raw baseline source SHA-256 `c49aab451d1500e1bd7e0d618a764efeb1815c172f74e1bc5174e38a793ff3fe`; Phase 3 feature source SHA-256 `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`; Phase 3B ablation source SHA-256 `1470a412ab42aa6015f94af01859615a96fb127b7fc124d045acf8c9a81cf8ae`.
- Two failed pre-run attempts were retained, not overwritten: `phase7_20261104T071500Z` stopped before model fits because the wrapper was imported instead of the preserved baseline module; `phase7_20261104T072000Z` stopped during the first development-fold preflight because an obsolete assertion expected 47 derived columns rather than the Phase 3 source's 51 registered definitions. The corrected run imports the baseline module directly and checks the feature count against the source definition catalog. No ablation OOF models or predictions were generated by either failed attempt.
- Completed command: `python D:\gci\truth\experiments\phase7_product_feature_discovery\run_phase7_product_discovery.py --run-id 20261104T073000Z`; exit code 0; runtime 1,015.158 seconds; Python 3.12.10 / Windows 11. The train input is `(171202, 34)`, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; test is `(61500, 33)`, SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. The Phase 6 raw OOF SHA-256 `03a88fc40fa8071cce342bc26eda221bc0359993c8688311a43e8c36521e8de3` matched its manifest and was checked for ID/TARGET alignment before the run. Test data use was limited to hashing plus header and ID integrity checks; no test predictors or TARGET were used.
- Protocol: holdout seed `20261112`, 20% (34,241 rows); development seed `20261113`, five stratified folds (136,961 rows); all labels were previously exposed in project OOF analyses, so neither the development OOF nor the one-time holdout is independent confirmation. The frozen primary was raw baseline plus the exact Phase 3B all-product block. Secondary tests were 16 single-feature additions, eight predeclared product blocks, and four negative controls. Shared boosting rounds were selected on inner splits using outer-fit raw rows only; product parameters and key maps were fit only on outer-fit rows. Total: 29 variants / 145 outer-fold model fits, plus inner round-selection and two frozen holdout fits. No submission, champion modification, pooled train/test fit, target encoding, test-to-test coupling, or external data.
- OOF integrity was independently checked: 136,961 unique development IDs; exact ID/TARGET alignment; five folds; all 29 prediction columns finite and complete. Recomputed ROC-AUCs for all 29 variants matched the comparison CSV. The run manifest contains 25 files; file existence, byte sizes, and SHA-256 values all verified. OOF prediction CSV SHA-256 `0a1585edf744d529aa120398857602636af857e856b6f1fd715d38224c6407bf`; partition-manifest SHA-256 `5f61f347650639d47bda74779046e1f7ab9b5397ad4b314cb43a30588fa7f56b`.

| Candidate | Development pooled OOF AUC | Delta vs same-fold raw | Positive folds |
|---|---:|---:|---:|
| Raw-only baseline | 0.756878879 | — | — |
| Raw + all 16 K/Q/P product features (primary) | 0.759320302 | +0.002441424 | 5/5 |
| Raw + Q/P | 0.759200 | +0.002321 | 5/5 |
| Raw + P | 0.758639 | +0.001761 | 5/5 |
| Raw + de-duplicated no-redundant set | 0.758465 | +0.001586 | 5/5 |
| Raw + K/P | 0.758092 | +0.001213 | 5/5 |
| Raw + Q | 0.757593 | +0.000714 | 4/5 |
| Raw + K | 0.757519 | +0.000640 | 4/5 |
| Raw + K/Q | 0.757474 | +0.000595 | 4/5 |

- Individual additions, ranked by development OOF AUC (all deltas vs this run's raw baseline): `PROD_P5_FIT_FREQUENCY_LOG` 0.757896 (+0.001017, 3/5); `PROD_P4` 0.757426 (+0.000547, 5/5); `PROD_P3` 0.757368 (+0.000489, 4/5); `PROD_Q5` 0.757209 (+0.000330, 3/5); `PROD_Q4` 0.757151 (+0.000272, 4/5); `PROD_K_NEAREST_INTEGER` 0.757127 (+0.000248, 3/5); `PROD_Q5_FIT_FREQUENCY_LOG` 0.757030 (+0.000151, 4/5); `PROD_Q3` 0.757009 (+0.000130, 3/5); `PROD_P5` 0.756951 (+0.000072, 3/5); `PROD_K_DECIMAL` 0.756928 (+0.000049, 2/5); `PROD_K_VALID` 0.756875 (-0.000003, 3/5); `PROD_PF4_FLAG` 0.756864 (-0.000015, 2/5); `PROD_K_FIT_FREQUENCY_LOG` 0.756847 (-0.000032, 2/5); `PROD_FEE_ADJUSTED_ANNUITY` 0.756726 (-0.000153, 2/5); `PROD_FEE_RECONSTRUCTION_RESIDUAL` 0.756633 (-0.000246, 1/5); `PROD_K_LATTICE_RESIDUAL` 0.756566 (-0.000313, 3/5). These are a multiple-comparison discovery screen, not independently confirmed feature gains.
- Negative controls: row-permuted products 0.755968 (-0.000911, 1/5); random keys 0.756240 (-0.000639, 2/5); false fixed fee offset 177.1 0.758047 (+0.001169, 3/5); false fixed lattice step 0.0097 0.757416 (+0.000538, 4/5). The gains from the false fee/step controls warn that discretization/representation capacity can explain part of the observed lift; they do not prove the actual fee or step interpretation. The primary's gain exceeded the best control by about 0.001273 AUC, but this remains exposed-fold exploratory evidence.
- The one-time holdout yielded raw AUC 0.761844 and primary AUC 0.762763 (delta +0.000919). Since these labels were included in earlier research OOF work before this partition was locked, this result is not independent confirmation and cannot promote a model. Do not compare the development raw-only 0.756879 directly with Phase 6's full-data five-fold 0.760658; these use different training sizes and fold protocols.
- Numerical pattern finding: on cash-loan rows with credit above goods price, the prespecified 0.0066 step had mean absolute integer residual 0.015282 and 94.8032% of held-out development rows below residual 0.1 across five folds (fold SD 0.002505). Step 0.0132 gave 89.7074%; placebo steps 0.0057, 0.0097, 0.0173, 0.0231 gave 8.0111%, 8.7086%, 5.4895%, and 3.1450%, respectively. Fit-only selected step was 0.0066 in four folds and 0.006604 in one. This confirms a stable numerical regularity in these columns, not a causal explanation, real financing tier, fee, term, or permission to claim contract semantics.
- Formula reproduction: the source defines 51 product-derived columns; the model ablation uses the exact 16 Phase 3B K/Q/P columns (K=5, Q=4, P=7). Cash-loan masks, rounding, fit-only frequency maps, step and offset fitting match source behavior. Independent Q5/P5/K-coordinate checks passed at max absolute error <=1e-10 in each development fold. The annuity payment period is unspecified in the dictionary; no duration or actual financial quantity is inferred.
- Interpretation and decision: product-derived representations add a promising but small and not yet independently verified increment over the same-split raw baseline. Q/P blocks account for more of the lift than K alone; the strong 0.0066 numeric grid and false-fee control both mean further causal interpretation must be restrained. **Research candidate only; not promoted.** No 0.77 claim is supported; historical V16/EXP85 verification remains unchanged.
- Deliverables: `PHASE7_EXECUTION_PLAN.md`, `PHASE7_EXECUTION_REPORT.md`, `EXPERIMENT_DECISION_REPORT.md`, `OOF_INTEGRITY_REPORT.md`, `PRODUCT_FEATURE_REPRODUCTION_REPORT.md`, `PRODUCT_FEATURE_FORMULA_REGISTER.csv`, `PRODUCT_INDIVIDUAL_ABLATION.csv`, `PRODUCT_BLOCK_ABLATION.csv`, `PHASE7_MODEL_COMPARISON.csv`, `DEVELOPMENT_FOLD_RESULTS.csv`, `DEV_OOF_PREDICTIONS.csv`, `LATTICE_HYPOTHESIS_REPORT.md`, `LATTICE_FOLD_STABILITY.csv`, `LATTICE_STEP_COMPARISON.csv`, `NEGATIVE_CONTROL_RESULTS.csv`, `LOCKED_HOLDOUT_ONE_TIME_METRICS.csv`, `COMPETITION_ELIGIBILITY_UPDATE.csv`, `PHASE7_SOURCE_PROVENANCE.csv`, `VALIDATION_PARTITION_MANIFEST.csv`, `LOCKED_HOLDOUT_PREDICTIONS.csv`, `LOCKED_HOLDOUT_ONE_TIME_SCORE_INPUT.csv`, `PAIRED_BOOTSTRAP_UNCERTAINTY.csv`, `NEXT_PHASE_RECOMMENDATION.md`, `EXECUTION_LOG.txt`, and `SHA256_MANIFEST.csv`, all under `D:\gci\truth\experiments\phase7_product_feature_discovery\runs\phase7_20261104T073000Z\`.
- Next action: do not tune or rescore the locked partition. Preserve this product block as an exploratory research candidate; obtain an actually unseen, organizer-approved labeled confirmation cohort before any promotion. If unavailable, next research should be one narrow, preregistered external-score-disagreement ablation on development folds only, and remain explicitly non-confirmatory.

#### 2026-11-04 — Phase 8 nested product-key target-encoding test

- Verified the latest ledger entry before execution: the Phase 7 fold-safe product-feature ablations and lattice stability entry immediately above. This record is append-only; no earlier lines were changed or removed.
- Question: does Claude's reported product-key target-encoding signal survive leakage-safe nesting and improve the stronger, fold-safe LightGBM raw-plus-Phase-3 product baseline? Claude's claimed HistGradientBoosting run and exact code were not supplied; this is a reconstruction of the written key descriptions, not an exact reproduction.
- Authoritative run: `D:\gci\truth\experiments\phase8_nested_product_target_encoding\runs\phase8_20261104T080000Z\`; runner `D:\gci\truth\experiments\phase8_nested_product_target_encoding\run_phase8_nested_product_te.py`; SHA-256 `50ce403edecb8172d684e32baa5da666b731f50b729c61c316b4067145f408cf`. Independent audit script `D:\gci\truth\experiments\phase8_nested_product_target_encoding\audit_phase8_results.py`; SHA-256 `e82e047ecdf60369f0fe9968816ff633684fbed68c85c2639d6bea2ff042fceb`. Preserved baseline source SHA-256 `c49aab451d1500e1bd7e0d618a764efeb1815c172f74e1bc5174e38a793ff3fe`; Phase 3 feature source SHA-256 `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`; Phase 7 helper runner SHA-256 `887f762901eb823d692e08db153fb17f1bdc9c1af47f38474717b099c38d2105`.
- Completed command: `python D:\gci\truth\experiments\phase8_nested_product_target_encoding\run_phase8_nested_product_te.py --run-id 20261104T080000Z`; exit code 0; runtime 877.871 seconds; Python 3.12.10 / Windows 11. Train `(171202,34)` SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`. The experiment and independent audit read no test file; no test predictor/TARGET, external data, submission, holdout score, or champion update.
- Validation: 3 stratified five-fold seeds (42, 7, 2026), 6 variants, 90 outer-fold fits plus 15 inner stopping fits. Shared boosting rounds were selected within each outer-fit partition on raw features and held constant across variants. Product transforms were fit within outer-fit folds. Every outer-fit row's target encoding was itself inner-five-fold OOF; every outer-validation key map used only outer-fit labels; smoothing alpha=30; unseen keys fall back to the fit prior. No outer-validation label was used for encoding. Independent synthetic perturbation changed only the application-row labels and verified their encodings/counts were unchanged; overlapping map-fit/apply indices were rejected.
- Independent integrity audit verified 171,202 unique OOF IDs, exact TARGET alignment, complete finite predictions, all 90 fold records, all 15 encoding-integrity records, and the run manifest's 17 file hashes/sizes. All 18 seed/variant pooled AUCs were recomputed from saved OOF predictions and matched within `1e-12`. The paired bootstrap used 500 stratified resamples and is descriptive only.

| Model / feature block | Mean of 3 seed-pooled OOF AUCs | Delta vs raw + Phase 3 product | Positive seed/fold comparisons |
|---|---:|---:|---:|
| Raw baseline | 0.760323852 | -0.001876470 | 0/3 pooled vs product |
| Raw + Phase 3 K/Q/P product block | 0.762200321 | — | reference |
| + nested target encoding of Q4 only | 0.765801007 | +0.003600685 | 3/3 seeds; 15/15 folds |
| + nested 8-key target means | 0.767622665 | +0.005422344 | 3/3 seeds; 15/15 folds |
| + same 8 means and log-counts | 0.767769948 | +0.005569627 | 3/3 seeds; 15/15 folds |
| + 10-key target means | 0.767433402 | +0.005233080 | 3/3 seeds; 15/15 folds |

- Per-seed 8-key pooled AUCs were `0.767869664` (seed 42), `0.767695558` (seed 7), and `0.767302773` (seed 2026), with positive deltas versus raw-plus-product in all 15 folds. The result is robust to these fold seeds but not independent: they reuse the same applicants and labels.
- Averaging the three held-out predictions per applicant produced raw-plus-product seed-averaged OOF AUC `0.764195071`; the 8-key target-encoded version was `0.769695603`, delta `+0.005500532`. The 8-key-plus-counts average was `0.769844976`; 10-key average was `0.769559052`. The descriptive paired stratified bootstrap for the 8-key averaged delta gave 95% interval `[+0.004318504,+0.006582593]` (500 resamples; bootstrap positive fraction 1.0). This is not an independent confidence interval: labels are repeatedly exposed, models overlap, and key selection is adaptive. No result reached 0.770 in this audit.
- Evidence about the mechanism: a single Q4 target mean accounts for a substantial portion of the lift (`+0.00360`); the eight-key set adds roughly another `+0.00182` relative to Q4-only. Adding two more keys did not improve the eight-key result. Log-counts added only about `+0.00015` on average, so most observed lift is in the nested target means rather than support counts. Key-level causal attribution remains untested; a block-level gain does not establish a true loan product, fee, term, or rate.
- Key reconstruction: `Q3/Q4/Q5` are rounded annuity/credit ratios; `P3/P4/P5` use annuity minus a fixed `130.5` hypothesis over credit; `KQ3/KQ4/KQ5` combine rounded annuity/credit with a cash-loan integerized `(credit/goods-1)/0.0066` coordinate; `AG3` is rounded annuity/goods. The 8-key set was `Q3,Q4,Q5,KQ3,KQ4,P4,P5,AG3`; the 10-key set adds `P3,KQ5`. These are reconstructed candidate keys based on Claude's prose, not recovered source definitions. The `130.5` offset and `0.0066` step are hypotheses, not confirmed financial meanings.
- Missingness/sentinel audit: the matched raw baseline already includes `DAYS_EMPLOYED_SENTINEL`, a cleaned employment sentinel, per-numeric-column `NA_*` indicators, `RAW_MISSING_COUNT`, and `EXT_COUNT`. Thus those suggestions were already represented; duplicate features were not added. No separate CatBoost category-cross, MLP, Optuna/hyperparameter search, or one-hot/native categorical comparison was run in Phase 8.
- Competition eligibility: the local official audit says target encoding is not expressly prohibited by supplied text but remains **UNRESOLVED in rule scope**. This implementation addresses fold leakage, not organizer authorization. Do not submit or promote it without written organizer clarification. All results remain exploratory on previously exposed labels; no validated 0.77 result or champion replacement is claimed.
- Deliverables: `PHASE8_EXECUTION_PLAN.md`, `PHASE8_RESULTS_REPORT.md`, `INDEPENDENT_VERIFICATION_REPORT.md`, `MISSINGNESS_SENTINEL_AUDIT.md`, `FOLD_RESULTS.csv`, `SEED_METRICS.csv`, `AGGREGATE_COMPARISON.csv`, `SEED_AVERAGED_OOF_DIAGNOSTIC.csv`, `SEED_AVERAGED_PAIRED_BOOTSTRAP.csv`, `INDEPENDENT_AUC_RECOMPUTATION.csv`, `TARGET_ENCODING_INTEGRITY.csv`, `PRODUCT_KEY_SUPPORT.csv`, `OOF_PREDICTIONS.csv`, `COMPETITION_ELIGIBILITY.csv`, `PHASE8_SOURCE_PROVENANCE.csv`, `INDEPENDENT_AUDIT_PROVENANCE.csv`, `EXECUTION_LOG.txt`, and `SHA256_MANIFEST.csv`, under `D:\gci\truth\experiments\phase8_nested_product_target_encoding\runs\phase8_20261104T080000Z\`.
- Next action: keep the 8-key nested target encoder as a **research-only candidate**. Before any further tuning, test one-key and leave-one-key-out contributions in a preregistered study; if the lift survives, request written rule clarification and evaluate CatBoost/native categorical or tuned LightGBM only on development folds. Do not use the Phase 7 locked holdout or treat repeated CV as fresh confirmation.

#### 2026-11-04 — Phase 9 mechanism discrimination and GPU CatBoost comparison

- Scope: completed both supplied workstreams. ChatGPT's ratio-key mechanism alternatives were compared against exact amount keys, the financing coordinate, ordinal-only representation, and precision levels. Claude's proposed native CatBoost categories, category crosses, cash/revolving and older/pensioner subgroup claims, residual heterogeneity, and adjacent 130.5 annuity gap were tested. The exact Claude source was not supplied; formula keys are reconstructed from prose and are not an exact source reproduction.
- Authoritative run: `D:\gci\truth\experiments\phase9_product_mechanism_and_catboost\runs\phase9_20261104T181500LOCAL\`. Runner `D:\gci\truth\experiments\phase9_product_mechanism_and_catboost\run_phase9_product_tests.py`, SHA-256 `7d07433e312e969e8b41fd0b31743f0963a6cbeee34a02d49451e896ab33deb8`; independent auditor `D:\gci\truth\experiments\phase9_product_mechanism_and_catboost\audit_phase9_results.py`, SHA-256 `daf59890c2d2d3601b6d7afca45cecb7c58a2c8ef8aecf31e044128a25df7f6b`. Training file `(171202,34)`, SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Completed command: `python D:\gci\truth\experiments\phase9_product_mechanism_and_catboost\run_phase9_product_tests.py --run-id 20261104T181500LOCAL`; exit code 0; runtime 6,120.591 seconds; Python 3.12.10, LightGBM 4.7.0, CatBoost 1.2.10, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.4. The NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB) was used for all CatBoost fitting; observed use was about 4,184 MiB. LightGBM GPU startup was tested and explicitly failed because the installed 4.7.0 build has no GPU tree learner, so LightGBM remained CPU. The completed run records `CatBoost_GPU` and `GPU:0`.
- Scope and safeguards: training CSV only; no test file, external data, or test-to-test coupling. Three five-fold seeds (42, 7, 2026); outer fold early stopping stayed inside the training partition. Each LightGBM target map used inner 5-fold OOF encodings for training rows and outer-fit-only maps for validation, alpha=30. No locked holdout, submission, or champion was touched. Eligibility for supervised target encoding/native ordered categorical statistics remains unresolved; these results are research-only pending organizer clarification.
- Independent audit: all 45 seed/model pooled OOF AUCs and 225 fold AUCs recalculated from saved predictions and matched within `1e-10`; 171,202 IDs and TARGET values aligned exactly; all 45 prediction columns were finite. All 135 target-map integrity rows passed disjointness and no-outer-validation-label checks. Subgroup metrics, residual tables, 130.5 gap counts, and manifest hashes were independently recalculated/verified. Full result: `INDEPENDENT_VERIFICATION_REPORT.md`; OOF vectors, 156,927,375 bytes, remain in the run folder.

| Model / candidate | Mean three-seed pooled OOF AUC | Delta vs same-family baseline | Positive outer folds |
|---|---:|---:|---:|
| LightGBM raw + Phase 3 product | 0.762503 | — | reference |
| LightGBM Q3 nested target mean | 0.764703 | +0.002200 | 15/15 |
| LightGBM Q4 nested target mean | 0.765577 | +0.003074 | 15/15 |
| LightGBM Q5 nested target mean | 0.764648 | +0.002145 | 15/15 |
| LightGBM cash-only Q4 | 0.765701 | +0.003199 | 15/15 |
| LightGBM exact (credit, annuity) | 0.762906 | +0.000403 | 9/15 |
| LightGBM exact (goods, credit) | 0.762086 | -0.000416 | 4/15 |
| LightGBM exact (goods, credit, annuity) | 0.762916 | +0.000414 | 11/15 |
| LightGBM K-only / Q4×K | 0.762194 / 0.763152 | -0.000308 / +0.000649 | 6/15 / 11/15 |
| LightGBM ordinal-only Q4 | 0.762499 | -0.000003 | 7/15 |
| CatBoost GPU native Q4 | 0.766511 | +0.004483 | 15/15 |
| CatBoost GPU native eight keys | **0.768240** | **+0.006213** | 15/15 |
| CatBoost GPU eight keys + two category crosses | 0.768032 | +0.006005 | 15/15 |

- Mechanism interpretation: Q4 beat Q3, but Q5 was lower than Q4, so the “finer key always wins” assertion did not hold. Exact amount keys and K-only were weak; a Q4×K cross added only a small, inconsistent increment. Ordinal Q4 was neutral, while CatBoost native categories captured larger gains. This is compatible with useful supervised group-risk information, not proof of contractual product identity, risk-based pricing, rate, or term.
- Claude-specific diagnostics: Phase 8 OOF subgroup lift averaged over the three seeds was +0.0068 for pensioners and +0.0064 for the oldest quartile, versus +0.0052 for the youngest; cash loans gained +0.0059 while revolving loans were essentially flat (-0.0001). Direction is broadly compatible with the proposed segment pattern, but Claude's specific magnitudes were not reproduced; these are post-hoc diagnostics. Among 128 Q4 groups with at least 300 rows, residual-z correlation with mean age ranged from -0.584 to -0.579 and with pensioner share from -0.531 to -0.525; only 3.1–5.5% exceeded absolute z=3. Repeated folds and within-key clustering preclude naive significance claims.
- Adjacent amount-gap check: under the exact cash-loan goods-price/credit grouping, 6,792 of 27,255 adjacent annuity gaps equalled 130.5 (24.9%). The listed placebo gaps were rare or absent. This supports a repeated numerical gap only, not a fee explanation or assignment mechanism; the denominator differs from the narrower gap count quoted in the supplied text.
- Iteration history: all earlier Phase 9 attempts are retained but are not evidence. The first attempts were stopped/failed for discovered key-format precision and parser/schema issues; the 12:55 run was stopped by the user during partial CatBoost folds. `PHASE9_RUN_ATTEMPT_STATUS.csv` identifies these attempts and the sole authoritative final run. The fixes were tested before the final execution; do not combine partial artifacts with the authoritative results.
- Deliverables: `PHASE9_EXECUTION_PLAN.md`, `PHASE9_RESULTS_REPORT.md`, `INDEPENDENT_VERIFICATION_REPORT.md`, `INDEPENDENT_AUC_RECOMPUTATION.csv`, `AGGREGATE_COMPARISON.csv`, `SEED_METRICS.csv`, `LIGHTGBM_FOLD_RESULTS.csv`, `CATBOOST_FOLD_RESULTS.csv`, `OOF_PREDICTIONS.csv`, `TARGET_MAP_INTEGRITY.csv`, `KEY_SUPPORT.csv`, `SUBGROUP_OOF_DIAGNOSTICS.csv`, `PRODUCT_RESIDUAL_HETEROGENEITY.csv`, three `Q4_PRODUCT_RESIDUALS_SEED_*.csv` files, `FEE_GAP_PLACEBO_AUDIT.csv`, `COMPETITION_ELIGIBILITY.csv`, provenance/log/GPU metadata, and `SHA256_MANIFEST.csv`, all under the authoritative run directory.
- Decision / next step: keep the product-key encodings as **research-only**; no 0.77 guarantee, independent confirmation, or champion change. Obtain written organizer guidance before any submission use. If permitted, the next useful step is a pre-registered nested evaluation with an untouched, organizer-approved confirmation cohort; do not continue selecting among features on the same exposed OOF labels.

#### 2026-11-04 — Phase 10 zero-EXP85 governance correction

- **Governance correction:** the Phase 10 draft depended on `build_rowwise_features` from the archived `research_archive\exp85_phase2_source_snapshot` helper. The user prohibited all such dependency. The draft runner and its bytecode cache were removed; no full Phase 10 ablation/grid was launched.
- **Prior action disclosed:** before the correction arrived, one exact-fold GPU parity probe had imported the prohibited helper through the draft runner. Its output was not saved as a Phase 10 artifact and is excluded from all Phase 10 results. The corrected run did not read, hash, import, copy, or execute the helper. No EXP85 data, model outputs, prediction cache, or holdout was accessed.
- **Phase 9 dependency trace:** Phase 9 directly loads the archived helper and also loads Phase 3, Phase 3B and Phase 7, which declare/transitively load that helper. Phase 9's Phase 8 OOF reference is only for post-hoc subgroup diagnostics; Phase 8 also has the same helper lineage. These legacy dependencies are listed but not loaded by corrected Phase 10.
- **Independent implementation:** created a new row-wise builder and eight product-key formulas from supplied training predictors, the official data dictionary, and explicit mathematical definitions. No target is accepted by the feature functions. Four formula/data-dictionary tests passed. This is a new feature specification, not a reconstruction of the undocumented Phase 9 matrix.
- **One-fold GPU diagnostic:** the independent matrix produced baseline AUC 0.755288 vs Phase 9 reference 0.760209 (delta -0.004921), and eight-key AUC 0.763850 vs 0.766152 (delta -0.002301); rank correlations were 0.975408 and 0.984758. Both used the saved Phase 9 iteration count of 465. These are changed-feature-space diagnostics, not parity or a model comparison.
- **Decision:** exact Phase 9 reproduction is **BLOCKED** under the zero-EXP85 rule because the complete historical feature-builder specification is not documented in allowed materials. Do not recover it from the archived helper, cached matrices, or historical outputs. No Phase 9 score was reproduced by the independent builder; no broader Phase 10 results, eligibility decision, submission, or champion change is claimed.
- **Authoritative zero-dependency audit run:** `D:\gci\truth\experiments\phase10_catboost_product_robustness\runs\phase10_zero_exp85_20261104T154946Z\`. The output manifest's 9 recorded file hashes/sizes passed independent verification; the 37-row dependency inventory contains no runtime dependency on the prohibited helper. Train SHA-256 remains `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; official dictionary SHA-256 is `4dcec9310a82dbd7d65a8290bad69bbac34e632494568a96281e8a779a88c25b`.
- **Deliverables:** `ZERO_EXP85_DEPENDENCY_AUDIT.md`, `PHASE9_PARITY_PREFLIGHT_REPORT.md`, `PHASE9_PARITY_PREFLIGHT.csv`, `DEPENDENCY_PROVENANCE.csv`, `INDEPENDENT_FEATURE_BUILDER_TESTS.md`, `EXECUTION_LOG.txt`, `NEXT_PHASE_RECOMMENDATION.md`, and `SHA256_MANIFEST.csv` are in the authoritative run directory. `PHASE9_FEATURE_BUILDER_INDEPENDENT_SPEC.md`, `phase10_independent_feature_builder.py`, and `test_phase10_independent_features.py` are in `D:\gci\truth\experiments\phase10_catboost_product_robustness\`.
- **Single next experiment:** freeze this independently specified baseline and run a preregistered paired five-fold CatBoost comparison of it against the independently defined eight-key block. Treat it as a new study, not Phase 9 reproduction; do not use the Phase 7 holdout or any historical EXP85-derived features.

#### 2026-11-04 — Phase 11 independent reconstruction and train-only search

- **Governance / data boundary:** used only `D:\gci\inpuy\train.csv` (171,202 rows; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`) and the independent Phase 10 builder. No EXP85 helper, Phase 9 cached matrix or OOF file, test data, locked holdout, external data, or submission was used. No champion was changed.
- **Independent primary comparison:** same five stratified folds (seed 2026), 700 iterations, depth 6, learning rate 0.05, two GPU CatBoost seeds averaged within each outer fold. Raw features scored **0.759455639** OOF AUC; the same features plus eight product keys scored **0.767399318** (**+0.007943680**). All five fold deltas were positive (+0.005861 to +0.009826). Paired stratified row bootstrap 95% interval: **[+0.006485, +0.009387]**, conditional on these folds and fitted models. Prediction Spearman correlation was 0.9533. Historical Phase 9's reported 0.768240 was not reproduced and is not directly comparable.
- **Independent key ablation:** 18 GPU CatBoost variants (raw, all eight, each key alone, each leave-one-out), five fixed folds, 350 iterations. All keys scored 0.765318885 versus 0.756335776 raw (+0.008983109). P4 alone scored 0.764562419; omitting Q3 scored 0.766046372 (+0.000727 versus all keys, 4/5 folds). These are adaptive same-fold diagnostics; do not promote a key drop based on this scan.
- **Broad train-only candidate screen:** 1,695 candidates across numeric singles, pairwise sums/differences/ratios, smooth controls, low-cardinality crosses, numeric-bin crosses, and the eight explicit keys. Best fixed 10% target-rate blend against the raw model was P4 at 0.762633 (+0.003178); next were P5 (+0.003074) and Q4 (+0.002735). The best low-cardinality triple and smooth-bin lifts were only +0.000125 and +0.000013. Forty frequency-preserving shuffled-key controls were near zero. Corrected label-free frequency-only AUCs were 0.471–0.508. Search rankings are adaptive, label-derived where target rates are used, and exploratory.
- **Target-rate increment over the eight-key model:** tested 14 predeclared product/rate candidates against the primary product-key OOF predictions. Best fixed 5% blend added only +0.000078; best 10% blend was -0.000044 and best 20% blend -0.000910. Forty shuffled-key controls were near zero. No meaningful extra value was established; target-derived encoding remains rule-ambiguous and research-only.
- **Label-free one-hot LightGBM and nested model branch:** fold-fitted one-hot keys raised LightGBM OOF AUC from 0.759686772 raw to 0.764493952 (+0.004807180; positive on all five folds), but remained below CatBoost. A three-config depth/learning-rate grid selected depth 6 / 0.05 in all five outer folds. The nested logistic stack scored 0.767768922, only +0.000369603 versus the primary CatBoost result; paired bootstrap 95% interval **[-0.000233, +0.001012]**, with positive fold changes in only three of five folds. Treat this as inconclusive, not a confirmed improvement.
- **Categorical-statistics audit / eligibility:** fitted CatBoost parameters used `one_hot_max_size=10`, `max_ctr_complexity=1`, and `Borders` target-dependent CTRs (`TargetBorderType=MinEntropy`) alongside `FeatureFreq`. Each product key had at least 96 observed categories. Therefore CatBoost's key processing does use internal target statistics; eligibility remains unresolved. The separate one-hot LightGBM feature transform is label-free, but no organizer approval or holdout validation was obtained.
- **Correctness fix:** candidate target-rate cross-fitting now factorizes missing/non-finite values into a real category and rejects negative codes, preventing `-1` from silently selecting the final category. It validates positive smoothing, disjoint fit/apply rows, complete single application, and binary labels. Six Phase 11 tests passed, including held-out-label invariance and fold-fitted unseen-category handling; the four Phase 10 builder tests also passed. All Phase 11 scripts compiled.
- **Authoritative outputs:** `D:\gci\truth\experiments\phase11_independent_research\PHASE11_RESULTS_REPORT.md`; primary run `runs\phase11_gpu_cv_20261105\` (OOF SHA-256 `483ec36819dbf2772846dcfe0e73607ef3316f75ab45cf36932a5df3d1e2bbbc`); ablation `runs\phase11_ablation_cv_20261105\` (OOF SHA-256 `d5c018899e4ad75012876b1ae059265703baa89cb00e8762ff2d7241fa2b6983`); candidate scan `runs\phase11_candidate_scan_20261105\`; target-rate increment `runs\phase11_te_vs_product_20261105\`; nested model/CTR audit `runs\phase11_nested_ensemble_20261105\` (OOF SHA-256 `85d81b513726725b7a2aaa95f559147e136b2a9d2e28ee67ca3c0f80192bde75`). CatBoost executed on the NVIDIA GeForce RTX 4050 Laptop GPU; LightGBM ran on CPU because the installed build lacks the required GPU learner.
- **Decision:** predictive value is independently demonstrated within this new feature specification, but no robust score above 0.77, Phase 9 reproduction, or competition-eligible CatBoost procedure was established. Keep target-rate encoding and CatBoost categorical statistics out of any submission pending written organizer guidance. Treat all Phase 11 OOF results as research evidence, not leaderboard or holdout confirmation.

#### 2026-11-04 — Phase 12 label-free product representations and eligibility audit

- **Scope / governance:** executed the standalone Phase 12 runner in `experiments/phase12_label_free_product_representation/`; authoritative run: `runs/phase12_20261104T173534Z/`. Used only `D:\gci\inpuy\train.csv` (171,202 rows; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`). Did not open test/sample-submission files, create a submission, use external data, use Phase 9/EXP85 code or caches, compute target encoding/target rates, or use target-derived categorical statistics. The feature builder is independent and reproduces all eight Phase 11 product-key columns exactly on every training row (including missing handling and string values); observed full-training cardinalities: Q3 96, Q4 797, Q5 5,102, KQ3 1,002, KQ4 3,576, P4 775, P5 4,079, AG3 131.
- **Protocol / runtime:** fixed stratified five-fold primary CV (seed `20261105`), with a 3-fold inner comparison of two LightGBM CPU configurations and inner-fold early stopping; outer folds were not used for selection. Compared numeric-only baseline, one-hot keys, fit-fold frequency-supported one-hot keys, numeric decompositions, predeclared interactions, combined features, and per-key feature hashing. Completed all-key, single-key, leave-one-out, and three pre-registered combination ablations; five repetitions each of frequency-preserving shuffled and randomized-assignment negative controls; and one two-model secondary-seed robustness replication (`20261106`). Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1; CPU, four LightGBM jobs. Primary execution window: 2026-11-04 17:35:34–18:25:59 UTC (about 50 minutes). Phase 11's GPU run manifest confirms its actual start was 2026-11-04 16:18:30 UTC; its `20261105` directory suffix is not its execution date.
- **Main results (Phase 12 numeric-only baseline):** raw numeric AUC **0.743256778**; raw numeric plus eight one-hot product keys **0.758729377** (delta **+0.015472598**, 5/5 positive folds; paired stratified bootstrap 95% interval **[+0.013742, +0.017175]**); fit-fold frequency-supported one-hot keys reached **0.759815774** (+0.016558996; 5/5; interval **[+0.014728, +0.018325]**); hashed keys reached 0.759298484 (+0.016041705; 5/5); numerical decompositions reached 0.753826137 (+0.010569359; 5/5); selected interactions alone reached 0.742790936 (-0.000465842; only 1/5 folds improved). The primary key OHE model's per-fold AUCs were 0.757411, 0.761762, 0.762345, 0.754326, 0.758234.
- **Interpretation / limitations:** the numeric-only Phase 12 baseline is deliberately not the Phase 11 LightGBM baseline, which also included raw categorical one-hot features and additional rowwise features. Thus Phase 12's larger absolute delta cannot establish that it preserved most of Phase 11's +0.004807 lift. Phase 12's raw numeric OOF AUC is also not a comparable competition benchmark. On the exposed Phase 12 folds, Q3 was the strongest single key (0.755227 pooled AUC), narrowly ahead of P4 (0.755052); all single keys were below the all-key representation. Leaving out KQ3 yielded 0.759257 versus all-key ablation 0.759051, only +0.000206 with 3/5 fold wins; this is exploratory and does not justify removing KQ3. Shuffled controls averaged 0.741184 AUC; randomized-assignment controls averaged 0.741384. Genuine key representations outperform these controls on this comparison, but this remains conditional same-data CV evidence, not independent confirmation.
- **Robustness / integrity:** second-seed raw numeric AUC 0.743681 and key OHE AUC 0.759590 (+0.015909, 5/5 folds); this is a robustness replication, not independent confirmation. Saved primary, ablation, control, and secondary OOF sets have zero missing predictions; 171,202 IDs are unique and targets/row order align. Input hash was unchanged. All 25 entries in the SHA-256 manifest were independently verified with no mismatch. Six Phase 12 formula, target-exclusion, frequency-fit/unseen-category, bootstrap correctness, and decomposition tests passed; the LightGBM pipeline smoke test and Python compilation passed. The first attempted full run stopped before fitting because the Windows process-memory API handle signature was incorrect; the handle was fixed and the preserved failed attempt is not treated as evidence.
- **Eligibility / decision:** no target-dependent processing, test fitting, test-to-test coupling, or external data were used. The existing course audit supports deterministic supplied-column feature construction, but does not document the provenance of the historical 130.5 fee offset or 0.0066 financing step used by P4/P5/KQ3/KQ4 and related numeric decompositions. These representations remain eligibility-pending, and no organizer approval or champion promotion is claimed. Next, establish permitted-data provenance or obtain organizer guidance for these constants; then compare raw-categorical and rowwise baselines plus keys on the same preregistered folds before asserting how much Phase 11 lift survives.
- **Authoritative outputs:** `experiments/phase12_label_free_product_representation/PHASE12_README.md`; run report `runs/phase12_20261104T173534Z/PHASE12_EXECUTION_REPORT.md`; formula audit `PHASE12_FEATURE_FORMULA_AUDIT.csv`; representation/model comparisons `LABEL_FREE_REPRESENTATION_RESULTS.csv` and `MODEL_COMPARISON.csv`; ablations `PRODUCT_KEY_INDIVIDUAL_ABLATION.csv`, `PRODUCT_KEY_LEAVE_ONE_OUT.csv`, and `PRODUCT_KEY_ABLATION_ALL_VARIANTS.csv`; controls `NEGATIVE_CONTROL_RESULTS.csv`; fold metrics `FOLD_LEVEL_METRICS.csv`; bootstrap `PAIRED_BOOTSTRAP_REPORT.md` and `.csv`; eligibility `COMPETITION_ELIGIBILITY_MATRIX.csv`; integrity `OOF_INTEGRITY_REPORT.md`; hashes `SHA256_MANIFEST.csv`; OOF predictions are preserved in the same run folder. Phase 12 did not change the champion or create a submission.

#### 2026-11-05 — Phase 12 exhaustive source, protocol, results, and artifact-hash supplement (append-only)

This supplement preserves the detailed code and result record for Phase 12 in the ledger itself. It extends, and does not replace, the Phase 12 summary above or any prior phase entry. Source/output paths below are relative to `D:\gci\truth\experiments\phase12_label_free_product_representation\`, except where stated otherwise.

##### A. Exact source code inventory and reproducibility

| Source file | Responsibility and material code details | Bytes | SHA-256 |
|---|---|---:|---|
| `phase12_feature_definitions.py` | Standalone, deterministic formula implementation. `build_product_keys` independently recreates Q3/Q4/Q5/KQ3/KQ4/P4/P5/AG3; `_safe_ratio` rejects non-finite inputs and zero denominators into NaN; `_integerized` preserves NumPy round-to-even and historical integer string encoding, with explicit int64 overflow failure; target columns are rejected. `build_numeric_predictors` keeps numeric supplied predictors and omits the identifier. `build_decomposed_features` creates the numerical ratio and financing components. `build_selected_interactions` implements four predeclared numeric interactions. `category_support` reports per-fit partition support and unseen apply categories. | 8,337 | `81f0861fffda89f44141faffdb3c30db3453b87a6e6175e793f5f4af7ffad099` |
| `run_phase12.py` | End-to-end runner. It validates input IDs/targets and data hash; builds features without importing EXP85, Phase 9, or Phase 10; constructs sparse fold-fitted one-hot, minimum-frequency one-hot, numeric and hash matrices; tunes two LightGBM configurations using 3-fold inner OOF and inner-fold early stopping; refits on each complete outer-fit partition; evaluates seven main representations; runs 20 key subsets (all-eight, 8 singles, 8 leave-one-out, 3 pre-registered combinations); runs two types of five-seed negative controls; performs the secondary-seed run; calculates pooled/fold AUC, paired stratified bootstrap, correlation, support, runtime, memory and collision reports; checks OOF integrity; writes a new timestamped folder with non-overwrite behavior and SHA-256 manifest. Windows peak-working-set telemetry uses correctly typed Win32 API signatures. | 84,479 | `ff9da3af45607654bc1f42be670578160cc24a91639a6d683192ede0bcc8e913` |
| `test_phase12_feature_definitions.py` | Six tests cover exact precision/cross formulas, missing and invalid denominators, target rejection, numeric components/interactions, fit-only frequency encoding and safe unknown categories, and paired bootstrap equivalence to direct ROC-AUC resampling. | 6,272 | `6037d275f6a5edd5a5589d2f293c49268b07038369b5b2abfdadd00f653ed828` |
| `PHASE12_README.md` | Phase scope, data/dependency restrictions, protocol and commands to test and execute the experiment. | 1,772 | `e33e4a48f1d3733504ce1fc57d68cf4ab672ecd647630459cbe02bfb76fc3ddb` |

The runner's primary configuration was two CPU LightGBM options: `leaves31_child80` (`num_leaves=31`, `min_child_samples=80`) and `leaves63_child100` (`num_leaves=63`, `min_child_samples=100`); common parameters were objective binary, learning rate 0.03, `max_depth=-1`, `colsample_bytree=0.8`, `subsample=0.8`, `subsample_freq=1`, `reg_lambda=1.0`, `max_bin=127`, CPU device, four jobs, deterministic mode, and force-column-wise mode. Maximum iterations were 1,200 with 80-round inner early stopping; median inner best iteration determined the outer refit rounds. The first configuration was selected for every primary outer-fold/representation combination and every second-seed main comparison. Main outer CV used seed `20261105`, five stratified folds; each outer fit used three stratified inner folds seeded with `seed + outer_fold`. A separate five-fold robustness run used seed `20261106`. Control seeds were `20261201` through `20261205`. Conditional paired stratified bootstrap used 1,000 repetitions and seed `21161105`.

The Phase 12 numeric baseline contains 24 supplied numeric predictors; categorical predictors are intentionally excluded from this baseline. The seven main variants were:

1. `raw_numeric`;
2. `raw_plus_keys_ohe`;
3. `raw_plus_keys_frequency_ohe` (frequency and minimum-frequency support fitted on the current fit partition only; unseen apply values map safely);
4. `raw_plus_decompositions`;
5. `raw_plus_interactions`;
6. `raw_plus_keys_decompositions_interactions`;
7. `raw_plus_keys_hash` (8,192 fixed hash buckets per key; collision diagnostics recorded).

The selected interactions were credit x region population, annuity x region population, EXT_SOURCE_1 x EXT_SOURCE_2, and EXT_SOURCE_2 x EXT_SOURCE_3. Numeric decompositions included annuity/credit, annuity/goods, (annuity-130.5)/credit, (credit/goods-1)/0.0066 for eligible Cash loans, and its eligibility indicator. Historical constants were preserved, not re-estimated or independently proven eligible.

##### B. Formula-by-formula audit record

All keys below are row-wise, label-free calculations from supplied columns. For Q/P/AG numeric keys, the exact encoding is `rint(value * 10^d)`, cast to signed int64, then decimal string; NumPy round-to-even is retained. Non-finite numerator/denominator or a zero denominator produces the literal `MISSING`. Equal rounded values intentionally share a category; these are precision-induced collisions, not hash collisions.

| Key | Exact formula | Decimal precision | Full-train category count | Special handling |
|---|---|---:|---:|---|
| Q3 | AMT_ANNUITY / AMT_CREDIT | 3 | 96 | Invalid ratio -> `MISSING` |
| Q4 | AMT_ANNUITY / AMT_CREDIT | 4 | 797 | Invalid ratio -> `MISSING` |
| Q5 | AMT_ANNUITY / AMT_CREDIT | 5 | 5,102 | Invalid ratio -> `MISSING` |
| KQ3 | For NAME_CONTRACT_TYPE=`Cash loans`, cross `rint(((AMT_CREDIT/AMT_GOODS_PRICE)-1)/0.0066)` with Q3; otherwise sentinel | Financing coordinate integer; Q3 at 3 | 1,002 | Only when finite, credit > goods > 0 and contract is Cash loans; invalid Q3 or ineligible row -> `NONCASH_OR_INVALID`; cross string is `<coordinate>|<Q3>` |
| KQ4 | Same financing coordinate crossed with Q4 | Financing coordinate integer; Q4 at 4 | 3,576 | Same eligibility and sentinel handling as KQ3 |
| P4 | (AMT_ANNUITY - 130.5) / AMT_CREDIT | 4 | 775 | Historical 130.5 offset; invalid ratio -> `MISSING` |
| P5 | (AMT_ANNUITY - 130.5) / AMT_CREDIT | 5 | 4,079 | Historical 130.5 offset; invalid ratio -> `MISSING` |
| AG3 | AMT_ANNUITY / AMT_GOODS_PRICE | 3 | 131 | Invalid ratio -> `MISSING` |

The full column-level formula text, inputs, missing-value policy, precision, cardinality, collision behavior, train-fold support statement, label-free status, test/external-data status, and eligibility note are in `runs\phase12_20261104T173534Z\PHASE12_FEATURE_FORMULA_AUDIT.csv`. The independent Phase 12 outputs were compared against the available historical formula implementation for every one of the 171,202 training rows; all eight resulting columns matched exactly.

##### C. Complete primary representation results

| Representation | Pooled OOF ROC-AUC | Delta vs numeric-only | Positive folds | Spearman correlation vs raw |
|---|---:|---:|---:|---:|
| raw_numeric | 0.743256778 | 0.000000000 | 0/5 (reference) | 1.000000 |
| raw_plus_keys_ohe | 0.758729377 | +0.015472598 | 5/5 | 0.919315 |
| raw_plus_keys_frequency_ohe | 0.759815774 | +0.016558996 | 5/5 | 0.913553 |
| raw_plus_decompositions | 0.753826137 | +0.010569359 | 5/5 | 0.948441 |
| raw_plus_interactions | 0.742790936 | -0.000465842 | 1/5 | 0.984726 |
| raw_plus_keys_decompositions_interactions | 0.758421880 | +0.015165101 | 5/5 | 0.914549 |
| raw_plus_keys_hash | 0.759298484 | +0.016041705 | 5/5 | 0.917690 |

Per-fold results are listed here in fold order 1, 2, 3, 4, 5; entries show `AUC (paired fold delta against raw_numeric)`:

| Representation | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| raw_numeric | 0.741754 (0) | 0.744910 (0) | 0.748290 (0) | 0.740864 (0) | 0.740608 (0) |
| raw_plus_keys_ohe | 0.757411 (+0.015657) | 0.761762 (+0.016852) | 0.762345 (+0.014054) | 0.754326 (+0.013462) | 0.758234 (+0.017626) |
| raw_plus_keys_frequency_ohe | 0.757215 (+0.015461) | 0.763682 (+0.018772) | 0.763754 (+0.015464) | 0.755242 (+0.014378) | 0.759414 (+0.018807) |
| raw_plus_decompositions | 0.752401 (+0.010647) | 0.756877 (+0.011967) | 0.758562 (+0.010272) | 0.749069 (+0.008205) | 0.752346 (+0.011738) |
| raw_plus_interactions | 0.740928 (-0.000826) | 0.745391 (+0.000482) | 0.747193 (-0.001097) | 0.740572 (-0.000291) | 0.740078 (-0.000530) |
| raw_plus_keys_decompositions_interactions | 0.756167 (+0.014413) | 0.762442 (+0.017532) | 0.762032 (+0.013742) | 0.753643 (+0.012779) | 0.758113 (+0.017505) |
| raw_plus_keys_hash | 0.756816 (+0.015062) | 0.763106 (+0.018196) | 0.764555 (+0.016265) | 0.753845 (+0.012981) | 0.758251 (+0.017643) |

Conditional paired stratified bootstrap 95% intervals for the change against `raw_numeric` were:

| Representation | Paired delta | 95% interval |
|---|---:|---:|
| raw_numeric | 0.000000 | [0.000000, 0.000000] |
| raw_plus_keys_ohe | +0.015473 | [+0.013742, +0.017175] |
| raw_plus_keys_frequency_ohe | +0.016559 | [+0.014728, +0.018325] |
| raw_plus_decompositions | +0.010569 | [+0.009271, +0.011878] |
| raw_plus_interactions | -0.000466 | [-0.001253, +0.000368] |
| raw_plus_keys_decompositions_interactions | +0.015165 | [+0.013355, +0.016825] |
| raw_plus_keys_hash | +0.016042 | [+0.014329, +0.017698] |

All intervals are conditional on fitted OOF predictions and previously examined training labels; they do not establish independent/dataset-level confirmation. Phase 11's +0.004807 label-free LightGBM result used a different raw-feature baseline that included categorical one-hot and rowwise-derived features, so Phase 12's larger numeric-only deltas do not answer the historical lift-preservation question.

##### D. Complete product-key ablation results

These are a separate screening grid. All variants used the same primary outer folds, and the all-eight row uses the OOF predictions from the ablation grid rather than the separately tuned main `raw_plus_keys_ohe` representation. They share per-fold configuration/iteration choices from the all-key representation; do not treat the results as unbiased selection or confirmation.

| Variant | Keys | Pooled OOF AUC | Paired fold wins vs ablation all-eight |
|---|---:|---:|---:|
| all_eight | 8 | 0.759050673 | reference |
| single_Q3 | 1 | 0.755227236 | 0/5 |
| single_Q4 | 1 | 0.752783777 | 0/5 |
| single_Q5 | 1 | 0.747693769 | 0/5 |
| single_KQ3 | 1 | 0.748648705 | 0/5 |
| single_KQ4 | 1 | 0.745474837 | 0/5 |
| single_P4 | 1 | 0.755051601 | 0/5 |
| single_P5 | 1 | 0.751960084 | 0/5 |
| single_AG3 | 1 | 0.747321035 | 0/5 |
| leave_out_Q3 | 7 | 0.758336 | 0/5 |
| leave_out_Q4 | 7 | 0.758481 | 1/5 |
| leave_out_Q5 | 7 | 0.758487 | 1/5 |
| leave_out_KQ3 | 7 | 0.759257 | 3/5 |
| leave_out_KQ4 | 7 | 0.759112 | 3/5 |
| leave_out_P4 | 7 | 0.758162 | 1/5 |
| leave_out_P5 | 7 | 0.758416 | 0/5 |
| leave_out_AG3 | 7 | 0.759028 | 3/5 |
| preregistered_Q3_Q4_Q5 | 3 | 0.756975 | 0/5 |
| preregistered_P4_P5_Q4 | 3 | 0.757817 | 1/5 |
| preregistered_P4_Q4_AG3 | 3 | 0.757629 | 0/5 |

Values marked to 6 decimals above are recorded more precisely in `PRODUCT_KEY_ABLATION_ALL_VARIANTS.csv`; the individual/leave-one-out files retain per-fold AUC, paired delta, model rounds, feature count and runtime. Q3 was strongest among single keys, only slightly above P4. Excluding KQ3 yielded the top leave-one-out score, only +0.000206 over ablation all-eight and winning three folds; this is insufficient to drop KQ3. No key was promoted or removed.

##### E. Negative controls, secondary seed, and computation

The frequency-preserving shuffled-key control retained each key's category-frequency distribution while breaking row correspondence. The randomized-assignment control drew categories using the fit-fold frequency distribution and assigned random keys to fit/apply rows. Each used the same five outer folds and the all-key-selected configuration/rounds; each repetition/fold prediction and metric is preserved.

| Control | Repetition seeds 1-5 | Pooled OOF AUC by repetition | Mean |
|---|---|---|---:|
| frequency_preserving_shuffle | 20261201, 20261202, 20261203, 20261204, 20261205 | 0.741288, 0.740655, 0.740966, 0.741539, 0.741471 | 0.741184 |
| randomized_assignment | 20261201, 20261202, 20261203, 20261204, 20261205 | 0.741629, 0.741098, 0.741239, 0.741200, 0.741756 | 0.741384 |

Secondary-seed robustness CV (seed `20261106`) results: numeric-only AUC `0.743680868` and eight-key one-hot AUC `0.759589880`, paired delta `+0.015909012`, positive folds `5/5`. Secondary training runtime was 141.65 seconds for numeric-only and 221.96 seconds for key one-hot. This reuses the same dataset/labels and is robustness replication, not independent confirmation.

For primary five-fold refits, selected iterations by fold 1-5 were: numeric-only `284, 301, 367, 385, 361`; key one-hot `469, 377, 340, 476, 380`; frequency-supported one-hot `442, 399, 368, 382, 373`; decompositions `366, 364, 401, 402, 523`; interactions `319, 459, 365, 356, 295`; combined `409, 388, 330, 395, 363`; hashed `427, 434, 416, 363, 456`. Primary peak process working set recorded was 640.37 MiB. Across primary outer fits, raw/frequency/key/hash and related tuning/final-fit elapsed seconds are detailed in `OUTER_MODEL_FITS.csv` and `LABEL_FREE_REPRESENTATION_RESULTS.csv`; this machine's recorded total training was roughly 50 minutes including controls, output and reports.

##### F. Integrity, attempted run, tests, and decisions

- Successful authoritative execution folder is `runs\phase12_20261104T173534Z\`, UTC start `2026-11-04T17:35:34.990669+00:00`, completion `2026-11-04T18:25:59.709616+00:00`. Python 3.12.10; pandas 3.1.0; NumPy 2.5.4; scikit-learn 1.9.1; LightGBM 4.7.0; SciPy 1.18.1. CPU LightGBM, four jobs; no GPU modeling.
- A prior attempt folder `runs\phase12_20261104T173440Z\` was preserved. It stopped before completing the first inner model because the process-memory Windows API handle was incorrectly declared; the API signature was corrected and independently smoke-tested before the successful run. The failed attempt is not metric evidence.
- Six Phase 12 unit tests passed; the run smoke test produced finite probabilities for 300 apply rows under current LightGBM early-stopping API; both feature and runner modules compiled. Full key parity against the historical builder passed for 171,202 rows and all eight columns.
- OOF integrity report records 171,202 rows, 171,202 unique IDs, 13,821 positives, complete primary and secondary fold assignments, zero missing predictions across 7 main models, 20 ablation models, 10 negative-control models, and 2 secondary-seed models; all saved predictions are finite and in [0,1]. Training ID/target order is aligned and the source data hash did not change.
- All 25 entries in the run's `SHA256_MANIFEST.csv` (22 output files plus three source files) were checked for existence, size and digest; no mismatch was found. The manifest itself is not self-hashed. The report and next-phase recommendation hashes were refreshed after their final text was edited.
- Competition gate remains unresolved for key formulas involving the undocumented 130.5 fee offset and 0.0066 financing step; label-free construction and fit-fold encoders do not by themselves establish organizer approval. Phase 12 results are exposed-label research CV, not official leaderboard, holdout, external or independent confirmation. No submission/champion change.
- Single best next experiment remains: document/resolve constant provenance, then preregister and compare numeric plus raw categorical/rowwise features and fold-fitted label-free keys on identical outer folds, with a secondary-seed robustness replication. Do not claim Phase 11 lift preservation until the baseline is matched.

##### G. Full SHA-256 inventory for the authoritative Phase 12 run

The following byte counts and SHA-256 values are copied from the verified `runs\phase12_20261104T173534Z\SHA256_MANIFEST.csv`. Paths are relative to that run folder; `source:` files refer to the phase source directory.

| File | Bytes | SHA-256 |
|---|---:|---|
| `CATEGORY_SUPPORT_REPORT.csv` | 74,818 | `780416714f775897d8d224da5e0d5c5a35e4783be2ff0222410afc42817293ff` |
| `COMPETITION_ELIGIBILITY_MATRIX.csv` | 2,106 | `14f8df981a63f0ca55950cf98723ffeecb82e7d2b05aa7a070d65d8c00f83e4d` |
| `EXECUTION_LOG.txt` | 37,884 | `d8017ece662cc94cd296a3853e0a78a5a291e29f913583f0347ba92896e636b1` |
| `FOLD_LEVEL_METRICS.csv` | 23,843 | `bba6edc0ff6955656ba0dc12d4e06937fe220ce794a7173de3dc21db695a950b` |
| `INNER_MODEL_SELECTION.csv` | 33,733 | `df67914b087dae851b7b290111bbfe1a397243ca46bd3024432561aa55eddde6` |
| `LABEL_FREE_REPRESENTATION_RESULTS.csv` | 3,027 | `5742e663552ed88b8945aa457add9ee03ec034ad179676bad1f7c50bab95e176` |
| `MODEL_COMPARISON.csv` | 3,027 | `5742e663552ed88b8945aa457add9ee03ec034ad179676bad1f7c50bab95e176` |
| `NEGATIVE_CONTROL_OOF_PREDICTIONS.csv.gz` | 16,402,432 | `95dc52c06e142b6981021cbb8b0aeb549e3a8c838bdc6d084f1c2beae6ee7451` |
| `NEGATIVE_CONTROL_RESULTS.csv` | 8,094 | `7f90b0b477e02c3bfcf5abea6c08178cdca2f1a36819420434a8003a487e9bce` |
| `NEXT_PHASE_RECOMMENDATION.md` | 944 | `e3664592a0c649989ba14e844293819a140c238ccdc177dc99ead19de6f47f47` |
| `OOF_INTEGRITY_REPORT.md` | 1,383 | `89e59b85defb82282b11fd5d034ea8fe66b18e1e53dcc779f0c578f938d34c7e` |
| `OOF_PREDICTIONS.csv` | 96,122,694 | `7d9dd16bf07a881cb618010cff35ddb92d48bc5a8cc102d512b84181a1560e40` |
| `OUTER_MODEL_FITS.csv` | 7,672 | `6ad8b9fdeab4fd60db66856f4a443ad1d19572e642118d340c260bf696a75adb` |
| `PAIRED_BOOTSTRAP_REPORT.csv` | 1,711 | `12f16d7d733e91363b535248e6c659b393d7fe1c3a77411e91688ab5f22b57d5` |
| `PAIRED_BOOTSTRAP_REPORT.md` | 909 | `5540c35ebc824a9b1365580855fac377c6677c32ff827582bcb486f02ff563b4` |
| `PHASE12_EXECUTION_REPORT.md` | 5,996 | `20e81f50b96064b15e24613a497e5236046f232773706d22a1bcf573fa64e76c` |
| `PHASE12_FEATURE_FORMULA_AUDIT.csv` | 7,194 | `1b073590cb581eafe0095ae0eaf844e8f8f71ca8db954e4495674ac68eb8cf18` |
| `PRODUCT_KEY_ABLATION_ALL_VARIANTS.csv` | 19,857 | `a51abfd5b3db833c0a69c1cbf07b9f2708fc06229c3817ef47aedd7d7a38067b` |
| `PRODUCT_KEY_INDIVIDUAL_ABLATION.csv` | 7,209 | `863036e53407bd8f55302196c07e83e7625cffaf035c51b43a5de34acc85f05b` |
| `PRODUCT_KEY_LEAVE_ONE_OUT.csv` | 9,923 | `f26a97c1dc62122f31de65be699e44077dbe1dbabaa1d642d45b1cef00ed1788` |
| `SECONDARY_SEED_OOF_PREDICTIONS.csv` | 10,267,144 | `64fd16d645d33355b0c2ada39c463921b7a064e725c88a11bdf1be5f53bc50fd` |
| `SECONDARY_SEED_REPLICATION.csv` | 408 | `972059cd16d47cdce04d34a1be9df8b82502452fa0062070e7769a735aaa56bc` |
| `source:run_phase12.py` | 84,479 | `ff9da3af45607654bc1f42be670578160cc24a91639a6d683192ede0bcc8e913` |
| `source:phase12_feature_definitions.py` | 8,337 | `81f0861fffda89f44141faffdb3c30db3453b87a6e6175e793f5f4af7ffad099` |
| `source:test_phase12_feature_definitions.py` | 6,272 | `6037d275f6a5edd5a5589d2f293c49268b07038369b5b2abfdadd00f653ed828` |

## Phase 13 — matched raw baseline, product-key incrementality, and exploratory hidden-key screen

**Recorded:** 2026-11-05  
**Status:** Executed and artifact-verified. This entry is append-only; earlier phase results and their conclusions remain unchanged.

### Executive result and decision

The matched comparison answers the main Phase 13 question: against the complete raw numeric + raw categorical + approved rowwise baseline C (`0.759619` pooled five-fold OOF AUC), ordinary one-hot product keys D scored `0.765270` (`+0.005651`) and fit-fold frequency-supported one-hot keys E scored `0.766038` (`+0.006418`). Both beat C in all five folds. Paired stratified bootstrap intervals against C exclude zero. E is nominally `+0.000767` above D, but its direct paired 95% interval includes zero; its secondary-seed run did not compare E against D. Frequency support is therefore promising, not established as superior.

Genuine product-key features beat each of the five frequency-preserving shuffled Phase 13 controls. The corrected same-seed ablation found that all eight keys together are useful as a group but no one key is indispensable; differences among leave-one-out subsets are small and mixed. The separate hidden-key screen is adaptive, target-dependent exploratory research; its best blends do not certify a latent product identifier or competition eligibility.

**No competition-ready champion was certified, no champion was changed, and no submission was made.** No test rows, sample submission, external data, or holdout were opened or used.

### Historical references and comparability

The historical reference scores remain:

| Phase / result | OOF AUC | Interpretation |
|---|---:|---|
| Phase 9 CatBoost + eight product keys | 0.768240 | Historical high; not independently reproduced |
| Phase 11 independent CatBoost + eight keys | 0.767399 | Independent construction; internal CTR eligibility unresolved |
| Phase 12 numeric-only LightGBM | 0.743257 | Different, weaker feature baseline |
| Phase 12 eight-key one-hot | 0.758729 | Not matched to Phase 11 raw-feature baseline |
| Phase 12 frequency-supported one-hot | 0.759816 | Not directly comparable to Phase 11 or Phase 13 |
| Phase 13 C complete raw + rowwise | 0.759619 | Matched Phase 13 reference |
| Phase 13 D, C + ordinary key one-hot | 0.765270 | Matched label-free benchmark |
| Phase 13 E, C + fit-fold frequency-supported key one-hot | 0.766038 | Highest primary matched benchmark |

The Phase 12 `+0.016559` over its numeric-only baseline must not be presented as the incremental value over the complete raw baseline. The Phase 13 C→D/E contrasts are the fair matched estimates.

### Data, code, protocol, and reproducibility

- Training input: `D:\gci\inpuy\train.csv`; 171,202 rows, 171,202 unique IDs, 13,821 positives. SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Five fixed stratified outer folds; 34,241, 34,241, 34,240, 34,240, and 34,240 validation rows respectively. Primary seed `20261113`; secondary robustness seed `20261114`. Three inner folds per outer fit.
- Model family: CPU LightGBM. Per outer training partition, two preregistered configurations (`leaves31_child80`, `leaves63_child100`) were compared using pooled inner OOF AUC; inner-fold early stopping selected iterations. Learning rate `0.03`, maximum 1,200 iterations, early stopping 80, `max_bin=127`, column and row sampling `0.8`, L2 lambda `1`, deterministic CPU execution, four jobs.
- A and B use raw numerical features, with B adding one-hot raw categories fitted on the applicable training partition. C adds 11 deterministic rowwise missingness, sentinel, age, and guarded financial-ratio features. D adds eight one-hot product keys. E adds the same keys with category-frequency support fitted in the applicable fit partition. F adds three direct numeric decompositions that omit the undocumented 130.5 and 0.0066 constants.
- Feature construction is independent in `experiments\phase13_matched_baseline\phase13_features.py`; it rejects a frame containing `TARGET`, excludes `SK_ID_CURR` from predictors, normalizes categorical missing values, guards invalid/zero-denominator ratios, and keeps rowwise matrix alignment. Product-key feature parity against the historical builder was checked on all 171,202 training rows.
- Five targeted feature tests passed, including noncontiguous input-index alignment; Python compilation passed. The smoke-test index-alignment defect was fixed before the full run.
- Primary execution: `2026-11-05T07:59:46.355550Z`–`2026-11-05T08:49:29.037820Z` (49m 42.7s). Primary peak working set was 772.66 MiB; secondary outer fit peak was 806.65 MiB. Environment: Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1.
- Main run folder: `experiments\phase13_matched_baseline\runs\phase13_20261105T075946Z`. Five-manifest/Oof checks passed: every original input row was predicted once out of fold, predictions had no missing/out-of-range values, IDs/targets remained in original row order, and the input file hash was unchanged.

### Preregistered matched experiments A–F

| Variant | Features | Pooled OOF AUC | Delta vs A | Paired fold wins vs C |
|---|---|---:|---:|---:|
| A | Raw numeric | 0.744128 | +0.000000 | 0/5 |
| B | Raw numeric + raw categorical one-hot | 0.752227 | +0.008100 | 0/5 |
| C | B + approved rowwise features | 0.759619 | +0.015492 | Reference |
| D | C + eight ordinary one-hot product keys | 0.765270 | +0.021143 | 5/5 |
| E | C + eight fit-fold frequency-supported one-hot keys | 0.766038 | +0.021910 | 5/5 |
| F | C + three documented-constant-free numeric decompositions | 0.760269 | +0.016141 | 3/5 |

Per-fold AUCs (folds 1–5):

| Variant | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| A | 0.748568 | 0.750451 | 0.738331 | 0.740718 | 0.742627 |
| B | 0.757170 | 0.756321 | 0.749326 | 0.748822 | 0.749613 |
| C | 0.764250 | 0.764120 | 0.755967 | 0.756907 | 0.757009 |
| D | 0.769678 | 0.770549 | 0.759578 | 0.763679 | 0.763085 |
| E | 0.770858 | 0.770764 | 0.760359 | 0.764418 | 0.763915 |
| F | 0.765046 | 0.764047 | 0.755864 | 0.757903 | 0.758723 |

Paired fold delta in AUC versus C:

| Variant | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Positive folds |
|---|---:|---:|---:|---:|---:|---:|
| D | +0.005428 | +0.006429 | +0.003611 | +0.006772 | +0.006076 | 5/5 |
| E | +0.006608 | +0.006645 | +0.004392 | +0.007511 | +0.006906 | 5/5 |
| F | +0.000796 | -0.000073 | -0.000103 | +0.000997 | +0.001714 | 3/5 |

Paired stratified row-bootstrap intervals (1,000 resamples) are conditional on the fitted OOF predictions and previously inspected labels, not independent confirmation:

| Candidate | Reference | Delta AUC | 95% interval |
|---|---|---:|---:|
| B | A | +0.008100 | [+0.006634, +0.009524] |
| C | A | +0.015492 | [+0.013901, +0.017208] |
| D | A | +0.021143 | [+0.019241, +0.023264] |
| D | C | +0.005651 | [+0.004519, +0.006779] |
| E | A | +0.021910 | [+0.019847, +0.023835] |
| E | C | +0.006418 | [+0.005315, +0.007590] |
| F | A | +0.016141 | [+0.014570, +0.017846] |

The direct E−D supplement: D `0.765270199`, E `0.766037689`, difference `+0.000767490`, paired stratified bootstrap 95% interval `[-0.000033246, +0.001512418]` (1,000 resamples). Since the interval includes zero, frequency support did not establish a conclusive advantage over ordinary one-hot encoding. Secondary seed compares E with C, not E with D.

### Product-key formulas, corrected ablations, and category support

The eight rowwise keys in `build_product_keys` are: Q3/Q4/Q5 = annuity/credit rounded to 3/4/5 decimal places; AG3 = annuity/goods-price rounded to 3 places; P4/P5 = (annuity−130.5)/credit rounded to 4/5 places; KQ3/KQ4 = the cross of Q3/Q4 with `round(((credit/goods_price)-1)/0.0066)` for valid `Cash loans` where credit > goods price > 0. Invalid/missing values use explicit missing or invalid sentinels. These are formula descriptions only, not claims about real fees, rates, or product tiers.

An ablation seed mismatch was found after the primary report. The initial ablation table is retained for audit but is not used for conclusions. The corrected rerun uses the same fold-specific LightGBM seed, D-selected configuration, and D-selected iteration count for every key subset; only the key subset changes. These same-OOF-label results remain exploratory.

| Corrected variant | Pooled OOF AUC | Mean fold delta vs all eight | Folds better |
|---|---:|---:|---:|
| leave_out_KQ4 | 0.765127 | +0.000159 | 3/5 |
| leave_out_P5 | 0.765116 | +0.000134 | 3/5 |
| leave_out_Q5 | 0.765019 | +0.000035 | 2/5 |
| all_eight | 0.764983 | +0.000000 | 0/5 |
| preregistered_P4_Q4_AG3 | 0.764947 | -0.000028 | 1/5 |
| leave_out_AG3 | 0.764922 | -0.000047 | 2/5 |
| leave_out_Q4 | 0.764919 | -0.000046 | 2/5 |
| leave_out_Q3 | 0.764787 | -0.000180 | 3/5 |
| leave_out_KQ3 | 0.764678 | -0.000302 | 3/5 |
| leave_out_P4 | 0.764585 | -0.000395 | 2/5 |
| preregistered_P4_P5_Q4 | 0.764372 | -0.000604 | 2/5 |
| preregistered_Q3_Q4_Q5 | 0.763559 | -0.001407 | 1/5 |
| single_Q3 | 0.762941 | -0.002032 | 0/5 |
| single_P4 | 0.762668 | -0.002319 | 0/5 |
| single_P5 | 0.761895 | -0.003087 | 0/5 |
| single_Q4 | 0.761771 | -0.003219 | 0/5 |
| single_KQ3 | 0.761454 | -0.003530 | 0/5 |
| single_AG3 | 0.760666 | -0.004309 | 0/5 |
| single_Q5 | 0.760504 | -0.004466 | 0/5 |
| single_KQ4 | 0.760418 | -0.004548 | 0/5 |

No single key is consistently indispensable. All single-key models trail all-eight on all five folds. Leave-out differences are small; KQ4 and P5 leave-outs are nominally above all-eight by only `+0.000159` and `+0.000134`, each on 3/5 folds. Do not select or drop keys based on these pooled ranks alone.

Key-category counts observed in the hidden-screen training rows: Q3 96; Q4 797; Q5 5,102; KQ3 1,002; KQ4 3,576; P4 775; P5 4,079; AG3 131. Inner/outer category support is fit-fold-local; the complete per-fold support, singleton, and unseen-category counts are in `CATEGORY_SUPPORT_REPORT.csv`.

### Negative controls and secondary-seed robustness

Five frequency-preserving shuffled-key control repetitions (seeds 20261121–20261125) were evaluated on the same folds. Every control’s pooled AUC was below genuine D `0.765270`; control mean was `0.758043` (range `0.757859`–`0.758323`).

| Control repetition | Seed | Pooled OOF AUC | Genuine D minus control |
|---|---:|---:|---:|
| 1 | 20261121 | 0.757934 | +0.007336 |
| 2 | 20261122 | 0.758201 | +0.007069 |
| 3 | 20261123 | 0.757859 | +0.007411 |
| 4 | 20261124 | 0.758323 | +0.006947 |
| 5 | 20261125 | 0.757899 | +0.007371 |

All 25 control fold comparisons are below genuine D. This is a negative-control check on previously examined OOF rows, not untouched validation.

Secondary seed `20261114` on the same applicants and labels: C `0.759620`, selected E `0.765533`, E−C `+0.005913`; E won in all five folds. This is robustness to a second seed, not independent confirmation, and it does not establish E>D.

### Formula provenance and competition eligibility

- **130.5:** unresolved; the Phase 9 audit observed 6,792 of 27,255 adjacent annuity gaps (24.9%) equal to 130.5 inside exact cash-loan goods-price/credit groups. That is a repeated numerical gap in supplied training data, not verified evidence of a fee, recurring charge, repayment schedule, or contractual rule.
- **0.0066:** unresolved; no permitted-data derivation or official source was found. It must not be called a financing rate, tier, payment term, multiplier, or contract attribute.
- D/E intentionally reproduce historical keys to measure predictive lift; inclusion does not certify eligibility. F excludes both undocumented constants and uses only annuity/goods-price, credit−goods-price, and annuity−credit rowwise components.
- Raw supplied columns, deterministic rowwise transformations, and fit-fold categorical preprocessing use no test data. Specific engineered formulas have no separate organizer certification. P4/P5 and KQ3/KQ4 remain research-only/eligibility-pending. Frequency support inherits any source-formula limitations.
- Hidden-screen target-rate encoding is label-dependent and its rule eligibility is unresolved; it is not part of A–F and is not promotable. CatBoost CTRs were not used; their eligibility remains unresolved.
- Official rules were not interpreted beyond the local course/competition materials reviewed. No external sources were used to manufacture formula provenance.

### Separate hidden-key screen (exploratory; not Phase 13 label-free benchmark)

Completed run folder: `experiments\phase13_matched_baseline\runs\hidden_key_screen_20261105T091314Z`; UTC `2026-11-05T09:13:14.736994Z`–`2026-11-05T09:25:32.228107Z`. It read the Phase 13 C OOF predictions and the same 171,202 training rows. It screened 1,968 candidates, including numeric singles, pair sums/differences, both ratio orientations at 3/4/5 significant digits, smooth-bin controls, low-cardinality crosses, selected numeric-bin/category crosses, and the eight product keys as positive controls.

For each candidate, category-rate estimates for a held-out fold used labels from the other four folds only, with smoothing 30. Fixed blend weights 0.05, 0.10, and 0.20 were screened. However, the candidate scan adaptively reuses inspected labels; train-wide predictor quantile cut points were computed before cross-fitting for smooth/numeric-bin candidate definitions. Those bin boundaries are not fold-local. The result is a hypothesis-prioritization screen, not confirmatory nested evaluation, a second independent score, or an eligible feature representation.

Baseline C pooled OOF AUC: `0.759619220`. Best fixed 10% blend: `PRODUCT_KEY_P4`, target-rate OOF AUC `0.680344`, 775 categories, blended AUC `0.762727` (delta `+0.003107`). P5 at 10% yielded `0.762656` (`+0.003037`). The highest numeric-pair 10% result was annuity/credit at 3 significant digits: `0.762161` (`+0.002542`). The screen did **not** reproduce Claude’s quoted `+0.0049` for A/C. P4 at 20% reached `0.764171` (`+0.004551`), but relies on unresolved 130.5 and is an adaptively selected exploratory blend—not a candidate to promote.

Top numeric-pair candidates from the fixed 10% screen:

| Candidate | Categories | Target-rate AUC | Blend delta vs C |
|---|---:|---:|---:|
| annuity/credit, 3 sig digits | 661 | 0.674582 | +0.002542 |
| credit/annuity, 4 sig digits | 2,754 | 0.670257 | +0.002461 |
| credit/annuity, 3 sig digits | 409 | 0.673865 | +0.002399 |
| annuity/credit, 4 sig digits | 4,402 | 0.663644 | +0.002277 |
| credit/annuity, 5 sig digits | 11,348 | 0.652035 | +0.001949 |
| annuity/credit, 5 sig digits | 15,024 | 0.645304 | +0.001767 |
| annuity/goods price, 4 sig digits | 6,201 | 0.641536 | +0.001621 |
| goods price/annuity, 4 sig digits | 3,997 | 0.644822 | +0.001488 |
| annuity/goods price, 5 sig digits | 19,670 | 0.634276 | +0.001479 |
| goods price/annuity, 5 sig digits | 15,063 | 0.635922 | +0.001433 |

Forty frequency-preserving shuffled target-rate controls averaged `0.759534` (range `0.759367`–`0.759636`). Full candidate rows, all controls, and cross-fit rate vectors are retained in the run folder.

The follow-up weight supplement recomputed all three fixed blend weights for all eight explicit key controls (using saved cross-fit rates). These are descriptive checks on exposed OOF labels, not nested model selection:

| Key | Categories | AUC at 0.05 (delta) | AUC at 0.10 (delta) | AUC at 0.20 (delta) |
|---|---:|---:|---:|---:|
| Q3 | 96 | 0.760641 (+0.001021) | 0.761310 (+0.001691) | 0.761536 (+0.001917) |
| Q4 | 797 | 0.761074 (+0.001454) | 0.762158 (+0.002539) | 0.763154 (+0.003535) |
| Q5 | 5,102 | 0.760931 (+0.001311) | 0.761930 (+0.002310) | 0.762926 (+0.003307) |
| KQ3 | 1,002 | 0.760633 (+0.001013) | 0.761365 (+0.001746) | 0.761925 (+0.002306) |
| KQ4 | 3,576 | 0.760511 (+0.000892) | 0.761168 (+0.001548) | 0.761713 (+0.002094) |
| P4 | 775 | 0.761376 (+0.001756) | 0.762727 (+0.003107) | 0.764171 (+0.004551) |
| P5 | 4,079 | 0.761326 (+0.001707) | 0.762656 (+0.003037) | 0.764145 (+0.004525) |
| AG3 | 131 | 0.759965 (+0.000346) | 0.760046 (+0.000427) | 0.759280 (-0.000339) |

An earlier hidden-screen attempt in `runs\hidden_key_screen_20261105T091020Z` was stopped after 250 candidates when the initial generator was found to omit the inverse orientation of pairwise ratios. Its partial checkpoint and log are retained for audit and explicitly marked incomplete; it contributes no result claims. The completed bidirectional screen supersedes it.

### Deliverables and run locations

- Main Phase 13: `experiments\phase13_matched_baseline\runs\phase13_20261105T075946Z` — `PHASE13_EXECUTION_REPORT.md`, matched-baseline/incremental-value/fold/control/ablation CSVs, paired-bootstrap and integrity reports, category-support, formula provenance and eligibility, secondary-seed predictions/metrics, execution log, and `SHA256_MANIFEST.csv`.
- Same-seed ablation correction: `experiments\phase13_matched_baseline\runs\phase13_ablation_seed_matched_20261105T085332Z` — corrected ablation CSV/OOF, summary, category support, correction report, execution log, manifest.
- Direct E-vs-D bootstrap supplement: `experiments\phase13_matched_baseline\runs\phase13_comparison_supplement_20261105T093015Z`.
- Completed hidden-key screen: `experiments\phase13_matched_baseline\runs\hidden_key_screen_20261105T091314Z` — all 1,968 candidate rows, 40 null controls, product-key cross-fit rates, report, log, manifest.
- Product-key blend-weight supplement: `experiments\phase13_matched_baseline\runs\hidden_key_weight_supplement_20261105T093323Z`.
- Incomplete ratio-screen attempt: `experiments\phase13_matched_baseline\runs\hidden_key_screen_20261105T091020Z`, retained but excluded from conclusions.
- Run instructions and governance: `experiments\phase13_matched_baseline\README.md`.

### SHA-256 ledger for Phase 13 run artifacts

The following inventory records the output and code hashes from each run’s verified `SHA256_MANIFEST.csv`. Paths are relative to the run folder unless prefixed `source:`. The original-run source snapshots are preserved so the original manifests remain verifiable; corrected code hashes are separately recorded in the correction and supplement manifests.

#### Main matched benchmark — `phase13_20261105T075946Z`

| File | Bytes | SHA-256 |
|---|---:|---|
| `CATEGORY_SUPPORT_REPORT.csv` | 59,011 | `eaf6e4a68b751cf2ac7204664f4dd07562b085d2aec57c0b9cee73c505a2f994` |
| `COMPETITION_ELIGIBILITY_MATRIX.csv` | 2,399 | `f27b508b3f830f5f74cccafd390186d594773f2f3ae08690f72d3baec0182922` |
| `EXECUTION_LOG.txt` | 32,295 | `3f4cb1eae000d87c7fde097da12cf7a3f35221d64862a45c1bf1e1e279012a69` |
| `FOLD_LEVEL_METRICS.csv` | 14,316 | `f11d6a9c75e951e98a1becd87bbeb5a30b7a2ef9394dc7f23358998458196a74` |
| `FORMULA_PROVENANCE_AUDIT.md` | 2,379 | `102d33805a9f62f253fc781d23136d8278b84a33f4c48ae9b428179d7ecddc03` |
| `INNER_MODEL_SELECTION.csv` | 29,266 | `1b0bf576ef0eff885b8b2af4c7830a9f3d560b8af050295ab8a90a7725466bbc` |
| `MATCHED_BASELINE_RESULTS.csv` | 1,785 | `ce07519a22f26b07897197b0c4c683478cdf40df01a3c1d85550772dad55988a` |
| `NEGATIVE_CONTROL_RESULTS.csv` | 4,333 | `78dcc5972293bd74bb150e4e32b165b5bbf91600d4c9bb99a9e509b8f54be1ac` |
| `NEXT_PHASE_RECOMMENDATION.md` | 663 | `d4fd9071fd227f1aa267f2434ba55744a70b5cd2c8b6b65867b042899904f445` |
| `OOF_INTEGRITY_REPORT.md` | 668 | `6adba6f425deb5ff8eba607c718a3ed6efe763b24df91fcfbe3c66a59f14386a` |
| `OOF_PREDICTIONS.csv` | 109,975,561 | `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488` |
| `OUTER_MODEL_FITS.csv` | 7,003 | `c7423a3b457f755223149d732641d3724e4ae7dfc2cda8c6b96c78140346217c` |
| `PAIRED_BOOTSTRAP_REPORT.csv` | 1,648 | `1e8260366d7051efa2c5937e8845bb7526fb887204415874f817e0942b8aa4b1` |
| `PAIRED_BOOTSTRAP_REPORT.md` | 836 | `dcbf88e7b69ce1a312f94ec6370fed83c3aa61dcecf10a324d0243ab68096a2d` |
| `PHASE13_EXECUTION_REPORT.md` | 3,710 | `6956ed4beec41d389fdf746032be8bf5e55489a9a7f75855b49151377022c0ce` |
| `PRODUCT_KEY_ABLATION.csv` | 24,851 | `53a80175a0e9bb7d82c30e827c53cde53181414b84a36757d833edb1eed59a83` |
| `PRODUCT_KEY_INCREMENTAL_VALUE.csv` | 704 | `708faddd7401d5062850d3dd93cb24d8ac674d5e9eb356e7f33093aa82387ef5` |
| `SECONDARY_OUTER_MODEL_FITS.csv` | 2,526 | `7d90fe43071da07d93f5fd2f2720832c7320e74e1593e8b3834141c9a8d331ed` |
| `SECONDARY_SEED_OOF_PREDICTIONS.csv` | 9,929,318 | `0efc528d0aa749d9c9caeedcc6c2461e308f280294012d5fde8c5fe8cc5c35a9` |
| `SECONDARY_SEED_REPLICATION.csv` | 405 | `309843a73e6870fcee6d8f06772b608aa9a2cf15322f197ddb0fd6c2d19e8261` |
| `source:run_phase13.py` (original run snapshot) | 63,343 | `20cb779c685d697aebf68faea2cb07d17dd8d5c6d971f8540b1734d21033ee76` |
| `source:phase13_features.py` | 8,294 | `a54563ccfaa7bbcb758f28d0bfc9036bba784273740038b403e734504c0deb2a` |
| `source:test_phase13_features.py` | 4,296 | `1180b040351457c991b83d6c3b5b8c487be33ca0c3e4e20e8b1ba1893a7e692a` |

#### Same-seed ablation correction — `phase13_ablation_seed_matched_20261105T085332Z`

| File | Bytes | SHA-256 |
|---|---:|---|
| `ABLATION_SEED_MATCHED_CORRECTION.md` | 2,159 | `a8475f5b401fafcb9b43af7477788c305dbfe7dc9ea7d78e2c0028b7f0920c9c` |
| `ABLATION_SUMMARY.csv` | 1,296 | `7086aacb807607427e1a76f72ca3334f5d851eae71d06a16b42b3bbd00e86284` |
| `CATEGORY_SUPPORT_REPORT.csv` | 33,656 | `82a024df4083bfc04f0bc6c87bcb14f81cdcba0d18168612be52e97831eac473` |
| `EXECUTION_LOG.txt` | 8,350 | `3e5167ce8fbd19d8025ef94a13371ee75828152e1f7170c799c6dc964c5a0839` |
| `PRODUCT_KEY_ABLATION_SEED_MATCHED.csv` | 24,755 | `860d10ebbb21099cc5caa779eabf45e03fb7925e94d0e7328f42729f9d097294` |
| `PRODUCT_KEY_ABLATION_SEED_MATCHED_OOF.csv` | 72,040,258 | `98ee404a0a454da408c3c3d24c7b22a2e254400c56760d6084e1ce7d5086f639` |
| `source:rerun_phase13_ablations.py` | 10,105 | `c91a94460fb4e7d1b18e137d3d7c68401e8310d4413a10602156fa41c3dffbf1` |
| `source:run_phase13.py` (corrected runner) | 63,331 | `5f1f642f092e637aa1fc5e784675c0cba1082ff22f239871ccec2990af3b2316` |
| `source:phase13_features.py` | 8,294 | `a54563ccfaa7bbcb758f28d0bfc9036bba784273740038b403e734504c0deb2a` |

#### Direct E-vs-D bootstrap — `phase13_comparison_supplement_20261105T093015Z`

| File | Bytes | SHA-256 |
|---|---:|---|
| `EXECUTION_LOG.txt` | 175 | `8ab92e99fcbba5878e54872a46865622e02a35a929d78aa7989b9b8461aaa98c` |
| `FREQUENCY_VS_ORDINARY_PAIRED_BOOTSTRAP.csv` | 405 | `af3d37e546540be08d01d90b839cc933ea187d5e71f2e20f6f6d1b14f4b4b0ee` |
| `REPORT.md` | 945 | `2609556e9525e320032701fc9c751e5af883a36526ab7d76314a1d67652c2aa5` |
| `source:run_phase13_comparison_supplement.py` | 5,962 | `ad250615ea1a675d1a4b035021acfed8a9bbae6a382716eafa959d9d1ae97e6f` |
| `source:run_phase13.py` | 63,331 | `5f1f642f092e637aa1fc5e784675c0cba1082ff22f239871ccec2990af3b2316` |

#### Completed hidden-key screen — `hidden_key_screen_20261105T091314Z`

| File | Bytes | SHA-256 |
|---|---:|---|
| `candidate_scan_checkpoint.csv` | 397,635 | `ee5211b4987914bdb69daa3621bac1aeb4a11270e328106764d0078d0110d78a` |
| `EXECUTION_LOG.txt` | 1,312 | `d045bd2b4447d4501662a7f171f71c8ae05f7628841884345f3bf345ce2b856b` |
| `FREQUENCY_PRESERVING_NULLS.csv` | 8,538 | `61e787ea0a41da7d1148d7e6ad90faac877c598d95467cbebce37eaac52c11da` |
| `HIDDEN_KEY_CANDIDATE_RESULTS.csv` | 410,043 | `daff0d6cf684ddd46f32d8c7a7c7c40c60d81bf449f419d153d8c4c81343b862` |
| `HIDDEN_KEY_SCREEN_REPORT.md` | 3,456 | `64f1ab359f23983be3511352c6ec7e76b242d255f8540ff06afffafb1fa663b5` |
| `PRODUCT_KEY_CROSSFIT_RATE_OOF.csv` | 21,988,300 | `851b7c78ddd4ef6270d275d111daecc4cc0389291aa3175e61df565855cd6f5c` |
| `source:run_hidden_key_screen.py` (original run snapshot) | 14,319 | `038213f2a6add3014ae514a07876940251a7cd1831856a4919d93205ca472103` |
| `source:phase13_features.py` | 8,294 | `a54563ccfaa7bbcb758f28d0bfc9036bba784273740038b403e734504c0deb2a` |
| `source:phase11_candidate_search.py` | 11,278 | `fe2685f7f10163989ece07a3f5ca6403e7fffecf34e421beef75349fc9c3df2d` |

#### Product-key blend-weight supplement — `hidden_key_weight_supplement_20261105T093323Z`

| File | Bytes | SHA-256 |
|---|---:|---|
| `EXECUTION_LOG.txt` | 190 | `d241748ec4f60ae44f4fd3b04211d42551607dd3694b127eeb821daf95f95e96` |
| `PRODUCT_KEY_BLEND_WEIGHTS.csv` | 3,253 | `8fd69deeace6c58a9a4e2505ed7ca518b05ab915735f30cd54f9a7b1b1bc0cb8` |
| `REPORT.md` | 2,280 | `ae8a89093246a75517dcb3a73f405fd06f0f082c609aaa6ad0530743c68757c7` |
| `source:run_hidden_key_weight_supplement.py` | 6,114 | `75c0481ed8be4b61ebf96e86b7bd4269308a84c732870049365bb732c50610ec` |
| `source:run_hidden_key_screen.py` (current corrected runner) | 14,447 | `5ba3491d25abc8d601337164fc8678f6324fb3055f83dfcfff013e201c351f22` |
| `source:phase13_features.py` | 8,294 | `a54563ccfaa7bbcb758f28d0bfc9036bba784273740038b403e734504c0deb2a` |

#### Incomplete initial screen attempt (excluded from results)

`runs\hidden_key_screen_20261105T091020Z\INCOMPLETE_ATTEMPT.md` was corrected to record the final checkpoint count of 250 (the process advanced from 200 before stopping). The partial checkpoint has 250 rows and the log records the same count. The note’s SHA-256 after correction is `c6b2d2edcef46175cb6ced94af5c636041ba8deaccb4ca4ad05c4305f400013e`; checkpoint SHA-256 is `e5056c6f03c9d959b1a3021b07cecf5fd6b689fe1c136a4a9c48eaf7e3ce48d6`; execution-log SHA-256 is `52a7d35d20f4608c61a2ecc298dc3388ccd204124d5f8df0136443092dfb00d9`.

### Final Phase 13 answers

1. **True incremental key value over complete raw baseline:** D `+0.005651`; E `+0.006418` pooled OOF AUC, each 5/5 positive folds; bootstrap intervals against C are positive. These are exposed-label CV estimates, not independent confirmation.
2. **Does frequency support outperform ordinary one-hot?** Nominal E−D `+0.000767`, but the paired 95% interval includes zero and there is no secondary-seed E-vs-D comparison; inconclusive.
3. **Which keys are consistently useful?** The eight-key set adds value as a group. Corrected same-seed ablations do not identify a single indispensable key; individual rankings are weak and variable.
4. **Do genuine keys beat negative controls?** Yes: all five Phase 13 frequency-preserving shuffled controls and all their fold comparisons are below genuine D. The 40 hidden-screen nulls are near baseline.
5. **Which features have documented eligibility?** No model or specific engineered product formula is organizer-certified. Raw supplied features and deterministic rowwise arithmetic have general local-material support; 130.5 and 0.0066 remain undocumented and P4/P5/KQ3/KQ4 are eligibility-pending. Target-rate hidden screening is exploratory and unresolved.
6. **Which model proceeds?** No model is promoted or certified. E is the strongest matched label-free *research representation* and may be the candidate for a preregistered robustness test only after formula/rule eligibility review. It is not a champion change.

## Phase 14 — label-free model development and stronger-feature comparison

**Recorded:** 2026-11-05  
**Status:** Completed research run; target score not achieved. The frozen Phase 13 E result remains ahead of every new candidate.

### Objective and evidence boundary

Phase 14 followed the recommendation to preserve the Phase 13 complete baseline and all eight product keys, add carefully specified label-free numerical features/interactions, compare different model families, and evaluate a fixed blend only after OOF predictions were checked. The target was to exceed the Phase 9 historical reference `0.768240`; it was not reached.

This cannot be called independent confirmation. The entire supplied training cohort and labels have been examined in earlier OOF studies. Re-splitting the same rows—even with an untouched fold inside a new run—does not undo prior researcher exposure. No genuinely untouched labeled cohort or external organizer-approved evaluation was available. All Phase 14 results are further model-development evidence on exposed labels.

No hidden-screen target-rate feature was used. No test rows, sample submission, external data, holdout labels, submissions, or champion files were opened or used. No champion or submission was changed.

### Frozen inputs, features, and model protocol

- Training file: `D:\gci\inpuy\train.csv`; 171,202 rows, 13,821 positives; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Phase 13 fold assignment was reproduced exactly: five stratified folds, seed `20261113`, validation sizes 34,241 / 34,241 / 34,240 / 34,240 / 34,240. The saved Phase 13 OOF reference SHA-256 is `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`.
- The existing Phase 13 A–F formulas were frozen. All eight keys were retained in the all-key track. A separate constant-free four-key comparison used Q3/Q4/Q5/AG3 and excluded P4/P5/KQ3/KQ4, which depend on unresolved 130.5 and 0.0066 constants.
- New deterministic, rowwise enhancements were implemented in `phase14_features.py`: log1p of nonnegative financial amounts and age/clean employment days; income, credit and annuity per family member; employment-to-age ratio; count/mean/min/max/range/population standard deviation of observed EXT_SOURCE values; pairwise EXT_SOURCE products/differences; selected EXT_SOURCE-by-financial-burden and age-by-employment interactions; raw financial and external-score missingness counts; document-flag count/missing count; and bureau-request sum/missing count. No label statistics, pooled bin cut points, or test-fitted statistics define these features. Invalid and zero-denominator ratios become missing.
- LightGBM compared two fixed settings inside three inner stratified folds for each outer-fit partition, with 1,200 maximum iterations, 80-round early stopping, CPU deterministic fitting, and median inner-selected rounds for the outer refit.
- CatBoost used depth 6, learning rate 0.05, 1,200 maximum iterations, 80-round inner early stopping and a fixed refit. It used raw categorical values and product-key strings with `one_hot_max_size=10,000`; the runner asserted each fit partition's category count stayed below that threshold. The recorded maximum was 4,564. This prevents CatBoost's target-derived CTR path for these categorical features in this run. Fit-partition key frequencies were numeric rowwise support columns.
- The fixed equal blend averaged augmented LightGBM and CatBoost probabilities 50:50; no weights were fitted on OOF labels.
- Run folder: `experiments\phase14_label_free_model_development\runs\phase14_20261105T114015Z`; execution UTC `2026-11-05T11:40:15.602Z` to `2026-11-05T12:16:17.167Z`; runtime 36.0 minutes; peak process working set 1,000.63 MiB. Environment: Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1, LightGBM 4.7.0, CatBoost 1.2.10.

### Results

| Variant | Pooled OOF ROC-AUC | Delta vs Phase 13 E | Positive folds vs E |
|---|---:|---:|---:|
| Phase 13 E reference | 0.766038 | +0.000000 | — |
| Augmented LightGBM, all eight keys | 0.765719 | -0.000318 | 2/5 |
| Phase 13 D reference | 0.765270 | -0.000767 | — |
| Fixed 50:50 augmented LightGBM/CatBoost blend | 0.764886 | -0.001152 | 0/5 |
| Augmented LightGBM, constant-free four-key set | 0.764053 | -0.001984 | 0/5 |
| Phase 13 C reference | 0.759619 | -0.006418 | — |
| Augmented CatBoost, one-hot categorical handling | 0.759529 | -0.006508 | 0/5 |

Fold AUCs, folds 1–5:

| Variant | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Phase 13 E | 0.770858 | 0.770764 | 0.760359 | 0.764418 | 0.763915 |
| Augmented LightGBM, all eight | 0.771878 | 0.769193 | 0.760512 | 0.763941 | 0.763256 |
| Augmented CatBoost, all eight | 0.763587 | 0.765657 | 0.756165 | 0.754511 | 0.757918 |
| Augmented LightGBM, constant-free keys | 0.768199 | 0.770127 | 0.759574 | 0.761958 | 0.760541 |
| Fixed 50:50 blend | 0.770643 | 0.769349 | 0.760120 | 0.761569 | 0.762919 |

Paired 1,000-resample stratified bootstrap deltas versus Phase 13 E (conditional on fitted, previously exposed OOF predictions):

| Candidate | Delta AUC | Conditional 95% interval |
|---|---:|---:|
| Augmented LightGBM, all eight | -0.000318 | [-0.001154, +0.000563] |
| Augmented CatBoost, all eight | -0.006508 | [-0.007856, -0.005224] |
| Augmented LightGBM, constant-free keys | -0.001984 | [-0.002891, -0.001025] |
| Fixed 50:50 blend | -0.001152 | [-0.002068, -0.000258] |

The augmented LightGBM was the strongest *newly trained* model but was below Phase 13 E and improved only two of five folds. Its conditional interval versus E includes zero. The equal-weight blend underperformed in every fold, so no ensemble gain was established.

The Phase 9 `0.768240` score remains an historical reference, not a held-out score; the best new model is lower by about `0.002521`. Also, the ledger contains the higher raw Phase 2 research score `0.771103`, but that result used unresolved constants and supervised encodings and is not certified or submission-eligible. Do not describe Phase 9 as the highest raw score across the entire ledger.

### Subgroup diagnostics

Subgroups were evaluated per outer fold for gender, contract type, and age/income quartiles whose cut points were fit on that fold's outer training rows. Paired subgroup deltas versus Phase 13 E are saved in `SUBGROUP_STABILITY_PAIRED.csv` and summarized in `SUBGROUP_STABILITY_SUMMARY.csv`.

For augmented LightGBM, the mean of subgroup mean-fold deltas was `-0.000307` across 12 valid subgroup levels; 19 of 60 valid subgroup-fold contrasts were positive. Subgroup mean deltas ranged from `-0.001173` to `+0.001398`, so no broad, stable subgroup improvement emerged. These small, exposed-label subgroup analyses are descriptive, unadjusted for multiple comparisons, and not fairness certification.

### OOF, metric, test, and eligibility verification

`OOF_INTEGRITY_REPORT.md` records PASS: training/Phase 13/Phase 14 IDs, targets, row order and fixed fold IDs aligned; 171,202 rows received finite predictions; all seven saved OOF prediction columns were finite; recomputed pooled AUCs matched all 7 result rows within `1e-12`; recomputed fold AUCs matched all 35 fold rows within `1e-12`. The Phase 13 and Phase 14 feature tests passed (5 + 3), and all Phase 14 Python sources compiled. The CatBoost one-hot smoke test confirmed a 10,000 threshold and maximum observed smoke-test category count 1,711; full-run inner-fit maximum was 4,564.

No formula or model is organizer-certified. Raw supplied features and deterministic transformations have general support in the local course materials, but no ruling is claimed for any particular engineered key. P4/P5/KQ3/KQ4 remain unresolved because 130.5 is a repeated numerical gap rather than an established fee and 0.0066 has no verified financing interpretation. The four-key constant-free track avoids those two constants, but still does not constitute organizer certification. CatBoost CTRs were excluded by the checked one-hot threshold. Hidden-screen target-rate candidates were excluded entirely.

### Stopped preliminary attempts (preserved; not evidence)

Two expensive alternatives were stopped before full OOF predictions or metrics were produced:

1. `runs\phase14_20261105T095345Z`: preliminary two-model search stopped during the sparse XGBoost workload after only partial fold-1 LightGBM log lines. No predictions or score were saved. Preserved `INCOMPLETE_RUN.md`; exact runner snapshot `run_phase14_incomplete_initial.py`, SHA-256 `3378f31597a39487015e77512d8961faced8650295f1f0267d3d3e5fc9510325`; feature-builder-attempt hash `7b094f3166cebb93af53ebfc7c56d950f059252abe5a9eaca0e3e3bfe730f931`; execution-log hash `917d00d205e7fd0f665ad5bf1f71f1ebd88282f99d583a04ebe2b7d442881017`.
2. `runs\phase14_20261105T105450Z`: bounded XGBoost attempt stopped for the same sparse high-cardinality cost after one partial fold-1 LightGBM fit. No predictions or score were saved. Preserved `INCOMPLETE_RUN.md`; exact runner snapshot `run_phase14_xgb_attempt.py`, SHA-256 `e9c10119b18c8c6be2081a2cc4ba41eb643d8c93188d630aff3b612649ffa8ae`; feature-builder-attempt hash `7b094f3166cebb93af53ebfc7c56d950f059252abe5a9eaca0e3e3bfe730f931`; execution-log hash `a433c0090f5fb55785001ba368591222c4f177618f9f2c9ee16fa4a4895657d3`.

These attempts contain no complete OOF experiment and are excluded from all Phase 14 results.

### Deliverables and reproducibility hashes

- Source and tests: `experiments\phase14_label_free_model_development\run_phase14.py`, `phase14_features.py`, `test_phase14_features.py`, `verify_phase14_run.py`, and `README.md`.
- Main report, OOF, fold metrics, paired bootstrap, model choices, subgroup outputs, eligibility matrix, integrity report and execution log are under `experiments\phase14_label_free_model_development\runs\phase14_20261105T114015Z`.
- All 20 rows of that run's verified `SHA256_MANIFEST.csv` are recorded below. File paths are relative to the run folder unless prefixed `source:`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `ELIGIBILITY_MATRIX.csv` | 1,290 | `edf3772960d7d42a218d89ffb33c15a479d76785c87397439bdf759eb0b9d620` |
| `EXECUTION_LOG.txt` | 2,650 | `31680b30efc6a40c8094fe7b5462021d12a7fc14284079fbb0777961b4802d82` |
| `FOLD_LEVEL_METRICS.csv` | 2,404 | `13e279443123b319f9714d15b1078beae2ba9010a69845d876b7bc0c61326eca` |
| `INNER_MODEL_SELECTION.csv` | 6,777 | `aa302059ad59e23452fcf9b82055646f92f94b118f16fadbcbb44d216c89153d` |
| `MODEL_RESULTS.csv` | 1,536 | `12ba24a2c5d1b0e65ad713dc931229e0f1211f162e4c6687d66b8f568510d569` |
| `OOF_INTEGRITY_REPORT.md` | 1,285 | `e94269c452969ce194e6e2b5ac6aed9b67df60d8a3807200fd25b2f29bbe82b1` |
| `OOF_PREDICTIONS.csv` | 27,176,117 | `4c3e31495dc9da5615aacb1fd87fbc529f51ee8128c500b924037866a8b5bfca` |
| `OUTER_MODEL_FITS.csv` | 2,937 | `c1daea9550293b88754950383c39276e46ea7caaddb0131796350a3cf5ba3854` |
| `PAIRED_BOOTSTRAP.csv` | 804 | `aa4c55cd197b08f86f2896f5536e414a22add14dde88e649755667159193816b` |
| `PHASE14_EXECUTION_REPORT.md` | 6,097 | `8c4698586b082036b64e32e6c2c545de12d08cb52f58f4f0a8061bf7e65030d9` |
| `PREDICTION_CORRELATIONS.csv` | 1,256 | `aa7c14590b83a24090876fa77d3f04d7ba2e7fa4f6e806613b75196c6bb894fd` |
| `SUBGROUP_STABILITY.csv` | 34,061 | `c528d6e2e316dfa16d94a01fc27175a77d0f117264ebc84655efcad260ded7dd` |
| `SUBGROUP_STABILITY_PAIRED.csv` | 50,347 | `6e663ac78f339651a025048018621f2f492e099c101694c22ecf04d49dd59919` |
| `SUBGROUP_STABILITY_SUMMARY.csv` | 7,538 | `58405408c29299b6f5d8642b375f5f1c97181925147b70da55f34690aee539f6` |
| `source:run_phase14.py` | 39,804 | `fe014e934c5083f2248b6de1697c30de148d09b089d538890e686b6650664ba9` |
| `source:phase14_features.py` | 6,027 | `a1c02d0951ff495515bec7099ee058739a20529174e7efcd588c81bc6cd11d28` |
| `source:test_phase14_features.py` | 2,185 | `1a1d22fb74d8c3f00be231cadacf0ac4e51b527b944865a8b4aebb8922bd1e74` |
| `source:verify_phase14_run.py` | 8,802 | `76d09e7f4bc45795014e130a75d5df7bdab29e99259c4355ed001643f6bbaefc` |
| `source:phase13_features.py` | 8,294 | `a54563ccfaa7bbcb758f28d0bfc9036bba784273740038b403e734504c0deb2a` |
| `source:run_phase13.py` | 63,331 | `5f1f642f092e637aa1fc5e784675c0cba1082ff22f239871ccec2990af3b2316` |

### Phase 14 decision

1. **Did Phase 14 exceed 0.768240?** No. Best new model was augmented LightGBM at `0.765719`; Phase 13 E remained highest in this matched comparison at `0.766038`.
2. **Did the numerical enhancements or fixed blend improve Phase 13 E?** No conclusive lift: LightGBM all-key augmentation was `-0.000318` with a conditional interval including zero; constant-free LightGBM and the fixed blend were lower, and CatBoost was substantially lower.
3. **Did this establish an eligible champion?** No. Eligibility of specific keys remains unresolved; OOF labels were previously exposed; no untouched evaluation occurred.
4. **Next action:** Do not broaden the adaptive feature search or promote a model from these OOF scores. Resolve organizer/rule eligibility of the keys and constants. Obtain a genuinely independent labeled evaluation design if one exists, otherwise explicitly treat further same-dataset runs as exploratory and avoid calling them confirmation. No submission or champion change.

## Phase 15 — Forensic Recovery and Adaptive Model Research (2026-11-05)

**Status:** Completed research cycle; neither `0.78000` nor aspirational
`0.78500` was reached. No champion or submission was changed. All Phase 15
results are development evidence on repeatedly exposed training labels, not
independent confirmation.

### Executive findings

1. The Phase 13/14 score decrease was not a like-for-like regression.
   Phase 9/11 used GPU CatBoost with native categorical target statistics;
   Phase 13 used CPU LightGBM with fold-fitted one-hot features; Phase 14
   intentionally configured CatBoost to one-hot all categories and suppress
   CTRs. Phase 15 restored the native-category CatBoost regime.
2. The highest complete Phase 15 result was **0.768824708 pooled OOF AUC**:
   rowwise average of three native-category CatBoost OOF seed predictions.
   Component seed AUCs were `0.767510928`, `0.767946055`, and `0.767081070`.
   It is a useful unresolved Track B research result, not an eligible model,
   independent confirmation, or champion.
3. This three-seed OOF average is higher than Phase 9's reported *mean of
   three seed AUCs* (`0.768240`) but lower than the rowwise average of Phase
   9's three saved OOF vectors (`0.769839127`). These are different
   aggregations and should not be conflated.
4. No new feature direction delivered a statistically established lift:
   Phase 14 numeric enhancements and seven date-offset lifecycle features
   improved the matched base by `+0.000227` and `+0.000507`, respectively;
   paired intervals included zero. The isolated time-feature lift was
   `+0.000279637`, with 4/5 positive folds and a 300-resample interval
   `[-0.000252807, +0.000816712]`.
5. Same-label post-hoc blends reached `0.771833` (Phase 2/Phase 9 at 60/40)
   and `0.771659` (25% Phase 15 / 75% Phase 2), but the weights were not
   nested and Phase 2 has unresolved target-derived features and constants.
   These numbers are descriptive diagnostics only and cannot be treated as
   validated scores or used for promotion.

### Historical reconciliation and data scope

| Reference | OOF AUC | Interpretation |
|---|---:|---|
| Phase 9 CatBoost | `0.768240` reported | Mean of seed AUCs; GPU, native high-cardinality categories and eight keys. |
| Phase 9 saved-vector mean | `0.769839127` recomputed | Rowwise average of three aligned OOF vectors; exploratory aggregation. |
| Phase 11 CatBoost | `0.767399` | Independent feature builder, GPU native categoricals, CTR complexity 1. |
| Phase 12 frequency-supported OHE | `0.759816` | Earlier numeric-only baseline comparison; not comparable to Phase 11/15. |
| Phase 13 E | `0.766037689` | Complete baseline plus keys, CPU LightGBM, frequency-supported OHE. |
| Phase 14 CatBoost | `0.759529` | CTR-free, forced one-hot categories. |
| Phase 14 augmented LightGBM | `0.765719` | Deterministic numeric expansion on Phase 13 folds. |
| Phase 15 single-seed CatBoost, numeric + time | `0.767510928` | Matched five-fold GPU native-CTR CatBoost. |
| Phase 15 three-seed OOF average | `0.768824708` | Best complete Phase 15 research result; unresolved Track B. |
| Phase 2 supervised-feature blend | `0.771102653` | Nested target statistics plus unresolved constants; research-only. |

Training input was `D:\gci\inpuy\train.csv`: 171,202 rows, 13,821
positives, 32 predictors, 8.0729% positive rate. Its SHA-256 is
`b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
Related bureau/history/installment/POS/cash/credit-card tables were not
available in the permitted input; no external data was used.

### Phase 15 controlled runs and outcomes

The pre-training research map was written before model fitting and is
preserved at
[`experiments/phase15_autonomous_research/PHASE15_RESEARCH_MAP.md`](./experiments/phase15_autonomous_research/PHASE15_RESEARCH_MAP.md).
The main matched run reused Phase 13's fixed five stratified folds
(seed `20261113`; validation sizes 34,241 / 34,241 / 34,240 / 34,240 /
34,240). GPU CatBoost settings were fixed at 700 iterations, depth 6,
learning rate 0.05, `one_hot_max_size=10`, `max_ctr_complexity=1`, L2 5,
random strength 1, and border count 128. No outer-validation early stopping
or model selection was used.

| Experiment | Pooled OOF AUC | Delta / interval | Finding |
|---|---:|---|---|
| Native CatBoost base, all eight product keys | `0.767004319` | Reference | Baseline for Phase 15 matched ablation. |
| Add Phase 14 numeric block | `0.767231290` | `+0.000227`; paired 95% bootstrap `[-0.000505, +0.001031]` | No established lift. |
| Add numeric block plus seven lifecycle features | `0.767510928` | `+0.000507` vs base; paired 95% bootstrap `[-0.000276, +0.001362]` | No established lift. |
| Time block isolated over numeric block | `+0.000279637` | 4/5 folds positive; 300-resample paired interval `[-0.000252807, +0.000816712]` | Small, uncertain. |
| Three-seed rowwise OOF average | `0.768824708` | Component scores `0.767510928`, `0.767946055`, `0.767081070` | Robustness across seeds, not independent confirmation. |
| Organization × external-score quartile, full OOF | `0.767641747` vs matched time baseline `0.767596343` | `+0.000045404`; 3/5 folds positive; paired 400-resample interval `[-0.000490, +0.000536]` | No useful gain. |
| CatBoost CTR complexity 2, one matched fold | `0.772544244` vs complexity 1 `0.772874781` | `-0.000330537`; 56.8s vs about 20.7s | Rejected: slower and lower in screen. |
| XGBoost native categorical screen, no keys | `0.737657993` | 30,000 rows, 24,000 fit / 6,000 validation | No full OOF run justified. |
| XGBoost screen, four constant-free keys | `0.723069890` | Same sample screen | No full OOF run justified. |
| Applicant ID alone | `0.502896563` | ID equals source row index `0..171201`; fold AUCs `0.507718`, `0.501050`, `0.494498`, `0.505169`, `0.506125` | No stable signal; retain ID for alignment only. |

The native CatBoost one-fold preflight scored `0.772628196` for numeric plus
time features; this was a screening fold, not a full OOF result. The failed
first preflight
(`phase15_native_catboost_preflight_20261105T123915Z`) stopped before model
fit on a feature-assembly error and is retained as incomplete.

### Exploratory blend and eligibility boundaries

An 18-row fixed blend grid was recalculated from saved OOF predictions.
Examples: Phase 15/Phase 9 at 50/50 scored `0.769896852`; Phase 15/Phase 9
at 25/75 scored `0.770006961`; Phase 15/Phase 2 at 25/75 scored
`0.771659151`; Phase 2/Phase 9 at 60/40 scored `0.7718327`. All are
post-hoc OOF diagnostics, not nested blend validation. Prediction Spearman
correlations were `0.990869` between the Phase 15 three-seed average and
Phase 9 mean, and `0.974369` between Phase 15 and Phase 2.

All eight product keys include unresolved hypotheses using constants 130.5
and 0.0066. Native CatBoost uses fold-training target statistics internally;
competition-rule eligibility for these formulas and CTRs remains unresolved.
Phase 2 additionally uses nested target-derived encodings and constants
129.5/0.0132, also unresolved. Therefore every Phase 15 CatBoost result and
Phase 2 blend remains in Track B/research-only. No feature or model is
certified competition-eligible merely because it derives from supplied
columns.

### Reproducibility, integrity, resources, and artifacts

- Independent verifier recalculated all saved Phase 15 AUCs and checked row
  IDs, targets, alignment, finite predictions, fold coverage, and run hashes.
  All full OOF files have 171,202 aligned rows.
- Artifact registry contains 95 entries: 92 manifest hashes passed, two
  stale execution-log hashes were recorded with corrections, and one failed
  preflight had no manifest because it failed before training.
- Stale log corrections: preflight `...T124003Z`, declared
  `a281db1c89eab28f76f633e6e6e6ce0167416887262ed40ae5f47b953bb438d9`,
  actual `c8796fca164964aa78b06480caff0133e10dfab1fe70448650fead437cada737`;
  full OOF `...T124238Z`, declared
  `d0be435c5230e0a7c0bd121bb7919576a47f69c42b334e06cd4c49a6c0805aa7`,
  actual `e4de77fab78be4d9e541c7aff0dfa9f1c8bf3b6feaefa4c8b28e95a40b2d6fbd`.
  Original manifests were preserved; supplemental corrections are in
  `PHASE15_MANIFEST_CORRECTIONS.csv`.
- Six targeted unit tests passed. Main CatBoost peak process working set was
  about 902 MiB; GPU was RTX 4050 Laptop with 6,141 MiB. Full three-variant
  five-fold run took about 12 minutes including bootstrap work.
- No test rows, sample submission, external data, submissions, or champion
  files were used or changed.

Implementation and diagnostics source include `phase15_features.py`,
`phase15_interactions.py`, all seven `run_phase15_*.py` runners,
`test_phase15_features.py`, `verify_phase15_artifacts.py`,
`generate_phase15_diagnostics.py`, and `build_phase15_final_manifest.py`.
Primary outputs include the execution report, pre-training research map,
OOF files, model results, fold metrics, manifest registry/corrections,
independent AUC verification, fixed blend diagnostics, prediction
correlations, ID-order diagnostics, and paired bootstrap JSON. The full
execution report is
[`experiments/phase15_autonomous_research/PHASE15_EXECUTION_REPORT.md`](./experiments/phase15_autonomous_research/PHASE15_EXECUTION_REPORT.md).

The final 33-file SHA-256 index is
[`experiments/phase15_autonomous_research/PHASE15_FINAL_SHA256_MANIFEST.csv`](./experiments/phase15_autonomous_research/PHASE15_FINAL_SHA256_MANIFEST.csv),
SHA-256
`f3adb40e27fe27823ab20de8fcd7be9701e7d41b112ff0e682b6f6cc7591dfc8`.
It includes report/map/source/diagnostic hashes, selected OOF files and
manifests, training data, and historical OOF inputs. Key documentation
hashes: report
`20e304680cc801a0c19b0b8935201eec20235158ea939880a08268a12f96e0db`;
research map
`75ac5ea3cf664820fe51a56e740242bbb5df4484e83885ae4746085b9558f86a`.

### Phase 15 decision

Carry the three-seed native-category CatBoost only as an unresolved Track B
research reference; do not promote it. For the conservative label-free
track, Phase 13 E (`0.766038`) remains the best established comparison in
this set. Phase 15 recovered modeling-regime performance but did not
establish a route to `0.78000` or `0.78500`. Further same-table OOF searches
would be exploratory; do not claim independent confirmation without an
untouched labeled evaluation or permitted new data. No champion or
submission change.

## Phase 16 — Nested CatBoost Parameter Selection (2026-11-05)

**Status:** Completed. Parameter-tuning research did not materially improve
the matched OOF result. No champion or submission was changed.

The Phase 15 best complete research OOF remains `0.768824708` (three-seed
average). Phase 16 tested if nested parameter selection would improve the
same-fold Phase 15 single-seed result without selecting settings on the
outer validation labels.

### Frozen experimental design

- Input: `D:\gci\inpuy\train.csv`, 171,202 rows and 13,821 positive targets.
- Five stratified outer folds frozen to Phase 13 seed `20261113`.
- Three inner folds per outer training partition; each of four fixed
  configurations received inner OOF predictions over the complete outer
  training partition. The maximum pooled inner OOF AUC configuration was
  refit on the outer training rows and used to predict only that outer
  validation fold.
- Features were unchanged: Phase 13 raw, rowwise and all-eight-key features,
  Phase 14 deterministic numeric enhancements, and seven Phase 15 date-offset
  features.
- GPU CatBoost used Logloss, AUC reporting metric, `one_hot_max_size=10`,
  `max_ctr_complexity=1`, and 128 borders. No outer-validation early stopping
  was used.
- Candidate settings were preregistered in
  [`experiments/phase16_nested_catboost_research/PHASE16_PROTOCOL.md`](./experiments/phase16_nested_catboost_research/PHASE16_PROTOCOL.md):
  historical depth 6 / LR 0.05 / 700 iterations / L2 5 / random strength 1;
  depth 5 / LR 0.05 / 850 / L2 5 / random strength 1;
  depth 7 / LR 0.04 / 800 / L2 8 / random strength 1; and
  depth 6 / LR 0.035 / 1,000 / L2 8 / random strength 0.5.

### Results

| Metric | Phase 16 nested tuning | Matched Phase 15 primary OOF |
|---|---:|---:|
| Pooled OOF ROC-AUC | `0.767652809` | `0.767510928` |
| Paired pooled delta | `+0.000141881` | — |
| Paired 95% bootstrap interval | `[-0.000442134, +0.000659163]` | — |
| Positive outer-fold deltas | `4/5` | — |

| Fold | Inner-selected configuration | Phase 16 AUC | Phase 15 AUC | Delta |
|---:|---|---:|---:|---:|
| 1 | depth 5, LR 0.05, 850 iterations | 0.773427082 | 0.772874781 | +0.000552301 |
| 2 | depth 6, LR 0.035, 1,000 iterations, L2 8 | 0.775154537 | 0.774919672 | +0.000234864 |
| 3 | depth 6, LR 0.05, 700 iterations | 0.760985652 | 0.761288394 | -0.000302743 |
| 4 | depth 6, LR 0.035, 1,000 iterations, L2 8 | 0.765225817 | 0.765141478 | +0.000084339 |
| 5 | depth 7, LR 0.04, 800 iterations, L2 8 | 0.763926674 | 0.763401437 | +0.000525238 |

The paired interval includes zero, pooled gain is negligible, and four
different fold choices were selected. Parameter tuning on this grid is not
a supported route to `0.78000` or `0.78500`; no configuration was promoted.

### Validation and governance

The independent Phase 16 verifier passed: it recalculated pooled and all
five fold AUCs, checked all 171,202 row/ID/target alignments and frozen fold
IDs, verified the chosen configuration won on inner OOF scores, confirmed
finite predictions and input hashes, and validated all 13 run-manifest
entries. Runtime was 1,427 seconds. No test rows, external data, submissions,
or champion files were used or changed.

Phase 16 remains Track B research-only: all-eight product keys use unresolved
130.5 and 0.0066 hypotheses, and CatBoost native CTR eligibility remains
unresolved. Local input inventory contains only the supplied train/test and
sample-submission CSVs; no related bureau/application-history/installment/
POS/cash/credit-card tables are available. The local rules prohibit external
data, so those missing relational histories cannot be introduced as a route
to higher scores.

Full results and artifacts:
[`experiments/phase16_nested_catboost_research/PHASE16_EXECUTION_REPORT.md`](./experiments/phase16_nested_catboost_research/PHASE16_EXECUTION_REPORT.md),
[`experiments/phase16_nested_catboost_research/PHASE16_PROTOCOL.md`](./experiments/phase16_nested_catboost_research/PHASE16_PROTOCOL.md),
[`experiments/phase16_nested_catboost_research/run_phase16_nested_tuning.py`](./experiments/phase16_nested_catboost_research/run_phase16_nested_tuning.py),
and
[`experiments/phase16_nested_catboost_research/verify_phase16_run.py`](./experiments/phase16_nested_catboost_research/verify_phase16_run.py).
Run output folder:
`experiments/phase16_nested_catboost_research/runs/phase16_nested_catboost_20261105T132953Z`.

**Decision:** retain Phase 16 for auditability; do not promote. Next, if
continuing model development, isolate nested target statistics on the
constant-free Q3/Q4/Q5/AG3 keys against a matched complete raw-feature
baseline. Keep that method in a separate research track pending rule
clarification.

## Phase 17 — Nested Constant-Free Product-Key Target Statistics (2026-11-05)

**Status:** Completed. A small, uncertain incremental lift over its matched
four-key baseline was found, but it did not beat the strongest existing
research model. No champion or submission was changed.

### Frozen method

- Train only, five stratified outer folds with seed `20261113`.
- Constant-free product keys only: Q3, Q4, Q5, AG3. The 130.5 fee-offset and
  0.0066 financing-step key hypotheses were excluded.
- Matched baseline: supplied raw and rowwise features; the four keys with
  fold-fitted one-hot and fit-fold frequency support.
- Treatment: eight additional columns, one smoothed target rate and log count
  for each key; alpha `30.0`, fixed from Phase 8.
- Training-row target encodings were five-fold inner OOF inside each outer
  fit partition. Validation encodings used outer-fit labels only. Model
  stopping was selected on a baseline-only 10% partition inside each outer
  fit and the same selected iteration count/seed was used for both variants.

### Results

| Variant | Pooled OOF ROC-AUC | Delta vs matched four-key baseline |
|---|---:|---:|
| Constant-free one-hot/frequency baseline | `0.764330449` | — |
| Nested target rates + log counts | `0.765290753` | `+0.000960304` |

Paired 1,000-resample 95% interval: `[-0.000065046, +0.002013920]`;
positive fold deltas: `3/5`. Fold deltas were `+0.002486609`,
`+0.002549605`, `-0.000860245`, `+0.001838042`, and `-0.001071050`.
Thus target statistics showed mixed fold behavior and a confidence interval
including zero; they did not exceed Phase 13 E (`0.766037689`) or the Phase
15 three-seed CatBoost OOF average (`0.768824708`).

A post-hoc same-fold blend diagnostic was calculated because Phase 15
CatBoost, Phase 16 tuned CatBoost, and Phase 17 LightGBM all use the same
outer folds. Their equal three-model mixture scored `0.769061967`; correlations
were 0.990237, 0.954683, and 0.956839 for the Phase 15/16, 15/17, and 16/17
pairs, respectively. This is post-hoc, same-label blend exploration—not
nested validation, not promotion evidence, and not an eligible model.

### Verification and compliance

The Phase 17 independent verifier passed: all 171,202 row/ID/target/fold
alignments, pooled and fold AUCs, paired bootstrap interval, and all nine
run-manifest hashes checked. Runtime was 300 seconds.

Although no undocumented constant was used, target rates are label-derived
and their competition eligibility remains unresolved. Phase 17 is research
only. No test rows, external data, submissions, or champion files were used
or changed.

Full details and code:
[`experiments/phase17_nested_constant_free_target_encoding/PHASE17_EXECUTION_REPORT.md`](./experiments/phase17_nested_constant_free_target_encoding/PHASE17_EXECUTION_REPORT.md),
[`experiments/phase17_nested_constant_free_target_encoding/PHASE17_PROTOCOL.md`](./experiments/phase17_nested_constant_free_target_encoding/PHASE17_PROTOCOL.md),
[`experiments/phase17_nested_constant_free_target_encoding/run_phase17_nested_te.py`](./experiments/phase17_nested_constant_free_target_encoding/run_phase17_nested_te.py),
[`experiments/phase17_nested_constant_free_target_encoding/verify_phase17_run.py`](./experiments/phase17_nested_constant_free_target_encoding/verify_phase17_run.py),
and
[`experiments/phase17_nested_constant_free_target_encoding/diagnose_matched_oof_blends.py`](./experiments/phase17_nested_constant_free_target_encoding/diagnose_matched_oof_blends.py).
Outputs are under
`experiments/phase17_nested_constant_free_target_encoding/runs/phase17_nested_te_20261105T140620Z`.

**Decision:** do not promote. The measured best same-fold blend is an
exploratory diagnostic only. The next experiment should use inner OOF
predictions to select blend weights inside outer training folds, then
evaluate once on the held-out outer folds.

## Phase 18 — Nested CatBoost / Target-Encoded LightGBM Blend (2026-11-05)

**Status:** Completed and independently verified; research-only. The blend
improved on both single-seed components, but did not exceed the Phase 15
three-seed OOF average. No champion or submission was changed.

### Frozen experiment and code

- Train only, 171,202 rows; five stratified outer folds with seed `20261113`.
- Inside each outer training partition, fit three-fold inner OOF predictions
  for GPU CatBoost and Phase 17 LightGBM. The LightGBM stopping count was
  selected on an internal split within each inner-fit partition.
- Select a CatBoost weight from `{0.00, 0.25, 0.50, 0.75, 1.00}` by pooled
  inner OOF AUC; apply it to the matched Phase 15 and Phase 17 outer OOF
  component predictions.
- Selected CatBoost weights by fold: `0.50, 0.75, 0.50, 0.75, 0.75`.
- Phase 15 contributes native categorical CTRs and eight product keys,
  including unresolved formula hypotheses. Phase 17 contributes nested,
  label-derived target rates and counts on the constant-free Q3/Q4/Q5/AG3
  keys.

### Results

| Model/vector | Pooled OOF ROC-AUC | Delta vs Phase 15 single seed |
|---|---:|---:|
| Phase 15 single-seed CatBoost | `0.767510928` | — |
| Phase 17 nested target-encoded LightGBM | `0.765290753` | `-0.002220175` |
| Phase 18 nested blend | `0.768668866` | `+0.001157938` |
| Phase 15 three-seed OOF average (prior best complete run) | `0.768824708` | `+0.001313780` |

Paired, class-stratified 1,000-resample 95% bootstrap intervals for the
Phase 18 delta:

- Versus Phase 15 single-seed CatBoost: `[+0.000690214, +0.001589754]`.
- Versus Phase 17 target-encoded LightGBM:
  `[+0.002590654, +0.004231289]`.
- Outer-fold blend deltas vs Phase 15 were positive on `5/5` folds; see
  report for per-fold results.

The blend is `0.000155842` below the earlier Phase 15 three-seed OOF
average, so it does not set a new best score. These are exploratory
comparisons on extensively studied labels, not independent confirmation.
There is no evidence here of reaching `0.785`.

### Verification, eligibility, and disposition

The independent Phase 18 verifier passed all 171,202 row/ID/target/fold
alignments; recomputed pooled/fold AUCs and both paired bootstrap intervals;
checked all five selected weights against the complete 25-candidate inner
grid; and validated all 16 run-manifest entries. Training CSV SHA-256:
`b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
Runtime was `723.35` seconds. No test rows, external data, submissions, or
champion files were used or changed.

The method remains research-only. Phase 15 uses native-category CTRs and
product-key hypotheses with unresolved formula provenance; Phase 17 uses
label-derived statistics with unresolved competition-rule status. Do not
promote this blend or treat its OOF AUC as a competition-ready score. Stop
same-label blend tuning; future claims require a predeclared untouched
evaluation set or genuinely independent permitted data.

Details and implementation:
[`experiments/phase18_nested_oof_ensemble/PHASE18_EXECUTION_REPORT.md`](./experiments/phase18_nested_oof_ensemble/PHASE18_EXECUTION_REPORT.md),
[`experiments/phase18_nested_oof_ensemble/PHASE18_PROTOCOL.md`](./experiments/phase18_nested_oof_ensemble/PHASE18_PROTOCOL.md),
[`experiments/phase18_nested_oof_ensemble/run_phase18_nested_blend.py`](./experiments/phase18_nested_oof_ensemble/run_phase18_nested_blend.py),
and
[`experiments/phase18_nested_oof_ensemble/verify_phase18_run.py`](./experiments/phase18_nested_oof_ensemble/verify_phase18_run.py).
Run artifacts:
`experiments/phase18_nested_oof_ensemble/runs/phase18_nested_blend_20261105T142048Z`.

## Phase 19 continuation — matched GPU feature screens (2026-11-06)

**Status:** Two additional feature screens completed on the Phase 15 primary
five-fold partition. Neither produced a material improvement. These are
exploratory comparisons on repeatedly examined labels; do not promote either
candidate.

### Frozen reference and validation

- Training rows: `171,202`; target rows and IDs checked against the saved
  Phase 15 multiseed OOF file.
- Reference: `phase15_primary_seed_20261113`, pooled OOF AUC `0.767510928`.
- Five stratified folds, seed `20261113`; saved reference fold IDs were
  recomputed and matched before fitting.
- Each candidate reused the Phase 15 raw, rowwise, numeric, time, and
  eight-product-key feature blocks. Product-key formula provenance and native
  CatBoost CTR eligibility remain unresolved; Track B/research-only.
- GPU CatBoost settings matched the Phase 15 primary run: 700 iterations,
  depth 6, learning rate `0.05`, `one_hot_max_size=10`,
  `max_ctr_complexity=1`, L2 `5`, random strength `1`, border count `128`.
  Model seed per outer fold: `20261113 + 100 * fold`. No outer-fold target
  was used for fitting or stopping.

### GPU candidate results

| Candidate | Added rowwise features | Pooled OOF AUC | Delta vs Phase 15 primary | Five fold AUCs |
|---|---|---:|---:|---|
| Phase 15 primary reference | None | `0.767510928` | — | `0.772875, 0.774920, 0.761288, 0.765141, 0.763401` |
| External-score polynomial + payment burden | For each EXT_SOURCE: square, cube, and score x annuity/income | `0.767589601` | `+0.000078673` | `0.772775, 0.775079, 0.761825, 0.765191, 0.763207` |
| External-score lifecycle/credit interactions | For each EXT_SOURCE: score x age, score x employment/age, and score x credit/income | `0.767506683` | `-0.000004245` | `0.773149, 0.774411, 0.761464, 0.764929, 0.763640` |

Paired fold deltas against the reference, in fold order:

- Polynomial/payment-burden candidate:
  `-0.000100, +0.000159, +0.000537, +0.000049, -0.000195` (3/5 positive).
- Lifecycle/credit candidate:
  `+0.000275, -0.000509, +0.000176, -0.000212, +0.000239` (3/5 positive).

Both experiments ran five GPU fits and took `57.6` and `55.2` seconds,
respectively. The first pooled gain is negligible, is not consistent across
folds, and is far below the Phase 15 three-seed OOF score. The second is
effectively unchanged. Neither is evidence of a new signal or an independent
generalization gain.

### Earlier exploratory CPU screen noted for completeness

Before the explicit GPU-only direction, a scratch CPU screen tested raw
feature blocks with broad financial ratios, product keys, and Phase 14 numeric
enhancements using five folds (seed `20261113`). Its exploratory CatBoost
results were `0.758993` for the raw-block baseline, `0.758690` after adding
the broad raw/financial block, `0.766962` after adding product keys, and
`0.766636` after also adding Phase 14 enhancements. The corresponding
LightGBM pooled results were `0.753009`, `0.753703`, `0.729063`, and
`0.730393`. This screen used reduced, different model settings and is not
comparable to the Phase 15 reference; it is not GPU evidence and is not used
to select a candidate. The GPU-matched tests above are the relevant follow-up.

### Reproducibility and disposition

The scratch probes validated row/ID/target/fold alignment and kept
predictions in memory; no prediction files, NPZ files, or experiment-run
directories were created. Consequently there is no candidate prediction
artifact or SHA-256 manifest for these probes. Formula and parameter details
are recorded here, but the in-memory runs are not independently replayable
from a saved standalone runner. No test data, external data, submission, or
champion was used or changed.

The best completed Phase 15 three-seed result remains `0.768824708`; the
retrospective Phase 9 three-seed rowwise average remains `0.769839` and is
not an independent or directly matched confirmation. Neither target `0.780`
nor `0.785` was reached.

## Phase 20 — Feature-routed GPU expert pilot (2026-11-06)

**Status:** Three predeclared, feature-only two-expert routing rules tested
with GPU CatBoost on the same five outer folds. All three underperformed the
global Phase 15 primary model; reject these routed systems. This is a
bounded feasibility pilot, not the requested 150-variant search and not
independent confirmation.

### Protocol

- Gate rules use only inference-time supplied features, never `TARGET`:
  `NAME_CONTRACT_TYPE`; `DAYS_EMPLOYED == 365243` versus recorded employment;
  and all three `EXT_SOURCE` values observed versus at least one missing.
- Group sizes were respectively `154,988/16,214`,
  `140,304/30,898`, and `36,164/135,038`; every train and validation branch
  exceeded the predeclared 5,000/1,000-row feasibility floors.
- Each outer fold trained one CatBoost model per gate group, then routed each
  held-out row to the model determined by its feature-only group. This was
  30 GPU fits total (three rules x two experts x five folds).
- The Phase 15 reference fold/ID/target alignment was rechecked. The expert
  models used the same Phase 15 feature blocks and GPU configuration listed
  above; no validation labels were used in training.

### Results

| Feature-only routing rule | Routed pooled OOF AUC | Delta vs global Phase 15 primary | Routed fold AUCs | Paired fold deltas vs global |
|---|---:|---:|---|---|
| Contract type | `0.765035164` | `-0.002475763` | `0.769627, 0.772275, 0.758265, 0.762225, 0.762937` | `-0.003248, -0.002645, -0.003023, -0.002917, -0.000464` |
| Employment sentinel | `0.764849931` | `-0.002660996` | `0.769237, 0.772419, 0.759208, 0.762362, 0.761562` | `-0.003638, -0.002500, -0.002081, -0.002780, -0.001840` |
| External-score completeness | `0.763804674` | `-0.003706253` | `0.767712, 0.773321, 0.759075, 0.759857, 0.759398` | `-0.005163, -0.001599, -0.002213, -0.005285, -0.004003` |
| Global Phase 15 reference | `0.767510928` | — | `0.772875, 0.774920, 0.761288, 0.765141, 0.763401` | — |

All 15 routed fold comparisons lost to the global reference. Runtime was
`618.8` seconds on the available RTX 4050 GPU. The result indicates that
these coarse groups do not justify separate CatBoost experts under the tested
settings; it does not rule out every possible conditional model.

### Decision and follow-up

Do not promote or blend these routed predictions. A 150-way search against
these already-examined labels would increase winner's-curse/selection bias,
not create 150 independent confirmations. Any further conditional-model
work should use a small predeclared set of meaningful gates and nested
selection inside training folds, with a final evaluation reserved from
selection. Keep all candidates research-only: the shared Phase 15 keys and
native CTRs retain unresolved eligibility/provenance, and no untouched
labeled holdout is available. No submission or champion was changed.

## Phase 19 — unified OOF recovery and audit completed (2026-11-06)

**Status:** Completed from saved OOF evidence; no new model training was
needed. All outputs remain exploratory, and no champion or submission was
changed.

- Audited the saved OOF sources for Phases 9, 11, 13, 15, 16, 17, 18, and 19.
  The unified matrix contains 15 prediction vectors; native fold comparisons
  are marked and only use the matched Phase 15/13 fold partition where valid.
- Independently verified all 171,202 Phase 19 MLP prediction rows, row order,
  applicant IDs, target alignment, fold assignments, finite prediction
  coverage, pooled AUC, and five fold AUCs. Recomputed MLP pooled AUC:
  `0.742694381201`; fold AUCs:
  `0.742212424, 0.751729335, 0.739719684, 0.744159638, 0.736261729`.
- The existing MLP run was historical CPU training (`gpu_available=false`,
  493.03 seconds). It was not rerun or represented as a GPU experiment.
- Unified score position remains Phase 9 retrospective rowwise mean
  predictions `0.769839127` (not independent confirmation), Phase 15
  three-seed average `0.768824708`, and Phase 18 nested blend `0.768668866`.
  The highest conservative no-custom-product-key reference remains Phase 13 C
  at `0.759619220`; eligibility remains subject to organizer interpretation.
- Found and fixed a Phase 19 report-generation bug: empty seed-mean fields
  were treated as numeric during Markdown formatting. The corrected report
  generator now handles missing values explicitly.
- Independently checked all 21 entries in the Phase 19 final manifest; all
  SHA-256 values match.

Deliverables:
[`experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_EXECUTION_REPORT.md`](./experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_EXECUTION_REPORT.md),
[`experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_MODEL_COMPARISON.csv`](./experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_MODEL_COMPARISON.csv),
[`experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_FOLD_METRICS.csv`](./experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_FOLD_METRICS.csv),
and
[`experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_FINAL_SHA256_MANIFEST.csv`](./experiments/phase19_unified_model_research/runs/phase19_completed_audit_20261106/PHASE19_FINAL_SHA256_MANIFEST.csv).
The corrected audit implementation is
[`experiments/phase19_unified_model_research/run_phase19_unified_audit.py`](./experiments/phase19_unified_model_research/run_phase19_unified_audit.py).

## Phase 20 — segmentation study preregistered (2026-11-06)

**Status:** Registry frozen and code/unit/GPU preflight checks completed;
Stage 1 model experiments have not yet been executed.

- Preregistered exactly 150 distinct experiment IDs: 15 segmentation
  families x 10 non-seed-only CatBoost model/feature configurations; also
  preregistered 30 placebo controls (three matched model configurations per
  configuration family, random 50/70/90% partitions or ID parity).
- Discovery/holdback contract: fixed stratified 60/40 split, seed `20261106`;
  five discovery folds, seed `20261113`. Because all labels have prior phase
  exposure, the 40% holdback is not an untouched confirmation set.
- Stage 1 is predeclared as R2 fixed 50/50 global plus routed-expert
  probabilities, with R1 hard routing reported diagnostically. The two
  learned-gate definitions use cross-fitted GPU CatBoost membership gates.
- All candidate model fits explicitly use CatBoost `task_type="GPU"`,
  `one_hot_max_size=256` (the installed CatBoost maximum), `max_ctr_complexity=1`,
  and no early stopping. Fit/apply category cardinalities are checked so that
  target-statistic CTRs cannot silently be activated. Inputs, routes, and
  transformations use only supplied training features; no test data or
  external data is read.
- GPU smoke test passed on the RTX 4050 Laptop GPU: the Phase 20 matrix/model
  path fit 10,000 rows and predicted 2,000 rows in `0.94` seconds with
  `one_hot_max_size=256`.
- The first smoke attempt failed before training because CatBoost rejects
  `one_hot_max_size=10000` (maximum supported is 256). The configuration was
  corrected, smoke-tested, and the corrected threshold included in the
  frozen registry.
- The two unsupervised clustering families (20 experiment definitions) are
  preregistered but GPU-blocked: installed PyTorch is CPU-only, and CuPy,
  cuML, and cuDF are unavailable. No CPU clustering fallback will be used.
- Registry SHA-256:
  `9e40ecc9fe060470c58604e5e1ca0b569fe7767b1bc00425214e1466ccf78f81`.
  Registry source and tests:
  [`experiments/phase20_segmentation_expert_research/run_phase20_segmentation_study.py`](./experiments/phase20_segmentation_expert_research/run_phase20_segmentation_study.py)
  and
  [`experiments/phase20_segmentation_expert_research/test_phase20_registry.py`](./experiments/phase20_segmentation_expert_research/test_phase20_registry.py).
  Generated pre-run definitions:
  [`experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106/PHASE20_EXPERIMENT_REGISTRY.csv`](./experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106/PHASE20_EXPERIMENT_REGISTRY.csv).
- Unit tests pass: four registry tests verify counts, unique configurations,
  label-free route definitions, and the 20 explicit GPU-blocked cluster
  definitions. All results remain pending model execution; no Phase 20 score
  is claimed yet.

## 2026-11-06 — Independent raw-data profile and hypothesis checks (append-only)

- **Scope / provenance:** Read the training data directly from
  `D:\gci\inpuy\train.csv` (171,202 rows, 34 columns; SHA-256
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`) and
  the supplied `HomeCredit_columns_description.xlsx` dictionary (SHA-256
  `4dcec9310a82dbd7d65a8290bad69bbac34e632494568a96281e8a779a88c25b`).
  There are 13,821 positive targets (8.0729%), 171,202 unique IDs, and no
  duplicated full rows. Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, and
  scikit-learn 1.9.1 were used. The analysis read no test rows, external data,
  cached features, or experiment OOF files; no submission or model artifacts
  were created. Analysis code was run inline and not persisted as a separate
  runner, in keeping with the request to update only this ledger.
- **Protocol / evidence status:** Descriptive target rates and the small
  logistic-regression comparison below use labels already examined in prior
  project work. The five-fold comparison uses stratified shuffled folds with
  seed `20261106`; imputation, scaling, and model fitting are fitted inside
  each training fold. It was not tuned and is `EXPLORATORY`, not independent
  confirmation, a competition benchmark, or evidence for champion promotion.

### Measurements against the proposed hypotheses

1. **Information availability / missingness**
   - In the three external scores, missing counts are 118,928
     (`EXT_SOURCE_1`), 369 (`EXT_SOURCE_2`), and 54,586 (`EXT_SOURCE_3`).
     Target rates by number of available external scores were: 0 available,
     117 rows / 9.40%; 1 available, 38,611 / 9.06%; 2 available, 96,310 /
     8.04%; all 3 available, 36,164 / 7.11%. The zero-score group is too
     small for a stable standalone conclusion.
   - All four bureau inquiry columns are jointly missing on exactly 23,116
     rows (10.378% target rate); all are present on 148,086 rows (7.713%).
     Within rows with a non-sentinel `DAYS_EMPLOYED`, bureau-missing versus
     observed rates were 11.116% (n=18,883) versus 8.274% (n=121,421).
     Within the sentinel group they were 7.087% (n=4,233) versus 5.160%
     (n=26,665). This association is not explained by that one split alone,
     but is not a multivariable causal or incremental-value claim.
   - The selected 10-field availability signature (three external scores,
     occupation, organization, income type, and four bureau fields) had only
     12 observed patterns. The common all-observed pattern had 36,164 rows
     and 7.11% target rate; the `EXT_SOURCE_1`-missing-only pattern had
     80,229 and 7.97%; `EXT_SOURCE_1` and `EXT_SOURCE_3` missing with bureau
     fields present had 21,815 and 7.88%; the same external-score missing
     pattern with all four bureau values missing had 16,602 and 10.65%.
     These are descriptive pattern contrasts, not evidence that missingness
     itself causes risk.
   - A deliberately narrow conditional test used only the three external
     score values as a baseline, then added predeclared availability
     indicators. Median imputation, standardization, and logistic regression
     (`C=1`, `lbfgs`) were fit within each of five folds:

     | Features | Pooled OOF ROC-AUC | Delta vs score-only | Folds improved |
     |---|---:|---:|---:|
     | Three external-score values only | 0.708635 | reference | — |
     | Plus count of missing external scores | 0.713269 | +0.004634 | 5/5 |
     | Plus all-bureau-missing indicator | 0.711847 | +0.003213 | 5/5 |
     | Plus employment/organization sentinel indicator | 0.711112 | +0.002478 | 5/5 |
     | Plus all three indicators above | 0.717047 | +0.008412 | 5/5 |

     This establishes a conditional signal in this simple model and split
     setup, not incremental lift over the project's stronger baselines. The
     employment/organization indicator is redundant with a data identity
     noted below.

2. **Employment history relative to age**
   - For negative `DAYS_EMPLOYED` values only, calculated
     `(-DAYS_EMPLOYED / 365.25) / (-DAYS_BIRTH / 365.25)`. The median ratio
     was 0.1188, the 90th percentile 0.3450, and no valid row exceeded 1.0.
     The sentinel value 365243 was excluded from this ratio.
   - Default rates by ratio were 10.62% below 5% (n=31,085), 8.89% at
     5–25% (81,069), 6.07% at 25–50% (23,674), and 4.38% at 50–75%
     (4,476). The 365243 sentinel group had 30,898 rows and a 5.42% rate.
     The decreasing pattern was broadly present within age quartiles too:
     under-5%-ratio rates ranged from 8.26% to 12.22%, while the 25–50%
     group ranged from 4.88% to 8.44%; some older-tenure cells are small.
     This is a descriptive association, not proof of actual age at first
     employment or a valid career-history feature.

3. **Financial-ratio disagreement**
   - Full-data medians were 3.267 for credit/income, 0.1630 for
     annuity/income, and 2.94 for goods-price/income. Quartile target rates
     were non-monotone: credit/income 7.52%, 8.66%, 8.98%, 7.14%;
     annuity/income 7.28%, 8.03%, 8.52%, 8.46%; goods-price/income 7.94%,
     8.96%, 8.59%, 6.80%. This supports testing nonlinearities rather than
     presuming a single increasing ratio-risk relationship.
   - Median-split disagreement groups had: high credit/income + low
     annuity/income, 16,910 rows / 6.74%; low credit/income + high
     annuity/income, 16,908 / 8.91%; high/high, 68,693 / 8.39%; low/low,
     68,691 / 7.88%. The direction was similar within cash loans; the
     corresponding revolving-loan disagreement group was very small
     (17 rows). Median splits are descriptive bins, not financial-contract
     interpretations or validated features.

4. **Bureau inquiry distribution**
   - The supplied dictionary and actual schema contain only
     `AMT_REQ_CREDIT_BUREAU_HOUR`, `_MON`, `_QRT`, and `_YEAR`; there are no
     day- or week-specific columns. All four fields are missing together on
     the same 23,116 rows. Thus the original day/week distribution
     hypothesis cannot be run as stated.
   - On the 148,086 complete rows, summing HOUR+MON as the nearer pair and
     QRT+YEAR as the older pair gave target rates of 6.70% for nearer-only
     positive activity (n=5,211), 7.95% for older-only (93,795), and 7.43%
     for both positive (19,925); no positive counts in either pair had
     7.32% (29,155). Total-count bins had 7.32% at zero, 7.48% at one, and
     7.92% at 2+. For nearer activity at least half the total, the rate was
     6.63% (n=12,993), versus 7.95% below half (105,938). These unadjusted
     groups are not evidence of recency causality; the dictionary's period
     descriptions should be checked before interpreting summed windows.

5. **Household-adjusted income**
   - `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` had a median of 75,000.
     Quartile target rates were 8.62%, 8.08%, 8.09%, and 7.44% from lowest
     to highest per-family-member income. This is a modest descriptive
     gradient with no adjustment for income type, age, or household
     composition; it does not establish available disposable income.

6. **Attribute-combination unusualness**
   - Occupation/organization had 842 observed pair levels. Pair support
     below 10 rows covered 1,163 applicants (6.79% rate); support 10–49
     covered 5,643 (8.20%); support at least 50 covered 164,396 (8.08%).
     The rarest-pair group therefore did not show an elevated crude rate.
     Education/occupation pairs below 10 rows contained only 49 applicants,
     and income-type/occupation pairs below 10 contained 26; their rates
     are too noisy to support an unusualness-risk claim. No applicant
     combination was labeled incorrect.

### Data-integrity finding and decision

- **Exact encoding alias:** `DAYS_EMPLOYED == 365243` and
  `ORGANIZATION_TYPE == "XNA"` identify exactly the same 30,898 rows
  (zero mismatches; Jaccard 1.0). Both groups have the same 5.424% target
  rate, compared with 8.656% for the other 140,304 rows. Treat these two
  columns as duplicate indicators in any follow-up attribution; the match
  does not explain the underlying sentinel's real-world meaning.
- `OCCUPATION_TYPE` had no null or `XNA` values in this file. The
  `ORGANIZATION_TYPE` `XNA` value is the exact employment sentinel alias,
  not a separate missingness dimension. This corrects the initial
  hypothesis's assumption that the two fields might provide independent
  missingness information.
- **Conclusion:** the strongest new leads are (a) fold-safe missingness
  indicators contain signal beyond external-score values in a simple
  baseline, (b) a perfectly duplicated employment-sentinel/organization
  category encoding, and (c) a broadly age-stratified association between
  employment-to-age ratio and default rate. All claims remain exploratory
  on historically exposed labels. No model-selection or submission
  decision follows from this entry.

## 2026-11-06 — Phase 22 pairwise AUC error decomposition (append-only)

- **Purpose:** test where the current model loses positive-negative ranking
  pairs, rather than treating row-level error counts or logloss as the
  competition objective. No model was trained and no blend weight was tuned.
- **Inputs and alignment:** supplied train data `(171202, 34)`,
  SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  Phase 15 single-seed OOF SHA-256
  `ce53321d2b301a65636c6c129bf44e4ecfc9a5602f26948a5cbbafbace95cbc4`;
  Phase 18 OOF SHA-256
  `bf80c8ca4a2e139a145174e7c5c6dde32c90625b977ac3584c592800906c37ac`.
  Applicant IDs, targets, row order, and all five fold IDs were checked;
  Phase 18's CatBoost component exactly matches the Phase 15 baseline.
- **Method:** partitioned rows into ten quantile bands of the Phase 15 OOF
  score; calculated exact positive-versus-negative concordance in each of
  the 100 band pairs using sorted score searches, including half-credit for
  ties. The band totals reproduce `roc_auc_score`; the full comparison
  represents all `2,175,162,801` positive-negative pairs, not a row sample.
- **Results:** prevalence is `8.073%` (13,821 positives, 157,381 negatives).
  Phase 15 AUC is `0.767510928`, corresponding to `505,701,581.5`
  discordant-equivalent pairs. Just 2,256 positives (16.3%) in the
  bottom-half baseline-risk bands account for `46.45%` of baseline ranking
  errors. The Phase 18 nested blend AUC is `0.768668866` (`+0.001157938`);
  its net gain is `2,518,703.5` correctly ordered pairs, of which
  `1,742,886.5` (69.2%) come from positive applicants in those bottom-half
  baseline-risk bands. It improves positive-risk deciles 1–8 and loses
  `350,395.5` net pair orderings in deciles 9–10.
- The Phase 17 target-encoded LightGBM component gains `6,750,027.5` net
  pair orderings among bottom-half risk positives, but loses
  `4,829,241.5` overall (`0.765290753` AUC); its high-risk-band losses
  outweigh the low-risk gains. This is a useful diagnostic of why a
  complementary model can help a blend despite a lower standalone AUC, not
  evidence that target encoding is competition-eligible.
- **Interpretation / decision:** the results support investigating
  low-scored positives and cross-model score compatibility; they do not
  establish that a dedicated low-risk expert will generalize. A single
  strictly increasing calibration transform cannot change that model's AUC,
  but model-specific transformations can alter a blend's cross-model
  ordering. The existing one-fold YetiRankPairwise pilot in the Phase 21
  runner scored `0.710221858`, so do not replace the classifier loss with a
  generic pairwise ranker on this evidence. First finish the active Phase 21
  suite, then consider one predeclared, nested low-risk conditional
  correction and score it over all pairs. All labels have prior phase
  exposure; these remain exploratory results, not independent confirmation.
  Competition eligibility of Phase 15/17 representations remains unresolved.
- **Artifacts / reproducibility:** runner
  `experiments/phase22_pairwise_auc_diagnostics/run_pairwise_auc_diagnostics.py`,
  SHA-256
  `a0b5a1ec759f448eba6af3af99ea5ac31c87388174ef09e996bd459110ca440f`;
  unit tests: 3 passed. The runner reproduces outputs under
  `experiments/phase22_pairwise_auc_diagnostics/runs/phase22_pairwise_20261106_final/`:
  `PHASE22_EXECUTION_REPORT.md`, `PHASE22_PAIRWISE_CELLS.csv`,
  `PHASE22_POSITIVE_RISK_BANDS.csv`, `PHASE22_SUMMARY.json`, and
  `PHASE22_SHA256_MANIFEST.csv`. All eight manifest entries were verified.
  The diagnostic reads no test data and writes no submission.

## 2026-11-06 — Phase 20 completion correction and Phase 21 stop/resume audit (append-only)

### Phase 20 status correction

The earlier Phase 20 preregistration entry above says that results were still
pending. That was true at preregistration time but is no longer the current
status. The later corrected run is complete; the historical entry is retained
unchanged.

- The corrected result registry has 180 unique rows: 150 real experiments and
  30 placebo controls. Of the real experiments, 130 completed on GPU and 20
  clustering definitions were GPU-blocked. All 30 placebos completed.
- Sixty-five of 130 completed real candidates had positive pooled deltas;
  median delta was `-0.000015532`. The 30-control placebo 95th-percentile
  delta was `-0.000074085`; 31 candidates passed that empirical cutoff plus
  the predeclared fold-consistency rule. None of these filters is an
  independent significance test.
- Independently recounted from `PHASE20_STAGE1_RESULTS.csv` and
  `PHASE20_SEGMENTATION_SUMMARY.csv`: contract-type routing was positive in
  all 10 configurations (mean delta `+0.000872324`); external-score
  completeness and goods/credit ratio were also positive in all 10
  (`+0.000374022` and `+0.000402708` mean deltas). The largest contract-type
  discovery lift was `F01_C06`, `+0.001203792`, with 5/5 positive discovery
  folds. These are 60%-discovery-sample findings, not full-data scores.
- Employment-duration and annuity/income-burden routing lost in all 10
  configurations. The two clustering families remain untested, not rejected.
  The learned-gate family lost its two tested configurations. Thirty-one
  discovery candidates passed the empirical placebo cutoff and predeclared
  fold-consistency rule; this is a ranking filter, not a significance test.
- The once-used 40% holdback was previously exposed in project phases and is
  not independent confirmation; the highest of its 10 one-time candidates,
  `F07_C08`, had a `+0.001527318` lift. The corrected report notes that the
  first run stopped at 177/180 due to a placebo parser error and resumed the
  last three rows; it also notes that GPU telemetry covers only the successful
  resume/finalization window. No row-level OOF cache was saved. No champion or
  submission changed, and no 0.780 result was reached.
- Phase 20's 7 registry tests pass. The 25-entry final manifest was checked:
  24 entries match; the only mismatch is the ledger source entry (manifest:
  230,767 bytes; current ledger: 225,465 bytes before this addendum). The
  experiment data, scripts, registry, and reported outputs match their
  manifest hashes. The historical ledger version referenced by that run is
  not present as a separate file.

Corrected evidence:
[`experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106_corrected/PHASE20_EXECUTION_REPORT.md`](./experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106_corrected/PHASE20_EXECUTION_REPORT.md),
[`PHASE20_STAGE1_RESULTS.csv`](./experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106_corrected/PHASE20_STAGE1_RESULTS.csv),
[`PHASE20_SEGMENTATION_SUMMARY.csv`](./experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106_corrected/PHASE20_SEGMENTATION_SUMMARY.csv),
and
[`PHASE20_FINAL_SHA256_MANIFEST.csv`](./experiments/phase20_segmentation_expert_research/runs/phase20_stage1_20261106_corrected/PHASE20_FINAL_SHA256_MANIFEST.csv).

### Phase 21 execution stop and recovery

- Initial inventory found two older Phase 21 directories containing only
  `PHASE21_FEATURE_SCHEMA.json` and `PHASE21_GPU_PREFLIGHT.json`; neither had
  model results or checkpoints.
- The first full-data attempt, in
  `experiments/phase21_focused_validation/runs/phase21_resume_20261106_2132/`,
  passed preflight, completed fold 1's global baseline (`AUC 0.772956`), then
  exited with code `-1` after fold 2 began. It left only schema and preflight
  files; no fold outputs were persisted, so its fold-1 work cannot be reused.
  No cause was reported by the process.
- The runner was corrected before the next model run:
  - learning-curve frequency features are now rebuilt from each sampled
    training subset, not from the full outer-training fold;
  - each completed outer fold is atomically checkpointed and can be resumed
    only if input, code, configuration, and feature-schema hashes match;
  - row-aligned IDs, targets, folds, all candidate predictions, and seed
    predictions are saved to `PHASE21_OOF_PREDICTIONS.csv` for independent
    metric and error analysis; checkpoint files are included in the manifest.
- Updated Phase 21 tests pass (7 tests), including subset-frequency
  recomputation and checkpoint round-trip/identity validation. Python syntax
  compilation passes.
- The new run directory is
  `experiments/phase21_focused_validation/runs/phase21_checkpointed_20261106/`.
  It has its identity, schema, preflight, and telemetry-progress files but
  no fold checkpoint yet. A separate GPU training process was already using
  the shared GPU, so this run was stopped during preflight without terminating
  that process. A bounded waiter is configured to resume this exact run with
  `--resume` after GPU memory remains below 1,500 MiB for two minutes.
- Therefore Phase 21 has **no completed result yet**. The predeclared scope is
  a matched full-data five-fold comparison: 20/40/60/80% learning curve;
  1/3/5/10/20-seed scaling; four feature-group ablations; four label-free
  domain flags; Track A raw and constant-free frequency models; and hard,
  fixed-mixture, and contract-type calibrated specialists. No test data,
  external data, submission, or champion has been used or changed.

### Remaining governance and interpretation

- The Phase 21 runner prepares three organizer questions on nested target
  statistics, data-derived constants, and pooled train/test covariates. They
  are drafts, not sent questions; the supplied rules audit found no verified
  live organizer contact or later ruling.
- The local compliance matrix marks pooled validation/test covariates
  `AMBIGUOUS_AVOID`. Test-key coverage remains uninspected until that
  project-specific no-test-data protocol is explicitly changed. No target
  labels from test are available or used.
- GPU YetiRankPairwise was screened on one fold and scored about `0.710222`;
  it was excluded because its default PFound metric is CPU-computed and the
  result was weak. GPU monotone constraints were rejected as unsupported.
  Hierarchical target statistics remain deferred pending organizer guidance.
  A new logistic stack has not been run; Phase 18's existing nested blend
  (`0.768668866`) beat its matched Phase 15 single-seed component
  (`0.767510928`) by `+0.001157938`, but remained `0.000155842` below the
  Phase 15 three-seed OOF average. Thus the supplied recommendation that
  Phase 18 failed to beat the single-seed model is incorrect. A broader
  logistic stack is not run from globally cached OOF vectors because that
  would let outer-validation labels influence base-model scores used to train
  the meta-model; a valid stack needs nested base refits and is not a cheap
  post-processing step on these repeatedly examined labels.
- No Phase 20/21 result establishes 0.78. All follow-up scores remain
  exploratory because the labels have prior phase exposure; a result from
  repeated OOF analysis is not an independent confirmation or champion.

## 2026-11-06 - AUC-oriented interpretation and next-step clarification (append-only)

- This entry adds interpretation to the Phase 22 pairwise audit above; it
  does not replace or revise any prior result. For ROC-AUC, the unit of
  evaluation is a positive-negative pair, not an individual applicant.
  Therefore a "low-risk positive" matters only through how its score orders
  against the negative population, and a candidate must be judged by its
  net pairwise effect across the full evaluation set.
- The Phase 22 decile table supports a low-score opportunity: 16.3% of
  positives in the bottom-half baseline-risk bands account for 46.45% of
  baseline discordant-equivalent pairs. The nested Phase 18 blend's
  `+0.001157938` AUC difference corresponds to 2,518,703.5 additional
  concordant-equivalent pairs; 1,742,886.5 of the net gain arise from those
  bottom-half positive bands. This is diagnostic attribution on reused OOF
  labels, not proof a low-risk expert will improve a new evaluation.
- Class imbalance explains why a loss optimized for per-row probability
  accuracy may not directly maximize pairwise ranking, but reweighting or
  calibration is not automatically an AUC improvement. A strictly
  increasing transform applied uniformly to one model preserves its
  ordering and AUC. Transformations can affect a blend by changing
  cross-model ordering, so any such blend/correction must be selected inside
  training folds and evaluated on the outer fold.
- **Next experiment gate:** do not launch another broad feature/model sweep
  or fit an expert specifically to the bottom-risk subgroup and score only
  that subgroup. First verify completion and results of the already-running
  Phase 21 suite. If further work is justified, preregister one
  low-score-positive hypothesis, derive its gate from cross-fitted
  predictions inside each outer training fold, fit any adjustment without
  outer-fold labels, and evaluate its net AUC/pair effect over all outer-fold
  positive-negative pairs. Retain the raw/global model as comparator and
  report both pair gains and pair losses by baseline-risk band. All
  currently available labels are previously examined; such a result remains
  exploratory and is not independent confirmation.

## 2026-11-06 — Phase 24 AUC reliability and profile audit (append-only; GPU stage pending)

- **Purpose / scope:** started the separate reliability investigation motivated
  by the supplied notes: pooled-versus-fold AUC, repeated application
  profiles, ID intervals, and a predeclared cold-product-key stress test.
  Added a new standalone runner rather than editing the active Phase 23 or
  earlier experiment runners:
  `experiments/phase24_auc_reliability_research/run_phase24_auc_reliability.py`.
  The new test file is
  `experiments/phase24_auc_reliability_research/test_phase24_auc_reliability.py`.
- **Inputs / provenance:** train-only CSV
  `D:\gci\inpuy\train.csv`, 171,202 rows x 34 columns, SHA-256
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
  Historical comparison predictions:
  `experiments/phase15_autonomous_research/runs/phase15_native_catboost_oof_20261105T124238Z/OOF_PREDICTIONS.csv`,
  SHA-256
  `ce53321d2b301a65636c6c129bf44e4ecfc9a5602f26948a5cbbafbace95cbc4`.
  Phase 15 IDs, targets, and all five fold IDs were checked against the
  training rows. No test data, external data, or model cache was read for
  this audit. Runner SHA-256
  `68c606efcbf0ec01ca2f6fbb79250a28004786b8c83cf0fcdf3b03f1d77c32d3`;
  test SHA-256
  `aea69682bf46ec08bbdd4512c5e559f8ecf00db0570324010119dd616f1a7d72`.
- **Implementation / verification:** the CPU audit compares pooled AUC,
  unweighted/row-weighted/pair-weighted fold AUCs, fold variability, per-fold
  prediction distributions, and the share of positive-negative pairs that
  cross folds. It checks full predictor, financial-application, and
  categorical-applicant duplicate groups against the original random folds,
  plus 20 ordered ID intervals and an ID-only monotone AUC diagnostic.
  Three unit tests passed; Python compilation passed. The audit-only run
  completed successfully in
  `experiments/phase24_auc_reliability_research/runs/phase24_forensics_20261106/`.
- **Cross-fold score finding:** 80.0000% of positive-negative pairs span
  different folds in this balanced five-fold OOF layout. Nevertheless,
  pooled AUC versus mean-fold AUC deltas were small for the saved Phase 15
  vectors:

  | Prediction | Pooled AUC | Mean fold AUC | Pooled - mean fold |
  |---|---:|---:|---:|
  | `native_catboost_base_all8` | 0.767004319 | 0.767024089 | -0.000019770 |
  | `native_catboost_phase14_numeric` | 0.767231290 | 0.767253342 | -0.000022052 |
  | `native_catboost_phase14_plus_time` | 0.767510928 | 0.767525153 | -0.000014225 |

  For these three saved vectors, row- and pair-weighted mean fold AUCs also
  differed from the unweighted mean by less than 0.000002. Thus cross-fold
  comparisons exist in principle, but their measured net effect on these
  particular OOF scores is negligible; this does not establish that pooled
  and fold AUCs are interchangeable for other runs.
- **Duplicate-profile finding:** exact duplicates over all 32 non-ID
  predictors: zero. Repeated financial-application profiles
  (`contract, credit, annuity, goods price, income`) affected 92,162 rows
  across 20,605 duplicate groups (largest group 509); 87,412 rows in such
  groups crossed random folds. Repeated categorical-applicant profiles
  affected 156,888 rows across 10,894 duplicate groups (largest group
  10,235); 155,156 rows in those groups crossed folds. These profiles are
  coarse matches, not duplicate full applications. Mixed labels occurred
  in 4,867 financial and 3,805 categorical duplicate groups. The evidence
  documents shared profiles across folds, not leakage or invalid random-CV
  evaluation.
- **ID/batch diagnostic:** monotone ID-only AUC was 0.502897 (no meaningful
  univariate ranking signal). Across 20 ordered ID intervals, target rates
  ranged from 7.3248% to 8.5631%. This argues against a strong simple
  monotone ID effect; it does not rule out non-monotone batch structure or
  covariate drift.
- **Optimization experiment status:** the new runner also preregisters
  matched CatBoost GPU comparisons of raw predictors versus raw predictors
  plus `PRODUCT_KEY_Q4`, evaluated on identical five-fold partitions under
  (1) ordinary stratified applicant splits, (2) Q4 product-key-disjoint
  splits, and (3) contiguous ordered-ID holdout blocks. It reports pooled
  and fold AUCs and verifies no Q4-key overlap in the cold-key split.
  Product-key categorical statistics remain rule-ambiguous; this is a
  research stress test only. At this entry, GPU fitting is **PENDING**:
  another independent Phase 23 process was actively using the shared RTX
  4050, so no concurrent Phase 24 model was launched. The runner requires
  at least 1,600 MiB free before starting GPU fits. No optimization lift is
  claimed yet.
- **Selection-bias limitation:** the historical candidate universe and
  all tried scores are not fully enumerated in this run, so the magnitude
  of winner's curse is not estimable from the selected OOF files. All
  results use previously exposed labels and are exploratory, not independent
  confirmation or a champion decision.

### Phase 24 GPU comparison completion supplement (2026-11-06; append-only)

- **Run / provenance:** completed
  `experiments/phase24_auc_reliability_research/runs/phase24_gpu_20261106/`.
  The Phase 24 runner SHA-256 remained
  `68c606efcbf0ec01ca2f6fbb79250a28004786b8c83cf0fcdf3b03f1d77c32d3`.
  Used the same train and Phase 15 reference OOF inputs and hashes recorded
  above; no test data or external data were accessed. Python 3.12.10,
  CatBoost 1.2.10, RTX 4050 Laptop GPU. GPU memory preflight observed
  1,954 MiB free. All 30 model fits reported `task_type=GPU`; 600 iterations,
  depth 6, learning rate 0.05, `one_hot_max_size=10`, and
  `max_ctr_complexity=1`. Runtime was 2,141.9 seconds. Fourteen output,
  source, and input hashes in the final run manifest were verified.
- **Protocol:** compared raw predictors (ID/TARGET excluded) with the same
  predictors plus only the deterministic `PRODUCT_KEY_Q4` category, under:
  five-fold stratified random applicants (seed `20261113`), five-fold
  `GroupKFold` with Q4 keys disjoint, and five contiguous equal-row ID
  intervals held out in turn. No candidate or threshold was selected on
  the outer folds. The group and ID protocols deliberately test different
  deployment conditions and are stress tests, not replacements for random
  competition CV.

  | Split / model | Pooled AUC | Mean fold AUC | Delta vs same-split raw | Positive folds |
  |---|---:|---:|---:|---:|
  | Random / raw | 0.752349442 | 0.752368644 | reference | — |
  | Random / raw + Q4 | 0.764449445 | 0.764476620 | +0.012100004 | 5/5 |
  | Q4-disjoint / raw | 0.747942157 | 0.748380930 | reference | — |
  | Q4-disjoint / raw + Q4 | 0.740437344 | 0.740145863 | -0.007504813 | 0/5 |
  | ID-block / raw | 0.751940916 | 0.752036257 | reference | — |
  | ID-block / raw + Q4 | 0.764029221 | 0.764213793 | +0.012088306 | 5/5 |

- **Fold behavior / uncertainty:** paired fold deltas for random CV were
  `+0.013627, +0.014518, +0.008010, +0.014763, +0.009620`; Q4-disjoint
  deltas were `-0.009343, -0.006942, -0.009330, -0.008754, -0.006807`;
  ID-block deltas were `+0.014151, +0.014716, +0.011129, +0.011692,
  +0.009199`. A 500-repetition paired stratified row bootstrap conditional
  on these saved OOF vectors gave 95% delta intervals: random
  `[+0.010596,+0.013585]`, Q4-disjoint `[-0.008881,-0.006110]`, and
  ID-block `[+0.010496,+0.013655]`. These intervals do not account for
  model-fitting variability, prior experiment selection, or exposure of
  the labels.
- **Cold-start diagnostic:** the random folds had only 47 validation rows
  with unseen Q4 values in total; Q4-disjoint validation contained no key
  overlap by construction (797 fold-specific held-out category instances).
  The large random-fold gain did not transfer to unseen Q4 groups; it
  reversed under that stress test. This is evidence that the gain is
  conditional on recurring Q4 values and is compatible with exploitation
  of repeated-key associations. It does not prove target memorization or
  that random CV is invalid, because the group split changes the test
  distribution. Contiguous ID-block results were close to random-fold
  results, providing no evidence of a large ID-interval shift for these
  particular models.
- **Independent verification:** independently rebuilt all three fold
  schemes, checked OOF row/ID/TARGET alignment and finite predictions, and
  recalculated pooled and mean-fold AUCs for all six split/model pairs.
  Every metric matched the saved results to `1e-12`; fold-delta signs and
  Q4 separation checks passed. The baseline's random-fold AUC is not
  directly comparable to Phase 15's stronger feature-set score because
  this controlled test intentionally uses only raw columns plus the single
  Q4 ablation.
- **Decision / eligibility:** Q4 is a strong *random-fold* increment over
  this narrow raw-feature baseline, but it does not demonstrate cold-key
  generalization and is not a robust generalization win. CatBoost's
  high-cardinality Q4 handling can use target-dependent categorical
  statistics; competition eligibility remains unresolved. Do not promote,
  submit, or call this a champion improvement. No tuning beyond this
  predeclared one-key comparison was performed.

## 2026-11-06 — Phase 23 AUC-directed feature research (append-only)

- **Question and protocol:** tested whether application-process/missingness
  patterns, external-score disagreement, financial relationship residuals,
  applicant-profile consistency, positive-class weighting, or a low-score
  specialist improve ROC-AUC. Used the same five stratified folds and seed
  (`20261113`) for all candidates, with 700-iteration CatBoost GPU fits
  (depth 6, learning rate 0.05). The run completed 60 GPU fits in about
  1,427 seconds. All feature fitting was restricted to training folds;
  training-derived vocabularies, residuals, and categorical frequencies
  were fold-safe/cross-fitted. No test rows, test labels, external data, or
  submissions were used.
- **Matched OOF results:** internal baseline pooled AUC was `0.767488294`
  (saved Phase 15 reference: `0.767510928`, a reproduction difference of
  `-0.000022633`). The strongest tested candidate was a fixed 50/50 blend
  of baseline and application-process-feature OOF predictions:
  `0.768177708` (`+0.000689414` versus internal baseline), improving all
  five folds. The conditional paired row-stratified bootstrap 95% interval
  was `[+0.000452954, +0.000943791]`; this interval is conditional on these
  OOF predictions and does not include training variability or research
  selection.

  | Candidate | Pooled OOF AUC | Delta vs baseline | Folds improved |
  |---|---:|---:|---:|
  | Application-process features | 0.767949642 | +0.000461348 | 4/5 |
  | External-score disagreement | 0.767615182 | +0.000126888 | 3/5 |
  | Financial relationship residuals | 0.767672508 | +0.000184213 | 3/5 |
  | Applicant-pair consistency/frequency | 0.767643520 | +0.000155226 | 3/5 |
  | All four feature families combined | 0.767540195 | +0.000051901 | 3/5 |
  | Fixed 50/50 baseline + application-process OOF blend | 0.768177708 | +0.000689414 | 5/5 |
  | Fixed 50/50 baseline + external-disagreement OOF blend | 0.767995554 | +0.000507260 | 5/5 |
  | Fixed 50/50 baseline + financial-residual OOF blend | 0.767977611 | +0.000489317 | 5/5 |
  | Fixed 50/50 baseline + applicant-consistency OOF blend | 0.768020705 | +0.000532411 | 5/5 |
  | `scale_pos_weight=2` | 0.767686338 | +0.000198044 | 4/5 |
  | `scale_pos_weight=4` | 0.767792631 | +0.000304336 | 3/5 |
  | Low-score specialist, hard routing | 0.765031626 | -0.002456668 | 0/5 |
  | Low-score specialist, fixed 50/50 mix | 0.767162972 | -0.000325322 | 0/5 |

- **Interpretation:** application-process patterns were the strongest single
  feature family and their fixed blend gave the best result in this study.
  The four-family combination was weaker than any corresponding fixed
  single-family blend, consistent with overlapping or noisy signals rather
  than additive gains. The tested positive weights did not establish a
  robust improvement. Both low-score-specialist variants lost on every
  fold, so this specialist setup is rejected. Fixed blend gains are
  exploratory: their OOF labels have been examined and no
  multiple-comparison correction was applied.
- **Train-only profile checks:** 26 distinct application missingness
  profiles and zero exact duplicate rows over the 32 non-ID predictors
  were observed. Default rates by number of available external scores were
  9.40% for zero (n=117), 9.06% for one, 8.04% for two, and 7.11% for
  three. These are descriptive associations, not proof of incremental
  predictive value.
- **Decision and next step:** none of the candidates exceeds the historical
  Phase 18 nested blend (`0.768668866`) or the retrospective Phase 9
  seed-average claim (`0.769839127`). Do not promote a champion or claim a
  submission gain. If pursued, evaluate the application-process blend and
  blend-weight selection in a new nested or otherwise untouched validation
  protocol before any promotion.
- **Reproducibility:** runner:
  `experiments/phase23_auc_feature_research/run_phase23_auc_feature_research.py`
  (SHA-256
  `b573c97521968cecb699752420c346363676f6a3fb2e3382645cb719977d276b`);
  tests:
  `experiments/phase23_auc_feature_research/test_phase23_auc_feature_research.py`
  (SHA-256
  `79e5a6dc874fd571b478c4f98bd7aa1dee8b3e7d70b4b28257ee79769e52467f`).
  Four feature tests passed; the OOF AUCs and folds were independently
  recomputed, and all 11 output-manifest hashes matched. Detailed protocol,
  results, bootstrap intervals, and outputs are under
  `experiments/phase23_auc_feature_research/runs/phase23_targeted_20261106/`.
- **Shared-GPU note:** an earlier Phase 21 resume process had only produced
  preflight artifacts and was stopped to free GPU capacity. A separate
  Phase 21 resume job and the Phase 24 reliability job were later observed
  in the shared environment; they were not stopped or treated as part of
  this experiment. The Phase 23 run completed and its outputs passed the
  checks above; its elapsed time may reflect shared-GPU contention.

## 2026-11-06 - EXP3 OOF and competition-rule cross-check (append-only)

- **Artifacts inspected read-only:** `D:\gci\gemini\independentexperiments\origin_exp3.ipynb`,
  `D:\gci\gemini\build_origin_exp3.py`, and
  `D:\gci\submission_origin_exp3_from_scratch.csv`. The builder starts from
  `origin_exp2.ipynb` and patches cells; it does not itself execute training.
  Notebook outputs show the supplied train/test paths and row/column counts,
  but do not record input hashes.
- **Rule interpretation:** Local competition rules do not require OOF or
  prescribe a cross-validation design. Local OOF on training rows is not
  prohibited, but it is not mandatory and does not by itself prove a
  leakage-free evaluation. The rules require using only the supplied
  `train.csv` and `test.csv`, prohibit manual/EDA-rule decisions for any
  subset of test predictions, and require reproducibility. The text does not
  expressly say that every cached prediction is automatically disqualifying.
- **Reported OOF:** Notebook cell 7 stores a pooled eight-engine OOF ROC-AUC
  of `0.771857`. This is not a trustworthy independent estimate:
  1. Cell 3 builds target encodings with the same five-fold,
     `random_state=42` split as the outer models; outer-training rows'
     encodings can therefore use labels from the model's held-out fold.
     The encoding also uses a full-training-label prior. The
     `UIDY_NEAR_Y`/`UIDY_ANYBAD` encodings reuse that fold assignment too.
  2. All seven cell 4 model streams use the reported validation fold for
     early stopping; cell 5 XGBoost does the same on its ten-fold split.
  3. Cell 7 optimizes blend weights against all OOF labels and reports AUC
     on those same labels. It does not check `res.success`; displayed weights
     are effectively the supplied initial weights, so the output does not
     establish a reliable optimization gain.
- **Test-feature scope and post-processing:** Cell 3 concatenates train and
  test rows before fitting group summaries, category/count statistics, and
  ranks. Whether such unlabeled test-covariate fitting is permitted is
  unresolved in the local rules; it also means this OOF pipeline does not
  “touch zero test data.” Cell 6 separately adjusts linked test predictions
  using training labels and hand-set `+3.50`/`-1.20` shifts and `0.20`
  coupling, and smooths test-to-test pairs. This is high-risk under the
  hand-labeling rule and requires organizer clarification. The specific
  `submission_origin_exp3_from_scratch.csv` is the pure suite export;
  the separate `submission_origin_exp3.csv` is the 85/15 anchor/suite blend.
- **External-reference use:** Although the model suite trains in memory and
  does not load `.npz`/`.npy` prediction arrays, cell 8 checks a historical
  champion CSV and cell 9 reads a pre-saved prediction CSV for Spearman
  comparison. The stored output reports correlations `0.922328` and
  `0.923995`, explicitly not AUC or leaderboard scores. This contradicts
  the notebook's blanket “zero external prediction files” claim; under the
  literal “only supplied datasets at any stage” rule, this analysis is not
  strictly train/test-only. The local materials do not establish an
  automatic disqualification outcome for this use.
- **Submission artifact check:** The supplied 61,500-row CSV has the expected
  `SK_ID_CURR,TARGET` columns, no nulls, unique finite rank scores, and IDs
  exactly matching the provided test-file order. Its SHA-256
  (`f2417e321bfa311bfc28466930fd6c17d376da3c133b781fe4d24f03a3c274f7`)
  matches the notebook's stored cell 8 checksum. This verifies file
  integrity and formatting, not its platform score or rule eligibility.
- **`0.77704` claim:** No `0.77704` value or official submission receipt is
  present in the inspected notebook or builder. The notebook records
  `0.771857` OOF, not the claimed leaderboard score. The supplied rules name
  Omnicampus, while the claim describes a Kaggle leaderboard. Therefore
  `0.77704` remains unverified without an official submission record tied to
  this exact CSV/hash. Do not treat either number as a compliant benchmark or
  promote the submission based on these artifacts.

### User-provided Omnicampus score clarification (2026-11-06; append-only)

- The user clarifies that Omnicampus is the official GCI Global 2026 platform;
  “Kaggle” was shorthand for the dataset's heritage, not the scoring platform.
  The local competition README confirms Omnicampus.
- The user reports an Omnicampus public score of `0.77704` for
  `submission_origin_exp3_from_scratch.csv`, with local OOF AUC `0.771857`.
  The local file's size (`1,291,519` bytes), timestamp
  (`2026-10-30 14:55:51` local time), and SHA-256 match the user's supplied
  details and the notebook's stored checksum. This corroborates the local
  artifact identity, not the remote score record; no portal receipt,
  submission ID, or screenshot was attached for independent verification.
- A real public score can be absent from the notebook because test labels
  are held by the platform. Accept `0.77704` as user-reported, not independently
  authenticated. Even if confirmed, it does not repair the OOF leakage or
  establish the private/final score; the rules describe the public leaderboard
  as a subset and the final leaderboard as the full test set.

## 2026-11-06 - Phase 25 EXP02 GPU ablation completion (append-only)

- **Run:** `experiments/phase25_exp02_compliance/runs/phase25_exp02_gpu_20261106/`.
  After the existing Phase 21 process exited, the RTX 4050 was idle at
  preflight (0% GPU utilization; 5,648 MiB free). Phase 25 then completed all
  20 model fits in 321.9 seconds. It read only the provided training CSV,
  made no submission, and did not read test data or external prediction files.
- **Protocol:** matched five-fold StratifiedKFold (seed `20261113`), fixed
  CatBoost GPU settings (700 iterations, depth 6, learning rate 0.05), with
  no validation-fold early stopping, feature selection, or blend fitting.
  Baseline, formula/peer, nested ratio target-encoding, and combined
  candidates used identical outer folds. Target encoding remains
  research-only pending organizer clarification.
- **Verified scores (ROC-AUC):**

  | Candidate | Pooled AUC | Mean fold AUC | Pooled delta vs baseline | Mean fold delta | Folds improved |
  |---|---:|---:|---:|---:|---:|
  | Baseline | 0.765682908 | 0.765688795 | — | — | — |
  | Formula/peer | 0.766157572 | 0.766168920 | +0.000474664 | +0.000480124 | 3/5 |
  | Nested ratio target encoding | 0.765970451 | 0.766039189 | +0.000287543 | +0.000350394 | 5/5 |
  | Combined | 0.765822543 | 0.765876192 | +0.000139636 | +0.000187397 | 2/5 |

- **Interpretation:** Formula/peer features have the largest pooled lift, but
  win only three folds. Nested target encoding has a small, consistent lift,
  but is not cleared for submission. Combining both reduces the gain versus
  either separate candidate. These are exploratory single-seed comparisons,
  not independent confirmation, and none establishes a lift over stronger
  historical baselines or validates EXP3's `0.771857` OOF.
- **Independent verification:** All 171,202 saved OOF rows and labels match
  the training CSV; all predictions are finite. The four pooled and 20
  per-fold AUCs were recomputed from saved OOF predictions and match the run
  outputs. Every output, input, and code SHA-256 in the manifest matched.
  Phase 25 unit tests passed (4/4); Python compilation and `git diff --check`
  passed. See the run directory for OOF predictions, fold/results CSVs,
  execution report, and SHA-256 manifest.

### Phase 21 focused validation completion (2026-11-06; append-only)

- **Run and provenance:** the checkpointed run in
  `experiments/phase21_focused_validation/runs/phase21_checkpointed_20261106/`
  completed all five folds after resuming in a second process session. Runtime
  was `5,040.97` seconds; 240 GPU model fits completed (241 including the GPU
  smoke test). The earlier interrupted attempt produced no reusable fold
  outputs; all five folds in this completed run have identity-checked
  checkpoints. The runner's telemetry history is marked incomplete after the
  process restart (2,382 two-second samples), so it is not represented as
  continuous telemetry for the full execution.
- **Inputs and integrity:** only the supplied training CSV and the saved
  Phase 15 single- and multiseed OOF references were read. Training SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  Phase 15 reference SHA-256:
  `ce53321d2b301a65636c6c129bf44e4ecfc9a5602f26948a5cbbafbace95cbc4`;
  Phase 15 multiseed reference SHA-256:
  `14e92b323a7c5255b506a33c0f8b3925e5f9d110d388c44becf0612aec37a2d6`.
  The Phase 21 reference IDs, targets, and all fold assignments match the
  current run. Independently checked the 171,202 OOF rows, unique IDs, exact
  ID/target alignment, expected five fold assignments and sizes, finite
  values in all 42 prediction columns, and key pooled AUCs. Recomputed all
  1/3/5/10/20-seed pooled AUCs from the row-level cache. All 31 entries in
  `PHASE21_SHA256_MANIFEST.csv`, including the five fold checkpoints, matched.
  No test data, external data, submissions, or champion files were used or
  changed.
- **Matched model and seed scaling:** the reproduced Phase 15 single-seed
  baseline scored `0.767472835`, versus `0.767510928` in its saved reference
  (same folds; difference `-0.000038093`). On those same folds, 1/3/5/10/20
  seed means scored `0.767472835`, `0.768284549`, `0.768417428`,
  `0.768617947`, and `0.768650650`. The 20-seed result is
  `+0.001177814` versus the reproduced single-seed model; the 10-to-20 seed
  increment is only `+0.000032703`, indicating strong diminishing returns.
  The historical Phase 15 three-seed score `0.768824708` used different fold
  assignments and remains contextual only, not a direct matched comparison.
- **Learning curve and ablations:** 20/40/60/80% training subsets scored
  `0.749913639`, `0.761293435`, `0.764635114`, and `0.766481827`; the
  60-to-80% gain was `+0.001846713`, and 80% to the full-data baseline added
  `+0.000991008`. This curve is still rising; it does not support the
  previously proposed `0.0005` feature-limited cutoff. Removing external
  scores reduced AUC by `0.042403513`, product keys by `0.008846576`,
  demographics by `0.003484090`, and day columns by `0.001457644`. These
  ablations describe this fitted pipeline and do not establish feature
  eligibility under the competition rules.
- **Feature transfer and specialists:** adding the cash-contract flag scored
  `0.767959874` (`+0.000487038`, 4/5 folds positive); external-score
  completeness scored `0.767494048` (`+0.000021213`); the goods/credit flag
  scored `0.767581592` (`+0.000108756`). The contract hard route scored
  `0.765077337` (`-0.002395498`), while fixed and calibrated 50/50
  global/expert mixtures scored `0.767639978` (`+0.000167143`) and
  `0.767663401` (`+0.000190566`). The calibrated mixture's paired
  row-stratified 95% interval was `[-0.000238,+0.000596]`, crossing zero;
  that interval is conditional on the saved OOF predictions and omits model
  training variability and prior candidate selection. Combined
  contract/completeness routing scored `0.766501768`, below the global
  baseline. The sampled positive-negative pair corrections for the
  contract-calibrated specialist were nearly balanced (11,500 corrected
  global ordering errors versus 11,483 newly introduced errors in one
  million sampled pairs). The specialist results do not confirm a robust
  generalizable gain.
- **Conclusion and boundaries:** `0.768650650` is the best result within
  this exploratory Phase 21 study, not a new champion, independent
  confirmation, or a verified `0.780` score; the remaining difference is
  `0.011349350`. The modest contract mixture change is uncertain. Phase 15
  native category/product-key eligibility is still unresolved, and the
  label-free Track A raw and frequency models scored `0.757884471` and
  `0.761292399`. Organizer questions about target statistics,
  data-derived constants, and pooled train/test covariates remain drafts;
  no ruling or contact channel was available. Test-key coverage remains
  uninspected under the no-test-data protocol. GPU monotone constraints were
  unsupported, the one-fold YetiRankPairwise screen (`0.710221857`) was
  excluded, and a CUDA logistic solver was unavailable, so no CPU fallback
  stack was run. The seven Phase 21 tests and Python syntax compilation pass.

  Detailed outputs:
  `experiments/phase21_focused_validation/runs/phase21_checkpointed_20261106/PHASE21_EXECUTION_REPORT.md`,
  `PHASE21_MODEL_RESULTS.csv`, `PHASE21_LEARNING_CURVE.csv`,
  `PHASE21_SEED_SCALING.csv`, `PHASE21_SPECIALIST_SUMMARY.csv`,
  `PHASE21_ERROR_CORRECTIONS.json`, `PHASE21_OOF_PREDICTIONS.csv`, and
  `PHASE21_SHA256_MANIFEST.csv`. Runner SHA-256:
  `f48d295ae5a456834027c26bb0afcb54868282cb9832f011f70e93d5cfa9add3`;
  test SHA-256:
  `d243b3284f66f165da3230f8a4abbd69ac2654725edc8b309ca83e27f4891db1`.

## 2026-11-07 - V2 review and AUC-focused replacement research (append-only)

- **V2 source review:** inspected
  `D:\gci\gemini\run_and_build_audit_certified_v2.py` and its saved notebook.
  The V1 shared target-encoding overwrite is fixed: each outer fold builds
  separate training/validation/test encodings, with inner cross-fitting for
  outer-training rows and outer-training labels only for validation/test
  maps. Fixed model rounds also avoid outer-validation early stopping.
  However, the saved notebook outputs were not independently regenerated in
  this review, and task handle `task-8902` is unavailable. The runner does
  not save row-level OOF predictions, so its reported `0.768243` blend AUC
  could not be independently recalculated from predictions.
- **Remaining V2 validation caveats:** category vocabularies and the
  quantized-rate frequency dictionaries are fitted on all of `train.csv`
  before outer CV. These are label-free but not strictly outer-fold-local.
  Group-mean residuals for outer-training rows include each row in the
  corresponding group mean, whereas validation rows do not; this creates a
  small self-influence/train-validation feature mismatch. Specialist
  training rows use inner-OOF predictions, while validation/test use a
  full-outer-fit specialist. This is label-isolated but has a different
  prediction-generation regime. V2's reported log-loss is calculated on a
  rank blend and is not calibrated probability log-loss. Applying ordinal
  rank to test predictions removes ties but does not improve the underlying
  pairwise ordering.
- **V2 reported results and submission:** the saved notebook reports pooled
  OOF AUC `0.768243`, below Phase 21's reported 20-seed `0.768650650`.
  The saved V2 CSV was independently checked: 61,500 rows, expected columns,
  no missing predictions, all 61,500 scores unique, and IDs aligned to the
  sample submission. SHA-256:
  `469c3cb08eda6631a53d10048c124b8534a5698d475f5d46bb16b7434dded6e6`.
  This verifies file structure/identity only, not its OOF AUC or leaderboard
  score. No public leaderboard score was checked.
- **AUC-focused analysis:** for binary labels, empirical ROC-AUC is the
  fraction of positive-negative pairs correctly ordered (ties count one
  half). Here `13,821 * 157,381 = 2,175,162,801` pairs contribute, so an
  absolute `0.001` AUC gain corresponds to roughly 2.18 million net pair
  orderings. Calibration changes alone cannot improve a single model's AUC;
  they matter when scores from distinct models are pooled.
- **Matched OOF aggregation test:** on the aligned Phase 21 OOF predictions,
  averaging probabilities from 20 CatBoost seeds scored `0.768650650`;
  global rank averaging scored `0.768620740`; rank averaging within each
  outer fold scored `0.768674226`; mean-logit scored `0.768632404`; and the
  per-row median scored `0.768531809`. The rank-space variants therefore
  changed AUC by only about `+0.000024` at best versus probability averaging.
  This does not support a major gain from replacing probability averaging
  with rank blending.
- **Matched feature-blend check:** Phase 23's application-process candidate
  was mixed at a fixed exploratory weight of 20% with Phase 21's 20-seed
  probability average. The pooled OOF AUC was `0.768726834` versus
  `0.768650650` (`+0.000076184`); fold AUCs were `0.774031`, `0.775931`,
  `0.762936`, `0.766198`, `0.764716`. A 500-replicate paired, stratified
  row bootstrap gave a conditional 95% delta interval
  `[-0.000015497,+0.000166474]`, crossing zero. The weight was explored
  after prior label exposure; do not promote this blend or treat the
  bootstrap as independent confirmation.
- **Seed bagging and uncertainty:** the matched Phase 21 20-seed average
  improves by `+0.001177814` over its one-seed baseline
  (`0.767472835` to `0.768650650`); the 10-to-20-seed increment is only
  `+0.000032703`. A 500-replicate paired row bootstrap of the saved
  predictions estimated a conditional 95% interval
  `[+0.000766454,+0.001635630]` for 20 seeds versus one seed. This
  conditions on the fitted OOF predictions and does not account for model
  retraining, prior experiments, or selection of the seed count.
- **Applicant ID and shift check:** training IDs are exactly `0..171201`;
  test IDs are exactly `171202..232701`. Raw ID has pooled training AUC
  `0.502896563`; target rate across ten ordered ID deciles ranges from
  `0.07547` to `0.08417`, with no useful monotone pattern. Largest numeric
  train/test KS distance among audited columns is `0.0293` for
  `EXT_SOURCE_3` (missing rates `0.319/0.320`). This does not establish a
  temporal label shift; no ID-based correction is supported.
- **Research conclusion:** V2 does not beat the existing Phase 21
  multi-seed OOF result. The most defensible performance direction to
  investigate further is seed averaging of the established CatBoost
  representation, but the result is still exploratory, and its inherited
  product keys include the unresolved `130.5`/`0.0066` hypotheses and native
  categorical-statistic behavior. The lower-scoring constant-free Phase 25
  candidate remains the safer provenance track. No method is independently
  shown here to reach `0.77000`; the OOF labels have been repeatedly
  examined, so an untouched labeled set or organizer leaderboard result is
  required before claiming that threshold. No code, notebook, or submission
  was modified or generated in this review.

## V5 evaluator-champion runner: provisional source audit (2026-11-07)

This is a source-level audit of `../gemini/run_and_build_audit_certified_v5_evaluator_champion.py`
and `../gemini/test_evaluator_fixes.py`. It is append-only and does not certify the
competition-rule status or the reported performance.

### Execution evidence at review time

- The external task handle `task-9201` could not be retrieved in this session. The local
  V5 Python process (PID 9124) was running and responsive when checked.
- The promised OOF CSV, both V5 submission CSVs, and V5 notebook were not present at their
  stated paths at review time. Therefore no V5 metric, OOF checksum, submission, or
  notebook execution result has been independently verified here.
- `test_evaluator_fixes.py` passed its small arithmetic assertion for a five-row,
  two-group example. This checks only the basic LOO formula, not the pipeline or its
  fold isolation.

### Improvements confirmed in source

- Group aggregate features are built within the outer-fold loop, and training-row
  deviations use a leave-one-out calculation for groups with at least two observed
  values. Validation/test group means are fitted from outer-training rows.
- Target encodings use inner out-of-fold labels for outer-training rows and full
  outer-training statistics for validation/test.
- Row-aligned OOF predictions, fold IDs, and two test submissions are written in the
  runner after model fitting completes; submission IDs are asserted against the sample
  order before training.

### Material issues still unresolved

1. **Categorical codes are not fold-fitted or train/test-consistent.** At line 158, each
   dataset is independently converted with `astype("category").cat.codes` before CV.
   The category vocabulary can therefore depend on outer-validation rows, and separately
   coding test rows can assign different numbers to the same category. Use a documented
   fixed vocabulary or fit category maps using only each outer-training fold, applying
   those maps to its validation rows and test rows with an explicit unknown-category
   policy.
2. **Singleton LOO groups do not have a peer mean.** At lines 387-390,
   `maximum(count - 1, 1)` makes the no-peer mean zero, so a singleton's deviation
   becomes its raw value. This is not a leave-one-out peer deviation. Use a documented
   fallback (for example, missing plus a singleton indicator, or a training-fold global
   mean) and test singleton and missing-value cases.
3. **Expectation-residual training features are in-sample.** The Ridge models are fit on
   outer-training rows and then predict those same rows to construct residuals, while
   validation/test residuals are based on out-of-sample predictions. Cross-fit the
   residual feature for outer-training rows using inner folds.
4. **The specialist stacker is still distribution-mismatched.** Each outer-training row
   gets a prediction from the one inner model whose held-out fold contains that row
   (lines 421 onward); each outer-validation/test row gets the average of all five inner
   models. An individual inner-model prediction and a five-model average do not have
   identical prediction distributions. Use the same inner-ensemble construction for
   stacker-training rows, with an appropriately nested design, or remove the specialist
   features.
5. **The final OOF score still does not evaluate test-time model averaging.** Each
   training row receives predictions from one outer-fold model, while test predictions
   average all five outer-fold models (lines 512 onward and 631 onward). The OOF blend
   is useful as conventional CV evidence, but it does not directly validate the exact
   inference ensemble; quantify that with a nested/repeated validation procedure or
   clearly limit the claim.
6. **The generated notebook is not shown to be executable.** The builder AST-parses cell
   source and copies buffered runner logs into notebook outputs, but cell extraction
   starts after the module-level definitions of `log1` through `log7`. The extracted
   cells call those undefined functions. AST parsing proves syntax only; it neither
   executes a notebook cell nor verifies runner/notebook parity. Include cell-local
   dependencies and actually execute the generated notebook from a clean kernel before
   claiming standalone execution.
7. **Probability calibration and feature provenance are overstated.** A fixed weighted
   blend of model probabilities is not, by itself, a calibration procedure. In addition,
   the pricing feature formulas still use constants such as `129.50` and `0.0132`, and
   the Merton-style features rely on assumed quantities. The report documents these
   formulas but does not establish their official-data provenance or rule eligibility.
   Label the blend as uncalibrated unless calibration is fold-fitted and validated; cite
   evidence for each engineered constant or mark it an exploratory heuristic and seek
   organizer clarification where required.
8. **GPU use and audit certification remain unverified.** The script requests CUDA for
   XGBoost and CatBoost but logs the RTX 4050 as a hard-coded hardware statement. Preserve
   actual library/device startup evidence and verify successful training on those
   devices; source configuration alone does not establish GPU execution. The report's
   “100% compliance” conclusion is not supported by its own source checks or by an
   organizer ruling.

### Provisional conclusion

V5 addresses some earlier concerns, particularly basic LOO group features and OOF
artifact generation, but the source still contains validation/inference mismatches and
the generated notebook is not yet demonstrated to run. No V5 score or claimed
improvement is recorded here because the run was still active and its outputs were not
available. Re-audit the saved OOF file, submissions, actual execution logs, and a clean
notebook run after completion; then append those measured results separately.

## V5 completion: independent artifact verification and remaining caveats (2026-11-07)

Follow-up to the provisional V5 audit above. The runner process was no longer present,
and the promised artifacts were available. Checks below were performed directly against
the local files; they verify file contents and reported metrics, not competition-rule
eligibility or unbiased generalization.

### Independently verified results

- Read `D:\gci\oof_predictions_v5_evaluator_grade.csv`: 171,202 rows, 10 expected
  columns, zero null cells, no duplicate applicant IDs, and fold sizes
  `{1: 34241, 2: 34241, 3: 34240, 4: 34240, 5: 34240}`.
- OOF applicant IDs and targets match `train.csv` exactly, including order.
- Independently recalculated ensemble pooled OOF ROC-AUC:
  **0.7690801448**; pooled OOF log-loss: **0.2432694240**.
- Independently recalculated ensemble fold AUCs:
  **0.7663622330, 0.7705258715, 0.7740201732, 0.7666703632, 0.7680843457**.
  Engine pooled AUCs from the same OOF file:
  LightGBM shallow **0.7682485965**, LightGBM regularized **0.7680873276**,
  LightGBM ExtraTrees **0.7677775228**, LightGBM monotone **0.7670124269**,
  XGBoost **0.7658989885**, CatBoost **0.7679499710**.
- Both submissions contain 61,500 rows, expected columns, no nulls, and IDs matching
  `test.csv` exactly and in order. Probability-file score range is
  `[0.0025137418, 0.7449352741]`; rank-file range is
  `[0.0000081301, 0.9999918699]`. Each file has 61,500 unique scores, so although
  average-rank handling is implemented, there were no ties in this submission.
- SHA-256 digests independently match the reported values:
  - OOF: `d77522e3f99b7a10d0fdd64389c6e58fdca0265bd0de4ff0b0c4c1796cb8a5a1`
  - Probabilities: `dcfbbfa32dd2110af044b92ad20395fb6a71ab2b2856c54182205e2a605aed20`
  - Ranks: `26d878ca5fa5f9f693a8dd9e475e532c7d3d89ff0759dffe2f721c3bb490d691`
- The V5 code commit exists in local clone `D:\gci\_gci_push_tmp`:
  `5c275365493062fc2dc74967f72eadd32861886e`, subject
  `feat: complete V5 evaluator-grade leak-free benchmark (0.769080 OOF AUC)`.
  This verifies the local clone's commit, not the remote's current branch state.

### Corrections to the V5 certification claims

- **The score is verified as a calculation over the saved OOF artifact, not as a
  rule-certified or independent leaderboard score.** No public leaderboard receipt was
  verified. The reported `0.7775–0.7790+` translation remains an estimate.
- **The OOF estimate still does not match the final test-time ensemble protocol.** A
  training applicant receives a prediction from the one outer-fold model that held the
  applicant out; each test applicant receives the average of all five outer-fold models.
  The value `0.769080` must be presented as this CV procedure's pooled OOF score, not
  proof of the exact five-model test ensemble's expected performance.
- **The label “calibrated probabilities” and “Log-Loss Optimal” is unsupported.** The
  submission is a fixed weighted blend of model probabilities; the runner applies no
  probability calibration step. The OOF log-loss is numerically reproducible, but does
  not establish calibration or optimality.
- **The notebook is still not standalone-executable as generated.** All seven cells
  parse as Python, but static inspection confirms each calls `log1` through `log7`
  without defining those functions in any notebook cell. The builder also copies
  buffered output and assigns execution counts; AST validation is not notebook
  execution. A clean-kernel run is required after fixing the missing definitions and
  cell dependencies.
- **Other source-level concerns from the provisional audit remain:** categorical
  codes are generated before CV and separately on train/test; singleton LOO groups
  fall back to raw values; Ridge residuals for training rows are in-sample; specialist
  stacker training features are one inner-model predictions while validation/test use
  five-model averages; the engineered constants/formulas still lack verified
  competition-rule provenance. Source settings request GPU, but no runtime device log
  was independently available to verify actual GPU execution.
- The local pushed-ledger copy contains the reported OOF metrics, but those metrics do
  not resolve the preceding methodological issues. The earlier recommendation stands:
  do not call V5 “100% evaluator-certified,” and do not claim a verified `0.77` result.
  Treat the verified value as exploratory CV evidence pending correction of the
  validation mismatches, a clean notebook execution, and any needed organizer ruling.

## 2026-11-07 — V9 Artifact and Framing Follow-up

The V9 artifacts under `D:\gci` were checked against `RUN_MANIFEST_V9.json`:

- The runner, OOF CSV, both submission CSVs, and executed notebook SHA-256 values
  match the manifest. The notebook contains 7 cells, each with a saved output.
- The OOF CSV contains 171,202 rows and no null cells. Recalculation from
  `TARGET` and `oof_ensemble` gives pooled ROC-AUC **0.7683397019** and
  log-loss **0.2434721626**, consistent with the manifest's rounded values
  (**0.768340** and **0.243472**).
- The **0.763107** value is the reported intermediate 79-feature result, not the
  final V9 result. A previous response incorrectly presented it as the latest
  V9 score; this entry corrects that statement.

Limits and documentation corrections:

- These checks verify local artifact integrity and reproducibility of the metric
  calculation. They do not establish a public leaderboard score, official
  competition-rule approval, or that the OOF protocol exactly estimates the
  five-model test-time average. OOF uses one held-out model per training row;
  test inference averages five outer-fold models.
- The V9 heading **“100% Audit-Certified”** is too strong and conflicts with the
  explicit absence of organizer adjudication. Describe V9 as a locally
  reproducible benchmark with verified artifacts, not as officially certified.
- The V9 ledger text reports Python 3.11, but the run manifest records Python
  **3.12.10**. The manifest value is the verified run metadata; treat the earlier
  Python 3.11 wording as superseded.
- The claimed commit `8287782c` and remote push could not be verified in this
  checkout: that commit is not present locally, and the ledger changes here are
  uncommitted. Do not describe the update as pushed based only on this checkout.

## 2026-11-07 — Competition-Rule Compliance Gate for V9

This is a compliance-risk assessment based on the locally available
`OFFICIAL_RULES_AUDIT.md`, compliance matrix, and V9 runner. It is not an
organizer ruling.

### Supported by the available evidence

- The local rules audit identifies the supplied competition train/test files as
  the permitted data and explicitly prohibits external data and EDA-driven
  manual applicant-level prediction decisions.
- V9 reads the supplied train, test, and sample-submission CSVs. Its feature
  formulas operate on supplied applicant columns; predictions are generated by
  fitted models, with no identified per-applicant score override or test-to-test
  score coupling in the reviewed V9 source.
- The V9 source constructs outer-fold features from that fold's training data
  and uses inner cross-fitting for its target-derived features. No train/test
  pooled statistic was identified in the reviewed feature-generation path.
- The local rule audit says feature engineering from supplied fields is
  encouraged. Therefore, the `129.50` and `0.0132` formulas are not, by
  themselves, evidence of external-data use or manual prediction editing.

### Still unresolved or requiring stronger evidence

- The supplied rule text does not explicitly address target encoding. V9's
  nested, fold-safe implementation addresses validation leakage but does not
  resolve whether the organizer permits label-derived feature encodings.
- The empirical constants are documented as hypotheses rather than verified
  contract facts. Keep their source limited to supplied data and do not describe
  them as official product terms.
- Repeated feature, model, and version selection using the same five-fold
  results can make the final OOF score optimistic. This is a validation-quality
  risk; it is not, on the evidence reviewed, an explicit competition-rule
  violation. Do not present the OOF score as a guaranteed leaderboard score.
- Citation/license obligations for reused code and any AI-assistance disclosure
  remain subject to the local audit's unresolved findings.

### Recommended submission gate

1. For the lowest rule-ambiguity submission, obtain written organizer
   clarification on target encoding. If clarification is unavailable, create a
   separately versioned conservative ablation that removes all target-encoded
   features; do not overwrite V9 or relabel its results.
2. Keep empirical formula features only with exact source-field formulas,
   derivation records, and no manual per-row prediction adjustment. If seeking
   the most conservative path, compare a second version without the empirical
   constants and report the score difference rather than silently selecting
   one.
3. Freeze feature and model choices before a final validation run. Use a
   predeclared holdout or nested model-selection procedure for the final
   estimate; do not repeatedly tune against the public leaderboard.
4. Before submission, verify the CSV against `sample_submission.csv`: exact
   columns and order, exact ID set and row order, one finite prediction per ID,
   and a recorded SHA-256. Retain the source/input hashes, environment, seed,
   run manifest, and required code attribution.
5. Keep the final wording to “locally reproducible benchmark” unless and until
   organizers confirm any ambiguous technique and an actual leaderboard result
   is independently recorded.

## 2026-11-07 — V10 Compliance and Provenance Review

The V10 OOF metric was independently recalculated from
`oof_predictions_v10_conservative_grade.csv`: 171,202 rows, no null cells,
pooled ROC-AUC **0.7623714502**, and log-loss **0.2449938016**. The OOF file,
both submissions, and executed notebook match their hashes in
`RUN_MANIFEST_V10.json`. The notebook has 7 cells with outputs; both submissions
have the required `SK_ID_CURR,TARGET` columns, 61,500 rows, finite scores, unique
IDs, and IDs in sample-submission order.

### Rule assessment

- The reviewed V10 source has no target-encoding columns, uses only the supplied
  competition files plus the sample schema, and generates scores from fitted
  models. No external dataset, cached prediction input, manual applicant-level
  adjustment, or train/test pooled statistic was identified. On the available
  local rule text, **no explicit prohibited-method violation was identified**.
- This is a lower-ambiguity candidate, not a guarantee of official approval.
  The local rules audit is not an organizer ruling; reproducibility, attribution,
  and submission obligations still apply.
- The V10 report's “exact contribution” wording is too strong. V9-to-V10 removes
  target encodings and empirical pricing features together, so the **-0.005969**
  score difference is a combined ablation only. It cannot attribute a separate
  lift to target encoding versus either pricing feature family, nor prove
  causality.
- V10 is not literally assumption-free. For example, the source contains
  `DISPOSABLE_INC = income / 12 - annuity` and `D_debt = credit + 12 * annuity`,
  fixed copula parameters, volatility values, and term horizons. The official
  data dictionary describes `AMT_INCOME_TOTAL` as “Income of the client” and
  `AMT_ANNUITY` as “Loan annuity”; it does not establish a period for either.
  The existing audit therefore cautions against asserting monthly/annual
  semantics without a source. These formulas are **not established rule
  violations**, but their semantics are unsupported in the reviewed materials.
  Document them as modeling assumptions or remove/ablate them before claiming
  “zero assumptions.”

### Reproducibility discrepancy requiring correction

- `RUN_MANIFEST_V10.json` records runner SHA-256
  `ac24e125cf62371d37ab31aa8d53f6ce98516f558eb2cb000483f8a0bac70ffd`.
  The current runner at `D:\gci\gemini\run_and_build_audit_certified_v10_conservative.py`
  hashes to `76fef1809b61f598b97af614ec42aefe05313a5b80923a17c45c3a7b3053318e`
  (40,408 bytes), and that current content is what the V10 commit tracks.
  The runner's file timestamp is later than the manifest timestamp. The seven
  notebook cell sources match the current runner's embedded cell definitions
  (apart from trailing newlines), but the manifest does **not** authenticate
  the current runner file as the exact runner used for the recorded run.
- This is a reproducibility/evidence gap under the supplied README's
  “Ensuring Reproducibility” requirement, not proof that V10 used prohibited
  data or an automatic disqualification. Preserve the exact run-time script or
  regenerate the manifest from an unchanged source and rerun before claiming
  the runner, notebook, and predictions are one hash-verified execution.

Accordingly, describe V10 as a **lower-ambiguity locally verified candidate**,
not “zero ambiguity,” “100% rule-compliant,” or an officially approved entry.

## 2026-11-07 — Phase 27: Feature Composition Screen

**Status:** Completed and independently verified; exploratory only. This
screen did not reach 0.77 and did not produce a candidate for promotion.

### Protocol and scope

- Used only the supplied `train.csv` (171,202 rows); no test rows, external
  data, cached predictions, applicant identity pooling, or manual score
  adjustment.
- Five stratified folds, seed `20261113`. Every candidate used the same
  LightGBM 4.7.0 settings: 700 estimators, learning rate `0.03`, 31 leaves,
  minimum child samples 80, row/column sampling `0.8`, L2 `5.0`, and
  `max_bin=255`. One-hot vocabularies were fit on each outer training fold;
  unseen validation categories map to all-zero one-hot values.
- The baseline contains raw numeric predictors, deterministic Phase 13
  rowwise/decomposition features, Phase 14 numeric enhancements (excluding
  the aggregate that sums bureau counts over different time windows), Phase
  15 temporal features, and fold-fitted one-hot categories.
- The separate blocks re-used the strongest prior label-free findings:
  application-process missingness, external-score disagreement, financial
  covariate residuals, applicant-profile frequencies, and occupation-relative
  score percentiles. Financial residuals use only supplied monetary columns,
  not `TARGET`; profile frequencies and peer percentiles use outer-training
  rows only.
- Origin-ledger formulas were tested as separate blocks: external-score
  entropy/geometry, the credit-to-income nonlinear interaction, temporal
  invariant fingerprints, and fixed-parameter Clayton/Gumbel score
  transforms. The copula parameters are hypotheses, not official domain
  constants. Fee/tariff keys (`130.5`, `0.0066`, `0.0132`), target encoding,
  the `12`-period income/debt assumptions, unsupported repayment horizons, and
  repeat-applicant score coupling were excluded.

### Results

| Candidate | Pooled OOF AUC | Delta vs baseline | Positive folds |
|---|---:|---:|---:|
| Baseline | `0.760226964` | — | — |
| Application-process missingness | `0.760558419` | `+0.000331454` | `4/5` |
| External-score disagreement | `0.760458729` | `+0.000231765` | `2/5` |
| Financial covariate residuals | `0.760096915` | `-0.000130049` | `2/5` |
| Applicant-profile consistency | `0.760228469` | `+0.000001505` | `3/5` |
| Formula/peer percentiles | `0.760464505` | `+0.000237541` | `3/5` |
| Cross-domain interactions | `0.760449306` | `+0.000222342` | `4/5` |
| Origin formula geometry | `0.760110948` | `-0.000116017` | `2/5` |
| Origin copula transforms | `0.760326638` | `+0.000099674` | `2/5` |
| All blocks combined | `0.758748896` | `-0.001478068` | `1/5` |

Baseline mean fold AUC was `0.760252581 +/- 0.003868977`; baseline pooled
log-loss was `0.245474443`. The all-block combination materially underperformed
the matched baseline, supporting the earlier finding that adding individually
promising or correlated formulas indiscriminately can hurt. No candidate
reached `0.77`. These results use a different model/featureset from V9/V10
and are not directly comparable to those scores. Do not promote a candidate
or interpret small same-label gains as independent confirmation.

### Verification and artifacts

The independent verifier confirmed the 171,202-row OOF ID/target alignment,
all five fold assignments, all ten finite prediction vectors, recomputed AUC
and log-loss values, and the recorded hashes. Two focused unit tests, Python
compilation, and a real-data feature-construction smoke test passed. Runtime
was `665.1` seconds.

Run directory:
`experiments/phase27_feature_composition/runs/phase27_composition_20261107T161240Z`

- Runner:
  `experiments/phase27_feature_composition/run_phase27_feature_composition.py`
  SHA-256 `62986f3161f0404e809d0c03649e26398e071e59052cff835503848cb230c1fd`
- Input training CSV SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- OOF predictions SHA-256:
  `c259dab76a85b44ad9f832938117740ad2ff3d62aafc4f517e1f31c8b46c242b`
- Fold results SHA-256:
  `8675fa4bbb8a4026684c3c314b4017a63af745e5ee9d9914f26a46c497e33d03`
- Summary SHA-256:
  `b80dc3c3673fc298b10025feed664eecd3307e670b4c06976bd7dd9bd3319d5c`

### Decision and limitations

Keep the all-block result rejected. If continuing, first test only a
predeclared small combination of the application-process, external-score,
and peer-percentile blocks against a stronger matched baseline; do not select
weights or features on these same folds and then describe the result as an
unbiased estimate. All competition labels have been examined repeatedly.
The local rules audit is not organizer approval; the copula parameters remain
documented feature hypotheses.

## 2026-11-07 — Append-Only V10 Provenance Recheck

A fresh disk check supersedes the earlier V10 provenance discrepancy recorded
above; that discrepancy reflects an older runner/manifest state. The current
`RUN_MANIFEST_V10.json` runner hash (`32fe1312513ef807079d75b4b17f10780e51471d1819a674774df58312596060`)
matches the current runner. The manifest hashes for `train.csv`, `test.csv`,
the OOF file, both submissions, and the executed notebook also match their
current disk files.

The OOF artifact contains 171,202 rows and no null cells. Recalculation from
`oof_ensemble` gives pooled ROC-AUC `0.762355883080` and log-loss
`0.244998202269`, matching the manifest after rounding (`0.762356` and
`0.244998`). This recheck corrects the stale `0.7623714502` figure and runner
hash discrepancy in the earlier V10 review entry; the older entry is retained
unchanged under the append-only protocol. This confirms artifact provenance
and metric consistency, not competition-organizer approval.

## 2026-11-07 — Phase 28: Fixed V9 / Phase 15 OOF Blend Screen

**Status:** Crossed `0.77` on existing pooled OOF predictions; exploratory
diagnostic only, not a submission or an independently validated score.

### Frozen blend and results

Compared a fixed equal-probability blend of the V9 ensemble OOF vector and
the Phase 15 three-seed CatBoost OOF average:

`0.50 * V9 oof_ensemble + 0.50 * Phase15 phase15_three_seed_average`

| OOF vector | Pooled ROC-AUC | Log-loss |
|---|---:|---:|
| V9 ensemble | `0.768339702` | `0.243472163` |
| Phase 15 three-seed average | `0.768824708` | `0.243228382` |
| Fixed 50/50 blend | **`0.770246587`** | **`0.242807303`** |

Blend delta was `+0.001906885` versus V9 and `+0.001421879` versus Phase 15.
The component prediction correlation was `0.966076538`. IDs and targets were
identical and aligned for all 171,202 rows. Each row has an OOF prediction
from each component; the three Phase 15 seed vectors use different fold
assignments.

### Evidence limits and eligibility

- This exceeds `0.77` only as a **local pooled OOF diagnostic**. It is not a
  public leaderboard score or a reliable estimate from untouched data.
- Multiple existing systems and blend alternatives had already been inspected
  on these labels. The equal blend result is therefore exposed to accumulated
  model/combination selection bias; do not treat it as an unbiased gain or
  claim generalization from it.
- V9 uses target encodings and empirical product-key hypotheses; Phase 15
  uses native categorical target statistics and unresolved product-key
  hypotheses. Local rule interpretation remains unresolved. This result is
  not a compliance determination.
- No Phase 15 test-set predictions were present in the source artifact. No
  submission was generated or changed. Do not copy OOF values into a
  submission. A test-time counterpart would require a separately reproduced
  full-training inference run, artifact validation, and review of rule
  eligibility; even then the OOF result does not guarantee a leaderboard
  improvement.

### Reproducibility

The independent verifier checked exact ID/target alignment, 171,202 finite
rows, the exact 50/50 arithmetic, recalculated ROC-AUC and log-loss, and
source/output hashes.

Run directory:
`experiments/phase28_oof_blend_screen/runs/phase28_equal_blend_20261107`

- Runner SHA-256:
  `439940b0c8be08690210349f4c1cf2a05d1e927ce002740c03ba84d7a0688b53`
- V9 OOF source SHA-256:
  `30341c9fba32bf0767ec4f4371852e972f2439739d97af9c0371e7695a91a1cc`
- Phase 15 OOF source SHA-256:
  `14e92b323a7c5255b506a33c0f8b3925e5f9d110d388c44becf0612aec37a2d6`
- Blended OOF artifact SHA-256:
  `b248492f37153f8122e387245472aef69903e4eecb98917402490eee7519c093`
- Summary SHA-256:
  `9d895a6d2013e4b0f56fb794c138fd344beaae946c66c0d096ec0b5a5dbacedb`

## 2026-11-07 — Phases 29–31: Lower-Ambiguity Scratch Model Comparisons

These are exploratory local cross-validation results, not organizer-certified
results or independent estimates on untouched labels. They use the supplied
train/test files and retain row-level OOF predictions and test submissions.
Independent verification confirmed all three OOF files have 171,202 rows,
finite values, and ID/target alignment with the supplied training file; each
saved submission has 61,500 finite predictions in sample-submission ID order.

### Results

| Phase / model | Pooled OOF ROC-AUC | Pooled log-loss | Interpretation |
|---|---:|---:|---|
| 29 CatBoost, fold-fitted one-hot | `0.757828688` | `0.246224021` | Label-free feature pipeline baseline |
| 29 LightGBM, fold-fitted one-hot | `0.759444198` | `0.245754017` | Best Phase 29 individual model |
| 29 fixed 50/50 model blend | `0.759968349` | `0.245576915` | Does not reach `0.77` |
| 30 CatBoost native categories | `0.765978119` | `0.244075989` | Best new individual model; native category rule interpretation unresolved |
| 30 LightGBM native categories | `0.746144915` | `0.249637341` | Materially weaker |
| 30 fixed 50/50 model blend | `0.762231765` | `0.245264563` | Does not reach `0.77` |
| 31 XGBoost, fold-fitted one-hot | `0.760314927` | `0.245426624` | Clean-scope alternative; does not reach `0.77` |

Phase 29 used deterministic rowwise covariate transforms and one-hot encoding
fit on the training partition, with no custom target encoding or native
categorical target statistics. Phase 30 tested CatBoost/LightGBM native
categorical handling and rounded ratio categories, without the historical
`129.50` or `0.0132` constants or custom target-encoded columns. The
competition wording available in this repository does not expressly resolve
whether native categorical target-statistic internals are permitted; Phase 30
must therefore not be described as confirmed rule-compliant.

Phase 31 uses XGBoost on Phase 29 rowwise features, with one-hot vocabularies
fit within each inner/outer training partition. Its recorded controls exclude
custom target encoding, external data, ID as a predictor, test-derived
statistics, and manual prediction edits. This is a lower-ambiguity technical
description, not organizer approval. Phase 31 completed five outer folds
(pooled AUC `0.760314927`, mean fold AUC `0.760324207`, fold standard
deviation `0.004429922`) in about 59 seconds.

### Conclusions and limits

- None of Phases 29–31 reaches the `0.77` pooled OOF target. Phase 30
  CatBoost is the strongest newly trained single model among these runs, but
  remains about `0.00402` below `0.77`.
- The earlier Phase 28 value above `0.77` came from blending existing V9 and
  Phase 15 OOF vectors after repeated evaluation of the same labels. It is a
  selection-biased retrospective diagnostic and does not establish a
  generalizing score or an eligible test-time method.
- Repeated model/feature comparisons on these labels make all current OOF
  results exploratory. Do not select blend weights on these same OOF results
  and call the resulting score an unbiased estimate.
- No claim is made that any submission will exceed `0.77` on a public or
  private leaderboard. The next useful evidence should come from a
  predeclared, narrowly scoped comparison and, if possible, organizer
  clarification on native categorical target statistics.

### Reproducibility locations

- Phase 29 runner: `experiments/phase29_clean_scratch/run_phase29_clean_scratch.py`
- Phase 29 run: `experiments/phase29_clean_scratch/runs/phase29_clean_20261107T163808Z`
- Phase 29 OOF SHA-256:
  `1673d02d9676d61c711723e86af8fae3538973bde21e3d94b303854621249ed6`
- Phase 30 runner: `experiments/phase30_native_feature_scratch/run_phase30_native_feature_scratch.py`
- Phase 30 run: `experiments/phase30_native_feature_scratch/runs/phase30_native_20261107T164802Z`
- Phase 30 OOF SHA-256:
  `c180fc87f02771efc24b16b1aecf311879dd77308be4ac306eb89b8aefd6b335`
- Phase 31 runner: `experiments/phase31_xgboost_clean_scratch/run_phase31_xgboost_clean_scratch.py`
- Phase 31 run: `experiments/phase31_xgboost_clean_scratch/runs/phase31_xgboost_20261107T165240Z`
- Phase 31 runner SHA-256:
  `3eb24a1a01e299e31954f2d124a7d86b3c2ae70ac9614a581ebca2166560b346`
- Phase 31 OOF SHA-256:
  `8a91f67fc534e7d0bb9b09d8810c687db39d9004425e167011b918191ced87a7`
- Phase 31 pooled OOF metrics and input/artifact hashes are also recorded in
  `PHASE31_SUMMARY.json` in the run directory.

## 2026-11-07 — Phase 32: CatBoost Depth Screen

Ran a predeclared depth comparison on the Phase 30 native-category feature
matrix, keeping the five outer folds, seed, learning rate, regularization,
inner-split early stopping, and feature set fixed. CatBoost's native
categorical target-statistic internals remain unadjudicated by the local
competition rule wording; this is exploratory research, not a compliance
certification.

| CatBoost depth | Pooled OOF ROC-AUC | Mean fold ROC-AUC | Pooled log-loss |
|---:|---:|---:|---:|
| 6 | `0.766328508` | `0.766375343` | `0.244013513` |
| 8 | `0.764849955` | `0.764913182` | `0.244505989` |

Depth 6 is only `0.000350388` above the Phase 30 depth-7 CatBoost result
(`0.765978119`), while depth 8 is weaker. Neither reaches `0.77`. Because
both settings were evaluated on the same repeatedly examined labels, this
small difference is exploratory and should not be represented as a
generalization gain. The run generated OOF artifacts only, not a submission.

OOF IDs and targets were independently verified against the supplied
171,202-row training file, with no null values. Recalculated pooled metrics
match the recorded values. Run directory:
`experiments/phase32_catboost_depth_screen/runs/phase32_depth_20261107T165554Z`

- Runner:
  `experiments/phase32_catboost_depth_screen/run_phase32_catboost_depth_screen.py`
- Runner SHA-256:
  `59532b5276b2012d548702f4bfb01046b30c52028e9740a6eece8f31e2bedcb4`
- OOF SHA-256:
  `4879d595a42812896d644ed6650d16b7c1157533032371d31b609629cf15a13e`
- Fold results SHA-256:
  `86d4a59c1b7aee652f2afb86c07f44f998be58bb6588966f8d052ad4f342b352`

## 2026-11-07 — Phases 33–38: Rules-First Feature and Robustness Experiments

These are exploratory local cross-validation experiments using only the supplied
training data. They are not official competition results, do not establish rule
eligibility, and do not provide an unbiased estimate after repeated inspection
of these labels. No phase in this series reached the `0.77` pooled OOF ROC-AUC
target.

### Results

| Phase / experiment | Pooled OOF ROC-AUC | Pooled log-loss | Result |
|---|---:|---:|---|
| 33 LightGBM baseline | `0.758515` | — | Baseline for the rules-first feature comparison |
| 33 + EXP02 financial ratios | `0.759365` | — | Small increase over Phase 33 baseline |
| 33 + fit-fold peer features | `0.758662` | — | No reliable improvement |
| 33 constant-free rounded ratio keys | `0.763221` | — | Best Phase 33 individual feature variant |
| 33 keys + fit-fold frequencies | `0.763284` | — | Marginal change over keys alone |
| 33 all safe feature blocks | `0.763251` | — | Combining all tested blocks did not improve on keys + frequencies |
| 34 XGBoost, Phase 33 key representation | `0.764736` | `0.244382` | Five-fold one-hot XGBoost |
| 34 fixed 50/50 Phase 33 LightGBM + XGBoost | `0.765335` | — | Fixed blend; still below `0.77` |
| 35 CatBoost with numerical one-hot inputs | `0.759857` | — | No native categorical target statistics |
| 35 fixed three-model average | `0.765152` | — | Weaker than the Phase 33/34 two-model blend |
| 36 expanded financial ratio-key set | `0.764154` | — | Added keys reduced XGBoost performance |
| 36 fixed 50/50 base/expanded-key blend | `0.764894` | — | Did not justify retaining the expanded keys |
| 37 three-seed XGBoost average | `0.765825` | `0.244039` | Best XGBoost-only result in this series |
| 37 fixed 50/50 LightGBM + three-seed XGBoost | `0.765968` | `0.244069` | Strongest lower-ambiguity diagnostic; `0.004032` below `0.77` |
| 38 XGBoost ratio-key one-hot `min_frequency=10` | `0.764571` | `0.244402` | Slightly below the `min_frequency=2` reference |
| 38 XGBoost ratio-key one-hot `min_frequency=50` | `0.764835` | `0.244336` | Only `0.000099` above the Phase 34 `min_frequency=2` reference |

Phase 33's repeatable result was that the constant-free rounded annuity/credit
and annuity/goods ratio keys helped more than the other tested safe feature
blocks. The additional Phase 36 ratio keys did not add value. Phase 38 increased
the minimum key-category frequency used for one-hot encoding, fitting each
vocabulary and frequency table on that model's training partition only. The
`min_frequency=50` gain over its reference is very small, does not exceed the
stronger Phase 37 result, and should not be treated as a generalization gain.

### Rule-scope controls and limitations

- The new pipelines used the supplied training CSV only; they did not read test
  rows, external datasets, cached predictions, or submission files.
- The feature pipelines did not use custom target encoding, native categorical
  target-statistic handling, pooled train/test statistics, the unverified
  `129.50` or `0.0132` constants, or manual prediction edits.
- One-hot vocabularies, frequency counts, and other learned feature transforms
  were fitted on the relevant training partition. Labels were used for
  supervised model fitting and fold evaluation only.
- These controls lower ambiguity but are not an organizer ruling or a
  certification of compliance. The OOF labels have been repeatedly used to
  compare models and blends; Phase 37's best result is especially exposed to
  accumulated selection bias. It is not a public/private leaderboard score.
- No submission was created or modified by Phases 33–38. Do not copy OOF
  predictions into a submission or infer test performance from these metrics.

### Reproducibility and independent verification

Phases 33–37 are under `experiments/phase33_rules_first_features/` through
`experiments/phase37_multiseed_xgboost_rules_first/`. Phase 38 is under
`experiments/phase38_key_frequency_thresholds/`; its run directory is
`experiments/phase38_key_frequency_thresholds/runs/phase38_thresholds_20261107T173546Z`.

The Phase 37 OOF artifact was independently checked against all 171,202
training IDs and targets; all three per-seed fold columns contain each fold
with expected coverage, all predictions are finite and in `[0,1]`, and the
three-seed arithmetic average reproduces from its components. Recalculated
pooled AUC and log-loss match the recorded rounded metrics.

The Phase 38 artifact was independently checked for exact ID/target alignment,
171,202 finite predictions in `[0,1]`, and five-fold coverage. Recalculated
pooled metrics match the summary to CSV precision.

- Phase 37 OOF SHA-256:
  `33563509db7afb33d9876d8b6a8e8b862859adef54dacf0a94df821f0da9c44a`
- Phase 37 fold-results SHA-256:
  `7ea71f8e778d10572dbd85253c3b9ba21524ad8e12f4eaf4965fd8dd6512e29b`
- Phase 38 runner:
  `experiments/phase38_key_frequency_thresholds/run_phase38_key_frequency_thresholds.py`
- Phase 38 runner SHA-256:
  `4bf7bcaaca70b9c450fd4257fb566335437aa9f58aea510a3eb28895bf40f3aa`
- Phase 38 OOF SHA-256:
  `b7dd64efa160bec9ab3c99a01cd1597a12a673cf37a76a24abda25f4eafa7c76`
- Phase 38 fold-results SHA-256:
  `cf9882769316ab7f28bf5a912ace31a085ebaaf3aed9648a04eae829230bd9c3`

## 2026-11-07 — Phases 39–40: Financial Block and XGBoost Complexity Ablations

These follow-up experiments retained the Phase 33 constant-free ratio-key
representation, fold-fitted one-hot encoding, and the same five outer folds.
They used the supplied training data only and did not add target-derived
features, test data, external data, or manually adjusted predictions.

| Phase / experiment | Pooled OOF ROC-AUC | Pooled log-loss | Comparison |
|---|---:|---:|---|
| 39 keys + frequencies (`min_frequency=50`) reference | `0.764835` | `0.244336` | Phase 38 reference |
| 39 add seven EXP02 financial ratios | `0.764591` | `0.244419` | `-0.000244` AUC; rejected |
| 40 depth 4 reference | `0.764835` | `0.244336` | Phase 38 reference |
| 40 depth 3, child weight 20 | `0.764520` | `0.244454` | Lower than reference |
| 40 depth 5, child weight 60 | `0.764480` | `0.244511` | Lower than reference |

Neither the additional financial ratios nor the tested depth/regularization
alternatives improved the reference. The per-fold AUCs for some folds exceed
`0.77`, but pooled results do not; those individual folds are not evidence that
the target has been met. The strongest lower-ambiguity diagnostic in this
series therefore remains Phase 37's fixed blend at `0.765968`, which is
`0.004032` below `0.77`. All results remain exploratory and selection-biased
by repeated use of the same labels; none predicts a leaderboard score.

An initial Phase 40 output directory
(`phase40_xgb_depth_20261107T174314Z`) contained the reference column but
omitted the two candidate prediction columns because of an output-frame
construction error. It is superseded and must not be used. The runner was
corrected and rerun; the verified complete artifact is in
`experiments/phase40_xgb_depth_screen/runs/phase40_xgb_depth_20261107T174605Z`.

The final Phase 39 and corrected Phase 40 OOF artifacts were independently
verified against all 171,202 training IDs and targets. Fold coverage is
complete, all saved predictions are finite and in `[0,1]`, and recalculated
pooled metrics match the summaries to CSV precision.

- Phase 39 runner:
  `experiments/phase39_key_plus_financial_xgb/run_phase39_key_plus_financial_xgb.py`
- Phase 39 runner SHA-256:
  `3a663b9a0e290ea04397156f80ec6d15cf48967dc1b0046ba2157bfdd5d6f061`
- Phase 39 OOF SHA-256:
  `d359b1ff2e8f2236c39eed562ad6c935cfc4c4a608d764f7d668dd622343fbc5`
- Phase 39 fold-results SHA-256:
  `d02e3b8d0d9ffd1b01a2b715825cd30aab84bb068eb77fcd416b258482c0b8c5`
- Phase 40 runner:
  `experiments/phase40_xgb_depth_screen/run_phase40_xgb_depth_screen.py`
- Phase 40 runner SHA-256:
  `a422f8a3d3e74b02c1e358418bfd180778e69f50a204cfbee0f2fa3fc610a701`
- Phase 40 OOF SHA-256:
  `10b3e9aa14a70e0b0cc3ca5f209791ac17e19253518eb4a7e4d76eb64a4104a5`
- Phase 40 fold-results SHA-256:
  `5b14103f51918bf521fa198947b21d8edd926e5c950e9eeb4e1d5e869b217dbe`

## 2026-11-07 — Phases 41–42: Employment History and Information Availability

These experiments tested two unlabelled covariate hypotheses from the earlier
research notes, keeping the Phase 38 XGBoost feature set and fold design fixed.

| Phase / experiment | Pooled OOF ROC-AUC | Pooled log-loss | Comparison |
|---|---:|---:|---|
| 41 keys + frequencies (`min_frequency=50`) reference | `0.764835` | `0.244336` | Phase 38 reference |
| 41 add employment-age ratio and estimated pre-employment age | `0.764809` | `0.244362` | `-0.000026` AUC; no useful lift |
| 42 keys + frequencies (`min_frequency=50`) reference | `0.764835` | `0.244336` | Phase 38 reference |
| 42 add joint missingness/sentinel profile | `0.764495` | `0.244413` | `-0.000339` AUC; rejected |

Phase 41 used `DAYS_EMPLOYED` with the sentinel mapped to missing and derived
two covariates: employment-years divided by age-years, and age-years minus
employment-years. The latter is only an arithmetic gap, not a claim about
actual age at first employment. The test did not improve the reference.

Phase 42 formed a categorical bit-pattern from missingness in the three
external scores, occupation, organization, income type, two bureau inquiry
fields, and the employment sentinel. It also included the profile's bit count.
The profile vocabulary was one-hot encoded using only each model's training
partition. This information-availability hypothesis did not improve the
reference.

The final OOF artifacts for both phases were independently checked against all
171,202 training IDs and targets; fold coverage is complete and all saved
predictions are finite and in `[0,1]`. Recalculated pooled metrics match the
summaries to CSV precision. The rule-scope limitations documented above apply:
these are exploratory OOF diagnostics, not organizer rulings or leaderboard
scores.

- Phase 41 runner:
  `experiments/phase41_employment_trajectory_xgb/run_phase41_employment_trajectory_xgb.py`
- Phase 41 runner SHA-256:
  `3e5f02ecb0aa9de084e895d5ce6e191b44b103178051c455300d4690ce0f2455`
- Phase 41 OOF SHA-256:
  `bd1ba26e3945d7bad26464a5228e458b6b787f429cc68f1d0ce444ebd49e597e`
- Phase 41 fold-results SHA-256:
  `cc8f42a94d44379dc23d504d0b681291d94fa7e8d5a52490f23aec664e79b52b`
- Phase 42 runner:
  `experiments/phase42_information_profile_xgb/run_phase42_information_profile_xgb.py`
- Phase 42 runner SHA-256:
  `77e65a27766de642e7ce80a9be4e46a699f2326bd1ade20aac89ec6ae9514557`
- Phase 42 OOF SHA-256:
  `dbab1d2c8789ec6ef0389f41c48193523d74a5d5406a8d6f19c18463c2086814`
- Phase 42 fold-results SHA-256:
  `d967f3534af465d9815fb504f4720c92e549fd3768c2586d6d8bd4f3d4d10bcb`

## 2026-11-07 — Phase 43: Origin Formula Review and Safe Bureau-Interaction Ablation

Reviewed the formulas in `D:\gci\ORIGIN_MASTER_VERIFICATION_LEDGER.md` and the
user-provided formula compendium. The most defensible formulas to test under
the current conservative data boundary were the three-score product, the
three-score harmonic mean, and a direct income-to-annuity ratio interacted
with the squared `EXT_SOURCE_2 * EXT_SOURCE_3` term. They are deterministic
transforms of provided columns; they do not require outside data or target
statistics.

The test intentionally did not copy the compendium's full plug-in block:

- The three bureau scores are not established calibrated, independent
  probabilities, so the “joint likelihood” explanation for their product is
  a hypothesis, not a proven interpretation.
- The product and harmonic mean were kept missing unless all three scores
  were present (and positive for the harmonic mean). No global median
  imputation was used.
- The solvency interaction used `AMT_INCOME_TOTAL / AMT_ANNUITY` directly;
  no unsupported `/12` “monthly income” conversion was introduced.
- Fee offsets `129.5`/`130.5`, tariff steps `0.0066`/`0.0132`, arbitrary
  subsistence amounts, and prime-centroid anchors were excluded as
  unverified hypotheses. Repeat-application quantities such as `N_Apps` and
  `DAYS_SINCE_PREV_APP` are absent from the supplied 34-column training file,
  so formulas requiring them were not eligible for this test.
- The origin ledger's formula-level “within-split” AUC and tree split-gain
  shares are not comparable to pooled outer-fold OOF AUC and do not establish
  out-of-sample benefit or causal mechanism.

### Results

All variants used the same Phase 38 XGBoost baseline, five stratified outer
folds, and inner-fold early stopping. Formula construction was rowwise and
label-free; one-hot vocabularies and key-frequency features remained fit on
each training partition.

| Model / added formula block | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs Phase 38 |
|---|---:|---:|---:|
| Phase 38 `min_frequency=50` reference | `0.764835` | `0.244336` | — |
| Add `EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3` | `0.764319` | `0.244458` | `-0.000516` |
| Add three-score harmonic mean | `0.764692` | `0.244346` | `-0.000143` |
| Add `(income / annuity) * (EXT_SOURCE_2 * EXT_SOURCE_3)^2` | `0.764611` | `0.244388` | `-0.000224` |
| Add all three formulas | `0.764368` | `0.244455` | `-0.000467` |

The harmonic mean was the least harmful of the individual candidates and had
slightly better log-loss than the other additions, but it still reduced AUC
against the matched reference. Combining the features did not recover the
difference. **None is retained as a demonstrated improvement**, and no
Phase 43 result reaches `0.77`. The current strongest lower-ambiguity
diagnostic remains the Phase 37 fixed blend at `0.765968`; it remains
exploratory and selection-biased, not a leaderboard guarantee.

The supplied training data has all three external scores for only 36,164 of
171,202 rows (`21.124%`); the `EXT_SOURCE_2`/`EXT_SOURCE_3` solvency interaction
is defined for 116,393 rows (`67.986%`). This limited support is consistent
with treating the product/harmonic results cautiously; missingness itself is
already represented in the baseline inputs.

Phase 43 used only `train.csv`, produced OOF artifacts only, and independently
verified exact ID/target alignment, complete five-fold coverage, finite
predictions in `[0,1]`, and recalculated pooled metrics. No submission was
created, and `ORIGIN_MASTER_VERIFICATION_LEDGER.md` was read-only during this
review.

- Runner:
  `experiments/phase43_bureau_formula_ablation/run_phase43_bureau_formula_ablation.py`
- Runner SHA-256:
  `8a17ebd07510b67c3d7dcc64b13846c094c864d73a286c8ac661b2a1c6a6e1ec`
- OOF SHA-256:
  `94a7cde25eff2b8d076e335176f057023c5245b95509f0e7c2fffa4014e90215`
- Fold-results SHA-256:
  `f700b725615bc07fda2165904319ed05bb63cf378299bf2481d2a80197974e15`

## 2026-11-08 — Phase 44: Fold-Safe Feature-Routed Expert Models

This experiment revisited data splitting and specialized prediction using the
evidence from Phases 20–21. Those studies found contract-type routing to be the
most promising of the tested feature-only gates, while hard routing was weaker
than global prediction and fixed global/expert mixtures were more competitive.
Phase 44 tests that pattern with the newer Phase 38 XGBoost/constant-free-key
representation, on identical full-data outer folds.

### Method

- Five stratified outer folds (`random_state=20261113`); all models select
  iteration count on an inner 12% split of their own outer-training data and
  are refit on the full corresponding outer-training partition.
- Each gate is a deterministic function of supplied features only:
  `NAME_CONTRACT_TYPE` (cash versus revolving), or all-three-`EXT_SOURCE`
  completeness.
- For each gate, each specialist is trained only on its gate subgroup within
  the outer-training rows. Outer validation rows are scored only by the expert
  assigned by the same covariate-only gate.
- Compared a global XGBoost model, hard-routed experts, and a predeclared
  50/50 probability average of global and routed expert predictions. No
  prediction weight was fitted on outer-fold labels.
- No test data, external data, target encoding, native categorical target
  statistics, unverified constants, cached prediction vectors, or manual
  prediction adjustments were used.

### Pooled results

| Model | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs global |
|---|---:|---:|---:|
| Global XGBoost | `0.764835` | `0.244336` | — |
| Contract-type hard routing | `0.764016` | `0.244591` | `-0.000818` |
| Contract-type global/expert 50/50 mixture | `0.765192` | `0.244240` | `+0.000358` |
| External-completeness hard routing | `0.761118` | `0.245333` | `-0.003716` |
| External-completeness global/expert 50/50 mixture | `0.764303` | `0.244434` | `-0.000532` |

Contract-type specialists are the only positive split result in this matched
test, and only when blended with the global model. This replicates the
direction—not the exact score—of Phase 21's modest contract-type mixture
finding. Hard routing discards useful cross-segment ranking information; the
external-completeness experts are not useful in this configuration. The
contract-type mixture remains `0.000776` below the strongest prior
lower-ambiguity diagnostic (Phase 37 fixed blend `0.765968`) and does not
reach `0.77`. It is therefore a research finding, not a new best candidate or
leaderboard claim.

Phase 44 OOF predictions were independently verified against all 171,202
training IDs and targets; all five folds are fully represented and all
prediction vectors are finite and in `[0,1]`. Recalculated pooled AUC and
log-loss match the run summary. All findings remain exploratory because the
same labels have been examined in earlier experiments.

- Runner:
  `experiments/phase44_fold_safe_experts/run_phase44_fold_safe_experts.py`
- Runner SHA-256:
  `4d60d539a89d85a2600a95d36643e48ad06e41ffb16fe863d56a25b130e50f6f`
- OOF SHA-256:
  `75c91c7d5cf321d76f5034efcc7aed638883dfb0ea817c0b68a319d61f22e96c`
- Fold-results SHA-256:
  `57eebf691af98b505a6e4210e5a83f86061177522a0b0cd22982665568e78118`

## 2026-11-08 — Phase 45: Nested Calibration of Contract-Type Experts

Phase 45 followed the Phase 44 contract-type mixture with per-group Platt
calibration, matching the leakage-safe calibration idea used in Phase 21.
For each outer fold and contract subgroup, the calibrator was fit only to
three-fold OOF expert predictions generated inside that outer-training
subgroup. It was then applied to the outer-fold expert predictions; the
outer-validation labels were not used for calibration or model selection.

| Model | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs global |
|---|---:|---:|---:|
| Global XGBoost | `0.764835` | `0.244336` | — |
| Contract-type hard routing | `0.764016` | `0.244591` | `-0.000818` |
| Fixed global/expert 50/50 mixture | `0.765192` | `0.244240` | `+0.000358` |
| Per-group Platt-calibrated 50/50 mixture | `0.765198` | `0.244245` | `+0.000363` |

Nested calibration changed pooled AUC by only `+0.000005` versus the fixed
mixture and slightly worsened pooled log-loss. The difference is too small to
justify preferring the calibrated model; retain the simpler fixed mixture
only as an exploratory split-model candidate. Its pooled score remains below
Phase 37's `0.765968` diagnostic and does not reach `0.77`. All OOF evidence
remains selection-biased by repeated use of these labels.

The final OOF artifact was independently checked for exact training ID/target
alignment, complete fold coverage, finite predictions in `[0,1]`, calibration
record completeness, and recalculated metrics.

- Runner:
  `experiments/phase45_nested_expert_calibration/run_phase45_nested_expert_calibration.py`
- Runner SHA-256:
  `b16578b6ed2449b11209174e6c01eddadd9e1f2fada2c8cf9772a120b90b7925`
- OOF SHA-256:
  `02a50012df2c60dc38cfd7712fedebedd59dc60aa6780ddf2d2fd1ebcc00bdc0`
- Calibration-results SHA-256:
  `97c5a052c3ea969a81e66e55688f7a950201d0c23418a9b5c77a1e5c939447e0`
- Fold-results SHA-256:
  `147a1ba7bba537b52b6a93110e80fb8f1338b5f6aeb4748f621f1ccbc3cf39d7`

## 2026-11-08 — Phase 46: Nested OOF Selection of the Contract-Expert Weight

Phase 46 tested whether the contract-type expert mixture should use a
training-fold-selected global/expert weight rather than the fixed 50/50
average. For each of the five outer folds, three-fold OOF predictions were
generated entirely inside that fold's outer-training rows for both the global
model and contract-routed experts. A single expert weight was then selected
from the predeclared grid `{0.0, 0.1, ..., 1.0}` by inner OOF AUC and applied
to the outer validation predictions. The outer validation labels were not
used to select the weight or fit any model.

| Model | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs global |
|---|---:|---:|---:|
| Global XGBoost | `0.764835` | `0.244336` | — |
| Contract-type hard routing | `0.764016` | `0.244591` | `-0.000818` |
| Fixed 50/50 global/expert mixture | `0.765192` | `0.244240` | `+0.000358` |
| Nested-selected mixture weight | `0.765218` | `0.244232` | `+0.000383` |

Chosen expert weights by outer fold were `0.4, 0.3, 0.4, 0.2, 0.6`. The
variation is a caution against treating the inner-selected weights as a stable
global optimum. The nested-selected mixture beats the fixed mix by only
`0.000025` AUC and remains `0.000750` below Phase 37's stronger `0.765968`
diagnostic. It does not reach `0.77`; do not promote it as a new best model.
The fixed 50/50 contract mixture is simpler and has nearly the same measured
performance. Both remain exploratory and subject to cumulative selection
bias from prior label reuse.

Phase 46 used only the supplied training CSV and from-scratch fold predictions;
it did not load prior OOF caches, test data, external data, or submissions.
Independent verification confirmed exact ID/target alignment, complete folds,
finite predictions in `[0,1]`, five per-fold weights selected from training
data only, and recalculated pooled metrics.

- Runner:
  `experiments/phase46_nested_expert_weight/run_phase46_nested_expert_weight.py`
- Runner SHA-256:
  `cad34797abf7ead35b150f01707353ec48905088f7138b375d487c20c4e45ce6`
- OOF SHA-256:
  `3f1ff2e2add0871167284025b3ec2830225506451ba2ecb31272f87df6e6729d`
- Weight-selection SHA-256:
  `cd0305ccbbd7a8947be6ee025e2c6b351d437b229c6622aeff775f5513a5fd9d`
- Fold-results SHA-256:
  `8a48b3da094045047f565005b40b1034c8652a5c0e0b51d03ab2f6372a195d6f`

## 2026-11-08 — Phase 47: Fold-Safe Multi-Partition Expert Models

Phase 47 evaluated applicant partitions supported by prior evidence from
the Phase 20/21 segmentation studies, testing whether partitioning beyond
contract type improves pooled out-of-fold discrimination across all rows.
On identical 5-fold stratified outer folds (`random_state=20261113`) using
the Phase 38 XGBoost/constant-free-key representation:

- Tested gates:
  1. `contract_type`: Cash loans (`0`) vs Revolving loans (`1`).
  2. `goods_credit_ratio`: `AMT_GOODS_PRICE / AMT_CREDIT >= 0.90` (`0`, merchandise POS)
     vs `< 0.90` or missing (`1`, cash loans with financed fee margins).
  3. `pensioner_status`: Actively employed (`0`) vs Pensioner / employment
     sentinel (`1`, `DAYS_EMPLOYED == 365243`).
  4. `contract_goods_bivariate`: 3-way triage (Revolving=`0`, Cash POS=`1`, Cash fee=`2`).
  5. `consensus_expert_mixture`: Predeclared combination of `0.50` global +
     `0.25` contract specialist + `0.25` goods/credit specialist.

### Pooled results

| Model | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs global |
|---|---:|---:|---:|
| Global XGBoost | `0.764835` | `0.244336` | — |
| Contract-type hard routing | `0.764016` | `0.244591` | `-0.000818` |
| Contract-type 50/50 mixture | `0.765192` | `0.244240` | `+0.000358` |
| Goods-credit ratio hard routing | `0.759660` | `0.245788` | `-0.005175` |
| Goods-credit ratio 50/50 mixture | `0.763972` | `0.244570` | `-0.000863` |
| Pensioner status hard routing | `0.761684` | `0.245166` | `-0.003151` |
| Pensioner status 50/50 mixture | `0.764475` | `0.244413` | `-0.000360` |
| Contract-goods bivariate hard routing | `0.759839` | `0.245747` | `-0.004995` |
| Contract-goods bivariate 50/50 mixture | `0.764306` | `0.244467` | `-0.000529` |
| Consensus expert mixture | `0.765072` | `0.244265` | `+0.000237` |

### Key Findings
1. **Contract-type remains the sole positive gate**: Only the contract-type
   mixture (`+0.000358`) improved over the global model.
2. **Hard routing fails across all gates**: Discarding cross-cohort training
   data severely damages out-of-fold discrimination (`-8.2` to `-51.7 bps`).
3. **Goods/credit ratio and pensioner partitioning harm pooled performance**:
   While Phase 20 suggested potential within-segment differences on reduced
   samples, evaluating on full-data outer folds shows that gradient boosted
   trees handle these continuous covariates and indicators better globally than
   discrete expert routing.
4. **Conclusion on 0.77 Goal**: Specialist routing alone does not bridge the
   gap to `0.77`. Phase 37's multiseed blend (`0.765968`) remains the highest
   diagnostic benchmark.

- Runner:
  `experiments/phase47_evidence_partition_experts/run_phase47_evidence_partition_experts.py`
- Runner SHA-256:
  `aeee3ad3140a76fb5b9614540cdf14f0ce456fdea89cfe8f944e9d21dac48149`
- OOF SHA-256:
  `8a56099273646f58987d65006cfdc131cb0022e6f3d1a0290aaadfefb827a97f`
- Fold-results SHA-256:
  `0b0db74ca36b7a4711d46d5683cf3b8ca5bae982c54ae8fe98a1915e7ba5031a`

## 2026-11-08 — Phase 48: Multi-Architecture Contract Experts & Cross-Model Synthesis

Phase 48 addressed whether specialist routing on contract type improves upon
the strongest cross-model benchmark (LightGBM + XGBoost), or whether its
signal is absorbed when diverse model architectures are combined.
On identical 5-fold stratified outer folds (`random_state=20261113`) using
the Phase 38 representation (`KEY_MIN_FREQUENCY=50`):

- **Architectures evaluated**:
  - `LightGBM`: `num_leaves=31`, `lr=0.03`, `min_child_samples=50`, `colsample=0.8`, `subsample=0.8`
  - `XGBoost`: `max_depth=4`, `lr=0.035`, `min_child_weight=40`, `colsample=0.82`, `subsample=0.82`
- **Configurations compared**:
  1. Global baseline for each architecture (`lgb_global`, `xgb_global`).
  2. Contract-type hard routing for each architecture.
  3. Contract-type 50/50 mixture for each architecture.
  4. Cross-architecture global blend (`0.5 * lgb_global + 0.5 * xgb_global`).
  5. Cross-architecture expert blend (`0.5 * lgb_mixture + 0.5 * xgb_mixture`).

### Pooled results

| Model / Blend | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs XGB Global | Delta vs Cross Global |
|---|---:|---:|---:|---:|
| `lgb_global` | `0.764843` | `0.244413` | `+0.000008` | — |
| `lgb_contract_hard` | `0.763583` | `0.244839` | `-0.001252` | — |
| `lgb_contract_mixture_50_50` | `0.765126` | `0.244369` | `+0.000291` | — |
| `xgb_global` | `0.764835` | `0.244336` | — | — |
| `xgb_contract_hard` | `0.764016` | `0.244591` | `-0.000818` | — |
| `xgb_contract_mixture_50_50` | `0.765192` | `0.244240` | `+0.000358` | — |
| `cross_model_global_blend` | `0.765710` | `0.244093` | `+0.000875` | — |
| `cross_model_expert_blend` | `0.765858` | `0.244082` | `+0.001023` | `+0.000148` |

### Key Findings
1. **Specialist mixture behavior replicates across architectures**: LightGBM
   exhibits the exact same pattern as XGBoost: hard routing harms performance
   (`-12.5 bps`), while the 50/50 probability mixture yields a clean positive
   lift (`+2.9 bps` for LightGBM, `+3.6 bps` for XGBoost).
2. **Contract-expert signal is additive to cross-architecture ensembling**:
   Combining the two contract mixtures (`cross_model_expert_blend`) achieves
   `0.765858` (+1.5 bps over the cross-model global blend `0.765710`), with
   log-loss improving from `0.244093` down to `0.244082`.
3. **The Feature/Architecture Ceiling**: The cross-model expert blend reaches
   within `0.00011` of Phase 37's 3-seed benchmark (`0.765968`) on a single
   seed, confirming that ensembling across diverse algorithms and contract
   mixtures reliably achieves `~0.7659 - 0.7660`. However, without introducing
   new interaction terms (e.g., hidden pricing offsets or term quantizations),
   relying solely on split-model routing cannot reach `0.7700`.

- Runner:
  `experiments/phase48_cross_architecture_contract_experts/run_phase48_cross_architecture_contract_experts.py`
- Runner SHA-256:
  `60fe9a38e09b09e11b5127aae90cdac65f7b31b2200642789c7d921381d1a45d`
- OOF SHA-256:
  `2d76f452959c96bb464d2852b1b41498b7e097fd37c811ac2c34f8ba78e15277`

## 2026-11-08 - Phase 50: Multi-Seed Pricing Synthesis

Phase 50 tested whether the Phase 49 pricing-feature and contract-expert
signal was robust to model seed and complementary to the Phase 37 ensemble.
The Phase 49 pipeline and model settings were held constant; only the
stratified outer-fold/model seed changed. The three Phase 49 cross-model
expert OOF vectors (seeds `20261113`, `20261114`, `20261115`) were averaged
with fixed equal weights. That result was then averaged 50/50 with the
existing Phase 37 three-seed LightGBM/XGBoost OOF blend. No blend weights
were selected from these OOF labels.

### Results

| Model / blend | Pooled OOF ROC-AUC | Pooled log-loss |
|---|---:|---:|
| Phase 37 three-seed reference blend | `0.765968` | `0.244069` |
| Phase 49 expert blend, seed `20261113` | `0.765709` | `0.244023` |
| Phase 49 expert blend, seed `20261114` | `0.766612` | `0.243858` |
| Phase 49 expert blend, seed `20261115` | `0.765447` | `0.244158` |
| Phase 49 expert blend, three-seed average | `0.766866` | `0.243730` |
| Phase 37 + Phase 49 three-seed expert, fixed 50/50 | **`0.767070`** | **`0.243694`** |
| Phase 37 + Phase 49 three-seed global blend, fixed 50/50 | `0.766990` | `0.243690` |

The fixed expert synthesis improves pooled OOF AUC by `0.001102` over Phase
37, but remains `0.002930` below `0.770000`. The seed-level range
(`0.765447` to `0.766612`) shows that single-seed comparisons are noisy;
averaging across seeds improves stability. Contract experts provide only a
small additional gain over the Phase 49 global blend in this synthesis
(`0.000080` AUC for the final 50/50 blends), so the larger observed gain is
mainly from the multi-seed Phase 49 ensemble being complementary to Phase 37,
not evidence that routing alone solves the remaining gap.

### Validation and compliance scope

- Each prediction component is OOF for its row under that component's own
  stratified seed split. Predictions were aligned by unique training ID and
  target; all 171,202 rows are present, with all five Phase 37 reference
  folds represented and no missing predictions.
- Phase 50 used the supplied training data and previously generated OOF
  artifacts only; it used no test data, external data, target encoding, or
  label-fitted blend weights.
- The Phase 49 pricing formulas retain the empirical constants `129.50` and
  `0.0066`. Fold-fitting the external-score medians does not empirically
  validate those constants or resolve their competition-rule status. This
  experiment is not organizer-certified.
- These results are exploratory and selection-biased because the same
  training labels have informed many preceding phases. The equal weights
  were fixed for this run, but Phase 50 was motivated by observed Phase 37-49
  results; it is not an independent confirmation or a leaderboard guarantee.
- Phase 48's earlier `~0.7659-0.7660` wording describes the then-observed
  benchmark neighborhood only. It is not a theoretical or information-
  theoretic AUC ceiling. No valid Bayes-AUC ceiling was established by these
  experiments; an asserted mutual-information estimate alone cannot certify
  that `0.7700` is mathematically impossible.

Runner:
`experiments/phase50_multiseed_pricing_synthesis/run_phase50_multiseed_pricing_synthesis.py`

Run directory:
`experiments/phase50_multiseed_pricing_synthesis/runs/phase50_multiseed_20261108`

Runner SHA-256:
`516b0199f0226ce1f9e423998e1a425b352dab1083cdae4503079e8708a28ac6`

Test SHA-256:
`537aafbf69ef171c02f8ca7eaddc61dac77d0ba152f479f0118deec1928c15b4`

OOF SHA-256:
`214e1080b89b776b115354b73b01189f296643949dfb24416704a74dec93e4ed`

## 2026-11-08 - Phase 51: Fold-Local Fingerprint Label Statistics

The user supplied several external-agent summaries. Their claims were
classified before use:

- **Previously measured project evidence:** Phase 47 shows hard expert routing
  underperformed and Phase 48/49 found only a small contract-mixture lift.
  Phase 50's fixed multi-seed synthesis reached `0.767070` pooled OOF AUC.
  These refer to saved project artifacts and remain exploratory OOF results.
- **Unverified leads:** the supplied HistGradientBoosting feature ladder, the
  claimed exact twin-group counts, external-score correlations, conditional
  default rates, expected lift ranges, and projected leaderboard gains did
  not include code or hash-bound artifacts in the material supplied here.
  They were not treated as measured results.
- **Unsupported conclusion:** the claim that `0.7700` is mathematically
  impossible, including the asserted mutual-information-to-AUC ceiling, is
  not established by the reported experiments. OOF plateaus and missingness
  do not prove a Bayes-AUC upper bound.
- **Rule interpretation:** the local competition audit finds no explicit
  target-encoding clause, so use of training-label-derived group statistics
  remains unresolved and requires organizer clarification. EDA-driven
  row-specific prediction overrides are a separate, higher-risk method and
  were not used here. No test rows were read or matched in Phase 51.

Phase 51 tested a concrete version of the proposed fingerprint as a feature
inside an XGBoost model, not as a hand-applied score adjustment. The
fingerprint was explicitly defined as absolute `DAYS_BIRTH -
DAYS_REGISTRATION` and `DAYS_BIRTH - DAYS_ID_PUBLISH` gaps plus
`CODE_GENDER`, `NAME_EDUCATION_TYPE`, and `REGION_RATING_CLIENT`. Incomplete
date keys were made row-unique so missing values cannot spuriously join
unrelated applicants. Features were a smoothed group default rate, log group
count, and match indicator.

For each outer fold, the group-rate/count features for model-fit rows were
cross-fitted across five inner folds. Inner-stopping validation features used
only inner-fit labels. Outer-validation group features used only outer-fit
labels. The Phase 49 XGBoost pricing-global model on the same outer folds was
the matched reference; a fixed 50/50 reference/model blend was also reported.

### Results and direct fingerprint check

| Model | Pooled OOF ROC-AUC | Pooled log-loss | Delta vs Phase 49 XGBoost global |
|---|---:|---:|---:|
| Phase 49 XGBoost pricing-global reference | `0.764771` | `0.244227` | - |
| XGBoost with fold-local fingerprint features | `0.765235` | `0.244095` | `+0.000464` |
| Fixed 50/50 fingerprint/reference blend | `0.765192` | `0.244103` | `+0.000421` |

Under this exact fingerprint definition, the training file has 2,086 repeated
key groups containing 4,199 rows (2.45%); each outer validation fold had
approximately 1.9-2.1% of its rows match a key in that fold's training data.
For rows whose key had another training row with a default, the row-level
default rate was 31.0%; for rows with matched non-default peers it was 4.6%;
rows with no peer had an 8.1% default rate. These are descriptive statistics,
not independent causal evidence. They differ from the agent-supplied exact
figures (1,702 groups and 41.7% / 3.7% conditional rates); because the
agent's precise key construction was not specified with executable artifacts,
those figures are not considered replicated.

An exploratory, post-hoc equal blend of the Phase 51 model with the Phase 50
ensemble scored `0.766779`; this is not an independent confirmation or a
predeclared Phase 51 candidate. The model-to-Phase-50 prediction correlation
was `0.9874`, indicating substantial overlap and explaining why the
incremental blend gain is modest. The main Phase 51 result is the matched
`+0.000464` lift over Phase 49's XGBoost global reference; it does not reach
`0.7700`, and no test submission was generated.

### Use of the collected information

1. Continue to treat contract-type soft mixtures and multi-seed cross-model
   averaging as modest measured signals, not a path proven to reach `0.77`.
2. The fingerprint label-statistic feature is a promising but
   rule-ambiguous research branch. Keep it separate from a strictly
   label-free candidate; do not promote it for submission until the
   organizer clarifies whether such training-label-derived features are
   allowed.
3. Do not use manual logit shifts, selected-row overrides, or the agent's
   projected additive gains. The Phase 51 implementation instead lets the
   model consume nested, fold-local features.
4. Do not state that the feature space has a proven `0.766` ceiling. Current
   experiments establish only measured performance for the tested pipelines.

Phase 51 artifacts:

- Runner:
  `experiments/phase51_fold_local_twin_features/run_phase51_fold_local_twin_features.py`
- Tests:
  `experiments/phase51_fold_local_twin_features/test_phase51_fold_local_twin_features.py`
- Run:
  `experiments/phase51_fold_local_twin_features/runs/phase51_twin_20261108`
- Runner SHA-256:
  `ae45ceb2f173dcce835d4d4a9944425c9ed6be03f654a700632c122818119597`
- Test SHA-256:
  `ddcc3047d560458506062c255e80d6dd9fc852055e31f3521f605d89af1680d0`
- Summary SHA-256:
  `26fcdfc88352de027746ea206c6799b630357c55e3e1cf58880ddb90ba75ef40`
- OOF SHA-256:
  `357b192d9ad0a8b360f3e63bf30a437736b3ecd40a693732a8ffad10dd80d66b`

## Forensic audit of newly supplied materials

Completed a read-only audit of `zip.zip`, `notepade.txt`, `process.txt`,
`GCI_v7_extra_trees_colab_1.ipynb`, `GCI_v13_reweight_and_ET_colab_1.ipynb`,
and `GCI_hgb_product_test.npy`. The detailed per-file findings, feature
descriptions, score classification, archive coverage, caveats, SHA-256 values,
and complete top-level/nested member inventory are in
`reports/NEW_MATERIALS_FORENSIC_AUDIT.md`.

Key evidence distinction: both newly supplied notebooks have null execution
counts and empty outputs. Their quoted 0.769–0.771 OOF/holdout values and
0.77865 public score are not results generated by these notebook files. The
archive does contain historical saved notebook outputs, local OOF vectors,
and public-score claims; these are kept as separate evidence classes. Test
submission CSVs do not contain labels and cannot verify public AUC.

The archive review covered 373 root members and 429 members across 30
byte-distinct nested archives; code/notebooks and readable documents were
read statically, CSVs were read through the final row, and data-bearing OOF
vectors were rescored only when row alignment with the local training labels
was established. Some binary images/video were not visually decoded and one
nested NPZ was unreadable; these scope limits are documented in the report.
The supplied HGB test vector is a finite float64 array of shape `(61500,)`
without IDs or labels, so its score and row alignment cannot be verified.

The local RTX 4050 is visible to `nvidia-smi`, but PyTorch reports CUDA
unavailable; neither audited notebook configures GPU training, and neither
was run. No new GPU experiment or score is claimed.

### User-requested deeper file review and GPU translation follow-up

The initial materials report was too inventory-focused. It has now been
expanded with per-file findings for the archive's code/notebooks,
research/document files, and CSV content groups, including exact embedded or
recomputed scores, methods/features, validation evidence, score limitations,
and project classification. Telecom/churn artifacts are explicitly marked
unrelated to GCI; they were not run.

A clean-room Phase 52 XGBoost CUDA translation of the v7-style deterministic
feature block was run on the local training data with five fixed stratified
folds. The raw one-hot baseline scored OOF ROC-AUC `0.752413` / log-loss
`0.247813`; adding the label-free feature bundle scored `0.758640` /
`0.246102`, a matched gain of `+0.006227` AUC and `+0.001711` lower log-loss,
positive in all five folds. This is a new XGBoost GPU experiment, not an
exact v7/v13 LightGBM reproduction and not a public leaderboard result.
Feature-family-specific causes remain unisolated.

The runner's two tests passed. CatBoost GPU and XGBoost CUDA smoke tests
passed; LightGBM GPU failed because the installed build has no GPU learner.
The user-named notebooks remain unexecuted as written; their Colab/Drive
cache dependencies are not present. Full details and artifacts are in
`reports/NEW_MATERIALS_FORENSIC_AUDIT.md` and
`experiments/phase52_gpu_archive_port`.

#### Phase 52 matched feature-family ablations

Ran a second five-fold XGBoost CUDA experiment using the same data, folds,
seed, and training configuration. Each family variant adds one engineered
feature family to the raw one-hot baseline; the full-bundle result adds all
families. The financial-ratio/log-amount family was the strongest standalone
addition: pooled OOF AUC `0.759889` (gain `+0.007476` vs raw `0.752413`),
log-loss `0.245761` (improvement `0.002052`), and positive AUC delta in all
five folds. The full v7-style bundle scored `0.758640` (+`0.006227` AUC vs
raw), also positive in all folds, but was `0.001249` below the financial-only
addition in this run.

Other raw-plus-one-family results: clean/missingness AUC `0.752526`
(+`0.000113`, 3/5 folds positive); external-score geometry `0.751169`
(-`0.001244`, 0/5); age/employment `0.751708` (-`0.000706`, 1/5); bureau
activity `0.752035` (-`0.000378`, 2/5). These are exploratory, single-seed
ablation results, not causal proof or a public score; family interactions and
other learner/fold configurations may change the result. Here, "family only"
means raw baseline plus that family. Exact OOF predictions and summary metrics
are saved under
`experiments/phase52_gpu_archive_port/runs/phase52_gpu_family_ablation_20261108`.
The runner tests and compilation check passed after validating that the family
partition covers each engineered column exactly once.

## V22 Scratch Run: Rules and Replacement Audit (2026-11-13)

### Run evidence and limits

The requested `reports/scratch_v_1.py` run completed unchanged on CPU. It
created `reports/scratch_v_1_submission.csv` with 61,500 rows, columns
`SK_ID_CURR,TARGET`, unique IDs in sample-submission order, finite predictions,
and SHA-256
`a14d6fd843f9fd36199f6a2992910438565a6735cff9de7a0251b6ad150ff229`.
The script printed pooled OOF ROC-AUC `0.769889`, but did not save row-level
OOF predictions; that metric was not independently recomputed. Its printed
Spearman correlation to a historical V22 prediction file is not a score.
The historical public-score claim `0.78239` was not independently verified.

### V22 findings against supplied competition materials

The primary local rules source is
`reports/final_competition_audit/OFFICIAL_RULES_AUDIT.md`. It confirms ROC-AUC,
the required submission schema, a prohibition on external data, and a
prohibition on manual or EDA-driven prediction overrides. It does not
explicitly resolve unlabeled train/test pooling or target encoding.

1. `reports/scratch_v_1.py` concatenates train and test rows in
   `build_all_features` (line 84) before deriving category codes and pooled
   feature statistics. This is not demonstrably an explicit rule violation:
   both files are organizer-provided, but the supplied rules do not authorize
   fitting feature statistics on test covariates. Treat it as unresolved and
   avoid it pending organizer clarification.
2. The `partner_enc` feature uses labels from matched training rows to create
   prediction features (lines 477-497), and the underlying identity-like
   groups were formed using pooled train/test covariates. This is automated,
   not a manual override; however, it is EDA-informed label propagation and
   could fall under the hand-labeling rule. Eligibility is ambiguous; do not
   present it as certified compliant.
3. The memoization keys in `enc` and `train_enc` use only index-array lengths
   and the first fit index (lines 503-524), rather than the full index sets.
   Different folds can collide and reuse encodings calculated from a different
   training subset. This can invalidate fold isolation and the reported OOF
   estimate; it is a code correctness issue independent of rule interpretation.
4. After writing the submission, the script optionally reads
   `d:/gci/gemini/v_22_pure_champion_ensemble.csv` for a historical correlation
   diagnostic (lines 719-723). It does not feed that file into the prediction
   computation, but the optional non-raw read contradicts a strict
   raw-input-only/reproducibility claim and should be removed from a
   competition-delivery version.
5. Fold-specific model predictions are pooled and rank-transformed across
   folds before the reported OOF score. Since ROC-AUC compares positive and
   negative score pairs globally, cross-fold score-scale differences can
   affect this pooled estimate; report fold AUCs and preserve row-level OOF
   predictions for independent rescoring.

### Replacement candidate audit: V28

The existing `solo_champion_v28_certified.py` is the strongest available
replacement candidate found in this review. Direct source inspection and
`truth/audit_check_compliance.py` show that the standalone V28 script:

- reads only the supplied train, test, and sample-submission files;
- does not concatenate train and test to fit features, transfer partner
  labels, read historical prediction files, or adjust selected test rows;
- fits preprocessing and target encodings within the active training folds,
  using inner-fold-only labels and priors for training encodings;
- uses outer-held-out labels only for evaluation, with fixed training
  schedules rather than outer-fold early stopping; and
- checks submission ID uniqueness and alignment.

The target-encoding implementation is not expressly prohibited by the local
rules. Its nested, fold-local implementation avoids the identified validation
leakage; organizer clarification is still appropriate if a ruling is needed.
V28 also uses hard-coded numerical feature constants (including 128.5, 129.5,
and 0.0066). These are deterministic transformations of supplied columns,
not per-row prediction overrides. Their data/feature-selection provenance
should be retained, because development choices made after examining all
training labels can make an OOF estimate optimistic even when runtime
preprocessing is nested.

The saved V28 notebook output reports true two-level nested OOF ROC-AUC
`0.771679`, equal-weight probit OOF `0.771670`, five-fold mean `0.771688`
(standard deviation `0.001969`), and a GPU run using CatBoost GPU, XGBoost
CUDA, and an RTX 4050. The output submission has 61,500 rows, unique IDs, and
SHA-256
`ED0D062664A47676E78586016C076F479E0E0E59A136135D1B21CC7AFCDC4874`.
These are notebook-recorded execution results and artifact checks, not a
verified public leaderboard score. The notebook does not save row-level OOF
predictions, so its pooled AUC could not be independently recomputed here.

The repository compliance runner exits nonzero because its
`.py`/`.ipynb` source-equivalence check finds a small mismatch in CLI default
path handling: the Python script chooses paths conditionally, while the
notebook hard-codes `D:\gci\inpuy` and the output path. The model pipeline
lines otherwise match under the checker's normalization. The standalone
Python script passes the V28 semantic nesting/prior checks; the notebook
should not be called source-identical until this path mismatch is reconciled.

### Decision and score-comparison caveat

V28 is recommended as the next rule-conscious candidate instead of the V22
scratch script. Its reported nested OOF AUC is numerically 0.001790 above the
V22 script's printed OOF AUC, but these results are not a matched comparison:
the V22 cache-key and pooling issues undermine its validation, the fold
designs and ensembles differ, and neither run saved row-level OOF vectors
for an independent paired comparison. Therefore this difference does not
prove that the V28 public score matches or exceeds V22.

No evidence can guarantee an exactly unchanged public AUC or a reduction
smaller than 0.0001 before the replacement is scored on the same untouched
evaluation set. The public leaderboard score is not available from local
training data. Preserve both submissions; do not claim exact-score
preservation or overwrite the V22 artifact based on these non-comparable OOF
numbers. A defensible next comparison requires frozen identical outer folds,
row-level OOF outputs from both pipelines, paired fold and pooled ROC-AUC
reports, and organizer clarification on any unresolved rule interpretation.