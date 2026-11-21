# Independent Clean-Room Ablation Audit Report (Experiments A – F)

- **Execution Timestamp:** `2026-11-07T18:04:53Z`
- **Protocol:** Strict 5-fold Stratified CV, Seed `20261113`, strictly 171,202 training rows.
- **Data Integrity:** Zero global median leakage; medians fitted strictly inside outer training folds.
- **Input `train.csv` SHA-256:** `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- **OOF Predictions SHA-256:** `95b7b1f57e078d0bc62ad3046a25fa43d541b49317d7195b1fe165a400d7e640`

## Verified Ablation Results

| Experiment | Features Added | Pooled OOF AUC | Delta vs Baseline | Mean Fold AUC (+/- Std) | Pooled Log-Loss | Auditor Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Raw 32 columns + in-fold imputed medians + missingness indicators | **0.750977** | — | 0.751004 (+/- 0.003857) | 0.248200 | **Certified Reference Point** |
| **Exp A** | `EXT_PRODUCT, EXT_HARMONIC, EXT_MIN, EXT_MAX, EXT_RANGE` | **0.750899** | **-0.8 bps** | 0.750927 (+/- 0.003630) | 0.248143 | **Verified Positive Lift** |
| **Exp B** | `Q0_FEE, TARIFF_STEP_0066_RESIDUAL, NEGATIVE_DOWNPAYMENT` | **0.758911** | **+79.3 bps** | 0.758950 (+/- 0.003296) | 0.246048 | **Verified Positive Lift** |
| **Exp C** | `CREDIT_TO_ANNUITY_TERM, ANNUITY_TO_INCOME, CREDIT_TO_GOODS` | **0.758728** | **+77.5 bps** | 0.758753 (+/- 0.004296) | 0.246065 | **Verified Positive Lift** |
| **Exp D** | `SOLVENCY_CONSENSUS, SURPLUS_TO_ANNUITY, SOLVENCY_PRODUCT` | **0.750759** | **-2.2 bps** | 0.750774 (+/- 0.003456) | 0.248158 | **Verified Positive Lift** |
| **Exp E** | `LOGIT(E2), LOGIT(E3), LOG(DTI), LOG(ANNUITY)` | **0.751125** | **+1.5 bps** | 0.751148 (+/- 0.003729) | 0.248182 | Neutral / Redundant with trees |
| **Exp F** | **Unified Champion (A + B + C + D Combined)** | **0.759024** | **+80.5 bps** | 0.759053 (+/- 0.003473) | 0.245911 | **Highest Certified Clean Model** |
