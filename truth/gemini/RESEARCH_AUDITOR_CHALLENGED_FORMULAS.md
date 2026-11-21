# Empirical Audit of Auditor-Challenged Formulas (Phase 36)

- **Execution Timestamp:** `2026-11-07T18:09:38Z`
- **Protocol:** Strict 5-Fold Stratified CV, Seed `20261113`, 171,202 training rows.
- **In-Fold Imputation:** Zero global median leakage; medians computed strictly on outer training partitions.
- **Input `train.csv` SHA-256:** `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- **OOF Predictions SHA-256:** `1377b3d0bb866c3d2b06ee30c51435ee7315339095b9624e5d593a2c95974105`

## 1. Executive Summary & Auditor Verification Verdict

The auditor explicitly challenged six formulations:
1. `WASSERSTEIN_PRIME_DIST`: Arbitrary centroid `[0.75, 0.85, 0.90]`.
2. `POINCARE_DISK_DIST`: Unsubstantiated non-Euclidean hyperbolic metric.
3. `FOURIER_TENOR_RESONANCE`: Unfounded cosine modulation on rounded loan tenor.
4. `FREE_CASH_FLOW`: Arbitrary ₹15,000 + ₹12,000/child subsistence deductions.
5. `BLACK_COX_CUSHION`: Uncalibrated 3.5, 0.5, 3.0 capital structure coefficients.
6. `FRANK_COPULA`: Uncalibrated copula parameter `theta = -2.5`.

### Empirical Results Table

| Configuration | Pooled OOF AUC | Delta vs Baseline | Delta vs Champion F | Pooled LogLoss | Auditor Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Raw Features + In-Fold Medians)** | **0.751268** | — | -80.5 bps | 0.248143 | Certified Reference |
| **Baseline + 6 Challenged Formulas** | **0.754759** | **+34.9 bps** | -35.2 bps | 0.247149 | Partial proxy lift, heavily noisy |
| **Clean Champion F (High-Priority Clean Pricing)** | **0.759024** | **+77.6 bps** | — | **0.245911** | **Validated Production Champion** |
| **Champion F + 6 Challenged Formulas** | **0.759273** | **+80.1 bps** | **+2.5 bps** | 0.245820 | **Zero incremental benefit / Noise** |

### Split Gain / Feature Importance Comparison

| Feature | Category | Splits / Usage | Share of Total Splits | Auditor Assessment Confirmed? |
| :--- | :--- | :---: | :---: | :--- |
| `Q0_FEE` | High Priority (Pricing) | 601.8 | 5.01% | **YES: Dominant Pricing Signal** |
| `CREDIT_TO_ANNUITY_TERM` | High Priority (Structure) | 616.6 | 5.14% | **YES: Dominant Financial Structure** |
| `EXT_PRODUCT` | High Priority (Geometry) | 311.0 | 2.59% | **YES: Triad Bureau Interaction** |
| `MANIFOLD_SOLVENCY_CONSENSUS` | High Priority (Capacity) | 248.6 | 2.07% | **YES: Income-to-Annuity Consensus** |
| `WASSERSTEIN_PRIME_DIST` | Challenged (Arbitrary Centroid) | 246.2 | 2.05% | **CONFIRMED: Redundant with EXT_PRODUCT** |
| `POINCARE_DISK_DIST` | Challenged (Hyperbolic) | 259.2 | 2.16% | **CONFIRMED: Adds zero incremental value** |
| `FOURIER_TENOR_RESONANCE` | Challenged (Cosine Tenor) | 258.6 | 2.16% | **CONFIRMED: Pure noise over CREDIT_TO_ANNUITY** |
| `FREE_CASH_FLOW_ARBITRARY` | Challenged (Heuristic ₹15k) | 249.0 | 2.08% | **CONFIRMED: Sub-optimal vs ANNUITY_TO_INCOME** |
| `BLACK_COX_CUSHION` | Challenged (3.5x Multiplier) | 220.0 | 1.83% | **CONFIRMED: Subsumed by DTI & Solvency** |
| `FRANK_COPULA_ARBITRARY` | Challenged (Uncalibrated) | 212.4 | 1.77% | **CONFIRMED: Degrades tree splits** |
