# GCI World 2026: Deep Mathematical Feature Attribution & Categorical Sensitivity Analysis

- **Execution Timestamp:** `2026-11-07T17:11:16Z`
- **Runtime:** `115.19s` on RTX 4050 GPU
- **Final 5-Fold Pooled OOF AUC:** `0.761778`

## 1. Information Value (IV) & Entropy Hierarchy of All Categories

| Category Feature | Unique Classes | Information Value (IV) | Predictive Power Class | Highest-Risk Subgroup (Default %) |
| :--- | :---: | :---: | :--- | :--- |
| **`OCCUPATION_TYPE`** | 19 | **0.0836** | Weak (0.02 - 0.10) | Low-skill Laborers (16.2%) |
| **`ORGANIZATION_TYPE`** | 58 | **0.0724** | Weak (0.02 - 0.10) | Transport: type 3 (16.2%) |
| **`NAME_EDUCATION_TYPE`** | 5 | **0.0508** | Weak (0.02 - 0.10) | Lower secondary (11.1%) |
| **`NAME_INCOME_TYPE`** | 8 | **0.0551** | Weak (0.02 - 0.10) | Working (9.5%) |
| **`NAME_HOUSING_TYPE`** | 6 | **0.0176** | Unpredictive (<0.02) | Rented apartment (12.9%) |
| **`CODE_GENDER`** | 3 | **0.0383** | Weak (0.02 - 0.10) | M (10.1%) |
| **`NAME_FAMILY_STATUS`** | 6 | **0.0222** | Weak (0.02 - 0.10) | Civil marriage (9.9%) |
| **`NAME_CONTRACT_TYPE`** | 2 | **0.0141** | Unpredictive (<0.02) | Cash loans (8.3%) |

## 2. Feature Block Incremental AUC Sensitivity (5-Fold CV Ablation)

| Feature Block Name | Block Features | Cumulative Features | 5-Fold Pooled OOF AUC | Marginal Lift (bps) | Underlying Mathematical Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw External Bureaus** | 3 | 3 | 0.712461 | — | Base credit bureau agency scores (E1, E2, E3). |
| **Bureau Copula Manifolds** | 6 | 9 | 0.713284 | **+8.2** | Archimedean Clayton/Frank/Harmonic lower-tail non-linear distress. |
| **Loan Terms & Tariff Decoders** | 10 | 19 | 0.748155 | **+348.7** | Standard retail bank pricing lattices (Q3, Q4, CT1, Negative Downpayment). |
| **Income & Solvency Buffers** | 6 | 25 | 0.747666 | **-4.9** | Debt-to-income (DTI) and household subsistence cash flow stress. |
| **Demographics & Lifespan** | 7 | 32 | 0.752774 | **+51.1** | Career velocity, retirement flags, and Single Male Laborer Hazard. |
| **Administrative Tenure** | 5 | 37 | 0.753311 | **+5.4** | Address/ID registration lags and phone number tenure longevity. |
| **Relational Entity Graph** | 3 | 40 | 0.753898 | **+5.9** | Invariant multi-application clustering (`entity_key`, `N_APPS_ALL`). |
| **Credit Bureau Inquiries** | 5 | 45 | 0.754696 | **+8.0** | Trailing inquiry velocity, quarterly acceleration, and missingness flags. |
| **Hermetic Nested Target Encodings** | 6 | 51 | **0.761778** | **+70.8** | In-fold inner cross-fitted Bayesian target encodings on top categorical keys. |
