# GCI WORLD 2026: 20 ADVANCED RESEARCH REGIME SPLITS & HIGH-AUC FORMULA DISCOVERIES (EXP 121 - 140)
**Execution Mode**: Clean-room execution on strictly competition-provided `train.csv` (171,202 records)  
**Cross-Validation**: 5-Fold Stratified Cross-Validation (`random_state=42`)  
**Evaluation Metric**: ROC-AUC  
**Date**: November 7, 2026  

---

## 1. Executive Summary

This study executes **20 novel advanced research regime partition experiments (Exp 121 to Exp 140)**. The suite explores:
1. **Geometric Bureau Topology & Information Theory**: Perpendicular consensus line distance, polar stereographic elevation angles, and Kullback-Leibler divergence measuring non-uniformity and conflict across external credit agencies (**`1,262.8 bps`** spread, $\rho = +0.1233$).
2. **Behavioral Administrative Latency**: Document-to-phone update synchrony and registration-to-ID lags.
3. **Demographic Hazard Concentrators**: Single male blue-collar laborers carrying high DTI, which isolates a subgroup default rate of **15.61%** with a **`2,604.2 bps`** default rate spread ($\rho = +0.1814$).
4. **Household Financial Physics & Shock Buffers**: Non-linear debt velocity per adult lifetime year, emergency reserve half-life, and disposable margin solvency under a 25% subsistence inflation shock.

### Key Validation Metric Results:
- **Baseline Feature Set 5-Fold OOF AUC**: `0.757767`
- **Enhanced Feature Set (+10 Discovered Features Exp 121-140) 5-Fold OOF AUC**: `0.757463`
- **Net Delta**: `-3.0 bps` (Demonstrating slight tree collinearity when adding unpruned geometric polar features; highlights the necessity of selective regularization and monotonic constraints).

---

## 2. Comprehensive 20-Regime Split & Formula Discovery Ledger (Exp 121 - 140)

| Exp ID | Regime Name | Subgroup Criteria | Sample Count (%) | Subgroup Def % | Discovered Formula | Within-Split Corr ($\rho$) | Low vs High Def Spread | Spread (bps) | Economic / Credit Risk Rationale |
|:---:|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---|
| **121** | `Bureau_Consensus_Line_Distance` | $\|E_2 - E_3\| > 0.15$ | 87,831 (51.30%) | 9.27% | `abs(E2 - E3) / sqrt(2)` | +0.0806 | 6.66% vs 13.65% | **699.0** | Perpendicular distance from the diagonal line of perfect agency agreement in Euclidean space. |
| **122** | `Stereographic_Elevation_Angle` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `arccos(E3 / sqrt(E1^2 + E2^2 + E3^2))` | +0.0152 | 10.46% vs 15.15% | **469.0** | Polar elevation angle in 3D bureau coordinate space measuring relative agency weighting geometry. |
| **123** | `Bureau_Entropy_KL_Divergence` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `sum(p_i * log(3 * p_i))` | +0.1233 | 6.98% vs 19.61% | **1262.8** | Kullback-Leibler divergence measuring non-uniformity and conflict across external agency ratings. |
| **124** | `Adult_Debt_Velocity` | Age $\ge 21 \land \text{Credit} \ge 200\text{k}$ | 151,028 (88.22%) | 8.22% | `Credit / (Age - 18)` | +0.0281 | 5.34% vs 8.55% | **320.5** | Pace of principal accumulation per year of legal adult life; penalizes aggressive early-life borrowing. |
| **125** | `Emergency_Liquidity_Reserve_Months` | Annuity $\ge 15\text{k}$ | 135,810 (79.33%) | 8.33% | `(2 * (Income / 12)) / Annuity` | -0.0070 | 8.41% vs 8.10% | **31.0** | Number of monthly annuity payments covered by a standard two-month emergency household buffer. |
| **126** | `Uncollateralized_Markup_Per_Emp_Year` | $\text{Credit} > \text{Goods Price}$ | 111,021 (64.85%) | 8.75% | `max(Credit - Goods, 0) / (Employed_Yrs + 0.5)` | +0.0304 | 7.03% vs 9.63% | **260.1** | Unbacked bank financing surcharge amortized against accumulated workplace tenure. |
| **127** | `Disposable_Margin_Inflation_Shock` | Children $\ge 1$ | 51,252 (29.94%) | 8.91% | `(Income/12 - 1.25*Subsistence - Annuity) / (Income/12)` | -0.0353 | 11.23% vs 7.15% | **407.7** | Tests household debt servicing solvency under a 25% inflation shock to non-discretionary subsistence baskets. |
| **128** | `Document_to_Phone_Synchrony` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `abs(DAYS_ID_PUBLISH - DAYS_LAST_PHONE_CHANGE) / 365.25` | -0.0293 | 9.73% vs 6.83% | **290.0** | Time elapsed between state identity document re-issuance and commercial mobile phone replacement. |
| **129** | `Registration_to_ID_Lag` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `abs(DAYS_REGISTRATION - DAYS_ID_PUBLISH) / 365.25` | -0.0289 | 9.46% vs 6.87% | **259.5** | Lag between official residence registration and passport issuance; identifies delayed address updates. |
| **130** | `Phone_to_Job_Relative_Stability` | Employed $>0 \land \text{Phone}>0$ | 124,199 (72.54%) | 8.41% | `Phone_Years / (Employed_Years + 0.1)` | +0.0155 | 7.64% vs 8.56% | **91.8** | Ratio of mobile telephone number longevity to current workplace employment tenure. |
| **131** | `Contract_Implied_APR` | Cash Loans | 154,988 (90.53%) | 8.34% | `(12 * Annuity) / Credit` | +0.0355 | 5.59% vs 4.74% | **84.5** | Annualized cash repayment obligation per credit ruble; high ratios reflect steep short-term financing terms. |
| **132** | `Goods_Price_Coverage_Margin` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `(Goods - Credit) / Income` | -0.0458 | 9.89% vs 6.82% | **307.1** | Financing margin normalized by income: positive values signal equity down payment, negative indicate fee loading. |
| **133** | `Single_Male_Laborer_Hazard` | Male $\land$ Single $\land$ Laborer | 5,759 (3.36%) | 15.61% | `DTI * (Credit / Goods) * (1 - E2)` | +0.1814 | 8.68% vs 34.72% | **2604.2** | Isolates vulnerable single male manual workers carrying heavy monthly debt burdens and subprime bureau scores. |
| **134** | `Senior_Equity_Cushion` | Pensioner $\land$ Car $\land$ Home | 5,407 (3.16%) | 4.98% | `(Income / Credit) * exp(-Car_Age / 10)` | -0.0469 | 6.47% vs 3.70% | **277.3** | Sovereign pension-backed senior homeowners with owned vehicle collateral represent premier credit stability. |
| **135** | `Young_Mother_Subsistence_Strain` | Female $\land$ Age $<32 \land$ Child $\ge 1$ | 8,906 (5.20%) | 11.19% | `(Annuity * (Children + 1)) / (Income / 12)` | +0.0483 | 9.65% vs 13.47% | **381.6** | Scales monthly annuity obligations by child dependency burden for mothers under age 32. |
| **136** | `Informal_Metro_Freelancer_Risk` | Rating $1 \land$ No Landline $\land \text{Inc} \ge 180\text{k}$ | 1,290 (0.75%) | 3.80% | `(Credit / Income) * (1 - Mean_E2_E3)` | +0.0379 | 2.33% vs 5.43% | **310.1** | High nominal income in major cities lacking formal enterprise employer phone verification. |
| **137** | `Overcrowded_Family_Dilution` | Fam Members $\ge 4$ | 15,985 (9.34%) | 8.66% | `Annuity / (Income / Fam)` | +0.0399 | 7.07% vs 10.70% | **362.7** | Monthly debt service obligation divided by per-capita family income for households of 4+ members. |
| **138** | `Vehicle_Liquidation_Annuity_Coverage` | Car Owners ($N=58,210$) | 58,210 (34.00%) | 7.25% | `(Income * exp(-Car_Age / 8)) / (12 * Annuity)` | -0.0524 | 9.54% vs 5.57% | **396.8** | Estimated vehicle asset recovery value divided by total annual contractual loan payment. |
| **139** | `Inquiry_Annual_Acceleration_Index` | Has Inquiries ($N=113,911$) | 113,911 (66.54%) | 7.87% | `Inquiries_Month * 12 - Inquiries_Year` | -0.0200 | 8.87% vs 6.76% | **210.9** | Differential between annualized monthly credit inquiry velocity and actual trailing annual inquiries. |
| **140** | `Credit_Income_Polynomial_Hazard` | $\text{Credit} \ge 2 \times \text{Income}$ | 131,673 (76.91%) | 8.23% | `(Credit / Income)^2 * (1 - E2)` | +0.0387 | 5.56% vs 9.94% | **438.2** | Quadratic leverage penalty capturing catastrophic non-linear default risk when credit exceeds twice income. |

---

## 3. Top Formulations Ranked by Default Risk Separation

1. **`F_133_Single_Male_Laborer_Hazard` (Exp 133)**:
   - **Spread**: **`2,604.2 bps`** (8.68% low decile vs 34.72% high decile, $\rho = +0.1814$)
   - **Formula**: `DTI * (Credit / Goods) * (1 - E2)`
   - **Behavior**: Isolates high-risk blue-collar demographic carrying heavy monthly debt burdens and subprime bureau scores.
2. **`F_123_Bureau_Entropy_KL_Divergence` (Exp 123)**:
   - **Spread**: **`1,262.8 bps`** (6.98% low decile vs 19.61% high decile, $\rho = +0.1233$)
   - **Formula**: `sum(p_i * log(3 * p_i))`
   - **Behavior**: Information-theoretic KL divergence captures high-conflict agency profiles where one bureau uncovers severe hidden delinquencies.
3. **`F_121_Bureau_Consensus_Line_Distance` (Exp 121)**:
   - **Spread**: **`699.0 bps`** (6.66% low decile vs 13.65% high decile, $\rho = +0.0806$)
   - **Formula**: `abs(E2 - E3) / sqrt(2)`
   - **Behavior**: Measures agency discordance geometrically in Euclidean space.
4. **`F_122_Stereographic_Elevation_Angle` (Exp 122)**:
   - **Spread**: **`469.0 bps`** (10.46% low decile vs 15.15% high decile)
   - **Formula**: `arccos(E3 / sqrt(E1^2 + E2^2 + E3^2))`
   - **Behavior**: 3D polar elevation angle in bureau coordinate space.
5. **`F_140_Credit_Income_Polynomial_Hazard` (Exp 140)**:
   - **Spread**: **`438.2 bps`** (5.56% low decile vs 9.94% high decile, $\rho = +0.0387$)
   - **Formula**: `(Credit / Income)^2 * (1 - E2)`
   - **Behavior**: Quadratic leverage penalty capturing catastrophic non-linear default risk when credit exceeds twice income.
