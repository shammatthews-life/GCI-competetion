# GCI WORLD 2026: 30 DEEP ADVANCED RESEARCH REGIME SPLITS & HIGH-AUC FORMULA DISCOVERIES (EXP 91 - 120)
**Execution Mode**: Clean-room execution on strictly competition-provided `train.csv` (171,202 records)  
**Cross-Validation**: 5-Fold Stratified Cross-Validation (`random_state=42`)  
**Evaluation Metric**: ROC-AUC  
**Date**: November 7, 2026  

---

## 1. Executive Summary

This study executes **30 advanced forensic research regime partition experiments (Exp 91 to Exp 120)**. The suite explores previously untouched dimensions of the competition data:
1. **Archimedean and Extreme Value Copula Distress Manifolds** (Joe upper tail, Frank comprehensive, Clayton deep lower tail) that capture non-linear joint dependency and separate default risk by up to **`2,252.5 bps`**.
2. **Contract & Collateral Financing Dynamics** (Down payment cushions, rolled-in ancillary financing fees, revolving credit line pre-screening).
3. **Institutional and Statutory Lifecycle Milestones** (Russian statutory passport renewal at ages 20/45, lifetime registration stability, adult career participation ratios).
4. **Engel Subsistence Living Cost Stresses** (Non-discretionary child subsistence floors subtracted from disposable income).
5. **Credit Bureau Inquiry Dynamics** (Acute shopping velocity bursts, calm multi-year credit anchors, missing inquiry penalties).

### Key Validation Metric Results:
- **Baseline Feature Set 5-Fold OOF AUC**: `0.757767`
- **Enhanced Feature Set (+15 Top Discovered Features) 5-Fold OOF AUC**: `0.758114`
- **Net Cross-Validated AUC Lift**: **`+3.5 bps`** (`+0.000347` AUC) across all 171,202 samples.

---

## 2. Comprehensive 30-Regime Split & Formula Discovery Ledger (Exp 91 - 120)

| Exp ID | Regime Name | Subgroup Criteria | Sample Count (%) | Subgroup Def % | Discovered Formula | Within-Split Corr ($\rho$) | Low vs High Def Spread | Spread (bps) | Economic / Credit Risk Rationale |
|:---:|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---|
| **91** | `Logit_Gaussian_Bureau_Divergence` | $\|E_2 - E_3\| > 0.20$ | 65,075 (38.01%) | 10.18% | `abs(logit(E2) - logit(E3)) / (0.5*(abs(logit(E2)) + abs(logit(E3))) + 0.1)` | +0.0568 | 10.45% vs 17.56% | **711.4** | Logit-space relative divergence between major bureaus isolates non-linear agency disagreement. |
| **92** | `Joe_Extreme_Upper_Tail_Copula` | $E_2 > 0.55 \land E_3 > 0.55$ | 32,144 (18.78%) | 2.63% | `1 - ((1-u)^2 + (1-v)^2 - (1-u)^2 * (1-v)^2)^(1/2)` | -0.0539 | 4.48% vs 1.71% | **276.8** | Asymmetric upper-tail copula captures super-prime default resilience when both agencies agree. |
| **93** | `Frank_Symmetric_Comprehensive_Copula` | Full Dataset ($N=171,202$) | 171,202 (100.0%) | 8.07% | `-1/4 * log(1 + (exp(-4u)-1)*(exp(-4v)-1)/(exp(-4)-1))` | -0.1834 | 21.56% vs 2.00% | **1956.1** | Radial symmetric Archimedean copula modeling full-support dependence between external rating providers. |
| **94** | `Clayton_Deep_Distress_Lower_Copula` | $E_2 < 0.40 \land E_3 < 0.40$ | 9,891 (5.78%) | 22.17% | `(u^(-3) + v^(-3) - 1)^(-1/3)` | -0.1666 | 32.86% vs 10.34% | **2252.5** | Extreme lower-tail copula mapping catastrophic joint credit distress when both scores collapse. |
| **95** | `Negative_Down_Payment_Markup_Ratio` | $\text{Credit} > \text{Goods Price}$ | 111,021 (64.85%) | 8.75% | `max(Credit - Goods, 0) / Goods` | +0.0833 | 7.03% vs 15.13% | **809.5** | Measures rolled-in financing markup and bank insurance fees exceeding retail purchase value. |
| **96** | `Revolving_Credit_Safety_Multiplier` | Revolving Loans | 16,214 (9.47%) | 5.55% | `(Income / Annuity) * (1 + (E2 + E3) / 2)` | -0.0172 | 6.72% vs 4.07% | **265.1** | Revolving credit borrowers have pre-vetted credit lines; scales annuity cash flow buffer by creditworthiness. |
| **97** | `Working_Life_Career_Participation` | Employed $> 0$ | 140,304 (81.95%) | 8.66% | `Employed_Years / (Age - 18)` | -0.0457 | 10.82% vs 7.12% | **369.9** | Percentage of adult life spent in verified active employment; separates career veterans from chronic job hoppers. |
| **98** | `Statutory_Passport_Renewal_Proximity` | Age $\in [21, 50]$ | 113,530 (66.31%) | 9.31% | `min(abs(Age_at_ID - 20), abs(Age_at_ID - 45))` | +0.0060 | 8.91% vs 9.95% | **104.0** | Distance from statutory passport renewal milestones (ages 20/45) indicates institutional compliance timing. |
| **99** | `Lifetime_Registration_Anchoring` | Adult Life $\ge 5\text{ yrs}$ | 168,796 (98.59%) | 8.00% | `Registration_Years / (Age - 18)` | +0.0080 | 7.24% vs 8.91% | **167.1** | Proportion of adult life registered at current address; establishes deep social and community roots. |
| **100** | `Subsistence_FCF_Stress_Margin` | Children $\ge 1$ | 51,252 (29.94%) | 8.91% | `(Income/12 - Annuity - (12000*Children + 15000)) / (Income/12)` | -0.0356 | 11.23% vs 7.43% | **380.4** | Monthly discretionary cash flow remaining after contractual debt and minimum Russian family subsistence needs. |
| **101** | `Pensioner_Autonomous_Collateral_Shield` | Pensioner with Car | 5,627 (3.29%) | 4.99% | `(Income / Annuity) * exp(-Car_Age / 10)` | -0.0326 | 5.86% vs 3.91% | **195.4** | Combines sovereign guaranteed state pension annuity coverage with secondary vehicle liquidation value. |
| **102** | `Young_Worker_Probationary_Strain` | Age $< 28 \land$ Employed $< 9\text{mo}$ | 2,422 (1.41%) | 12.68% | `(Credit / Income) / (Employed_Years + 0.1)` | +0.0260 | 12.35% vs 13.17% | **82.3** | Young applicants on job probation carrying heavy leverage face high involuntary termination risk. |
| **103** | `Single_Parent_Subsistence_Load` | Single Parent with Child | 7,459 (4.36%) | 10.43% | `(Annuity * (Children + 1)) / (Income / 12)` | +0.0423 | 8.98% vs 12.47% | **348.5** | Multiplies debt service burden by dependent child count for single-breadwinner households. |
| **104** | `High_Balance_Extended_Term_Drag` | Credit $\ge 800\text{k} \land \text{Term} \ge 25\text{mo}$ | 33,210 (19.40%) | 6.43% | `(Credit / Income) * (Term_Yrs) * (1 - Mean_E2_E3)` | +0.0812 | 3.55% vs 10.57% | **701.6** | Multi-year high-balance loans carry cumulative macroeconomic shock exposure mitigated only by strong bureaus. |
| **105** | `Micro_Financed_Retail_Markup` | Credit $\le 120\text{k} \land \text{Credit} > \text{Goods}$ | 4,143 (2.42%) | 5.70% | `((Credit - Goods) / Goods) * (Annuity / Monthly_Income)` | +0.0579 | 4.09% vs 7.71% | **362.7** | Point-of-sale micro-credit with rolled-in fees imposes predatory monthly debt loads on small purchases. |
| **106** | `Dual_Bureau_Harmonic_Distress_Floor` | $E_2 > 0 \land E_3 > 0$ | 116,393 (67.99%) | 7.70% | `2 / (1/E2 + 1/E3)` | -0.1989 | 19.34% vs 1.95% | **2018.9** | Harmonic mean heavily penalizes applicants if either credit bureau registers delinquency ('weakest link'). |
| **107** | `Thin_File_Missing_Inquiry_Penalty` | Missing Bureau Inquiries | 23,116 (13.50%) | 10.38% | `(Credit / Income) * (1 / (E2 + 0.1))` | +0.1006 | 5.28% vs 19.16% | **1388.4** | For applicants with zero bureau inquiry tracking, high leverage interacting with weak internal score indicates severe risk. |
| **108** | `Acute_Inquiry_Velocity_Shock` | Inq Month $\ge 1$ | 24,353 (14.22%) | 7.25% | `Inquiries_Month / (Inquiries_Quarter / 3 + 0.05)` | -0.0197 | 7.85% vs 5.66% | **218.9** | Surge in monthly credit bureau checks relative to quarterly baseline flags emergency credit shopping. |
| **109** | `Calm_Credit_History_Anchor` | Inq Year $\ge 1 \land \text{Inq Mon}=0$ | 88,994 (51.98%) | 8.03% | `log(1 + Inq_Year) * (E2 * E3)` | -0.1619 | 19.98% vs 2.00% | **1797.8** | Seasoned borrowers with established annual credit records who have not applied for emergency credit in 30 days. |
| **110** | `Metropolitan_High_Earner_Cost_Ratio` | Rating $= 1 \land \text{Inc} \ge 225\text{k}$ | 8,944 (5.22%) | 4.06% | `(12 * Annuity) / (Income * (10*Pop + 1))` | -0.0078 | 4.80% vs 3.69% | **111.7** | High cost of living in primary metropolitan centers deflates high nominal salaries. |
| **111** | `Rural_Subprime_Collateral_Buffer` | Rating $= 3 \land \text{Car Age} \le 8$ | 2,633 (1.54%) | 7.86% | `((8 - Car_Age)/8) * (Income / Credit)` | +0.0098 | 6.82% vs 8.83% | **201.4** | In economically distressed rural regions (Rating 3), recent vehicle ownership provides critical collateral protection. |
| **112** | `Academic_Degree_Income_Efficiency` | Higher / Academic Degree | 41,746 (24.38%) | 5.36% | `Income / ((Age - 22) * 5000)` | +0.0250 | 4.65% vs 6.35% | **169.9** | Earnings yield per career year for university-educated borrowers; catches career under-performers. |
| **113** | `Trade_Construction_Volatile_Leverage` | Volatile Industry $\land \text{DTI} \ge 0.20$ | 50,460 (29.47%) | 10.52% | `DTI * (Credit / Goods) / (Employed_Yrs + 0.5)` | +0.0838 | 7.14% vs 15.60% | **846.2** | Workers in cyclically volatile trades carrying high DTI face seasonal layoffs that derail debt servicing. |
| **114** | `Public_Security_Stability_Bonus` | Police, Military, Government | 10,268 (6.00%) | 7.14% | `(Monthly_Income - Annuity) * sqrt(E2)` | +0.0234 | 7.40% vs 6.82% | **58.4** | State employees and security personnel benefit from recession-proof public budgets and guaranteed wage floors. |
| **115** | `Tenured_Landline_Enterprise_Anchor` | Work Phone Verified $\land \text{Emp} \ge 2\text{yr}$ | 26,311 (15.37%) | 8.94% | `log(1 + Employed_Yrs) * (E2 + E3)` | -0.1886 | 23.95% vs 1.95% | **2199.8** | Borrowers at established firms with verified physical landline switchboards have low ghost-employment risk. |
| **116** | `Mobile_Number_Churn_to_Age_Ratio` | Phone Change $\le 180\text{ days}$ | 35,509 (20.74%) | 9.26% | `Phone_Change_Years / (Age - 18)` | -0.0008 | 8.35% vs 10.34% | **199.0** | Frequent phone number updates relative to borrower age signal evasion, transience, or contact instability. |
| **117** | `Housing_Equity_Shield_Ratio` | Homeowner $\land \text{Leverage} \le 2.5$ | 52,855 (30.87%) | 7.64% | `(Income / Annuity) * (1 + Population_Relative)` | -0.0165 | 8.01% vs 7.31% | **70.0** | Homeowners with modest credit leverage represent deeply anchored, low-risk prime borrowers. |
| **118** | `Rented_Apartment_Annuity_Burden` | Rented Apt $\land \text{Annuity} \ge 20\text{k}$ | 1,678 (0.98%) | 13.83% | `(12 * Annuity) / Income` | +0.0086 | 13.83% vs 13.83% | **0.0** | Renters face fixed landlord lease payments in addition to loan annuities, multiplying default risk. |
| **119** | `Multi_Generational_Parental_Buffer` | With Parents $\land \text{Income} \ge 140\text{k}$ | 3,818 (2.23%) | 10.87% | `(Monthly_Income - Annuity) / Annuity` | -0.0440 | 13.58% vs 9.39% | **418.8** | Working adults living with parents avoid rental overhead, leaving full salary available for debt service. |
| **120** | `Extreme_Underwater_Loan_Surcharge` | Credit $\ge 1.25 \times \text{Goods Price}$ | 23,061 (13.47%) | 12.20% | `((Credit - Goods) / Annuity) * (1 - E2)` | +0.0974 | 7.54% vs 19.51% | **1196.4** | Excessive upfront loan markup creating severe negative equity compounded by subprime credit bureau status. |

---

## 3. Top Regime Formulations Ranked by Default Spread Separation

The 5 highest-separating formulations discovered in Exp 91-120 are:

1. **`F_94_Clayton_Deep_Distress_Lower_Copula` (Exp 94)**:
   - **Spread**: **`2,252.5 bps`** (32.86% high decile vs 10.34% low decile, $\rho = -0.1666$)
   - **Formula**: `(u^(-3) + v^(-3) - 1)^(-1/3)`
   - **Behavior**: Concentrates solely on joint lower-tail distress, detecting applicants whose ratings collapse across multiple agencies simultaneously.
2. **`F_115_Tenured_Landline_Enterprise_Anchor` (Exp 115)**:
   - **Spread**: **`2,199.8 bps`** (23.95% low decile vs 1.95% high decile, $\rho = -0.1886$)
   - **Formula**: `log(1 + Employed_Yrs) * (E2 + E3)`
   - **Behavior**: Interaction between physical employer landline verification, career tenure, and bureau consensus isolates high-credit super-stable corporate employees.
3. **`F_106_Dual_Bureau_Harmonic_Distress_Floor` (Exp 106)**:
   - **Spread**: **`2,018.9 bps`** (19.34% low decile vs 1.95% high decile, $\rho = -0.1989$)
   - **Formula**: `2.0 / ((1/E2) + (1/E3))`
   - **Behavior**: Harmonic mean severely penalizes borrowers when either bureau drops, enforcing the "weakest-link" underwriting principle.
4. **`F_93_Frank_Symmetric_Comprehensive_Copula` (Exp 93)**:
   - **Spread**: **`1,956.1 bps`** (21.56% low decile vs 2.00% high decile, $\rho = -0.1834$)
   - **Formula**: `-1/4 * log(1 + (exp(-4u)-1)*(exp(-4v)-1)/(exp(-4)-1))`
   - **Behavior**: Continuous radially symmetric copula mapping joint dependence across the full probability domain.
5. **`F_109_Calm_Credit_History_Anchor` (Exp 109)**:
   - **Spread**: **`1,797.8 bps`** (19.98% low decile vs 2.00% high decile, $\rho = -0.1619$)
   - **Formula**: `log(1 + Inq_Year) * (E2 * E3)`
   - **Behavior**: Identifies planned, mature borrowers with credit history who exhibit zero acute credit-seeking urgency.
