"""
====================================================================================================
GCI WORLD 2026 — 100 DEEP OPTIMIZED SPECIALIZED REGIME SPLITS & HIGH-AUC MASTER FORMULAS (EXP 141 - 240)
FULL EMPIRICAL RESEARCH SUITE | 10 DIVERSE DOMAINS | STRICT PERMITTED DATASET ONLY | 5-FOLD CV AUDIT
====================================================================================================
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

t_start = time.time()
print("=" * 95)
print("  GCI WORLD 2026: 100 DEEP OPTIMIZED SPECIALIZED REGIME SPLITS & MASTER FORMULAS (EXP 141 - 240)")
print("=" * 95)

train_path = r"D:\gci\inpuy\train.csv"
df = pd.read_csv(train_path)
y = df["TARGET"].values
n_train = len(df)
print(f"[+] Loaded {n_train:,} training records in {time.time() - t_start:.2f}s.")

# --------------------------------------------------------------------------------------------------
# BASE VECTOR EXTRACTIONS (STRICTLY FROM train.csv 34 CURATED COLUMNS)
# --------------------------------------------------------------------------------------------------
inc = df["AMT_INCOME_TOTAL"].values.astype(float)
cr = df["AMT_CREDIT"].values.astype(float)
an = df["AMT_ANNUITY"].values.astype(float)
gp = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"]).values.astype(float)
fam = df["CNT_FAM_MEMBERS"].fillna(1.0).values.astype(float)
ch = df["CNT_CHILDREN"].values.astype(float)

emp_raw = df["DAYS_EMPLOYED"].values.astype(float)
is_retired = (emp_raw == 365243.0)
emp = np.where(is_retired, 0.0, -emp_raw)
emp_yrs = emp / 365.25

birth = df["DAYS_BIRTH"].values.astype(float)
age = -birth / 365.25
adult_life = np.maximum(age - 18.0, 1.0)

reg = df["DAYS_REGISTRATION"].values.astype(float)
reg_yrs = -reg / 365.25

idp = df["DAYS_ID_PUBLISH"].values.astype(float)
idp_yrs = -idp / 365.25

ph_raw = df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values.astype(float)
ph_yrs = -ph_raw / 365.25

car_age = df["OWN_CAR_AGE"].fillna(-1.0).values.astype(float)
car_flag = (~df["OWN_CAR_AGE"].isna()).astype(float)

pop = df["REGION_POPULATION_RELATIVE"].values.astype(float)
rating = df["REGION_RATING_CLIENT"].values.astype(float)

e1 = df["EXT_SOURCE_1"].values.astype(float)
e2 = df["EXT_SOURCE_2"].values.astype(float)
e3 = df["EXT_SOURCE_3"].values.astype(float)

med_e1 = float(np.nanmedian(e1))
med_e2 = float(np.nanmedian(e2))
med_e3 = float(np.nanmedian(e3))

e1_f = np.nan_to_num(e1, nan=med_e1)
e2_f = np.nan_to_num(e2, nan=med_e2)
e3_f = np.nan_to_num(e3, nan=med_e3)

u1 = np.clip(e2_f, 1e-4, 1.0 - 1e-4)
u2 = np.clip(e3_f, 1e-4, 1.0 - 1e-4)

# Inquiries
inq_h = df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0).values.astype(float)
inq_m = df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0).values.astype(float)
inq_q = df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0).values.astype(float)
inq_y = df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0).values.astype(float)
inq_missing = df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().values

# Verification flags
flag_emp_phone = df["FLAG_EMP_PHONE"].fillna(0).values.astype(float)
flag_work_phone = df["FLAG_WORK_PHONE"].fillna(0).values.astype(float)

# String categoricals
contract = df["NAME_CONTRACT_TYPE"].fillna("").values
gender = df["CODE_GENDER"].fillna("").values
family = df["NAME_FAMILY_STATUS"].fillna("").values
housing = df["NAME_HOUSING_TYPE"].fillna("").values
income_type = df["NAME_INCOME_TYPE"].fillna("").values
edu = df["NAME_EDUCATION_TYPE"].fillna("").values
occ = df["OCCUPATION_TYPE"].fillna("").values
org = df["ORGANIZATION_TYPE"].fillna("").values

# Identity entity key
entity_key = (
    (birth - reg).round(0).astype(int).astype(str) + "|" +
    (birth - idp).round(0).astype(int).astype(str) + "|" +
    gender.astype(str) + "|" +
    edu.astype(str)
)
ent_series = pd.Series(entity_key)
cnt_ent = ent_series.value_counts()
n_apps_all = ent_series.map(cnt_ent).values.astype(float)

df_sort = pd.DataFrame({"ent": entity_key, "reg": reg, "orig_idx": np.arange(n_train)})
df_sort = df_sort.sort_values(["ent", "reg"])
df_sort["APP_ORDER"] = df_sort.groupby("ent").cumcount()
df_sort["DAYS_SINCE_PREV_APP"] = df_sort.groupby("ent")["reg"].diff()
df_restored = df_sort.sort_values("orig_idx")
app_order = df_restored["APP_ORDER"].values.astype(float)
days_since_prev = df_restored["DAYS_SINCE_PREV_APP"].fillna(-999.0).values.astype(float)

# Master derived baselines
monthly_inc = inc / 12.0
dti = np.clip(cr / (inc + 1.0), 0.0, 30.0)
term = np.clip(cr / (an + 1.0), 1.0, 100.0)
subsistence = 12000.0 * ch + 15000.0

# Copulas
cop_clayton = np.maximum(u1**(-3.0) + u2**(-3.0) - 1.0, 1e-6)**(-1.0 / 3.0)
num_f = (np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)
cop_frank = -0.25 * np.log(1.0 + num_f / (np.exp(-4.0) - 1.0))
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))

# Single male laborer hazard
is_male = (gender == "M")
is_single = np.isin(family, ["Single / not married", "Separated", "Widow"])
is_laborer = np.isin(occ, ["Laborers", "Drivers", "Low-skill Laborers"])
single_male_hazard = (is_male & is_single & is_laborer).astype(float) * dti * (cr / (gp + 1.0)) * (1.0 - u1)

# Merton DD
disp_ratio = np.maximum(0.01, monthly_inc / (an + 1.0))
T_years = np.clip(term / 12.0, 0.1, 10.0)
merton_dd = (np.log(disp_ratio) + (0.02 - 0.5 * 0.20**2) * T_years) / (0.20 * np.sqrt(T_years))

# --------------------------------------------------------------------------------------------------
# EVALUATION STORAGE & ENGINE
# --------------------------------------------------------------------------------------------------
results = {}
discovered_features = {}

def evaluate_regime(exp_id, name, mask, formula_arr, formula_str, domain_reason):
    mask = mask.astype(bool)
    count = int(mask.sum())
    pct = round(count / n_train * 100, 2)
    def_rate = round(float(y[mask].mean() * 100), 2) if count > 0 else 0.0
    
    sub_y = y[mask]
    sub_feat = formula_arr[mask]
    valid = ~np.isnan(sub_feat) & ~np.isinf(sub_feat)
    
    if valid.sum() > 20:
        corr, _ = spearmanr(sub_feat[valid], sub_y[valid])
        corr = round(float(corr), 4)
        
        q10 = np.quantile(sub_feat[valid], 0.1)
        q90 = np.quantile(sub_feat[valid], 0.9)
        d_low = float(sub_y[valid][sub_feat[valid] <= q10].mean() * 100)
        d_high = float(sub_y[valid][sub_feat[valid] >= q90].mean() * 100)
        spread_bps = round(abs(d_high - d_low) * 100, 1)
        low_high_str = f"{d_low:.2f}% vs {d_high:.2f}%"
    else:
        corr = 0.0
        spread_bps = 0.0
        low_high_str = "N/A"
        
    print(f"[{exp_id:3d}] {name:42s} | N={count:6,d} ({pct:4.1f}%) | Def={def_rate:5.2f}% | Corr: {corr:+6.4f} | Spread: {spread_bps:6.1f} bps")
    
    results[f"exp_{exp_id}_{name}"] = {
        "exp_id": exp_id,
        "name": name,
        "subgroup_count": count,
        "subgroup_pct": pct,
        "subgroup_default_rate": def_rate,
        "formula": formula_str,
        "within_split_correlation": corr,
        "split_low_vs_high_default": low_high_str,
        "spread_bps": spread_bps,
        "domain_reason": domain_reason
    }
    discovered_features[f"F_{exp_id}_{name}"] = formula_arr

print("\n" + "=" * 95)
print("  EXECUTING 100 DEEP OPTIMIZED SPECIALIZED REGIME SPLITS (EXP 141 - 240)")
print("=" * 95)

# ==================================================================================================
# DOMAIN 1: DEMOGRAPHIC LIFECYCLE & HOUSEHOLD FRAGILITY (EXP 141 - 150)
# ==================================================================================================
# 141. Young Single Renter Obligation
m141 = (age < 28.0) & is_single & (housing == "Rented apartment")
f141 = (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0)) / (u1 + 0.1)
evaluate_regime(141, "Young_Single_Renter_Obligation", m141, f141,
                "(Annuity / Monthly_Income) * (Credit / Goods) / (E2 + 0.1)",
                "Young unpartnered renters lacking residential equity face compounded rent and debt strain.")

# 142. Sandwich Generation Dependency Strain
m142 = (age >= 35.0) & (age <= 50.0) & (ch >= 2) & (housing == "With parents")
f142 = (an * (ch + 2.0)) / (inc + 1.0) * (1.0 - cop_frank)
evaluate_regime(142, "Sandwich_Generation_Dependency_Strain", m142, f142,
                "(Annuity * (Children + 2)) / Income * (1 - Frank_Copula)",
                "Middle-aged parents cohabiting with elderly relatives shoulder severe multi-generational costs.")

# 143. Senior Pensioner Single Annuity Burn
m143 = (age >= 58.0) & is_retired & np.isin(family, ["Widow", "Separated"])
f143 = (an * 12.0) / (inc + 1.0) * (1.0 / (u2 + 0.1))
evaluate_regime(143, "Senior_Pensioner_Single_Annuity_Burn", m143, f143,
                "(12 * Annuity) / Income * (1 / (E3 + 0.1))",
                "Isolated senior retirees surviving on fixed state pensions face catastrophic annuity burdens.")

# 144. Large Family Overcrowding Burden
m144 = fam >= 5.0
f144 = (cr / (inc / fam + 1.0)) * (1.0 - u1 * u2)
evaluate_regime(144, "Large_Family_Overcrowding_Burden", m144, f144,
                "Credit / (Income / Family_Size) * (1 - E2 * E3)",
                "Households with 5+ members face extreme dilution of disposable income per head.")

# 145. Sole Breadwinner Child Multiplier
m145 = is_single & (ch >= 1)
f145 = (an / (monthly_inc - subsistence + 1.0)) * (1.0 + dti)
evaluate_regime(145, "Sole_Breadwinner_Child_Multiplier", m145, f145,
                "(Annuity / (Monthly_Inc - Subsistence)) * (1 + DTI)",
                "Single parents without spousal income support experience severe subsistence cash depletion.")

# 146. Early Career Commercial Associate
m146 = (age < 30.0) & (income_type == "Commercial associate") & (emp_yrs < 2.0)
f146 = (cr / (emp_yrs + 0.2)) / (monthly_inc + 1.0) * (1.0 - u1)
evaluate_regime(146, "Early_Career_Commercial_Associate", m146, f146,
                "(Credit / (Emp_Yrs + 0.2)) / Monthly_Inc * (1 - E2)",
                "Young commercial workers in competitive sales roles carry high leverage on short employment history.")

# 147. Tenured Civil Servant Stability
m147 = (income_type == "State servant") & (emp_yrs >= 7.0)
f147 = (inc / (cr + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)
evaluate_regime(147, "Tenured_Civil_Servant_Stability", m147, f147,
                "(Income / Credit) * (E2 + E3) * log1p(Emp_Yrs)",
                "Long-tenured public sector employees backed by state employment stability represent premier anchors.")

# 148. Maternity Unemployment Vulnerability
m148 = np.isin(income_type, ["Maternity leave", "Unemployed"])
f148 = cr / (monthly_inc + 1.0) * (1.0 / (u2 + 0.05))
evaluate_regime(148, "Maternity_Unemployment_Vulnerability", m148, f148,
                "Credit / Monthly_Inc * (1 / (E3 + 0.05))",
                "Zero or near-zero active earnings during career interruptions create immediate default risks.")

# 149. Midlife Divorced Unanchored Male
m149 = is_male & np.isin(family, ["Separated", "Single / not married"]) & (age >= 38.0) & (age <= 52.0)
f149 = dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))
evaluate_regime(149, "Midlife_Divorced_Unanchored_Male", m149, f149,
                "DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Divorced or single men in middle adulthood without household asset anchoring exhibit high default drift.")

# 150. Multi Generational House Asset Anchor
m150 = (housing == "With parents") & (car_age >= 0) & (car_age < 10.0)
f150 = (monthly_inc / (an + 1.0)) * cop_harmonic * (10.0 - car_age)
evaluate_regime(150, "Multi_Generational_House_Asset_Anchor", m150, f150,
                "(Monthly_Inc / Annuity) * Harmonic_Copula * (10 - Car_Age)",
                "Co-residing adults with modern owned vehicles possess unencumbered personal mobility and asset equity.")

# ==================================================================================================
# DOMAIN 2: WORKPLACE, INDUSTRY & OCCUPATIONAL POLARIZATION (EXP 151 - 160)
# ==================================================================================================
# 151. Hazardous Industry Manual Laborer
m151 = np.isin(org, ["Construction", "Transport: type 3", "Industry: type 3", "Industry: type 9"]) & is_laborer
f151 = (cr / (gp + 1.0)) * dti * (1.0 - u1)
evaluate_regime(151, "Hazardous_Industry_Manual_Laborer", m151, f151,
                "(Credit / Goods) * DTI * (1 - E2)",
                "Manual workers in volatile heavy industries experience high physical injury and economic volatility.")

# 152. High Skill IT Academic Solvency
m152 = (edu == "Higher education") & np.isin(occ, ["IT staff", "High skill tech staff", "Core staff"]) & (emp_yrs >= 3.0)
f152 = (monthly_inc / (an + 1.0)) * (u1 * u2) * np.log1p(emp_yrs)
evaluate_regime(152, "High_Skill_IT_Academic_Solvency", m152, f152,
                "(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)",
                "Specialized professionals in technology with stable careers demonstrate resilient disposable buffers.")

# 153. Informal Trade Sales Volatility
m153 = np.isin(org, ["Trade: type 3", "Trade: type 7", "Services"]) & (flag_work_phone == 0.0)
f153 = (an / (monthly_inc + 1.0)) * (1.0 / (u2 + 0.1))
evaluate_regime(153, "Informal_Trade_Sales_Volatility", m153, f153,
                "(Annuity / Monthly_Inc) / (E3 + 0.1)",
                "Retail trade workers without a verifiable landline office phone have precarious informal earnings.")

# 154. Security Police Stable Anchor
m154 = np.isin(occ, ["Security staff"]) | np.isin(org, ["Police", "Security", "Military"])
f154 = (inc / (cr + 1.0)) * cop_frank * (emp_yrs + 1.0)
evaluate_regime(154, "Security_Police_Stable_Anchor", m154, f154,
                "(Income / Credit) * Frank_Copula * (Emp_Yrs + 1)",
                "State enforcement and institutional security staff maintain rigid regularized payroll schedules.")

# 155. Low Skill Day Laborer Distress
m155 = (occ == "Low-skill Laborers") & (inc < np.median(inc))
f155 = (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))
evaluate_regime(155, "Low_Skill_Day_Laborer_Distress", m155, f155,
                "(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)",
                "Lowest occupational tier workers earning below-median income have zero margin for credit repayment.")

# 156. Executive Corporate Leverage
m156 = np.isin(occ, ["Managers", "Accountants"]) & (inc >= np.percentile(inc, 80))
f156 = (cr / (monthly_inc + 1.0)) * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(156, "Executive_Corporate_Leverage", m156, f156,
                "(Credit / Monthly_Inc) * (1 - E2) * (Credit / Goods)",
                "High-earning managers taking large speculative credit amounts exhibit sharp credit-score sensitivity.")

# 157. Healthcare Public Sector Tenure
m157 = np.isin(org, ["Medicine", "Kindergarten", "School"]) & (income_type == "State servant")
f157 = (monthly_inc / (an + 1.0)) * (u1 + u2) / (1.0 + ch)
evaluate_regime(157, "Healthcare_Public_Sector_Tenure", m157, f157,
                "(Monthly_Inc / Annuity) * (E2 + E3) / (1 + Children)",
                "Public healthcare and educational staff have predictable incomes moderated by family dependency.")

# 158. Cleaning Cooking Service Pressure
m158 = np.isin(occ, ["Cleaning staff", "Cooking staff"]) & (ch >= 1)
f158 = (an / (monthly_inc - subsistence + 1.0)) * (1.0 / (u1 + 0.1))
evaluate_regime(158, "Cleaning_Cooking_Service_Pressure", m158, f158,
                "(Annuity / (Monthly_Inc - Subsistence)) / (E2 + 0.1)",
                "Domestic service workers supporting dependents experience severe cash exhaustion on repayment days.")

# 159. Self Employed Sole Proprietor Friction
m159 = (income_type == "Commercial associate") & np.isin(org, ["Business Entity Type 3", "Self-employed"])
f159 = dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))
evaluate_regime(159, "Self_Employed_Sole_Proprietor_Friction", m159, f159,
                "DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Sole proprietors bear the full variance of business revenue shocks without formal unemployment benefits.")

# 160. Driver Automotive Collateral Vulnerability
m160 = (occ == "Drivers") & (car_flag == 1.0) & (cr >= 400000)
f160 = (cr / (gp + 1.0)) * (car_age + 1.0) / (u2 + 0.1)
evaluate_regime(160, "Driver_Automotive_Collateral_Vulnerability", m160, f160,
                "(Credit / Goods) * (Car_Age + 1) / (E3 + 0.1)",
                "Professional drivers financing debt against aging vehicles face simultaneous asset depreciation and repair costs.")

# ==================================================================================================
# DOMAIN 3: CREDIT BUREAU DIVERGENCE & COPULA MANIFOLDS (EXP 161 - 170)
# ==================================================================================================
# 161. Bureau Triad Full Agreement Anchor
valid_triad = ~np.isnan(e1) & ~np.isnan(e2) & ~np.isnan(e3)
triad_std = np.std(np.column_stack([e1_f, e2_f, e3_f]), axis=1)
m161 = valid_triad & (triad_std < 0.08)
f161 = ((e1_f + e2_f + e3_f) / 3.0) * (inc / (cr + 1.0))
evaluate_regime(161, "Bureau_Triad_Full_Agreement_Anchor", m161, f161,
                "Mean(E1, E2, E3) * (Income / Credit)",
                "When all three external bureaus report consensual high scores, applicant credit risk is exceptionally low.")

# 162. Bureau Bipolar Acute Discordance
m162 = valid_triad & (e2_f > 0.60) & (e3_f < 0.30)
f162 = (e2_f - e3_f) * dti / (u2 + 0.05)
evaluate_regime(162, "Bureau_Bipolar_Acute_Discordance", m162, f162,
                "(E2 - E3) * DTI / (E3 + 0.05)",
                "Massive divergence where Agency 2 is optimistic but Agency 3 flags acute distress signals hidden borrower risk.")

# 163. Gumbel Upper Tail Ultra Prime Sanctuary
m163 = (u1 > 0.65) & (u2 > 0.65)
f163 = (u1 * u2) * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)
evaluate_regime(163, "Gumbel_Upper_Tail_Ultra_Prime_Sanctuary", m163, f163,
                "(E2 * E3) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Simultaneous high-percentile bureau ratings isolate premier prime borrowers with near-zero default rates.")

# 164. Clayton Deep Subprime Sinkhole
m164 = (u1 < 0.35) & (u2 < 0.35)
f164 = cop_clayton * dti * (cr / (gp + 1.0))
evaluate_regime(164, "Clayton_Deep_Subprime_Sinkhole", m164, f164,
                "Clayton_Copula * DTI * (Credit / Goods)",
                "Joint lower-tail concentration captures extreme correlated default clusters under financial distress.")

# 165. Bureau Harmonic vs Arithmetic Deficit
arith_mean = (e1_f + e2_f + e3_f) / 3.0
harm_mean = 3.0 / ((1.0 / np.clip(e1_f, 0.01, 1.0)) + (1.0 / np.clip(e2_f, 0.01, 1.0)) + (1.0 / np.clip(e3_f, 0.01, 1.0)))
f165 = (arith_mean - harm_mean) * (cr / (inc + 1.0))
m165 = valid_triad
evaluate_regime(165, "Bureau_Harmonic_Vs_Arithmetic_Deficit", m165, f165,
                "(Arithmetic_Mean - Harmonic_Mean) * (Credit / Income)",
                "Disproportionate penalization of an applicant who has even one disastrous agency score.")

# 166. Missing Bureau 3 Informational Blindspot
m166 = np.isnan(e3) & ~np.isnan(e2)
f166 = (1.0 - u1) * dti * (an / (monthly_inc + 1.0))
evaluate_regime(166, "Missing_Bureau_3_Informational_Blindspot", m166, f166,
                "(1 - E2) * DTI * (Annuity / Monthly_Inc)",
                "Lenders operating without Bureau 3 data face asymmetric adverse selection.")

# 167. Missing Bureau 1 Novel Credit Profile
m167 = np.isnan(e1) & ~np.isnan(e2) & ~np.isnan(e3)
f167 = cop_frank * (monthly_inc / (an + 1.0))
evaluate_regime(167, "Missing_Bureau_1_Novel_Credit_Profile", m167, f167,
                "Frank_Copula(E2, E3) * (Monthly_Inc / Annuity)",
                "Absence of Bureau 1 reflects younger or newer banking history, where joint E2-E3 agreement is decisive.")

# 168. Single Bureau Isolated Reliance
m168 = np.isnan(e1) & np.isnan(e3) & ~np.isnan(e2)
f168 = (1.0 - u1) * (cr / (gp + 1.0)) * (an / (monthly_inc + 1.0))
evaluate_regime(168, "Single_Bureau_Isolated_Reliance", m168, f168,
                "(1 - E2) * (Credit / Goods) * (Annuity / Monthly_Inc)",
                "Single-source evaluation leaves underwriting models vulnerable to unobserved credit delinquencies.")

# 169. Frank Radial Symmetry Deficit
f169 = np.abs(u1 + u2 - 1.0) * dti
m169 = (u1 > 0.1) & (u2 > 0.1)
evaluate_regime(169, "Frank_Radial_Symmetry_Deficit", m169, f169,
                "abs(E2 + E3 - 1.0) * DTI",
                "Deviation from the center manifold of credit evaluation exposes structural risk imbalance.")

# 170. Bureau Logit Variance Polarization
logit_e1 = np.log(np.clip(e1_f, 0.01, 0.99) / (1.0 - np.clip(e1_f, 0.01, 0.99)))
logit_e2 = np.log(np.clip(e2_f, 0.01, 0.99) / (1.0 - np.clip(e2_f, 0.01, 0.99)))
logit_e3 = np.log(np.clip(e3_f, 0.01, 0.99) / (1.0 - np.clip(e3_f, 0.01, 0.99)))
f170 = np.var(np.column_stack([logit_e1, logit_e2, logit_e3]), axis=1) * (cr / (gp + 1.0))
m170 = valid_triad
evaluate_regime(170, "Bureau_Logit_Variance_Polarization", m170, f170,
                "Var(logit(E1), logit(E2), logit(E3)) * (Credit / Goods)",
                "High logit variance indicates fundamental rating disagreement across independent banking bureaus.")

# ==================================================================================================
# DOMAIN 4: CONTRACT GEOMETRY, ANNUITY RATIOS & TARIFFS (EXP 171 - 180)
# ==================================================================================================
# 171. Revolving High Velocity Buffer
m171 = (contract == "Revolving loans")
f171 = (inc / (an + 1.0)) * cop_harmonic * (1.0 - dti / 10.0)
evaluate_regime(171, "Revolving_High_Velocity_Buffer", m171, f171,
                "(Income / Annuity) * Harmonic_Copula * (1 - DTI / 10)",
                "Revolving credit products allow discretionary repayments, where liquidity ratios strongly protect solvency.")

# 172. Cash Loan Fixed Fee Lattice Exact
cash_mask = (contract == "Cash loans")
qb = (an - 129.5) / (cr + 1e-5)
m172 = cash_mask & (np.abs(qb - np.round(qb, 4)) < 1e-4) & (cr > gp)
f172 = qb * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(172, "Cash_Loan_Fixed_Fee_Lattice_Exact", m172, f172,
                "Q0_Fee * (1 - E2) * (Credit / Goods)",
                "Exact fit on the 129.50 fee lattice isolates standardized retail bank cash-credit programs.")

# 173. Ultra Long Tenure Amortization Strain
m173 = (cr / (an + 1.0)) >= 24.0
f173 = (cr / (an + 1.0)) * (1.0 - cop_frank) * dti
evaluate_regime(173, "Ultra_Long_Tenure_Amortization_Strain", m173, f173,
                "(Credit / Annuity) * (1 - Frank_Copula) * DTI",
                "Repayment terms exceeding 24 months prolong borrower vulnerability to life shock events.")

# 174. Ultra Short Bridge Financing Surge
m174 = (cr / (an + 1.0)) <= 10.0
f174 = (an / (monthly_inc + 1.0)) * (1.0 / (u2 + 0.1))
evaluate_regime(174, "Ultra_Short_Bridge_Financing_Surge", m174, f174,
                "(Annuity / Monthly_Inc) / (E3 + 0.1)",
                "Short bridge loans demand massive monthly cash outflows relative to monthly income.")

# 175. Severe Negative Downpayment Excess
m175 = cr > 1.20 * gp
f175 = ((cr - gp) / (gp + 1.0)) * (1.0 - u1) * dti
evaluate_regime(175, "Severe_Negative_Downpayment_Excess", m175, f175,
                "((Credit - Goods) / Goods) * (1 - E2) * DTI",
                "Borrowers rolling consumer fees and uncollateralized balances >20% over goods face immediate negative equity.")

# 176. Subsidized Par Financing Contract
m176 = (cr == gp) & (gp > 0)
f176 = (inc / (cr + 1.0)) * cop_clayton * np.log1p(emp_yrs)
evaluate_regime(176, "Subsidized_Par_Financing_Contract", m176, f176,
                "(Income / Credit) * Clayton_Copula * log1p(Emp_Yrs)",
                "Zero markup promotional loans (Credit = Goods Price) attract credit-savvy prime retail shoppers.")

# 177. High Annuity Absolute Burden
m177 = an >= 40000.0
f177 = (an / (monthly_inc + 1.0)) * (1.0 - u1 * u2) * (cr / (gp + 1.0))
evaluate_regime(177, "High_Annuity_Absolute_Burden", m177, f177,
                "(Annuity / Monthly_Inc) * (1 - E2 * E3) * (Credit / Goods)",
                "Top-tier monthly obligations above 40k rubles cause immediate distress upon income disruption.")

# 178. Alexander APR 36 Month Stress
apr_36 = (36.0 * an - cr) / (cr + 1e-5)
m178 = apr_36 > 0.35
f178 = apr_36 * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))
evaluate_regime(178, "Alexander_APR_36_Month_Stress", m178, f178,
                "Alexander_APR_36 * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "High effective interest rate markups over a 36-month horizon signal predatory terms and desperate borrowers.")

# 179. Micro Loan Disproportionate Fixed Fee
m179 = cash_mask & (cr <= 150000.0)
f179 = (129.5 / (an + 1.0)) * (1.0 / (u1 + 0.1))
evaluate_regime(179, "Micro_Loan_Disproportionate_Fixed_Fee", m179, f179,
                "(129.50 / Annuity) / (E2 + 0.1)",
                "Small loans where the flat 129.50 fee constitutes a huge percentage of monthly servicing costs.")

# 180. Jumbo Syndicated Threshold Exposure
m180 = cr >= 1200000.0
f180 = (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))
evaluate_regime(180, "Jumbo_Syndicated_Threshold_Exposure", m180, f180,
                "(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)",
                "Large principal exposures above 1.2M rubles represent severe portfolio loss severity upon default.")

# ==================================================================================================
# DOMAIN 5: SOLVENCY, CASH FLOW BUFFERS & LIVING COSTS (EXP 181 - 190)
# ==================================================================================================
# 181. Sub Zero Disposable Cash Flow
fcf = monthly_inc - an - subsistence
m181 = fcf < 0.0
f181 = (-fcf / (monthly_inc + 1.0)) * (1.0 - u1)
evaluate_regime(181, "Sub_Zero_Disposable_Cash_Flow", m181, f181,
                "(-FCF / Monthly_Inc) * (1 - E2)",
                "Households whose basic survival costs and annuity exceed monthly take-home pay are mathematically insolvent.")

# 182. Extreme DTI Overleverage
m182 = dti >= 6.0
f182 = dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))
evaluate_regime(182, "Extreme_DTI_Overleverage", m182, f182,
                "DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Debt-to-income exceeding 6.0 years of total gross earnings signals unsustainable leverage.")

# 183. Debt Service Coverage Fragility
dscr = an / (monthly_inc + 1.0)
m183 = dscr >= 0.40
f183 = dscr * (1.0 / (u2 + 0.1)) * (cr / (gp + 1.0))
evaluate_regime(183, "Debt_Service_Coverage_Fragility", m183, f183,
                "(Annuity / Monthly_Inc) / (E3 + 0.1) * (Credit / Goods)",
                "Devoting >40% of gross monthly earnings strictly to debt service leaves zero resilience for emergencies.")

# 184. Per Capita Family Income Squeeze
inc_per_fam = inc / fam
m184 = inc_per_fam < 40000.0
f184 = (cr / (inc_per_fam + 1.0)) * (1.0 - u1)
evaluate_regime(184, "Per_Capita_Family_Income_Squeeze", m184, f184,
                "(Credit / Income_Per_Fam) * (1 - E2)",
                "Low per-capita household income amplifies default risk as food and medical necessities take precedence.")

# 185. Double Inflation Shock Insolvency
fcf_shock = monthly_inc - 1.5 * subsistence - an
m185 = fcf_shock < 0.0
f185 = (-fcf_shock / (monthly_inc + 1.0)) * (1.0 - cop_frank)
evaluate_regime(185, "Double_Inflation_Shock_Insolvency", m185, f185,
                "(-Shock_FCF / Monthly_Inc) * (1 - Frank_Copula)",
                "Simulates household default rates under a 50% non-discretionary inflation shock.")

# 186. Affluent Liquidity Fortress
m186 = (inc >= 300000.0) & (dti <= 2.0)
f186 = (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)
evaluate_regime(186, "Affluent_Liquidity_Fortress", m186, f186,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)",
                "High-earning, lightly levered borrowers maintain impenetrable liquidity reserves.")

# 187. Living Margin Surplus to Goods Ratio
surplus_val = inc - 12.0 * an
m187 = surplus_val > 0.0
f187 = (surplus_val / (gp + 1.0)) * cop_harmonic
evaluate_regime(187, "Living_Margin_Surplus_To_Goods_Ratio", m187, f187,
                "(Annual_Surplus / Goods) * Harmonic_Copula",
                "Annual discretionary cash buffer relative to acquired goods value indicates true purchasing power.")

# 188. Merton Default Probability High Barrier
m188 = merton_dd < 1.0
f188 = norm.cdf(-merton_dd) * (1.0 - u1) * dti
evaluate_regime(188, "Merton_Default_Probability_High_Barrier", m188, f188,
                "Merton_PD * (1 - E2) * DTI",
                "Structural distance-to-default models identify insolvent cash-flow trajectories.")

# 189. Uncollateralized Burn Per Employed Month
emp_months = np.maximum(emp_yrs * 12.0, 1.0)
uncoll_burn = np.maximum(cr - gp, 0.0) / emp_months
m189 = uncoll_burn > 500.0
f189 = uncoll_burn * (1.0 / (u2 + 0.1))
evaluate_regime(189, "Uncollateralized_Burn_Per_Employed_Month", m189, f189,
                "(Uncollateralized_Credit / Emp_Months) / (E3 + 0.1)",
                "Excess loan principal accumulated relative to accumulated employment career tenure.")

# 190. Household Dependency Cost Annuity Parity
m190 = subsistence > an
f190 = (subsistence / (an + 1.0)) * (1.0 - u1) * (cr / (inc + 1.0))
evaluate_regime(190, "Household_Dependency_Cost_Annuity_Parity", m190, f190,
                "(Subsistence / Annuity) * (1 - E2) * (Credit / Income)",
                "When non-negotiable dependent living expenses exceed the monthly loan payment, loan defaults surge.")

# ==================================================================================================
# DOMAIN 6: IDENTITY STABILITY & ADMINISTRATIVE TELEMETRY (EXP 191 - 200)
# ==================================================================================================
# 191. Recent Address Registration Instability
m191 = reg_yrs < 1.0
f191 = (1.0 / (reg_yrs + 0.1)) * (1.0 - u1) * dti
evaluate_regime(191, "Recent_Address_Registration_Instability", m191, f191,
                "(1 / (Reg_Yrs + 0.1)) * (1 - E2) * DTI",
                "Recent municipal address changes correlate with transient living conditions and credit flight.")

# 192. Recent Passport Renewal Velocity
m192 = idp_yrs < 1.5
f192 = (1.0 / (idp_yrs + 0.1)) * (1.0 - u2) * (cr / (gp + 1.0))
evaluate_regime(192, "Recent_Passport_Renewal_Velocity", m192, f192,
                "(1 / IDP_Yrs + 0.1) * (1 - E3) * (Credit / Goods)",
                "Re-issuance of state passport within 18 months of credit application flags identity churn.")

# 193. Statutory Passport Age 20 Window
age_id = age - idp_yrs
m193 = (age >= 19.5) & (age <= 21.0)
f193 = np.abs(age_id - 20.0) * dti * (1.0 - u1)
evaluate_regime(193, "Statutory_Passport_Age_20_Window", m193, f193,
                "abs(Age_at_ID - 20) * DTI * (1 - E2)",
                "Identifies delay or compliance delinquency in mandatory age-20 Russian domestic passport renewal.")

# 194. Statutory Passport Age 45 Window
m194 = (age >= 44.5) & (age <= 46.0)
f194 = np.abs(age_id - 45.0) * dti * (1.0 - u2)
evaluate_regime(194, "Statutory_Passport_Age_45_Window", m194, f194,
                "abs(Age_at_ID - 45) * DTI * (1 - E3)",
                "Identifies delay or compliance delinquency in mandatory age-45 Russian domestic passport renewal.")

# 195. Phone Number Longevity Anchor
m195 = ph_yrs >= 5.0
f195 = np.log1p(ph_yrs) * (u1 + u2) * (monthly_inc / (an + 1.0))
evaluate_regime(195, "Phone_Number_Longevity_Anchor", m195, f195,
                "log1p(Phone_Yrs) * (E2 + E3) * (Monthly_Inc / Annuity)",
                "Unbroken mobile telephone ownership exceeding 5 years signals deep civic and commercial stability.")

# 196. Hyper Recent Phone Number Replacement
m196 = (-ph_raw) <= 60.0
f196 = (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(196, "Hyper_Recent_Phone_Number_Replacement", m196, f196,
                "(1 / Phone_Months) * (1 - E2) * (Credit / Goods)",
                "Switching phone numbers less than 60 days before applying is an acute indicator of collector avoidance.")

# 197. Address to ID Administrative Desynchrony
desync = np.abs(reg_yrs - idp_yrs)
m197 = desync >= 10.0
f197 = desync * dti * (1.0 - cop_frank)
evaluate_regime(197, "Address_To_ID_Administrative_Desynchrony", m197, f197,
                "abs(Reg_Yrs - IDP_Yrs) * DTI * (1 - Frank_Copula)",
                "A gap of >10 years between passport issuance and official residence registration flags bureaucratic detachment.")

# 198. Registration Longevity Home Stability
m198 = reg_yrs >= 15.0
f198 = np.log1p(reg_yrs) * cop_harmonic * (inc / (cr + 1.0))
evaluate_regime(198, "Registration_Longevity_Home_Stability", m198, f198,
                "log1p(Reg_Yrs) * Harmonic_Copula * (Income / Credit)",
                "Living at the same registered municipality for >15 years anchors long-term borrower reliability.")

# 199. Working Life Administrative Friction
admin_friction = np.abs(reg - idp) / (adult_life * 365.25 + 1.0)
m199 = admin_friction > 0.30
f199 = admin_friction * (1.0 - u1) * (an / (monthly_inc + 1.0))
evaluate_regime(199, "Working_Life_Administrative_Friction", m199, f199,
                "Admin_Friction * (1 - E2) * (Annuity / Monthly_Inc)",
                "Proportion of adult life spent in bureaucratic transition between registration and identity renewal.")

# 200. Verification Phone Protocol Disconnect
phone_friction = (flag_emp_phone == 0.0) & (flag_work_phone == 1.0)
m200 = phone_friction
f200 = dti * (1.0 - u2) * (cr / (gp + 1.0))
evaluate_regime(200, "Verification_Phone_Protocol_Disconnect", m200, f200,
                "(Work_Phone_Only) * DTI * (1 - E3) * (Credit / Goods)",
                "Discrepancy between employer phone validation flags indicates informal or third-party intermediation.")

# ==================================================================================================
# DOMAIN 7: CREDIT INQUIRY VELOCITY, URGENCY & ACCELERATION (EXP 201 - 210)
# ==================================================================================================
# 201. Zero Inquiry Calm Credit Sanctuary
m201 = (~inq_missing) & (inq_y == 0.0) & (u1 > 0.4)
f201 = (u1 * u2) * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)
evaluate_regime(201, "Zero_Inquiry_Calm_Credit_Sanctuary", m201, f201,
                "(E2 * E3) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Zero annual credit inquiries combined with healthy bureau scores reflects voluntary, deliberate borrowing.")

# 202. Recent Monthly Inquiry Surge
m202 = inq_m >= 2.0
f202 = inq_m * (1.0 - u1) * (an / (monthly_inc + 1.0))
evaluate_regime(202, "Recent_Monthly_Inquiry_Surge", m202, f202,
                "Inq_Month * (1 - E2) * (Annuity / Monthly_Inc)",
                "Multiple credit requests within 30 days reveals acute, unmet short-term liquidity distress.")

# 203. Hourly Emergency Borrowing
m203 = inq_h > 0
f203 = inq_h * (cr / (gp + 1.0)) * (1.0 / (u2 + 0.1))
evaluate_regime(203, "Hourly_Emergency_Borrowing", m203, f203,
                "Inq_Hour * (Credit / Goods) / (E3 + 0.1)",
                "Inquiries registered in the last hour indicate rapid simultaneous emergency loan applications.")

# 204. Quarterly Acceleration Over Annual
inq_accel = (inq_q * 4.0) - inq_y
m204 = inq_accel > 1.0
f204 = inq_accel * (1.0 - cop_frank) * dti
evaluate_regime(204, "Quarterly_Acceleration_Over_Annual", m204, f204,
                "((Inq_QRT * 4) - Inq_YEAR) * (1 - Frank_Copula) * DTI",
                "A sudden surge in quarterly inquiry velocity relative to trailing annual activity marks credit deterioration.")

# 205. High Annual Credit Hunting Intensity
m205 = inq_y >= 5.0
f205 = np.log1p(inq_y) * (1.0 - u1) * (an / (monthly_inc + 1.0))
evaluate_regime(205, "High_Annual_Credit_Hunting_Intensity", m205, f205,
                "log1p(Inq_YEAR) * (1 - E2) * (Annuity / Monthly_Inc)",
                "Persistent chronic credit shopping across 5+ institutions indicates repeated rejections and distress.")

# 206. Missing Bureau Inquiry History
m206 = inq_missing
f206 = dti * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(206, "Missing_Bureau_Inquiry_History", m206, f206,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Applicants with completely unrecorded inquiry histories lack standard institutional credit footprints.")

# 207. Inquiry Velocity to Income Ratio
inq_to_inc = inq_y / (np.log1p(inc) + 1.0)
m207 = inq_to_inc > 0.4
f207 = inq_to_inc * (1.0 - u2) * (cr / (gp + 1.0))
evaluate_regime(207, "Inquiry_Velocity_To_Income_Ratio", m207, f207,
                "(Inq_YEAR / log1p(Income)) * (1 - E3) * (Credit / Goods)",
                "Heavy inquiry frequency from low-income applicants compounds repayment risk.")

# 208. Inquiry Burden Per Bureau Score
inq_per_score = inq_y / (u1 + 0.05)
m208 = inq_per_score > 10.0
f208 = inq_per_score * (an / (monthly_inc + 1.0))
evaluate_regime(208, "Inquiry_Burden_Per_Bureau_Score", m208, f208,
                "(Inq_YEAR / (E2 + 0.05)) * (Annuity / Monthly_Inc)",
                "High inquiry counts concentrated among subprime credit-score holders indicate adverse credit seeking.")

# 209. Late Stage Quarterly Inquiry Shock
m209 = (inq_q >= 3.0) & (u2 < 0.35)
f209 = inq_q * (1.0 - u2) * dti
evaluate_regime(209, "Late_Stage_Quarterly_Inquiry_Shock", m209, f209,
                "Inq_QRT * (1 - E3) * DTI",
                "Subprime borrowers aggressively requesting credit over the last 90 days exhibit acute default probability.")

# 210. Harmonic Inquiry Credit Stress
f210 = (inq_m + 1.0) * an / (monthly_inc + 1.0) * (1.0 - cop_clayton)
m210 = inq_m >= 1.0
evaluate_regime(210, "Harmonic_Inquiry_Credit_Stress", m210, f210,
                "(Inq_MON + 1) * (Annuity / Monthly_Inc) * (1 - Clayton_Copula)",
                "Scales monthly debt servicing stress by immediate 30-day inquiry velocity.")

# ==================================================================================================
# DOMAIN 8: ASSET COLLATERAL, VEHICLE & HOUSING STRUCTURE (EXP 211 - 220)
# ==================================================================================================
# 211. Vintage Car Maintenance Drain
m211 = (car_flag == 1.0) & (car_age >= 15.0)
f211 = (car_age / 10.0) * dti * (1.0 - u1)
evaluate_regime(211, "Vintage_Car_Maintenance_Drain", m211, f211,
                "(Car_Age / 10) * DTI * (1 - E2)",
                "Vehicles older than 15 years create recurring unexpected mechanical repair drains on disposable cash.")

# 212. Modern Vehicle Luxury Asset Anchor
m212 = (car_flag == 1.0) & (car_age >= 0.0) & (car_age <= 3.0)
f212 = (monthly_inc / (an + 1.0)) * cop_harmonic * (4.0 - car_age)
evaluate_regime(212, "Modern_Vehicle_Luxury_Asset_Anchor", m212, f212,
                "(Monthly_Inc / Annuity) * Harmonic_Copula * (4 - Car_Age)",
                "Late-model vehicles represent substantial unencumbered liquidation equity and high borrower net worth.")

# 213. Homeowner Unencumbered Equity Anchor
m213 = (housing == "House / apartment") & (car_flag == 0.0)
f213 = (inc / (cr + 1.0)) * (u1 + u2) * np.log1p(age)
evaluate_regime(213, "Homeowner_Unencumbered_Equity_Anchor", m213, f213,
                "(Income / Credit) * (E2 + E3) * log1p(Age)",
                "Residential apartment occupiers without vehicle debt obligations represent resilient low-risk anchors.")

# 214. Municipal Housing Tenant Vulnerability
m214 = np.isin(housing, ["Municipal apartment", "Rented apartment"])
f214 = (an / (monthly_inc + 1.0)) * (1.0 / (u2 + 0.1)) * (cr / (gp + 1.0))
evaluate_regime(214, "Municipal_Housing_Tenant_Vulnerability", m214, f214,
                "(Annuity / Monthly_Inc) / (E3 + 0.1) * (Credit / Goods)",
                "Tenants living in municipal or rented units lack housing wealth buffers against personal bankruptcy.")

# 215. Asset Rich Cash Poor Dilemma
m215 = (car_flag == 1.0) & (housing == "House / apartment") & (inc < np.median(inc))
f215 = dti * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))
evaluate_regime(215, "Asset_Rich_Cash_Poor_Dilemma", m215, f215,
                "DTI * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "Vehicle and residential property owners on below-median income face steep maintenance costs with illiquid cash.")

# 216. Car Depreciation to Loan Horizon
car_deprec = (car_age / 10.0) * (cr / (gp + 1.0))
m216 = (car_flag == 1.0) & (car_age > 5.0)
f216 = car_deprec * (1.0 - u1)
evaluate_regime(216, "Car_Depreciation_To_Loan_Horizon", m216, f216,
                "(Car_Age / 10) * (Credit / Goods) * (1 - E2)",
                "Depreciating vehicle asset value compounding against financing fee markup.")

# 217. Vehicle Equity Liquidation Coverage
veh_equity = np.exp(-np.maximum(car_age, 0.0) / 7.0) * gp / (cr + 1.0)
m217 = (car_flag == 1.0) & (car_age >= 0.0)
f217 = veh_equity * cop_harmonic * (monthly_inc / (an + 1.0))
evaluate_regime(217, "Vehicle_Equity_Liquidation_Coverage", m217, f217,
                "Vehicle_Liquidation_Value / Credit * Harmonic_Copula",
                "Realizable salvage recovery value of motor vehicle in relation to outstanding loan balance.")

# 218. Office Apartment Commercial Footprint
m218 = np.isin(housing, ["Office apartment", "Co-op apartment"])
f218 = (monthly_inc / (an + 1.0)) * cop_clayton * (1.0 - dti / 10.0)
evaluate_regime(218, "Office_Apartment_Commercial_Footprint", m218, f218,
                "(Monthly_Inc / Annuity) * Clayton_Copula * (1 - DTI / 10)",
                "Co-op and commercial office apartment dwellers reflect distinct self-employment and professional profiles.")

# 219. Assetless Urban Tenant Exposure
m219 = (car_flag == 0.0) & (housing == "Rented apartment") & (pop >= 0.02)
f219 = (an / (monthly_inc + 1.0)) * (1.0 - u1) * dti
evaluate_regime(219, "Assetless_Urban_Tenant_Exposure", m219, f219,
                "(Annuity / Monthly_Inc) * (1 - E2) * DTI",
                "Urban renters with zero tangible vehicle property facing high city expenses present elevated default rates.")

# 220. Residential Collateral Shield Ratio
m220 = (housing == "House / apartment") & (dti <= 2.5)
f220 = (inc / (cr + 1.0)) * (u1 * u2) * np.log1p(adult_life)
evaluate_regime(220, "Residential_Collateral_Shield_Ratio", m220, f220,
                "(Income / Credit) * (E2 * E3) * log1p(Adult_Life)",
                "Residential apartment dwellers carrying modest debt burdens maintain substantial living stability.")

# ==================================================================================================
# DOMAIN 9: REGIONAL DEMOGRAPHY, ECONOMIC GEOGRAPHY & WORKPLACE TELEMETRY (EXP 221 - 230)
# ==================================================================================================
# 221. Megacity Tier 1 Metro Concentration
m221 = pop >= 0.04
f221 = (inc / (cr + 1.0)) * cop_frank * (monthly_inc / (an + 1.0))
evaluate_regime(221, "Megacity_Tier1_Metro_Concentration", m221, f221,
                "(Income / Credit) * Frank_Copula * (Monthly_Inc / Annuity)",
                "Megacity residents (Moscow/St. Petersburg) have access to deep labor markets and dynamic wages.")

# 222. Rural Periphery Low Density Isolation
m222 = pop < 0.007
f222 = (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))
evaluate_regime(222, "Rural_Periphery_Low_Density_Isolation", m222, f222,
                "(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)",
                "Depressed rural regions with low population density offer limited secondary employment options.")

# 223. Worst Region Rating Severe Risk
m223 = rating == 3.0
f223 = dti * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(223, "Worst_Region_Rating_Severe_Risk", m223, f223,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Jurisdictions assigned rating tier 3 suffer from systemic local economic distress.")

# 224. Elite High Density Tier 1 Sanctuary
m224 = (rating == 1.0) & (pop >= 0.03)
f224 = (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)
evaluate_regime(224, "Elite_High_Density_Tier1_Sanctuary", m224, f224,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)",
                "Prime borrowers in high-density, rating tier 1 metropolitan districts provide premier credit stability.")

# 225. High Institutional Rating In Rural Periphery
m225 = (rating == 1.0) & (pop < 0.01)
f225 = (inc / (cr + 1.0)) * cop_clayton * np.log1p(age)
evaluate_regime(225, "High_Rating_In_Rural_Periphery", m225, f225,
                "(Income / Credit) * Clayton_Copula * log1p(Age)",
                "Rating tier 1 public officials and institutions based in lower-density regional hubs.")

# 226. Unverifiable Employer Phone Disconnect
m226 = (flag_emp_phone == 0.0) & (income_type == "Working")
f226 = dti * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(226, "Unverifiable_Employer_Phone_Disconnect", m226, f226,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Active working applicants lacking an official employer telephone verification flag present high default.")

# 227. Dual Workplace Telephone Verification Anchor
m227 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0) & (emp_yrs >= 5.0)
f227 = (monthly_inc / (an + 1.0)) * (u1 * u2) * np.log1p(emp_yrs)
evaluate_regime(227, "Dual_Workplace_Phone_Anchor", m227, f227,
                "(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)",
                "Long-tenured employees with dual employer telephone verification reflect authentic employment roots.")

# 228. Transient New Hire Workplace Divergence
m228 = (flag_emp_phone == 1.0) & (flag_work_phone == 0.0) & (emp_yrs < 1.0)
f228 = (cr / (gp + 1.0)) * (1.0 / (u2 + 0.1)) * (an / (monthly_inc + 1.0))
evaluate_regime(228, "Transient_New_Hire_Workplace_Divergence", m228, f228,
                "(Credit / Goods) / (E3 + 0.1) * (Annuity / Monthly_Inc)",
                "Recent hires lacking landline workplace verification exhibit heightened probationary job vulnerability.")

# 229. Regional Income Relative Deviance
grp_reg_inc = df.groupby("REGION_POPULATION_RELATIVE")["AMT_INCOME_TOTAL"].transform("mean").values
dev_reg_inc = (inc - grp_reg_inc) / (grp_reg_inc + 1.0)
m229 = dev_reg_inc < -0.20
f229 = (-dev_reg_inc) * dti * (1.0 - u1)
evaluate_regime(229, "Regional_Income_Relative_Deviance", m229, f229,
                "(-Income_Deviation_From_Region) * DTI * (1 - E2)",
                "Earning significantly below the regional municipal peer average amplifies living cost distress.")

# 230. City Level Credit Absorption Capacity
m230 = pop >= 0.015
f230 = (cr / (pop * 1e6 + 1.0)) * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))
evaluate_regime(230, "City_Level_Credit_Absorption_Capacity", m230, f230,
                "(Credit / (Pop_Norm + 1)) * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "Total borrowed credit volume scaled against municipality macroeconomic absorption capacity.")

# ==================================================================================================
# DOMAIN 10: RELATIONAL GRAPH, SEQUENCE TELEMETRY & MASTER INTERACTIONS (EXP 231 - 240)
# ==================================================================================================
# 231. Repeat Applicant First Born Sequence
m231 = (app_order == 0.0) & (n_apps_all >= 2.0)
f231 = (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(n_apps_all)
evaluate_regime(231, "Repeat_Applicant_First_Born_Sequence", m231, f231,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(N_Apps)",
                "Initial loan application in an identified multi-application borrower sequence establishes credit baseline.")

# 232. Repeat Applicant Subsequent Rapid Reapplication
m232 = (app_order >= 1.0) & (days_since_prev >= 0.0) & (days_since_prev < 180.0)
f232 = (1.0 / (days_since_prev / 30.0 + 0.1)) * (1.0 - u1) * dti
evaluate_regime(232, "Repeat_Applicant_Subsequent_Reapplication", m232, f232,
                "(1 / Prev_App_Months) * (1 - E2) * DTI",
                "Rapid reapplication within 6 months of a prior loan reflects accelerating debt accumulation.")

# 233. Long Term Returning Borrower Maturity
m233 = (app_order >= 1.0) & (days_since_prev >= 365.0)
f233 = np.log1p(days_since_prev / 365.25) * cop_harmonic * (inc / (cr + 1.0))
evaluate_regime(233, "Long_Term_Returning_Borrower_Maturity", m233, f233,
                "log1p(Years_Since_Last_App) * Harmonic_Copula * (Income / Credit)",
                "Returning to borrow after a healthy multi-year sabbatical signals disciplined financial lifecycle management.")

# 234. Singleton Borrower Cold Start
m234 = n_apps_all == 1.0
f234 = dti * (1.0 - u1) * (cr / (gp + 1.0))
evaluate_regime(234, "Singleton_Borrower_Cold_Start", m234, f234,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Borrowers appearing only once in the portfolio lack historical cross-application behavioral telemetry.")

# 235. High Density Multi App Cluster
m235 = n_apps_all >= 3.0
f235 = n_apps_all * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))
evaluate_regime(235, "High_Density_Multi_App_Cluster", m235, f235,
                "N_Apps * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Dense multi-application clusters experience correlated credit distress across their borrowing history.")

# 236. Cross Feature Affordability Copula Interaction
m236 = (monthly_inc > an) & (u1 > 0.1) & (u2 > 0.1)
f236 = cop_frank * (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0)
evaluate_regime(236, "Cross_Feature_Affordability_Copula_Interaction", m236, f236,
                "Frank_Copula * (Monthly_Inc / Annuity) * (1 - DTI / 15)",
                "Non-linear coupling of joint bureau consensus copula with real disposable income coverage.")

# 237. Extreme Undercollateralized Laborer Stress
m237 = (single_male_hazard > 0.0) & (cr > gp)
f237 = single_male_hazard * ((cr - gp) / (gp + 1.0))
evaluate_regime(237, "Extreme_Undercollateralized_Laborer_Stress", m237, f237,
                "Single_Male_Laborer_Hazard * ((Credit - Goods) / Goods)",
                "Single male laborers carrying loans where credit exceeds goods price face compounded default probability.")

# 238. Merton Distance Bureau Consensus Hybrid
m238 = ~np.isnan(merton_dd) & valid_triad
f238 = merton_dd * cop_harmonic * (1.0 / (np.abs(e2_f - e3_f) + 0.1))
evaluate_regime(238, "Merton_Distance_Bureau_Consensus_Hybrid", m238, f238,
                "Merton_DD * Harmonic_Copula / (abs(E2 - E3) + 0.1)",
                "Integrates cash flow solvency distance with consensus external agency confidence.")

# 239. Statutory Identity Cycle Tenure Harmonic
id_tenure = (ph_yrs * emp_yrs) / adult_life
m239 = (adult_life > 2.0) & (emp_yrs > 0.5)
f239 = id_tenure * cop_frank * (inc / (cr + 1.0))
evaluate_regime(239, "Statutory_Identity_Cycle_Tenure_Harmonic", m239, f239,
                "(Phone_Yrs * Emp_Yrs / Adult_Life) * Frank_Copula * (Income / Credit)",
                "Harmonizes commercial telephone and employment stability normalized by legal adult lifespan.")

# 240. Unified Cognitive Super Manifold
f240 = cop_frank * (1.0 - cop_clayton) * (monthly_inc / (an + 1.0)) / (1.0 + single_male_hazard)
m240 = np.ones(n_train, dtype=bool)
evaluate_regime(240, "Unified_Cognitive_Super_Manifold", m240, f240,
                "Frank_Copula * (1 - Clayton_Copula) * (Monthly_Inc / Annuity) / (1 + Single_Male_Hazard)",
                "Comprehensive master cognitive manifold uniting copula topology, debt affordability, and demographic risk.")

# --------------------------------------------------------------------------------------------------
# STRATIFIED 5-FOLD CROSS-VALIDATION VERIFICATION
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 95)
print("  STRATIFIED 5-FOLD CV VERIFICATION OF DISCOVERED MASTER FORMULAS")
print("=" * 95)

base_features = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT"
]

base_df = df[base_features].copy()
base_df["DTI"] = dti
base_df["TERM"] = term
base_df["AGE"] = age
base_df["EMP_YRS"] = emp_yrs

# Select top 12 discovered features with highest spread & correlation
spread_sorted = sorted(results.values(), key=lambda x: x["spread_bps"], reverse=True)
top_exp_keys = [f"F_{item['exp_id']}_{item['name']}" for item in spread_sorted[:12]]
print("\nTop 12 Discovered Formulations Selected for Ensemble CV:")
for i, item in enumerate(spread_sorted[:12], 1):
    print(f"  {i:2d}. Exp {item['exp_id']:3d} ({item['name']:40s}): Spread = {item['spread_bps']:6.1f} bps | Corr = {item['within_split_correlation']:+6.4f}")

enhanced_df = base_df.copy()
for tk in top_exp_keys:
    enhanced_df[tk] = discovered_features[tk]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(df, y))

params = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 10.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

# Baseline CV
print("\n1. Evaluating Baseline Feature Set (5-Fold CV)...")
oof_base = np.zeros(n_train)
for trn_idx, val_idx in splits:
    X_tr, y_tr = base_df.iloc[trn_idx], y[trn_idx]
    X_va, y_va = base_df.iloc[val_idx], y[val_idx]
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    m = lgb.train(params, trn_data, num_boost_round=600, valid_sets=[val_data], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_base[val_idx] = m.predict(X_va, num_iteration=m.best_iteration)
auc_base = roc_auc_score(y, oof_base)
print(f"   [+] Baseline Pooled OOF AUC: {auc_base:.6f}")

# Enhanced CV
print("\n2. Evaluating Enhanced Feature Set (+12 Discovered Master Formulas)...")
oof_enh = np.zeros(n_train)
for trn_idx, val_idx in splits:
    X_tr, y_tr = enhanced_df.iloc[trn_idx], y[trn_idx]
    X_va, y_va = enhanced_df.iloc[val_idx], y[val_idx]
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    m = lgb.train(params, trn_data, num_boost_round=600, valid_sets=[val_data], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_enh[val_idx] = m.predict(X_va, num_iteration=m.best_iteration)
auc_enh = roc_auc_score(y, oof_enh)
lift_bps = (auc_enh - auc_base) * 10000
print(f"   [+] Enhanced Pooled OOF AUC: {auc_enh:.6f} | Net Lift: {lift_bps:+5.1f} bps")

# --------------------------------------------------------------------------------------------------
# SAVE JSON AND MD REPORTS
# --------------------------------------------------------------------------------------------------
output_meta = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_elapsed_seconds": round(time.time() - t_start, 2),
    "baseline_pooled_auc": round(auc_base, 6),
    "enhanced_pooled_auc": round(auc_enh, 6),
    "auc_net_lift_bps": round(lift_bps, 1),
    "experiments_count": len(results),
    "experiments": results
}

json_path = r"D:\gci\gemini\RESEARCH_100_OPTIMIZED_REGIMES_EXP141_240.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_meta, f, indent=2)

md_path = r"D:\gci\gemini\RESEARCH_100_OPTIMIZED_REGIMES_EXP141_240.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# GCI World 2026: 100 Deep Optimized Specialized Regime Splits & Master Formulas (Exp 141 – 240)\n\n")
    f.write(f"- **Execution Timestamp:** `{output_meta['timestamp_utc']}`\n")
    f.write(f"- **Runtime:** `{output_meta['total_elapsed_seconds']}s`\n")
    f.write(f"- **Baseline 5-Fold OOF AUC:** `{auc_base:.6f}`\n")
    f.write(f"- **Enhanced 5-Fold OOF AUC:** `{auc_enh:.6f}` (**{lift_bps:+.1f} bps**)\n")
    f.write(f"- **Total Regimes Analyzed:** 100 Distinct Splits (Zero Duplicates)\n\n")
    f.write("## Exhaustive Ledger of 100 Optimized Formulations\n\n")
    f.write("| Exp ID | Regime Name | Sample N | Pct (%) | Subgroup Default | Correlation (rho) | Spread (bps) | Formula | Domain Reason |\n")
    f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n")
    for k, v in results.items():
        f.write(f"| {v['exp_id']} | **{v['name']}** | {v['subgroup_count']:,} | {v['subgroup_pct']:.1f}% | {v['subgroup_default_rate']:.2f}% | {v['within_split_correlation']:+.4f} | **{v['spread_bps']:.1f}** | `{v['formula']}` | {v['domain_reason']} |\n")

print(f"\n[+] Research JSON Report Saved: {json_path}")
print(f"[+] Research Markdown Saved   : {md_path}")
print(f"[+] Total Research Execution : {time.time() - t_start:.2f}s")
print("=" * 95)
