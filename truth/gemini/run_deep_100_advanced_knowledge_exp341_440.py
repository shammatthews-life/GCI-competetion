"""
====================================================================================================
  GCI WORLD 2026: 100 DEEP RESEARCH DOMAIN & ACTUARIAL KNOWLEDGE FORMULATIONS (EXP 341 - 440)
  + MATHEMATICAL FEATURE IMPORTANCE PROFILER & SPLIT GAIN RECONSTRUCTION
====================================================================================================
  - Pure In-Memory, Clean-Room Compliant (Strictly inpuy/train.csv & inpuy/test.csv)
  - 10 Deep Mathematical Domains (Black-Cox, Bühlmann-Straub, EVT Gumbel/Clayton,
    Tariff Harmonics, Subsistence Free Cash Flow, Occupational Matrix, Asset Decay,
    Inquiry Acceleration, Spatial Divergence, Relational Trajectories)
  - Exhaustive 100-Regime Evaluation + Stratified 5-Fold Cross Validation Verification
====================================================================================================
"""

import sys
import os
import time
import json
import warnings
import numpy as np
from numpy import log1p
import pandas as pd
from scipy.stats import spearmanr, norm, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

t_start = time.time()
print("=" * 115)
print("  GCI WORLD 2026: 100 DEEP RESEARCH DOMAIN & ACTUARIAL FORMULATIONS (EXP 341 - 440)")
print("  + FEATURE IMPORTANCE MECHANICS (SPLIT GAIN & INFORMATION VALUE GENERATION)")
print("=" * 115)

train_path = r"D:\gci\inpuy\train.csv"
test_path = r"D:\gci\inpuy\test.csv"
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
y = df["TARGET"].values
n_train = len(df)
n_test = len(df_test)
prior_pd = float(y.mean())
print(f"[+] Loaded {n_train:,} training records (Unconditional Baseline Default = {prior_pd*100:.2f}%) in {time.time() - t_start:.2f}s.")

# Construct Invariant Entity Key & Relational Graph for Sequence Features
df_all = pd.concat([df.drop(columns=["TARGET"]), df_test], ignore_index=True)
b_all = df_all["DAYS_BIRTH"].values.astype(float)
r_all = df_all["DAYS_REGISTRATION"].values.astype(float)
id_all = df_all["DAYS_ID_PUBLISH"].values.astype(float)
df_all["entity_key"] = (
    (b_all - r_all).round(0).astype(int).astype(str) + "|" +
    (b_all - id_all).round(0).astype(int).astype(str) + "|" +
    df_all["CODE_GENDER"].astype(str) + "|" +
    df_all["NAME_EDUCATION_TYPE"].astype(str)
)
cnt_ent = df_all["entity_key"].value_counts()
n_apps_full = df_all["entity_key"].map(cnt_ent).values.astype(float)
df_all["row_idx"] = np.arange(len(df_all))
df_sorted = df_all.sort_values(["entity_key", "DAYS_REGISTRATION"])
df_sorted["APP_ORDER"] = df_sorted.groupby("entity_key").cumcount()
df_sorted["DAYS_SINCE_PREV_APP"] = df_sorted.groupby("entity_key")["DAYS_REGISTRATION"].diff()
df_restored_full = df_sorted.sort_values("row_idx")

df_restored = df_restored_full.iloc[:n_train].copy()
n_apps_all = n_apps_full[:n_train]

# --------------------------------------------------------------------------------------------------
# BASE VECTOR EXTRACTIONS
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

reg_yrs = -df["DAYS_REGISTRATION"].values.astype(float) / 365.25
idp_yrs = -df["DAYS_ID_PUBLISH"].values.astype(float) / 365.25
ph_yrs = -df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values.astype(float) / 365.25

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

u0 = np.clip(e1_f, 1e-4, 1.0 - 1e-4)
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

# Core financial metrics
monthly_inc = inc / 12.0
dti = np.clip(cr / (inc + 1.0), 0.0, 30.0)
term = np.clip(cr / (an + 1.0), 1.0, 100.0)
subsistence = 12000.0 * ch + 15000.0
surplus = inc - 12.0 * an
free_cash_flow = np.maximum(monthly_inc - subsistence - an, -50000.0)

# --------------------------------------------------------------------------------------------------
# PART 1: EXACT FEATURE IMPORTANCE (SPLIT GAIN & COUNT) PROFILING
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 115)
print("  PART 1: FEATURE IMPORTANCE PROFILING (SPLIT GAIN VS SPLIT COUNT)")
print("=" * 115)

base_feat_dict = {
    "EXT_SOURCE_1": e1_f,
    "EXT_SOURCE_2": e2_f,
    "EXT_SOURCE_3": e3_f,
    "EXT_PRODUCT": u0 * u1 * u2,
    "EXT_HARMONIC": 3.0 / ((1.0 / u0) + (1.0 / u1) + (1.0 / u2)),
    "EXT_MIN": np.minimum(np.minimum(u0, u1), u2),
    "AMT_CREDIT": cr,
    "AMT_ANNUITY": an,
    "AMT_INCOME_TOTAL": inc,
    "AMT_GOODS_PRICE": gp,
    "DAYS_BIRTH": birth,
    "DAYS_EMPLOYED": emp,
    "DAYS_REGISTRATION": df["DAYS_REGISTRATION"].values.astype(float),
    "DAYS_ID_PUBLISH": df["DAYS_ID_PUBLISH"].values.astype(float),
    "DAYS_LAST_PHONE_CHANGE": df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values.astype(float),
    "REGION_POPULATION_RELATIVE": pop,
    "REGION_RATING_CLIENT": rating,
    "CREDIT_TO_ANNUITY_TERM": term,
    "CREDIT_TO_GOODS_RATIO": cr / (gp + 1.0),
    "DTI_RATIO": dti,
    "ANNUITY_TO_MONTHLY_INC": an / (monthly_inc + 1.0),
    "Q0_FEE": np.where((contract == "Cash loans") & ((an - 129.5) / (cr + 1e-5) > 0), (an - 129.5) / (cr + 1e-5), an / (cr + 1e-5)),
    "NEGATIVE_DOWNPAYMENT": np.maximum(cr - gp, 0.0) / (gp + 1.0),
    "OWN_CAR_AGE": car_age,
    "N_APPS_ALL": n_apps_all,
    "APP_ORDER": df_restored["APP_ORDER"].values.astype(float),
    "INQ_BURDEN_YEAR": inq_y
}

X_base = pd.DataFrame(base_feat_dict)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gain_collector = np.zeros(X_base.shape[1])
split_collector = np.zeros(X_base.shape[1])

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_base, y)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_base.iloc[trn_idx], y[trn_idx])
    gain_collector += m.booster_.feature_importance(importance_type="gain") / 5.0
    split_collector += m.booster_.feature_importance(importance_type="split") / 5.0

gain_df = pd.DataFrame({
    "Feature": X_base.columns,
    "Split_Gain": gain_collector,
    "Split_Count": split_collector
}).sort_values("Split_Gain", ascending=False).reset_index(drop=True)

gain_df["Gain_Pct"] = (gain_df["Split_Gain"] / gain_df["Split_Gain"].sum()) * 100

print(f"{'Rank':<5} | {'Feature Name':<28} | {'Total Gain':<12} | {'Gain Pct (%)':<12} | {'Split Count':<12}")
print("-" * 75)
for i, r in gain_df.head(15).iterrows():
    print(f"{i+1:<5} | {r['Feature']:<28} | {r['Split_Gain']:<12.1f} | {r['Gain_Pct']:<12.2f} | {r['Split_Count']:<12.0f}")

# --------------------------------------------------------------------------------------------------
# PART 2: 100 DEEP RESEARCH DOMAIN & ACTUARIAL FORMULATIONS (EXP 341 - 440)
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 115)
print("  PART 2: EXECUTING 100 DEEP ACTUARIAL & QUANTITATIVE MANIFOLDS (EXP 341 - 440)")
print("=" * 115)

# Quantitative Mathematics Infrastructure
# Copulas
cop_clayton_2 = np.maximum(u1**(-2.0) + u2**(-2.0) - 1.0, 1e-6)**(-0.5)
cop_gumbel_2 = np.exp(-((-np.log(u1))**2.0 + (-np.log(u2))**2.0)**0.5)
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))
num_f4 = (np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)
cop_frank_4 = -0.25 * np.log(1.0 + num_f4 / (np.exp(-4.0) - 1.0))

# Black-Cox Absorbing Barrier Setup
asset_proxy = inc * 3.5 + cr * 0.5
barrier_subsistence = (15000.0 * fam) * 3.0
black_cox_cushion = np.maximum(asset_proxy - barrier_subsistence, 1e-3) / (cr + 1.0)

# Bühlmann-Straub Credibility Defaults on Occupation
occ_s = pd.Series(occ)
occ_cnt = occ_s.map(occ_s.value_counts()).values
occ_mean = pd.Series(y).groupby(occ_s).transform("mean").values
K_cred = 250.0  # Bühlmann-Straub credibility parameter
Z_cred = occ_cnt / (occ_cnt + K_cred)
buhlmann_occ_pd = Z_cred * occ_mean + (1.0 - Z_cred) * prior_pd

# Tariff Lattices & Tenor Harmonics
tenor_months = np.round(cr / (an + 1e-5)).astype(int)
is_standard_tenor = np.isin(tenor_months, [12, 18, 20, 24, 30, 36, 48, 60])
lattice_step_0066 = np.abs((cr / (gp + 1e-5) - 1.0) - 0.0066 * np.round((cr / (gp + 1e-5) - 1.0) / 0.0066))
q0_fee = np.where((contract == "Cash loans") & ((an - 129.5) / (cr + 1e-5) > 0), (an - 129.5) / (cr + 1e-5), an / (cr + 1e-5))

# Demographics
single_male_hazard = ((gender == "M") & np.isin(family, ["Single / not married", "Separated", "Widow"])).astype(float) * dti * (1.0 - u1)

results_100 = {}
discovered_features = {}

def evaluate_regime(exp_id, name, mask, formula_arr, formula_str, math_family, domain_reason):
    mask = mask.astype(bool)
    count = int(mask.sum())
    pct = round(count / n_train * 100, 2)
    def_rate = round(float(y[mask].mean() * 100), 2) if count > 0 else 0.0
    
    sub_y = y[mask]
    sub_feat = formula_arr[mask]
    valid = ~np.isnan(sub_feat) & ~np.isinf(sub_feat)
    
    if valid.sum() > 25 and len(np.unique(sub_feat[valid])) > 5:
        corr_val, _ = spearmanr(sub_feat[valid], sub_y[valid])
        corr = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0
        
        try:
            auc_raw = roc_auc_score(sub_y[valid], sub_feat[valid])
            auc_score = round(float(max(auc_raw, 1.0 - auc_raw)), 4)
        except Exception:
            auc_score = 0.50
            
        q10 = np.quantile(sub_feat[valid], 0.10)
        q90 = np.quantile(sub_feat[valid], 0.90)
        d_low = float(sub_y[valid][sub_feat[valid] <= q10].mean() * 100) if (sub_feat[valid] <= q10).sum() > 0 else def_rate
        d_high = float(sub_y[valid][sub_feat[valid] >= q90].mean() * 100) if (sub_feat[valid] >= q90).sum() > 0 else def_rate
        spread_bps = round(abs(d_high - d_low) * 100, 1)
    else:
        corr = 0.0
        auc_score = 0.50
        spread_bps = 0.0
        
    results_100[exp_id] = {
        "exp_id": exp_id,
        "name": name,
        "count": count,
        "pct": pct,
        "def_rate": def_rate,
        "within_auc": auc_score,
        "spearman_corr": corr,
        "spread_bps": spread_bps,
        "math_family": math_family,
        "formula": formula_str,
        "domain_reason": domain_reason
    }
    discovered_features[f"F_{exp_id}_{name}"] = formula_arr
    print(f"[{exp_id}] {name:<35} | N={count:>7,} ({pct:>4.1f}%) | Def={def_rate:>5.2f}% | AUC={auc_score:.4f} | Corr={corr:+.4f} | Spread={spread_bps:>6.1f} bps | {math_family:<18}")

# ==================================================================================================
# DOMAIN 1: BLACK-COX FIRST-PASSAGE ABSORBING BARRIER MODELS (EXP 341 - 350)
# ==================================================================================================
# 341. Black-Cox Subsistence Barrier Proximity
f341 = (1.0 / (black_cox_cushion + 0.01)) * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m341 = black_cox_cushion < 1.0
evaluate_regime(341, "BlackCox_Subsistence_Barrier_Proximity", m341, f341,
                "(1 / Barrier_Cushion) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Black-Cox Barrier", "First-passage default occurs when assets deplete below essential living expenses.")

# 342. Black-Cox High Volatility Barrier Touch
f342 = ((cr - gp) / (gp + 1.0)) / (black_cox_cushion + 0.05) * (1.0 - cop_frank_4)
m342 = cr > gp
evaluate_regime(342, "BlackCox_Negative_Equity_Touch", m342, f342,
                "((Credit - Goods) / Goods) / Barrier_Cushion * (1 - Frank_Copula)",
                "Black-Cox Barrier", "Uncollateralized negative downpayments accelerate asset decay toward the default barrier.")

# 343. Black-Cox Multi-Child Barrier Compression
f343 = (ch / fam) / (black_cox_cushion + 0.05) * (1.0 - u2)
m343 = ch >= 2.0
evaluate_regime(343, "BlackCox_Child_Barrier_Compression", m343, f343,
                "(Children / Family) / Barrier_Cushion * (1 - E3)",
                "Black-Cox Barrier", "Dependent family members elevate the subsistence barrier, narrowing safety buffers.")

# 344. Black-Cox Asset Barrier Solvency Anchor
f344 = black_cox_cushion * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m344 = black_cox_cushion >= 2.0
evaluate_regime(344, "BlackCox_Solvency_Anchor", m344, f344,
                "Barrier_Cushion * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Black-Cox Barrier", "Extensive distance from the absorbing barrier provides structural resilience.")

# 345. Black-Cox Tenured Senior Buffer
f345 = black_cox_cushion * log1p(emp_yrs) * cop_frank_4
m345 = emp_yrs >= 5.0
evaluate_regime(345, "BlackCox_Tenured_Senior_Buffer", m345, f345,
                "Barrier_Cushion * log1p(Emp_Yrs) * Frank_Copula",
                "Black-Cox Barrier", "Sticky employment seniority prevents downward asset drift toward insolvency.")

# 346. Black-Cox Early Career Vulnerability
f346 = (dti / (black_cox_cushion + 0.1)) * (1.0 - u1)
m346 = (age < 30.0) & (black_cox_cushion < 1.5)
evaluate_regime(346, "BlackCox_Early_Career_Vulnerability", m346, f346,
                "(DTI / Barrier_Cushion) * (1 - E2)",
                "Black-Cox Barrier", "Young borrowers lacking asset accumulation face immediate first-passage risk.")

# 347. Black-Cox Short Loan Rapid Amortization Barrier
f347 = (an / (monthly_inc + 1.0)) / (black_cox_cushion + 0.1) * (1.0 - u2)
m347 = (term < 18.0) & (black_cox_cushion < 1.0)
evaluate_regime(347, "BlackCox_Rapid_Amortization_Barrier", m347, f347,
                "(Annuity / Monthly_Inc) / Barrier_Cushion * (1 - E3)",
                "Black-Cox Barrier", "Steep monthly payments rapidly deplete available liquid buffers.")

# 348. Black-Cox Laborer Barrier Hazard
f348 = (single_male_hazard / (black_cox_cushion + 0.05))
m348 = single_male_hazard > 0.0
evaluate_regime(348, "BlackCox_Laborer_Barrier_Hazard", m348, f348,
                "Single_Male_Laborer_Hazard / Barrier_Cushion",
                "Black-Cox Barrier", "Precarious manual employment couples with razor-thin asset buffers.")

# 349. Black-Cox Sovereign Pensioner Shield
f349 = black_cox_cushion * (u1 + u2) * np.log1p(age)
m349 = is_retired
evaluate_regime(349, "BlackCox_Pensioner_Shield", m349, f349,
                "Barrier_Cushion * (E2 + E3) * log1p(Age)",
                "Black-Cox Barrier", "Guaranteed lifetime state pensions guarantee income remains strictly above the absorbing barrier.")

# 350. Black-Cox Municipal Metro Insulation
f350 = black_cox_cushion * (pop / 0.02) * (monthly_inc / (an + 1.0)) * cop_harmonic
m350 = pop >= 0.02
evaluate_regime(350, "BlackCox_Metro_Insulation", m350, f350,
                "Barrier_Cushion * (Pop / 0.02) * (Monthly_Inc / Annuity) * Harmonic_Copula",
                "Black-Cox Barrier", "Metropolitan centers provide immediate alternative liquidity and asset monetization.")

# ==================================================================================================
# DOMAIN 2: BÜHLMANN-STRAUB ACTUARIAL CREDIBILITY WEIGHTINGS (EXP 351 - 360)
# ==================================================================================================
# 351. Bühlmann Credibility Occupational Risk Anchor
f351 = buhlmann_occ_pd * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m351 = buhlmann_occ_pd > 0.10
evaluate_regime(351, "Buhlmann_High_Risk_Occupation", m351, f351,
                "Buhlmann_Occ_PD * (Annuity / Monthly_Inc) * (1 - E2)",
                "Actuarial Credibility", "Credibility-weighted occupational default odds scale debt burden.")

# 352. Bühlmann Prime Occupation Shield
f352 = (1.0 - buhlmann_occ_pd) * (monthly_inc / (an + 1.0)) * (u1 * u2)
m352 = buhlmann_occ_pd < 0.06
evaluate_regime(352, "Buhlmann_Prime_Occupation_Shield", m352, f352,
                "(1 - Buhlmann_Occ_PD) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Actuarial Credibility", "Low-risk professions backed by high sample credibility anchor prime credit.")

# 353. Bühlmann Laborer Markup Compounding
f353 = buhlmann_occ_pd * ((cr - gp) / (gp + 1.0)) * dti
m353 = (occ == "Low-skill Laborers") | (occ == "Drivers")
evaluate_regime(353, "Buhlmann_Laborer_Markup_Compounding", m353, f353,
                "Buhlmann_Occ_PD * ((Credit - Goods) / Goods) * DTI",
                "Actuarial Credibility", "High-credibility manual risk combined with excessive credit markups.")

# 354. Bühlmann Small Category Bayesian Shrinkage
f354 = buhlmann_occ_pd * dti * (1.0 - cop_frank_4)
m354 = Z_cred < 0.70
evaluate_regime(354, "Buhlmann_Small_Category_Shrinkage", m354, f354,
                "Buhlmann_Occ_PD * DTI * (1 - Frank_Copula)",
                "Actuarial Credibility", "Small occupational niches shrunk toward population baseline prevent overfitting.")

# 355. Bühlmann High Credibility Stability Anchor
f355 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * Z_cred
m355 = Z_cred >= 0.95
evaluate_regime(355, "Buhlmann_High_Credibility_Anchor", m355, f355,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * Z_cred",
                "Actuarial Credibility", "Large, well-observed employment pools provide rock-solid statistical certainty.")

# 356. Bühlmann High Tech Wage Premium
f356 = (1.0 - buhlmann_occ_pd) * (inc / (cr + 1.0)) * (E2_f := u1)
m356 = occ == "High skill tech staff"
evaluate_regime(356, "Buhlmann_High_Tech_Premium", m356, f356,
                "(1 - Buhlmann_Occ_PD) * (Income / Credit) * E2",
                "Actuarial Credibility", "Exceptional wage growth in tech offsets consumer borrowing.")

# 357. Bühlmann Construction Downside Exposure
f357 = buhlmann_occ_pd * (1.0 - cop_clayton_2) * (an / (monthly_inc + 1.0))
m357 = (org == "Construction") | (occ == "Laborers")
evaluate_regime(357, "Buhlmann_Construction_Downside", m357, f357,
                "Buhlmann_Occ_PD * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Actuarial Credibility", "Cyclical construction industry exposure magnified by actuarial odds.")

# 358. Bühlmann Healthcare Essential Resilience
f358 = (1.0 - buhlmann_occ_pd) * (monthly_inc / (an + 1.0)) * cop_harmonic
m358 = (occ == "Medicine staff") | (org == "Medicine")
evaluate_regime(358, "Buhlmann_Healthcare_Resilience", m358, f358,
                "(1 - Buhlmann_Occ_PD) * (Monthly_Inc / Annuity) * Harmonic_Copula",
                "Actuarial Credibility", "Permanent societal demand for medical personnel guarantees recession immunity.")

# 359. Bühlmann Security Staff Precariousness
f359 = buhlmann_occ_pd * dti * (1.0 - u2)
m359 = (occ == "Security staff")
evaluate_regime(359, "Buhlmann_Security_Staff_Hazard", m359, f359,
                "Buhlmann_Occ_PD * DTI * (1 - E3)",
                "Actuarial Credibility", "Low-wage private security contractors face high default rates.")

# 360. Bühlmann Educational Civil Service Shield
f360 = (1.0 - buhlmann_occ_pd) * (monthly_inc / (an + 1.0)) * (u1 + u2)
m360 = (org == "School") | (org == "Kindergarten") | (org == "University")
evaluate_regime(360, "Buhlmann_Education_Shield", m360, f360,
                "(1 - Buhlmann_Occ_PD) * (Monthly_Inc / Annuity) * (E2 + E3)",
                "Actuarial Credibility", "Public sector educators maintain stable, default-resistant employment.")

# ==================================================================================================
# DOMAIN 3: EXTREME VALUE THEORY (EVT) GUMBEL-CLAYTON ASYMMETRIC MANIFOLDS (EXP 361 - 370)
# ==================================================================================================
# 361. EVT Gumbel Upper Tail Concordance
f361 = cop_gumbel_2 * (monthly_inc / (an + 1.0)) * (inc / (cr + 1.0))
m361 = cop_gumbel_2 > 0.60
evaluate_regime(361, "EVT_Gumbel_Upper_Tail_Concordance", m361, f361,
                "Gumbel_Copula * (Monthly_Inc / Annuity) * (Income / Credit)",
                "Extreme Value Theory", "Gumbel copula captures upper tail concordance across multiple bureau score agencies.")

# 362. EVT Clayton Lower Tail Bankruptcy Cascade
f362 = (1.0 / (cop_clayton_2 + 0.05)) * (an / (monthly_inc + 1.0)) * dti
m362 = cop_clayton_2 < 0.20
evaluate_regime(362, "EVT_Clayton_Lower_Tail_Cascade", m362, f362,
                "(1 / Clayton_Copula) * (Annuity / Monthly_Inc) * DTI",
                "Extreme Value Theory", "Clayton copula isolates severe joint lower-tail collapse across external bureaus.")

# 363. EVT Asymmetric Tail Ratio
f363 = (cop_gumbel_2 / (cop_clayton_2 + 0.05)) * (monthly_inc / (an + 1.0))
m363 = (u1 > 0.2) & (u2 > 0.2)
evaluate_regime(363, "EVT_Asymmetric_Tail_Ratio", m363, f363,
                "(Gumbel_Copula / Clayton_Copula) * (Monthly_Inc / Annuity)",
                "Extreme Value Theory", "Ratio of upper-to-lower tail dependence measures asymmetric credit skewness.")

# 364. EVT Extreme Shock Downpayment Compression
f364 = (1.0 - cop_gumbel_2) * ((cr - gp) / (gp + 1.0)) * dti
m364 = cr > 1.15 * gp
evaluate_regime(364, "EVT_Downpayment_Tail_Compression", m364, f364,
                "(1 - Gumbel_Copula) * ((Credit - Goods) / Goods) * DTI",
                "Extreme Value Theory", "Tail agency collapse compounded by instant negative downpayment equity.")

# 365. EVT Gumbel Prime Borrower Sanctuary
f365 = cop_gumbel_2**2 * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)
m365 = cop_gumbel_2 >= 0.75
evaluate_regime(365, "EVT_Gumbel_Prime_Sanctuary", m365, f365,
                "Gumbel_Copula^2 * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Extreme Value Theory", "Simultaneous extreme upper-tail agency consensus proves unassailable solvency.")

# 366. EVT Clayton Laborer Distress Squeeze
f366 = (single_male_hazard / (cop_clayton_2 + 0.05)) * (an / (monthly_inc + 1.0))
m366 = single_male_hazard > 0.0
evaluate_regime(366, "EVT_Clayton_Laborer_Distress", m366, f366,
                "(Single_Male_Hazard / Clayton_Copula) * (Annuity / Monthly_Inc)",
                "Extreme Value Theory", "Lower-tail bureau failure concentrated in single manual laborers.")

# 367. EVT Multi-Agency Harmonic Discordance
f367 = np.abs(cop_gumbel_2 - cop_clayton_2) * dti * (1.0 - u1)
m367 = np.abs(cop_gumbel_2 - cop_clayton_2) > 0.30
evaluate_regime(367, "EVT_Tail_Asymmetry_Discordance", m367, f367,
                "abs(Gumbel - Clayton) * DTI * (1 - E2)",
                "Extreme Value Theory", "Large divergence between upper and lower tail dependence flags model ambiguity.")

# 368. EVT Gumbel Senior Pensioner Stability
f368 = cop_gumbel_2 * (u1 + u2) * np.log1p(age)
m368 = is_retired & (cop_gumbel_2 > 0.50)
evaluate_regime(368, "EVT_Gumbel_Pensioner_Stability", m368, f368,
                "Gumbel_Copula * (E2 + E3) * log1p(Age)",
                "Extreme Value Theory", "Joint upper tail bureau alignment for retirees confirms low risk.")

# 369. EVT Clayton Low Income Micro-Burden
f369 = (1.0 / (cop_clayton_2 + 0.05)) * (129.50 / (an + 1.0)) * dti
m369 = inc < 100000.0
evaluate_regime(369, "EVT_Clayton_Low_Income_Burden", m369, f369,
                "(1 / Clayton_Copula) * (129.50 / Annuity) * DTI",
                "Extreme Value Theory", "Fixed fee burden on low-earners experiencing joint lower-tail credit degradation.")

# 370. EVT Bounded Gumbel Debt Velocity
f370 = (cop_gumbel_2 / (dti + 0.1)) * (monthly_inc / (an + 1.0))
m370 = dti > 5.0
evaluate_regime(370, "EVT_Gumbel_Debt_Velocity_Buffer", m370, f370,
                "(Gumbel_Copula / DTI) * (Monthly_Inc / Annuity)",
                "Extreme Value Theory", "High bureau tail consensus counteracting heavy debt-to-income multiples.")

# ==================================================================================================
# DOMAIN 4: CONTRACTUAL TARIFF TENOR HARMONICS & STEP RESIDUALS (EXP 371 - 380)
# ==================================================================================================
# 371. Standard Tenor Contractual Stability
f371 = (monthly_inc / (an + 1.0)) * (u1 * u2) * is_standard_tenor.astype(float)
m371 = is_standard_tenor
evaluate_regime(371, "Tariff_Standard_Tenor_Anchor", m371, f371,
                "(Monthly_Inc / Annuity) * (E2 * E3) * Is_Standard_Tenor",
                "Tariff Harmonics", "Standard loan tenors (12, 18, 24, 36, 48 mos) follow predictable institutional repayment schedules.")

# 372. Non-Standard Tenor Friction
f372 = (an / (monthly_inc + 1.0)) * (1.0 - u1) * (~is_standard_tenor).astype(float)
m372 = ~is_standard_tenor
evaluate_regime(372, "Tariff_Non_Standard_Tenor_Friction", m372, f372,
                "(Annuity / Monthly_Inc) * (1 - E2) * (~Is_Standard_Tenor)",
                "Tariff Harmonics", "Irregular contractual durations signal non-standard renegotiated or distressed financing.")

# 373. Integer Step 0066 Micro-Lattice Precision
f373 = (1.0 / (lattice_step_0066 + 0.001)) * (1.0 - u2) * (cr / (gp + 1.0))
m373 = lattice_step_0066 < 0.002
evaluate_regime(373, "Tariff_Step_0066_Precision", m373, f373,
                "(1 / Step_0066_Residual) * (1 - E3) * (Credit / Goods)",
                "Tariff Harmonics", "Exact fit on 0.0066 step grid identifies retail POS loans with fixed rate schedules.")

# 374. Off-Lattice Arbitrage Distortion
f374 = lattice_step_0066 * dti * (1.0 - cop_frank_4)
m374 = lattice_step_0066 >= 0.003
evaluate_regime(374, "Tariff_Off_Lattice_Distortion", m374, f374,
                "Step_0066_Residual * DTI * (1 - Frank_Copula)",
                "Tariff Harmonics", "Deviation from standard tariff pricing grids indicates customized high-interest products.")

# 375. Fixed Fee 129.50 Over-Amortization Elasticity
f375 = (129.50 / (an + 1.0)) * (cr / (monthly_inc * 12.0 + 1.0)) * (1.0 - u1)
m375 = (contract == "Cash loans") & (an < 10000.0)
evaluate_regime(375, "Tariff_Fee_12950_Micro_Elasticity", m375, f375,
                "(129.50 / Annuity) * (Credit / Income) * (1 - E2)",
                "Tariff Harmonics", "Small cash loans where fixed overhead consumes substantial fractions of total payment.")

# 376. Negative Equity Tariff Surcharge
f376 = ((cr - gp) / (gp + 1.0))**2 * (an / (monthly_inc + 1.0)) / (u2 + 0.1)
m376 = cr > 1.20 * gp
evaluate_regime(376, "Tariff_Negative_Equity_Surcharge", m376, f376,
                "((Credit - Goods) / Goods)^2 * (Annuity / Monthly_Inc) / E3",
                "Tariff Harmonics", "Severe negative equity on loans carrying substantial wrapped dealer charges.")

# 377. Clean Par-Financing Prime Anchor
f377 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m377 = np.abs(cr - gp) < 1.0
evaluate_regime(377, "Tariff_Zero_Markup_Par_Anchor", m377, f377,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Tariff Harmonics", "Credit exactly equals goods price (0% promotional retail rate for prime borrowers).")

# 378. Micro-Term High Amortization Strain
f378 = (an / (monthly_inc + 1.0)) * (12.0 / (tenor_months + 1.0)) * (1.0 - u1)
m378 = tenor_months <= 12
evaluate_regime(378, "Tariff_Micro_Term_Strain", m378, f378,
                "(Annuity / Monthly_Inc) * (12 / Tenor) * (1 - E2)",
                "Tariff Harmonics", "Very short loan terms create severe monthly liquidity demands.")

# 379. Long-Horizon Macro Exposure
f379 = (tenor_months / 60.0) * dti * (1.0 - cop_frank_4)
m379 = tenor_months >= 48
evaluate_regime(379, "Tariff_Long_Horizon_Exposure", m379, f379,
                "(Tenor / 60) * DTI * (1 - Frank_Copula)",
                "Tariff Harmonics", "Long multi-year loan maturities expose borrower to multi-year employment disruptions.")

# 380. High-Yield Retail Tariff Decoder
f380 = q0_fee * (monthly_inc / (an + 1.0)) * (u1 * u2)
m380 = q0_fee > 0.05
evaluate_regime(380, "Tariff_High_Yield_Decoder", m380, f380,
                "Q0_Fee * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Tariff Harmonics", "Decoded effective interest rate tariff calibrated against disposable debt coverage.")

# ==================================================================================================
# DOMAIN 5: NET DISPOSABLE FREE CASH FLOW & SUBSISTENCE STRAIN (EXP 381 - 390)
# ==================================================================================================
# 381. Severe Negative Free Cash Flow Deficit
f381 = (-free_cash_flow / (monthly_inc + 1.0)) * dti * (1.0 - u1)
m381 = free_cash_flow < 0.0
evaluate_regime(381, "Subsistence_Negative_FCF_Deficit", m381, f381,
                "(-FCF / Monthly_Inc) * DTI * (1 - E2)",
                "Free Cash Flow", "Monthly income fails to cover basic family subsistence plus contractual annuity.")

# 382. Substantial Liquid Free Cash Flow Buffer
f382 = (free_cash_flow / (an + 1.0)) * (u1 * u2)**2
m382 = free_cash_flow > 25000.0
evaluate_regime(382, "Subsistence_Liquid_FCF_Buffer", m382, f382,
                "(FCF / Annuity) * (E2 * E3)^2",
                "Free Cash Flow", "Abundant monthly cash flow remaining after living expenses ensures guaranteed debt service.")

# 383. Dependent Dilution Subsistence Squeeze
f383 = (subsistence / (monthly_inc + 1.0)) * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m383 = subsistence > 0.60 * monthly_inc
evaluate_regime(383, "Subsistence_Dependent_Squeeze", m383, f383,
                "(Subsistence / Monthly_Inc) * (Annuity / Monthly_Inc) * (1 - E3)",
                "Free Cash Flow", "Basic dependent maintenance costs consume the vast majority of borrower income.")

# 384. Sole Earner Large Family Stress
f384 = (fam / (monthly_inc / 10000.0 + 1.0)) * dti * (1.0 - u1)
m384 = fam >= 4.0
evaluate_regime(384, "Subsistence_Sole_Earner_Large_Family", m384, f384,
                "(Family / (Monthly_Inc / 10k)) * DTI * (1 - E2)",
                "Free Cash Flow", "Heavy per-capita household strain across 4+ family members.")

# 385. FCF Margin to Annuity Coverage
f385 = (free_cash_flow / (monthly_inc + 1.0)) * cop_frank_4
m385 = (free_cash_flow > 0.0) & (u1 > 0.2)
evaluate_regime(385, "Subsistence_FCF_Margin_Coverage", m385, f385,
                "(FCF / Monthly_Inc) * Frank_Copula",
                "Free Cash Flow", "Discretionary cash margin normalized by joint bureau alignment.")

# 386. Subsistence Burn Velocity
f386 = (an / (np.maximum(free_cash_flow, 1000.0))) * (1.0 - u1)
m386 = (free_cash_flow > 0.0) & (free_cash_flow < 10000.0)
evaluate_regime(386, "Subsistence_Burn_Velocity", m386, f386,
                "(Annuity / FCF) * (1 - E2)",
                "Free Cash Flow", "Annuity consumes nearly all remaining discretionary cash flow.")

# 387. Pensioner Guaranteed Baseline Surplus
f387 = (surplus / (cr + 1.0)) * (u1 + u2) * np.log1p(age)
m387 = is_retired & (surplus > 0.0)
evaluate_regime(387, "Subsistence_Pensioner_Guaranteed_Surplus", m387, f387,
                "(Annual_Surplus / Credit) * (E2 + E3) * log1p(Age)",
                "Free Cash Flow", "Elderly retirees with annual pension income strictly exceeding total annuity payments.")

# 388. Single Provider Extreme Deficit
f388 = (dti / (free_cash_flow / 10000.0 + 0.1)) * (1.0 - cop_clayton_2)
m388 = (ch >= 1.0) & np.isin(family, ["Single / not married", "Separated", "Widow"])
evaluate_regime(388, "Subsistence_Single_Parent_Deficit", m388, f388,
                "(DTI / (FCF / 10k)) * (1 - Clayton_Copula)",
                "Free Cash Flow", "Single parents without spousal backup facing tight free cash flow margins.")

# 389. Discretionary Disposable Cushion
f389 = (free_cash_flow / (cr * 0.1 + 1.0)) * (u1 * u2)**2
m389 = free_cash_flow > 15000.0
evaluate_regime(389, "Subsistence_Discretionary_Cushion", m389, f389,
                "(FCF / (Credit * 0.1)) * (E2 * E3)^2",
                "Free Cash Flow", "Monthly discretionary cash exceeds 10% of total loan balance.")

# 390. Micro-Loan Sub-Subsistence Paradox
f390 = (an / (subsistence + 1.0)) * dti * (1.0 - u1)
m390 = cr < 150000.0
evaluate_regime(390, "Subsistence_Micro_Loan_Paradox", m390, f390,
                "(Annuity / Subsistence) * DTI * (1 - E2)",
                "Free Cash Flow", "Borrowing very small credit lines despite severe baseline living cost burdens.")

# ==================================================================================================
# DOMAIN 6: MULTI-DIMENSIONAL OCCUPATIONAL HAZARD MATRICES (EXP 391 - 400)
# ==================================================================================================
# 391. Low-Skill Laborer Negative Downpayment Compound
f391 = ((cr - gp) / (gp + 1.0)) * dti * (1.0 - u1) * 2.0
m391 = occ == "Low-skill Laborers"
evaluate_regime(391, "Occupational_LowSkill_Markup_Hazard", m391, f391,
                "((Credit - Goods) / Goods) * DTI * (1 - E2) * 2",
                "Occupational Matrix", "Low-skill manual workers (16.2% default) borrowing above retail goods price.")

# 392. Professional Driver Vehicle Wear Burden
f392 = (car_age / 10.0) * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m392 = occ == "Drivers"
evaluate_regime(392, "Occupational_Driver_Vehicle_Wear", m392, f392,
                "(Car_Age / 10) * (Annuity / Monthly_Inc) * (1 - E3)",
                "Occupational Matrix", "Commercial drivers (11.3% default) burdened by vehicle repair overhead.")

# 393. Specialized Core Staff Stability
f393 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m393 = occ == "Core staff"
evaluate_regime(393, "Occupational_Core_Staff_Stability", m393, f393,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Occupational Matrix", "Corporate white-collar staff maintain continuous, low-volatility payrolls.")

# 394. Executive Management Speculative Debt Ratio
f394 = (cr / (monthly_inc * 12.0 + 1.0)) * (1.0 - cop_frank_4)
m394 = (occ == "Managers") | (occ == "Accountants")
evaluate_regime(394, "Occupational_Managerial_Leverage", m394, f394,
                "(Credit / Income) * (1 - Frank_Copula)",
                "Occupational Matrix", "High-earning managers taking large personal loans with agency disagreement.")

# 395. Cleaning & Cooking Staff Cash Fragility
f395 = (an / (monthly_inc + 1.0)) / (u1 + 0.1) * dti
m395 = (occ == "Cleaning staff") | (occ == "Cooking staff")
evaluate_regime(395, "Occupational_Service_Cash_Fragility", m395, f395,
                "(Annuity / Monthly_Inc) / E2 * DTI",
                "Occupational Matrix", "Low-wage domestic service employees bearing tight disposable margins.")

# 396. Public Sector Teaching Municipal Security
f396 = (monthly_inc / (an + 1.0)) * cop_harmonic * (u1 + u2)
m396 = (org == "School") | (org == "Kindergarten")
evaluate_regime(396, "Occupational_School_Teacher_Security", m396, f396,
                "(Monthly_Inc / Annuity) * Harmonic_Copula * (E2 + E3)",
                "Occupational Matrix", "Public school staff supported by municipal payrolls rarely default.")

# 397. High-Skill Tech Specialist Human Capital
f397 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * np.log1p(adult_life)
m397 = occ == "High skill tech staff"
evaluate_regime(397, "Occupational_Tech_Human_Capital", m397, f397,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Adult_Life)",
                "Occupational Matrix", "Rapid human capital appreciation among specialized technological talent.")

# 398. Sales Worker Commission Volatility
f398 = (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton_2) * dti
m398 = (occ == "Sales staff") | (org == "Trade: type 7")
evaluate_regime(398, "Occupational_Sales_Commission_Volatility", m398, f398,
                "(Annuity / Monthly_Inc) * (1 - Clayton_Copula) * DTI",
                "Occupational Matrix", "Retail sales employees vulnerable to commission revenue fluctuations.")

# 399. Private Security Enforcement Risk
f399 = dti * (1.0 - u2) * (cr / (gp + 1.0))
m399 = occ == "Security staff"
evaluate_regime(399, "Occupational_Security_Risk", m399, f399,
                "DTI * (1 - E3) * (Credit / Goods)",
                "Occupational Matrix", "Private guard personnel with below-average bureau scores.")

# 400. Transport Type 3 High Systemic Exposure
f400 = ((cr - gp) / (gp + 1.0)) * dti * (1.0 - u1) * 2.5
m400 = org == "Transport: type 3"
evaluate_regime(400, "Occupational_Transport3_Systemic_Hazard", m400, f400,
                "((Credit - Goods) / Goods) * DTI * (1 - E2) * 2.5",
                "Occupational Matrix", "Transport Type 3 employees (16.2% default rate) carry severe credit risk.")

# ==================================================================================================
# DOMAIN 7: ASSET LIQUIDITY DECAY & VEHICLE MAINTENANCE DRAG (EXP 401 - 410)
# ==================================================================================================
# 401. Vehicle Salvage Value Salvage Buffer
salvage_val = np.where(car_age >= 0, gp * np.exp(-car_age / 6.0), 0.0)
f401 = (salvage_val / (cr + 1.0)) * (u1 * u2)
m401 = car_age >= 0
evaluate_regime(401, "Asset_Vehicle_Salvage_Buffer", m401, f401,
                "(Salvage_Val / Credit) * (E2 * E3)",
                "Asset Dynamics", "Estimated vehicle liquidation value provides secondary loan collateral.")

# 402. Aging Vehicle Mechanical Liability Drag
f402 = (car_age / 8.0) * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m402 = car_age > 10.0
evaluate_regime(402, "Asset_Aging_Vehicle_Liability_Drag", m402, f402,
                "(Car_Age / 8) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Asset Dynamics", "Vehicles older than 10 years demand heavy ongoing maintenance outlays.")

# 403. Luxury Vehicle Equity Cushion
f403 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (5.0 - np.minimum(car_age, 4.0))
m403 = (car_age >= 0) & (car_age <= 4.0)
evaluate_regime(403, "Asset_Late_Model_Vehicle_Cushion", m403, f403,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * (5 - Car_Age)",
                "Asset Dynamics", "Late-model personal vehicles reflect high unencumbered physical equity.")

# 404. Homeowner Real Estate Solvency Anchor
f404 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(adult_life)
m404 = housing == "House / apartment"
evaluate_regime(404, "Asset_Homeowner_Solvency_Anchor", m404, f404,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Adult_Life)",
                "Asset Dynamics", "Stable residential property ownership anchors long-term credit reliability.")

# 405. Rented Tenant Living Friction
f405 = (an / (monthly_inc + 1.0)) * dti / (u2 + 0.1)
m405 = housing == "Rented apartment"
evaluate_regime(405, "Asset_Rented_Apartment_Friction", m405, f405,
                "(Annuity / Monthly_Inc) * DTI / E3",
                "Asset Dynamics", "Market rent obligations compound monthly loan debt service.")

# 406. Municipal Apartment Social Housing Strain
f406 = dti * (1.0 - cop_frank_4) * (an / (monthly_inc + 1.0))
m406 = housing == "Municipal apartment"
evaluate_regime(406, "Asset_Municipal_Housing_Strain", m406, f406,
                "DTI * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "Asset Dynamics", "Municipal social housing tenancy signals structural household economic distress.")

# 407. Living With Parents Financial Buffer
f407 = (monthly_inc / (an + 1.0)) * (u1 * u2) * log1p(emp_yrs)
m407 = housing == "With parents"
evaluate_regime(407, "Asset_Living_With_Parents_Buffer", m407, f407,
                "(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)",
                "Asset Dynamics", "Young adults living with parents benefit from shared living expenses.")

# 408. Zero-Asset Unanchored Urban Tenant
f408 = (an / (monthly_inc + 1.0)) * dti * (1.0 - u1)
m408 = (car_age < 0) & (housing == "Rented apartment")
evaluate_regime(408, "Asset_Zero_Asset_Urban_Tenant", m408, f408,
                "(Annuity / Monthly_Inc) * DTI * (1 - E2)",
                "Asset Dynamics", "Tenants without car equity face extreme downside vulnerability.")

# 409. Salvage Coverage-to-Debt Multiplier
f409 = (salvage_val / (cr + 1.0)) * cop_frank_4
m409 = salvage_val > cr
evaluate_regime(409, "Asset_Fully_Collateralized_Salvage", m409, f409,
                "(Salvage_Val / Credit) * Frank_Copula",
                "Asset Dynamics", "Personal vehicle value fully covers outstanding credit balance.")

# 410. Commercial Office Co-op Agility
f410 = (monthly_inc / (an + 1.0)) * (u1 + u2) * log1p(emp_yrs)
m410 = housing == "Office apartment"
evaluate_regime(410, "Asset_Office_Apartment_Agility", m410, f410,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)",
                "Asset Dynamics", "Co-op and office apartment dwellers demonstrate professional commercial adaptability.")

# ==================================================================================================
# DOMAIN 8: INQUIRY ACCELERATION SHOCK & HIGH-FREQUENCY VELOCITY (EXP 411 - 420)
# ==================================================================================================
# 411. Emergency Simultaneous Application Shock
f411 = inq_h * (cr / (gp + 1.0)) / (u2 + 0.1)
m411 = inq_h >= 1.0
evaluate_regime(411, "Inquiry_Emergency_Hourly_Shock", m411, f411,
                "Inq_Hour * (Credit / Goods) / E3",
                "Inquiry Dynamics", "Credit bureau requests in the immediate trailing hour indicate simultaneous panic loan applications.")

# 412. Monthly Credit Velocity Surge
f412 = inq_m * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m412 = inq_m >= 2.0
evaluate_regime(412, "Inquiry_Monthly_Velocity_Surge", m412, f412,
                "Inq_Month * (Annuity / Monthly_Inc) * (1 - E2)",
                "Inquiry Dynamics", "Multiple credit applications in 30 days reveals acute unmet cash demands.")

# 413. Quarterly-to-Annual Velocity Surge
inq_surge = np.maximum(inq_q * 4.0 - inq_y, 0.0)
f413 = inq_surge * dti * (1.0 - cop_frank_4)
m413 = inq_surge >= 2.0
evaluate_regime(413, "Inquiry_Quarterly_Surge_Index", m413, f413,
                "(Inq_QRT * 4 - Inq_YEAR) * DTI * (1 - Frank_Copula)",
                "Inquiry Dynamics", "Quarterly inquiry acceleration exceeding historical annual velocity signals accelerating distress.")

# 414. Chronic Institutional Rejection Fatigue
f414 = np.log1p(inq_y) * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m414 = inq_y >= 5.0
evaluate_regime(414, "Inquiry_Chronic_Rejection_Fatigue", m414, f414,
                "log1p(Inq_YEAR) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Inquiry Dynamics", "5+ annual inquiries indicates persistent institutional lending refusal.")

# 415. Zero Inquiry Prime Discretion Sanctuary
f415 = (u1 * u2)**2 * (monthly_inc / (an + 1.0)) * log1p(emp_yrs)
m415 = (inq_y == 0.0) & (~inq_missing)
evaluate_regime(415, "Inquiry_Zero_Inquiry_Prime_Sanctuary", m415, f415,
                "(E2 * E3)^2 * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Inquiry Dynamics", "Zero inquiries over trailing 12 months reflects voluntary, planned consumer financing.")

# 416. Unmonitored Thin File Informational Vacuum
f416 = dti * (cr / (gp + 1.0)) * (1.0 - u1)
m416 = inq_missing
evaluate_regime(416, "Inquiry_Thin_File_Vacuum", m416, f416,
                "DTI * (Credit / Goods) * (1 - E2)",
                "Inquiry Dynamics", "Absence of bureau inquiry records indicates unmonitored retail credit history.")

# 417. Subprime Inquiry Velocity Squeeze
f417 = (inq_y / (u1 + 0.05)) * (an / (monthly_inc + 1.0))
m417 = (inq_y >= 3.0) & (u1 < 0.3)
evaluate_regime(417, "Inquiry_Subprime_Velocity_Squeeze", m417, f417,
                "(Inq_YEAR / E2) * (Annuity / Monthly_Inc)",
                "Inquiry Dynamics", "Frequent credit searching among applicants already holding subprime external bureau scores.")

# 418. Late-Stage 90-Day Desperation Attempts
f418 = inq_q * dti * (1.0 - u2)
m418 = (inq_q >= 3.0) & (u2 < 0.25)
evaluate_regime(418, "Inquiry_Late_Stage_Desperation", m418, f418,
                "Inq_QRT * DTI * (1 - E3)",
                "Inquiry Dynamics", "Trailing 90-day inquiry storm from severely credit-impaired applicants.")

# 419. Inquiry-to-Income Elasticity Drain
f419 = (inq_y / np.log1p(inc)) * (cr / (gp + 1.0)) * (1.0 - u1)
m419 = (inq_y >= 2.0) & (inc < 120000.0)
evaluate_regime(419, "Inquiry_Low_Income_Elasticity", m419, f419,
                "(Inq_YEAR / log1p(Income)) * (Credit / Goods) * (1 - E2)",
                "Inquiry Dynamics", "Credit shopping by low earners compounds structural default vulnerability.")

# 420. Monthly Harmonic Velocity Damping
f420 = (1.0 / (inq_m + 1.0)) * (monthly_inc / (an + 1.0)) * cop_harmonic
m420 = ~inq_missing
evaluate_regime(420, "Inquiry_Harmonic_Velocity_Damping", m420, f420,
                "(1 / (Inq_MON + 1)) * (Monthly_Inc / Annuity) * Harmonic_Copula",
                "Inquiry Dynamics", "Low monthly inquiry velocity corroborating high joint bureau harmonic consensus.")

# ==================================================================================================
# DOMAIN 9: SPATIAL ECONOMIC CENTROID DIVERGENCE & MUNICIPAL BUFFERS (EXP 421 - 430)
# ==================================================================================================
# 421. Elite Tier-1 High-Density Megacity Sanctuary
f421 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (pop / 0.02)
m421 = pop >= 0.03
evaluate_regime(421, "Spatial_Tier1_Megacity_Sanctuary", m421, f421,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * (Pop / 0.02)",
                "Economic Geography", "Borrowers in high-density metropolitan districts benefit from robust labor liquidity.")

# 422. Isolated Rural Periphery Vulnerability
f422 = dti * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m422 = pop < 0.005
evaluate_regime(422, "Spatial_Rural_Periphery_Vulnerability", m422, f422,
                "DTI * (Annuity / Monthly_Inc) * (1 - E3)",
                "Economic Geography", "Low-density rural jurisdictions lack alternative employment buffers during localized plant shutdowns.")

# 423. Rating Tier 3 Structural Macro Distress
f423 = dti * (cr / (gp + 1.0)) * (1.0 - u1)
m423 = rating == 3.0
evaluate_regime(423, "Spatial_Rating_Tier3_Distress", m423, f423,
                "DTI * (Credit / Goods) * (1 - E2)",
                "Economic Geography", "Regions designated rating tier 3 suffer structural macroeconomic stagnation.")

# 424. Rating Tier 1 Institutional Safety
f424 = (monthly_inc / (an + 1.0)) * cop_frank_4 * (u1 + u2)
m424 = rating == 1.0
evaluate_regime(424, "Spatial_Rating_Tier1_Safety", m424, f424,
                "(Monthly_Inc / Annuity) * Frank_Copula * (E2 + E3)",
                "Economic Geography", "Tier-1 prime regional markets maintain lower baseline unemployment rates.")

# 425. Regional Centroid Income Deficit
grp_reg_inc = df.groupby("REGION_POPULATION_RELATIVE")["AMT_INCOME_TOTAL"].transform("mean").values
reg_inc_dev = (inc - grp_reg_inc) / (grp_reg_inc + 1.0)
f425 = (-reg_inc_dev) * dti * (1.0 - u1)
m425 = reg_inc_dev < -0.25
evaluate_regime(425, "Spatial_Regional_Income_Deficit", m425, f425,
                "(-Income_Deviation) * DTI * (1 - E2)",
                "Economic Geography", "Earning significantly below the local municipal average intensifies regional cost pressures.")

# 426. Regional Centroid Wealth Premium
f426 = reg_inc_dev * (monthly_inc / (an + 1.0)) * (u1 * u2)
m426 = reg_inc_dev > 0.50
evaluate_regime(426, "Spatial_Regional_Wealth_Premium", m426, f426,
                "Income_Deviation * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Economic Geography", "Earning well above regional peer average generates exceptional discretionary surplus.")

# 427. Dual Workplace Landline Verification Anchor
f427 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m427 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0)
evaluate_regime(427, "Spatial_Dual_Landline_Verification_Anchor", m427, f427,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Economic Geography", "Both employer and direct office landline phone numbers verified by loan officers.")

# 428. Disconnected Employer Phone Warning
f428 = dti * (1.0 - u1) * (cr / (gp + 1.0))
m428 = (flag_emp_phone == 0.0) & (emp_yrs > 0.5)
evaluate_regime(428, "Spatial_Disconnected_Employer_Warning", m428, f428,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Economic Geography", "Working applicant lacking confirmed employer telephone contact.")

# 429. Municipal Credit Absorption Capacity
f429 = (cr / (pop * 1e6 + 1.0)) * (1.0 - cop_frank_4) * (an / (monthly_inc + 1.0))
m429 = pop >= 0.015
evaluate_regime(429, "Spatial_Credit_Absorption_Capacity", m429, f429,
                "(Credit / (Pop_Norm + 1)) * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "Economic Geography", "Loan principal scaled against municipality credit capacity.")

# 430. Institutional Public Hubs Regional Periphery
f430 = (monthly_inc / (an + 1.0)) * cop_frank_4 * (u1 * u2)
m430 = (rating == 1.0) & (pop < 0.01)
evaluate_regime(430, "Spatial_Institutional_Periphery_Anchor", m430, f430,
                "(Monthly_Inc / Annuity) * Frank_Copula * (E2 * E3)",
                "Economic Geography", "Prime institutional personnel stationed in regional outposts.")

# ==================================================================================================
# DOMAIN 10: RELATIONAL SEQUENCE MULTI-LOAN TRAJECTORY DYNAMICS (EXP 431 - 440)
# ==================================================================================================
days_prev = df_restored["DAYS_SINCE_PREV_APP"].fillna(-999.0).values
app_order = df_restored["APP_ORDER"].values.astype(float)

# 431. Inaugural Application Trajectory Anchor
f431 = (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(n_apps_all)
m431 = (app_order == 0.0) & (n_apps_all >= 2.0)
evaluate_regime(431, "Relational_Inaugural_Application_Anchor", m431, f431,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(N_Apps)",
                "Relational Trajectories", "The first loan in a confirmed multi-application borrower's sequence establishes credit roots.")

# 432. Rapid Reapplication Stacking Hazard
f432 = (1.0 / (np.maximum(days_prev, 0.0) / 30.0 + 0.1)) * (1.0 - u1) * dti
m432 = (app_order >= 1.0) & (days_prev >= 0.0) & (days_prev < 120.0)
evaluate_regime(432, "Relational_Rapid_Stacking_Hazard", m432, f432,
                "(1 / Prev_App_Months) * (1 - E2) * DTI",
                "Relational Trajectories", "Reapplication within 4 months of prior financing reflects acute cash accumulation strain.")

# 433. Multi-Year Sabbatical Planned Borrowing
f433 = np.log1p(days_prev / 365.25) * (u1 * u2)**2
m433 = (app_order >= 1.0) & (days_prev >= 365.0)
evaluate_regime(433, "Relational_Multi_Year_Sabbatical", m433, f433,
                "log1p(Years_Since_Last_App) * (E2 * E3)^2",
                "Relational Trajectories", "Multi-year intervals between loans prove disciplined, non-habitual borrowing.")

# 434. Singleton Cold-Start Baseline
f434 = dti * (1.0 - u1) * (cr / (gp + 1.0))
m434 = n_apps_all == 1.0
evaluate_regime(434, "Relational_Singleton_Cold_Start", m434, f434,
                "DTI * (1 - E2) * (Credit / Goods)",
                "Relational Trajectories", "Applicants appearing only once lack repeat cross-application corroboration.")

# 435. High-Density Distress Cluster
f435 = (n_apps_all / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))
m435 = n_apps_all >= 3.0
evaluate_regime(435, "Relational_Dense_Distress_Cluster", m435, f435,
                "(N_Apps / (Harmonic_Copula + 0.1)) * (Annuity / Monthly_Inc)",
                "Relational Trajectories", "Borrowers with 3+ applications experience correlated systemic delinquency.")

# 436. Loan Order Progression Maturity
f436 = (app_order / (n_apps_all + 0.1)) * (monthly_inc / (an + 1.0)) * cop_frank_4
m436 = n_apps_all >= 2.0
evaluate_regime(436, "Relational_Order_Progression_Maturity", m436, f436,
                "(App_Order / N_Apps) * (Monthly_Inc / Annuity) * Frank_Copula",
                "Relational Trajectories", "Later loans in a mature customer trajectory benefit from accumulated repayment experience.")

# 437. Repeat Single Male Laborer Cluster
f437 = (single_male_hazard * np.log1p(n_apps_all)) / (cop_harmonic + 0.1)
m437 = (single_male_hazard > 0.0) & (n_apps_all >= 2.0)
evaluate_regime(437, "Relational_Repeat_Laborer_Hazard", m437, f437,
                "(Single_Male_Laborer_Hazard * log1p(N_Apps)) / Harmonic_Copula",
                "Relational Trajectories", "Single male manual laborers taking multiple serial consumer loans.")

# 438. Black-Cox Relational Solvency Hybrid
f438 = black_cox_cushion * (u1 * u2)**2 * np.log1p(days_prev / 365.25 + 1.0)
m438 = (black_cox_cushion > 1.5) & (app_order >= 1.0)
evaluate_regime(438, "Relational_BlackCox_Solvency_Hybrid", m438, f438,
                "Barrier_Cushion * (E2 * E3)^2 * log1p(Prev_App_Years)",
                "Relational Trajectories", "Couples structural asset barrier safety with disciplined repeat application pacing.")

# 439. Identity Invariant Registration Velocity
reg_id_gap = np.abs(df["DAYS_REGISTRATION"].values - df["DAYS_ID_PUBLISH"].values) / 365.25
f439 = (reg_id_gap / (adult_life + 0.1)) * dti * (1.0 - u1)
m439 = reg_id_gap > 10.0
evaluate_regime(439, "Relational_Identity_Administrative_Lag", m439, f439,
                "(Reg_ID_Gap / Adult_Life) * DTI * (1 - E2)",
                "Relational Trajectories", "Decade-long divergence between municipal registration and identity document renewals.")

# 440. Master Cognitive Synthesis Manifold V2
f440 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)) / (1.0 + single_male_hazard + (cr - gp) / (gp + 1.0))
m440 = np.ones(n_train, dtype=bool)
evaluate_regime(440, "Master_Cognitive_Super_Manifold_V2", m440, f440,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)) / (1 + Single_Male_Hazard + Neg_Downpayment)",
                "Cognitive Synthesis", "Master synthesis uniting squared bureau consensus, disposable income, career tenure, negative markup, and demographic hazard.")

# --------------------------------------------------------------------------------------------------
# PART 3: STRATIFIED 5-FOLD CV VERIFICATION OF TOP 15 ADVANCED FORMULATIONS
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 115)
print("  PART 3: STRATIFIED 5-FOLD CV VERIFICATION OF TOP 15 ADVANCED QUANTITATIVE FORMULAS")
print("=" * 115)

top_15_ids = sorted(results_100.keys(), key=lambda k: results_100[k]["within_auc"], reverse=True)[:15]

print("\nTop 15 Discovered Quantitative & Actuarial Formulations:")
for rank_i, exp_id in enumerate(top_15_ids, 1):
    r = results_100[exp_id]
    print(f"  {rank_i:>2}. Exp {exp_id} ({r['name']:<35}): Within AUC = {r['within_auc']:.4f} | Spread = {r['spread_bps']:>6.1f} bps | {r['math_family']}")

# Run 5-fold CV comparison
print("\n1. Evaluating Baseline Feature Set (5-Fold CV)...")
oof_base = np.zeros(n_train)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_base, y)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_base.iloc[trn_idx], y[trn_idx])
    oof_base[val_idx] = m.predict_proba(X_base.iloc[val_idx])[:, 1]

auc_base = roc_auc_score(y, oof_base)
print(f"   [+] Baseline Pooled OOF AUC: {auc_base:.6f}")

print("\n2. Evaluating Enhanced Feature Set (+Top 15 Advanced Quantitative Formulations)...")
X_enhanced = X_base.copy()
for exp_id in top_15_ids:
    r = results_100[exp_id]
    feat_name = f"ADV_{exp_id}_{r['name']}"
    X_enhanced[feat_name] = np.nan_to_num(discovered_features[f"F_{exp_id}_{r['name']}"], nan=0.0)

oof_enh = np.zeros(n_train)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_enhanced, y)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_enhanced.iloc[trn_idx], y[trn_idx])
    oof_enh[val_idx] = m.predict_proba(X_enhanced.iloc[val_idx])[:, 1]

auc_enh = roc_auc_score(y, oof_enh)
cv_lift_bps = (auc_enh - auc_base) * 10000.0
print(f"   [+] Enhanced Pooled OOF AUC: {auc_enh:.6f} | Net Cross-Validation Lift: {cv_lift_bps:+.1f} bps")

# --------------------------------------------------------------------------------------------------
# PART 4: SAVE ARTIFACTS
# --------------------------------------------------------------------------------------------------
json_path = r"D:\gci\gemini\RESEARCH_100_ADVANCED_KNOWLEDGE_EXP341_440.json"
md_path = r"D:\gci\gemini\RESEARCH_100_ADVANCED_KNOWLEDGE_EXP341_440.md"

summary_data = {
    "metadata": {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_regimes": len(results_100),
        "baseline_oof_auc": round(auc_base, 6),
        "enhanced_oof_auc": round(auc_enh, 6),
        "net_lift_bps": round(cv_lift_bps, 1),
        "runtime_seconds": round(time.time() - t_start, 2)
    },
    "feature_importance_top15": gain_df.head(15).to_dict(orient="records"),
    "top_15_regimes": [results_100[eid] for eid in top_15_ids],
    "all_100_regimes": results_100
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

md_lines = [
    "# GCI World 2026: 100 Deep Research Domain & Actuarial Knowledge Formulations (Exp 341 – 440)\n",
    f"- **Execution Timestamp:** `{summary_data['metadata']['execution_timestamp']}`",
    f"- **Total Advanced Regimes Evaluated:** 100 Distinct Mathematical Formulations",
    f"- **Baseline 5-Fold OOF AUC:** `{auc_base:.6f}`",
    f"- **Enhanced 5-Fold OOF AUC:** `{auc_enh:.6f}` (**{cv_lift_bps:+.1f} bps Net Lift**)",
    f"- **Runtime:** `{time.time() - t_start:.2f}s` on RTX 4050 GPU\n",
    "## 1. Feature Importance Profiler (Split Gain vs Split Count)\n",
    "| Rank | Feature Name | Total Split Gain | Gain Share (%) | Split Count | Mathematical Role |",
    "| :---: | :--- | :---: | :---: | :---: | :--- |"
]

role_map = {
    "EXT_SOURCE_2": "Primary unconstrained continuous bureau sensor",
    "EXT_SOURCE_3": "Secondary unconstrained continuous bureau sensor",
    "EXT_HARMONIC": "Lower-tail non-linear penalty for extreme single-agency collapse",
    "EXT_PRODUCT": "Joint Bayesian likelihood product of all three bureau probabilities",
    "EXT_MIN": "Worst-case agency floor bounding downside tail exposure",
    "Q0_FEE": "Decoded effective interest rate tariff minus 129.50 fee lattice",
    "DAYS_BIRTH": "Continuous demographic age / lifetime human capital maturity",
    "DAYS_EMPLOYED": "Sticky career tenure and workplace seniority capital",
    "CREDIT_TO_ANNUITY_TERM": "Implied loan amortization period and interest schedule",
    "ANNUITY_TO_MONTHLY_INC": "Contractual debt service burden relative to monthly income",
    "DAYS_REGISTRATION": "Municipal housing / address tenure stability",
    "DAYS_ID_PUBLISH": "Administrative document renewal friction and identity aging",
    "AMT_CREDIT": "Total borrowed principal / loss severity exposure at default",
    "AMT_ANNUITY": "Contractual monthly cash outflow demand",
    "NEGATIVE_DOWNPAYMENT": "Dealer fee markup exceeding retail goods price",
    "DTI_RATIO": "Aggregate leverage relative to annual income",
    "REGION_POPULATION_RELATIVE": "Municipal economic density and local labor market liquidity"
}

for i, r in gain_df.head(15).iterrows():
    role = role_map.get(r['Feature'], "Financial or demographic risk covariate")
    md_lines.append(f"| {i+1} | `{r['Feature']}` | {r['Split_Gain']:,.1f} | {r['Gain_Pct']:.2f}% | {r['Split_Count']:,.0f} | {role} |")

md_lines.extend([
    "\n## 2. Exhaustive Ledger of 100 Advanced Knowledge Formulations (Exp 341 – 440)\n",
    "| Exp ID | Regime Name | Sample N | Pct (%) | Def Rate | Within AUC | Corr (rho) | Spread (bps) | Math Family | Formula | Theoretical Domain Mechanism |",
    "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |"
])

for exp_id in sorted(results_100.keys()):
    r = results_100[exp_id]
    md_lines.append(
        f"| {r['exp_id']} | **{r['name']}** | {r['count']:,} | {r['pct']}% | {r['def_rate']:.2f}% | "
        f"**{r['within_auc']:.4f}** | {r['spearman_corr']:+.4f} | **{r['spread_bps']:.1f}** | "
        f"`{r['math_family']}` | `{r['formula']}` | {r['domain_reason']} |"
    )

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n[+] Research JSON Saved : {json_path}")
print(f"[+] Research MD Saved   : {md_path}")
print(f"[+] Total Execution Time: {time.time() - t_start:.2f}s")
print("=" * 115)
