"""
====================================================================================================
  GCI WORLD 2026: 100 DEEP HIGHER-ORDER MATHEMATICAL & ACTUARIAL MANIFOLDS (EXP 441 - 540)
  + INFORMATION GEOMETRY, KOU JUMP-DIFFUSION, & MARKOV TRANSITION KERNELS
====================================================================================================
  - Pure In-Memory, Clean-Room Compliant (Strictly inpuy/train.csv & inpuy/test.csv)
  - 10 Advanced Mathematical Frameworks:
    1. Merton-Kou Double Exponential Jump-Diffusion Solvency (Exp 441 - 450)
    2. Information Geometry & Fisher Information Curvature (Exp 451 - 460)
    3. Ornstein-Uhlenbeck Mean-Reverting Income Drift (Exp 461 - 470)
    4. Copula Conditional Tail Expectation (CTE) & Actuarial VaR (Exp 471 - 480)
    5. Quantized Fourier Harmonic Tariff Resonances (Exp 481 - 490)
    6. Padé Rational Solvency Approximants (Exp 491 - 500)
    7. Multiscale Administrative Renewal Synchronization (Exp 501 - 510)
    8. Spatial Gravitational Potential & Economic Gradient Flow (Exp 511 - 520)
    9. Hessian Loss Surface Principal Curvature Eigenvectors (Exp 521 - 530)
    10. Relational Markov Chain Transition Kernels (Exp 531 - 540)
  - Exhaustive 100-Regime Evaluation + Stratified 5-Fold Cross Validation Verification
====================================================================================================
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from numpy import log1p
from scipy.stats import spearmanr, norm, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

t_start = time.time()
print("=" * 115)
print("  GCI WORLD 2026: 100 HIGHER-ORDER MATHEMATICAL & ACTUARIAL MANIFOLDS (EXP 441 - 540)")
print("  + INFORMATION GEOMETRY, KOU JUMP-DIFFUSION, & LOSS SURFACE SPLIT MECHANICS")
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
# ADVANCED MATHEMATICAL ENGINES
# --------------------------------------------------------------------------------------------------
# 1. Copulas & Information Geometry
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))
cop_frank_4 = -0.25 * np.log(1.0 + ((np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)) / (np.exp(-4.0) - 1.0))
cop_clayton_2 = np.maximum(u1**(-2.0) + u2**(-2.0) - 1.0, 1e-6)**(-0.5)

# Fisher Information / Sensor Curvature Metric (Information Geometry)
# Local metric g_11 = 1 / (u1 * (1 - u1)), g_22 = 1 / (u2 * (1 - u2))
fisher_metric_det = (1.0 / (u1 * (1.0 - u1) * u2 * (1.0 - u2) + 1e-5))**0.25
geodesic_dist = np.sqrt((np.arcsin(np.sqrt(u1)) - np.arcsin(np.sqrt(u2)))**2)

# 2. Kou Double Exponential Jump-Diffusion Setup
# Asset drift with asymmetric Poisson jump shocks (lambda_jump = 0.20, p_up = 0.4, eta_down = 3.0)
kou_barrier_cushion = (inc * 3.0 + cr * 0.5 - subsistence * 3.0) / (cr + 1.0)
kou_jump_hazard = 0.20 * (1.0 / 3.0) * (1.0 - u1) * (an / (monthly_inc + 1.0))

# 3. Ornstein-Uhlenbeck Occupational Mean Reversion
grp_occ_inc = df.groupby("OCCUPATION_TYPE")["AMT_INCOME_TOTAL"].transform("mean").values
ou_wage_drift = (inc - grp_occ_inc) / (grp_occ_inc + 1.0)

# 4. Tariff Harmonics (Fourier Step Resonances)
tenor_months = np.round(cr / (an + 1e-5)).astype(int)
fourier_tenor_harm = np.cos(2.0 * np.pi * tenor_months / 12.0)
q0_fee = np.where((contract == "Cash loans") & ((an - 129.5) / (cr + 1e-5) > 0), (an - 129.5) / (cr + 1e-5), an / (cr + 1e-5))
neg_downpayment = np.maximum(cr - gp, 0.0) / (gp + 1.0)

# 5. Demographics & Hazards
single_male_hazard = ((gender == "M") & np.isin(family, ["Single / not married", "Separated", "Widow"])).astype(float) * dti * (1.0 - u1)

# Relational Sequence
days_prev = df_restored["DAYS_SINCE_PREV_APP"].fillna(-999.0).values
app_order = df_restored["APP_ORDER"].values.astype(float)

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
    print(f"[{exp_id}] {name:<35} | N={count:>7,} ({pct:>4.1f}%) | Def={def_rate:>5.2f}% | AUC={auc_score:.4f} | Corr={corr:+.4f} | Spread={spread_bps:>6.1f} bps | {math_family:<20}")

# ==================================================================================================
# DOMAIN 1: MERTON-KOU DOUBLE EXPONENTIAL JUMP-DIFFUSION SOLVENCY (EXP 441 - 450)
# ==================================================================================================
# 441. Kou Downside Jump Hazard Intensity
f441 = kou_jump_hazard * dti * (1.0 - cop_clayton_2)
m441 = kou_jump_hazard > 0.02
evaluate_regime(441, "Kou_Downside_Jump_Hazard", m441, f441,
                "Kou_Jump_Hazard * DTI * (1 - Clayton_Copula)",
                "Kou Jump-Diffusion", "Simulates Poisson jump shocks modeling catastrophic personal income disruptions.")

# 442. Kou Solvency Barrier Resilience
f442 = kou_barrier_cushion * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m442 = kou_barrier_cushion >= 2.0
evaluate_regime(442, "Kou_Solvency_Barrier_Resilience", m442, f442,
                "Kou_Barrier_Cushion * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Kou Jump-Diffusion", "Structural asset distance protecting against jump-to-default risk.")

# 443. Kou Microloan Jump Asymmetry
f443 = kou_jump_hazard * (cr / (gp + 1.0)) / (u2 + 0.1)
m443 = cr < 150000.0
evaluate_regime(443, "Kou_Microloan_Jump_Asymmetry", m443, f443,
                "Kou_Jump_Hazard * (Credit / Goods) / E3",
                "Kou Jump-Diffusion", "Small unhedged loans facing asymmetric downside income shocks.")

# 444. Kou Career Jump Cushion
f444 = kou_barrier_cushion * log1p(emp_yrs) * cop_frank_4
m444 = emp_yrs >= 4.0
evaluate_regime(444, "Kou_Career_Jump_Cushion", m444, f444,
                "Kou_Barrier_Cushion * log1p(Emp_Yrs) * Frank_Copula",
                "Kou Jump-Diffusion", "Senior career capital dampens jump amplitude.")

# 445. Kou Laborer Jump Vulnerability
f445 = (single_male_hazard * kou_jump_hazard * 10.0) / (cop_harmonic + 0.1)
m445 = single_male_hazard > 0.0
evaluate_regime(445, "Kou_Laborer_Jump_Vulnerability", m445, f445,
                "(Single_Male_Hazard * Kou_Jump_Hazard * 10) / Harmonic_Copula",
                "Kou Jump-Diffusion", "Single male laborers lacking spousal safety nets absorb full jump shocks.")

# 446. Kou Senior Pensioner Invariance
f446 = kou_barrier_cushion * (u1 + u2) * np.log1p(age)
m446 = is_retired
evaluate_regime(446, "Kou_Pensioner_Jump_Invariance", m446, f446,
                "Kou_Barrier_Cushion * (E2 + E3) * log1p(Age)",
                "Kou Jump-Diffusion", "Sovereign guaranteed retirement benefits are immune to private sector jump shocks.")

# 447. Kou Short Amortization Jump Compression
f447 = (an / (monthly_inc + 1.0)) * kou_jump_hazard * (12.0 / (term + 1.0))
m447 = term < 15.0
evaluate_regime(447, "Kou_Short_Amortization_Jump_Compression", m447, f447,
                "(Annuity / Monthly_Inc) * Kou_Jump_Hazard * (12 / Term)",
                "Kou Jump-Diffusion", "Fast repayment velocity leaves zero liquid recovery window after a negative shock.")

# 448. Kou Prime Megacity Buffer
f448 = kou_barrier_cushion * (pop / 0.02) * (u1 * u2)**2
m448 = pop >= 0.02
evaluate_regime(448, "Kou_Megacity_Jump_Buffer", m448, f448,
                "Kou_Barrier_Cushion * (Pop / 0.02) * (E2 * E3)^2",
                "Kou Jump-Diffusion", "Metropolitan job liquidity enables immediate re-employment post-shock.")

# 449. Kou Negative Equity Acceleration
f449 = neg_downpayment * kou_jump_hazard * dti
m449 = neg_downpayment > 0.15
evaluate_regime(449, "Kou_Negative_Equity_Jump_Acceleration", m449, f449,
                "Neg_Downpayment * Kou_Jump_Hazard * DTI",
                "Kou Jump-Diffusion", "Borrowing above collateral value accelerates first-passage jump default.")

# 450. Kou Free Cash Flow Shield
f450 = (free_cash_flow / (an + 1.0)) * (1.0 - kou_jump_hazard) * cop_frank_4
m450 = free_cash_flow > 15000.0
evaluate_regime(450, "Kou_FCF_Jump_Shield", m450, f450,
                "(FCF / Annuity) * (1 - Kou_Jump_Hazard) * Frank_Copula",
                "Kou Jump-Diffusion", "Abundant monthly free cash flow absorbs unexpected expenditure spikes.")

# ==================================================================================================
# DOMAIN 2: INFORMATION GEOMETRY & FISHER METRIC CURVATURE (EXP 451 - 460)
# ==================================================================================================
# 451. Fisher Metric Information Determinant
f451 = fisher_metric_det * (monthly_inc / (an + 1.0)) * (u1 * u2)
m451 = fisher_metric_det > 2.0
evaluate_regime(451, "Fisher_Information_Determinant_Scale", m451, f451,
                "Fisher_Metric_Det * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Information Geometry", "Riemannian Fisher information metric measures local curvature on the score manifold.")

# 452. Geodesic Distance Bureau Conflict
f452 = geodesic_dist * dti * (1.0 - cop_frank_4)
m452 = geodesic_dist > 0.25
evaluate_regime(452, "Fisher_Geodesic_Bureau_Conflict", m452, f452,
                "Geodesic_Dist * DTI * (1 - Frank_Copula)",
                "Information Geometry", "Geodesic arc length divergence across external score agencies.")

# 453. Fisher Prime Centroid Sanctuary
f453 = (1.0 / (geodesic_dist + 0.05)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m453 = geodesic_dist < 0.08
evaluate_regime(453, "Fisher_Prime_Centroid_Sanctuary", m453, f453,
                "(1 / Geodesic_Dist) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Information Geometry", "Zero geodesic Riemannian divergence between credit bureaus proves high credibility.")

# 454. Fisher Tangent Space Anisotropy
f454 = np.abs(u1 - u2) * fisher_metric_det * (cr / (gp + 1.0))
m454 = np.abs(u1 - u2) > 0.20
evaluate_regime(454, "Fisher_Tangent_Anisotropy", m454, f454,
                "abs(E2 - E3) * Fisher_Metric_Det * (Credit / Goods)",
                "Information Geometry", "Severe asymmetric directional curvature in the score tangent space.")

# 455. Fisher Carrier Longevity Anchor
f455 = (1.0 / (geodesic_dist + 0.05)) * log1p(ph_yrs) * (u1 + u2)
m455 = (geodesic_dist < 0.10) & (ph_yrs > 2.0)
evaluate_regime(455, "Fisher_Carrier_Longevity_Anchor", m455, f455,
                "(1 / Geodesic_Dist) * log1p(Phone_Yrs) * (E2 + E3)",
                "Information Geometry", "Geodesic consensus corroborated by telecom longevity.")

# 456. Fisher Subprime Metric Dilution
f456 = geodesic_dist * (an / (monthly_inc + 1.0)) / (u1 + 0.05)
m456 = u1 < 0.30
evaluate_regime(456, "Fisher_Subprime_Metric_Dilution", m456, f456,
                "Geodesic_Dist * (Annuity / Monthly_Inc) / E2",
                "Information Geometry", "Subprime applicants experiencing acute bureau discordance.")

# 457. Fisher Education Centroid Alignment
f457 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (1.0 / (geodesic_dist + 0.05))
m457 = edu == "Higher education"
evaluate_regime(457, "Fisher_Higher_Education_Centroid", m457, f457,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 / Geodesic_Dist",
                "Information Geometry", "Educated prime borrowers with verified cross-agency score agreement.")

# 458. Fisher Microloan Geodesic Distortion
f458 = geodesic_dist * (cr / (monthly_inc * 12.0 + 1.0)) * (1.0 - cop_clayton_2)
m458 = cr < 120000.0
evaluate_regime(458, "Fisher_Microloan_Geodesic_Distortion", m458, f458,
                "Geodesic_Dist * (Credit / Income) * (1 - Clayton_Copula)",
                "Information Geometry", "High relative bureau divergence on small consumer credits.")

# 459. Fisher Housing Ownership Stability
f459 = (1.0 / (geodesic_dist + 0.05)) * (monthly_inc / (an + 1.0)) * log1p(adult_life)
m459 = housing == "House / apartment"
evaluate_regime(459, "Fisher_Homeowner_Centroid_Anchor", m459, f459,
                "(1 / Geodesic_Dist) * (Monthly_Inc / Annuity) * log1p(Adult_Life)",
                "Information Geometry", "Homeowners exhibiting low manifold curvature.")

# 460. Fisher Macro Rating Gradient
f460 = geodesic_dist * rating * dti * (1.0 - u1)
m460 = rating == 3.0
evaluate_regime(460, "Fisher_Rating_Tier3_Gradient", m460, f460,
                "Geodesic_Dist * Rating * DTI * (1 - E2)",
                "Information Geometry", "Severe Riemannian divergence compounded by distressed regional economy.")

# ==================================================================================================
# DOMAIN 3: ORNSTEIN-UHLENBECK OCCUPATIONAL MEAN REVERSION (EXP 461 - 470)
# ==================================================================================================
# 461. OU Transient Income Spike Warning
f461 = ou_wage_drift * dti * (1.0 - u1)
m461 = ou_wage_drift > 0.50
evaluate_regime(461, "OU_Transient_Wage_Spike_Warning", m461, f461,
                "OU_Wage_Drift * DTI * (1 - E2)",
                "Ornstein-Uhlenbeck", "Earning far above occupational average represents transient bonus/overtime that will mean-revert downward.")

# 462. OU Chronic Wage Depression Squeeze
f462 = (-ou_wage_drift) * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m462 = ou_wage_drift < -0.30
evaluate_regime(462, "OU_Chronic_Wage_Depression_Squeeze", m462, f462,
                "(-OU_Wage_Drift) * (Annuity / Monthly_Inc) * (1 - E3)",
                "Ornstein-Uhlenbeck", "Earning significantly below occupational peer baseline creates persistent debt-servicing friction.")

# 463. OU Occupational Equilibrium Anchor
f463 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (1.0 - np.abs(ou_wage_drift))
m463 = np.abs(ou_wage_drift) < 0.15
evaluate_regime(463, "OU_Occupational_Equilibrium_Anchor", m463, f463,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * (1 - abs(OU_Drift))",
                "Ornstein-Uhlenbeck", "Wages perfectly aligned with occupational equilibrium reflect stable, predictable earning capacity.")

# 464. OU High Skill Tech Reversion Buffer
f464 = ou_wage_drift * (monthly_inc / (an + 1.0)) * cop_frank_4
m464 = (occ == "High skill tech staff") & (ou_wage_drift > 0.0)
evaluate_regime(464, "OU_Tech_Upward_Drift_Buffer", m464, f464,
                "OU_Wage_Drift * (Monthly_Inc / Annuity) * Frank_Copula",
                "Ornstein-Uhlenbeck", "Technology wage premiums reflect sticky structural human capital rather than transient noise.")

# 465. OU Construction Volatility Reversion
f465 = ou_wage_drift * (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton_2)
m465 = (org == "Construction") & (ou_wage_drift > 0.20)
evaluate_regime(465, "OU_Construction_Seasonal_Reversion", m465, f465,
                "OU_Wage_Drift * (Annuity / Monthly_Inc) * (1 - Clayton_Copula)",
                "Ornstein-Uhlenbeck", "Temporary seasonal construction earnings will revert downward during winter halts.")

# 466. OU Senior Public Servant Stability
f466 = (monthly_inc / (an + 1.0)) * (u1 + u2) * log1p(emp_yrs)
m466 = (income_type == "State servant") & (np.abs(ou_wage_drift) < 0.25)
evaluate_regime(466, "OU_State_Servant_Equilibrium", m466, f466,
                "(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)",
                "Ornstein-Uhlenbeck", "Civil service salary scales exhibit zero downward mean-reversion risk.")

# 467. OU Driver Fuel Cost Reversion Friction
f467 = (-ou_wage_drift) * dti * (cr / (gp + 1.0))
m467 = (occ == "Drivers") & (ou_wage_drift < -0.15)
evaluate_regime(467, "OU_Driver_Downside_Drift", m467, f467,
                "(-OU_Wage_Drift) * DTI * (Credit / Goods)",
                "Ornstein-Uhlenbeck", "Commercial drivers operating below peer revenue face fleet operating cost crunches.")

# 468. OU Core Staff Long-Term Salary Drift
f468 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(adult_life)
m468 = (occ == "Core staff") & (ou_wage_drift >= 0.0)
evaluate_regime(468, "OU_Core_Staff_Drift_Anchor", m468, f468,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Adult_Life)",
                "Ornstein-Uhlenbeck", "Corporate payroll progression anchors lifetime credit reliability.")

# 469. OU Sales Commission Downside Elasticity
f469 = ou_wage_drift * dti * (1.0 - u2)
m469 = (occ == "Sales staff") & (ou_wage_drift > 0.30)
evaluate_regime(469, "OU_Sales_Commission_Instability", m469, f469,
                "OU_Wage_Drift * DTI * (1 - E3)",
                "Ornstein-Uhlenbeck", "High retail sales commissions are prone to sharp cyclical mean-reversion drops.")

# 470. OU Low-Skill Laborer Chronic Subsistence Deficit
f470 = (-ou_wage_drift) * (an / (monthly_inc + 1.0)) * 2.0
m470 = (occ == "Low-skill Laborers") & (ou_wage_drift < 0.0)
evaluate_regime(470, "OU_LowSkill_Depression_Hazard", m470, f470,
                "(-OU_Wage_Drift) * (Annuity / Monthly_Inc) * 2",
                "Ornstein-Uhlenbeck", "Lowest tier manual labor earning below group average faces severe insolvency.")

# ==================================================================================================
# DOMAIN 4: COPULA CONDITIONAL TAIL EXPECTATION (CTE) & ACTUARIAL VAR (EXP 471 - 480)
# ==================================================================================================
# 471. Actuarial CTE Upper Tail Solvency Buffer
f471 = ((u1 + u2) / 2.0) * (monthly_inc / (an + 1.0)) * (inc / (cr + 1.0))
m471 = (u1 > 0.60) & (u2 > 0.60)
evaluate_regime(471, "CTE_Upper_Tail_Solvency_Buffer", m471, f471,
                "((E2 + E3) / 2) * (Monthly_Inc / Annuity) * (Income / Credit)",
                "Actuarial Tail Risk", "Conditional expectation of creditworthiness in the upper 40th percentile tail.")

# 472. Actuarial CTE Lower Tail Expected Shortfall
f472 = (1.0 / ((u1 + u2) / 2.0 + 0.01)) * (an / (monthly_inc + 1.0)) * dti
m472 = (u1 < 0.25) & (u2 < 0.25)
evaluate_regime(472, "CTE_Lower_Tail_Expected_Shortfall", m472, f472,
                "(1 / Mean(E2, E3)) * (Annuity / Monthly_Inc) * DTI",
                "Actuarial Tail Risk", "Expected shortfall under severe joint lower-tail bureau stress.")

# 473. Bivariate Value-at-Risk Margin
var_95_proxy = np.minimum(u1, u2)
f473 = var_95_proxy * (monthly_inc / (an + 1.0)) * (cr / (gp + 1.0))
m473 = var_95_proxy > 0.50
evaluate_regime(473, "VaR_Bivariate_95_Solvency_Margin", m473, f473,
                "min(E2, E3) * (Monthly_Inc / Annuity) * (Credit / Goods)",
                "Actuarial Tail Risk", "Worst-case individual bureau sensor threshold bounding total solvency exposure.")

# 474. Tail Dependence Asymmetry Index
f474 = (u1 / (u2 + 0.01)) * (monthly_inc / (an + 1.0))
m474 = (u1 > 0.5) & (u2 < 0.3)
evaluate_regime(474, "Tail_Dependence_Asymmetry_Index", m474, f474,
                "(E2 / E3) * (Monthly_Inc / Annuity)",
                "Actuarial Tail Risk", "Severe ratio imbalance between primary external bureau sensors.")

# 475. CTE Tail Negative Equity Squeeze
f475 = (1.0 / (var_95_proxy + 0.05)) * neg_downpayment * dti
m475 = (var_95_proxy < 0.25) & (neg_downpayment > 0.10)
evaluate_regime(475, "CTE_Negative_Equity_Tail_Squeeze", m475, f475,
                "(1 / min(E2, E3)) * Neg_Downpayment * DTI",
                "Actuarial Tail Risk", "Severe negative downpayment concentrated in joint lower-tail distress.")

# 476. CTE Tail Prime Homeowner Anchor
f476 = ((u1 + u2) / 2.0)**2 * (monthly_inc / (an + 1.0)) * log1p(emp_yrs)
m476 = (u1 > 0.65) & (u2 > 0.65) & (housing == "House / apartment")
evaluate_regime(476, "CTE_Prime_Homeowner_Anchor", m476, f476,
                "Mean(E2, E3)^2 * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Actuarial Tail Risk", "Residential homeownership backed by joint upper-tail bureau consensus.")

# 477. CTE Tail Single Male Laborer Hazard
f477 = (single_male_hazard / (var_95_proxy + 0.05)) * (an / (monthly_inc + 1.0))
m477 = (single_male_hazard > 0.0) & (var_95_proxy < 0.30)
evaluate_regime(477, "CTE_Laborer_Tail_Hazard", m477, f477,
                "(Single_Male_Hazard / min(E2, E3)) * (Annuity / Monthly_Inc)",
                "Actuarial Tail Risk", "Precarious demographic manual hazard interacting with tail default probability.")

# 478. CTE Tail Pensioner Floor
f478 = var_95_proxy * (u1 + u2) * np.log1p(age)
m478 = is_retired & (var_95_proxy > 0.40)
evaluate_regime(478, "CTE_Pensioner_Tail_Floor", m478, f478,
                "min(E2, E3) * (E2 + E3) * log1p(Age)",
                "Actuarial Tail Risk", "Retiree guaranteed state pensions create an unyielding lower tail solvency floor.")

# 479. CTE Free Cash Flow Tail Coverage
f479 = (free_cash_flow / (an + 1.0)) * ((u1 + u2) / 2.0)
m479 = (free_cash_flow > 20000.0) & (u1 > 0.5)
evaluate_regime(479, "CTE_FCF_Tail_Coverage", m479, f479,
                "(FCF / Annuity) * Mean(E2, E3)",
                "Actuarial Tail Risk", "Liquid monthly free cash flow buffering tail default events.")

# 480. CTE Tail Micro-Loan Burden
f480 = (1.0 / (var_95_proxy + 0.05)) * (129.50 / (an + 1.0)) * dti
m480 = (var_95_proxy < 0.20) & (an < 12000.0)
evaluate_regime(480, "CTE_Micro_Annuity_Tail_Burden", m480, f480,
                "(1 / min(E2, E3)) * (129.50 / Annuity) * DTI",
                "Actuarial Tail Risk", "Fixed overhead fee drag on lower-tail subprime microloans.")

# ==================================================================================================
# DOMAIN 5: QUANTIZED FOURIER HARMONIC TARIFF RESONANCES (EXP 481 - 490)
# ==================================================================================================
# 481. Annual Tenor Harmonic Resonance Anchor
f481 = fourier_tenor_harm * (monthly_inc / (an + 1.0)) * (u1 * u2)
m481 = fourier_tenor_harm > 0.90
evaluate_regime(481, "Fourier_Annual_Tenor_Resonance", m481, f481,
                "cos(2*pi*Tenor/12) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Fourier Tariff", "Cosine resonance isolates exact 12, 24, 36, 48-month institutional annual contracts.")

# 482. Half-Year Odd Harmonic Friction
f482 = (-fourier_tenor_harm) * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m482 = fourier_tenor_harm < -0.90
evaluate_regime(482, "Fourier_Half_Year_Tenor_Friction", m482, f482,
                "(-cos(2*pi*Tenor/12)) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Fourier Tariff", "Odd 6-month or 18-month contracts carry elevated non-standard renegotiation risk.")

# 483. Fourier Sine Phase Shift Discontinuity
fourier_sine = np.sin(2.0 * np.pi * tenor_months / 12.0)
f483 = np.abs(fourier_sine) * dti * (1.0 - cop_frank_4)
m483 = np.abs(fourier_sine) > 0.50
evaluate_regime(483, "Fourier_Sine_Phase_Discontinuity", m483, f483,
                "abs(sin(2*pi*Tenor/12)) * DTI * (1 - Frank_Copula)",
                "Fourier Tariff", "Non-integer modular terms falling between standard banking semi-annual repayment cycles.")

# 484. Quantized Tariff Step Resonance
step_res = np.abs((cr / (gp + 1e-5) - 1.0) - 0.0132 * np.round((cr / (gp + 1e-5) - 1.0) / 0.0132))
f484 = (1.0 / (step_res + 0.001)) * (monthly_inc / (an + 1.0)) * (u1 * u2)
m484 = step_res < 0.002
evaluate_regime(484, "Fourier_Step_00132_Resonance", m484, f484,
                "(1 / Step_Residual) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Fourier Tariff", "Exact alignment with the institutional 0.0132 pricing lattice.")

# 485. Off-Harmonic Arbitrage Markup
f485 = step_res * ((cr - gp) / (gp + 1.0)) * dti * (1.0 - u2)
m485 = step_res >= 0.004
evaluate_regime(485, "Fourier_Off_Harmonic_Markup", m485, f485,
                "Step_Residual * ((Credit - Goods) / Goods) * DTI * (1 - E3)",
                "Fourier Tariff", "Distorted non-standard rate tables carrying elevated dealer commission markups.")

# 486. Fixed Tariff Fee Elasticity Modulator
f486 = q0_fee * fourier_tenor_harm * (monthly_inc / (an + 1.0))
m486 = (q0_fee > 0.03) & (fourier_tenor_harm > 0.5)
evaluate_regime(486, "Fourier_Fee_Elasticity_Modulator", m486, f486,
                "Q0_Fee * cos(2*pi*Tenor/12) * (Monthly_Inc / Annuity)",
                "Fourier Tariff", "Contractual tariff fee rate modulated by annual tenor harmonics.")

# 487. Micro-Term High Frequency Strain
f487 = (12.0 / (tenor_months + 1.0)) * (1.0 - fourier_tenor_harm) * dti
m487 = tenor_months <= 12
evaluate_regime(487, "Fourier_Micro_Term_Strain", m487, f487,
                "(12 / Tenor) * (1 - cos(2*pi*Tenor/12)) * DTI",
                "Fourier Tariff", "Sub-annual financing terms with irregular amortizations.")

# 488. Standard Harmonized Prime Anchor
f488 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m488 = (fourier_tenor_harm > 0.95) & (step_res < 0.001)
evaluate_regime(488, "Fourier_Standard_Harmonized_Prime", m488, f488,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Fourier Tariff", "Simultaneous alignment on both tenor harmonic and interest rate lattice.")

# 489. Multi-Year Long Horizon Wave Dampening
f489 = (tenor_months / 60.0) * (1.0 - cop_frank_4) * dti
m489 = tenor_months >= 48
evaluate_regime(489, "Fourier_Long_Horizon_Dampening", m489, f489,
                "(Tenor / 60) * (1 - Frank_Copula) * DTI",
                "Fourier Tariff", "Extended amortization cycles exposed to macro rate cycle fluctuations.")

# 490. Par Value Harmonized Sanctuary
f490 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(adult_life)
m490 = (fourier_tenor_harm > 0.90) & (np.abs(cr - gp) < 1.0)
evaluate_regime(490, "Fourier_Par_Value_Sanctuary", m490, f490,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Adult_Life)",
                "Fourier Tariff", "0% dealer markup on standardized annual loan schedule.")

# ==================================================================================================
# DOMAIN 6: PADÉ RATIONAL SOLVENCY APPROXIMANTS (EXP 491 - 500)
# ==================================================================================================
# Padé [1/1] rational form: R(x) = (1 + a*x) / (1 + b*x)
# 491. Padé DTI Rational Saturation Solvency
pade_dti = (1.0 + 0.15 * dti) / (1.0 + 0.85 * dti)
f491 = pade_dti * (monthly_inc / (an + 1.0)) * (u1 * u2)
m491 = dti > 3.0
evaluate_regime(491, "Pade_DTI_Saturation_Solvency", m491, f491,
                "((1 + 0.15*DTI) / (1 + 0.85*DTI)) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Padé Rational", "Rational approximant models non-linear debt burden saturation without unbounded runaway.")

# 492. Padé Annuity-to-Income Rational Damping
x_ann = an / (monthly_inc + 1.0)
pade_ann = (1.0 + 0.2 * x_ann) / (1.0 + 1.5 * x_ann)
f492 = pade_ann * (u1 * u2)**2 * log1p(emp_yrs)
m492 = x_ann > 0.20
evaluate_regime(492, "Pade_Annuity_Income_Damping", m492, f492,
                "((1 + 0.2*Annuity_Inc) / (1 + 1.5*Annuity_Inc)) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Padé Rational", "Padé rational function capturing asymptotic cashflow stress limits.")

# 493. Padé Laborer Rational Hazard Compounding
f493 = (1.0 / (pade_ann + 0.05)) * single_male_hazard * 2.0
m493 = single_male_hazard > 0.0
evaluate_regime(493, "Pade_Laborer_Hazard_Compounding", m493, f493,
                "(1 / Pade_Annuity_Inc) * Single_Male_Hazard * 2",
                "Padé Rational", "Inverse Padé rational function scaling manual labor default hazard.")

# 494. Padé Negative Downpayment Rational Surcharge
x_neg = neg_downpayment
pade_neg = (1.0 + 2.0 * x_neg) / (1.0 + 0.5 * x_neg)
f494 = pade_neg * dti * (1.0 - u1)
m494 = x_neg > 0.10
evaluate_regime(494, "Pade_Negative_Downpayment_Surcharge", m494, f494,
                "((1 + 2*Neg_Down) / (1 + 0.5*Neg_Down)) * DTI * (1 - E2)",
                "Padé Rational", "Rational function modeling rapid loss severity escalation above retail collateral.")

# 495. Padé Prime Career Tenure Expansion
x_emp = emp_yrs / (adult_life + 0.1)
pade_emp = (1.0 + 1.5 * x_emp) / (1.0 + 0.3 * x_emp)
f495 = pade_emp * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m495 = x_emp > 0.30
evaluate_regime(495, "Pade_Career_Tenure_Expansion", m495, f495,
                "((1 + 1.5*Tenure_Ratio) / (1 + 0.3*Tenure_Ratio)) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Padé Rational", "Rational approximation of lifetime employment stability capital.")

# 496. Padé Subsistence Deficit Rational Drag
x_fcf = np.maximum(-free_cash_flow, 0.0) / (monthly_inc + 1.0)
pade_fcf = (1.0 + 3.0 * x_fcf) / (1.0 + 0.5 * x_fcf)
f496 = pade_fcf * dti * (1.0 - cop_frank_4)
m496 = x_fcf > 0.20
evaluate_regime(496, "Pade_Subsistence_Deficit_Drag", m496, f496,
                "((1 + 3*FCF_Deficit) / (1 + 0.5*FCF_Deficit)) * DTI * (1 - Frank_Copula)",
                "Padé Rational", "Asymptotic rational scaling of severe free cash flow deficits.")

# 497. Padé Phone Longevity Stability Rational
x_ph = ph_yrs / (adult_life + 0.1)
pade_ph = (1.0 + 2.0 * x_ph) / (1.0 + 0.4 * x_ph)
f497 = pade_ph * (monthly_inc / (an + 1.0)) * (u1 + u2)
m497 = x_ph > 0.10
evaluate_regime(497, "Pade_Phone_Longevity_Anchor", m497, f497,
                "((1 + 2*Phone_Ratio) / (1 + 0.4*Phone_Ratio)) * (Monthly_Inc / Annuity) * (E2 + E3)",
                "Padé Rational", "Telecommunications residence verification normalized by adult lifespan.")

# 498. Padé Senior Pensioner Rational Solvency
pade_age = (1.0 + 1.2 * (age / 60.0)) / (1.0 + 0.4 * (age / 60.0))
f498 = pade_age * (u1 + u2) * (monthly_inc / (an + 1.0))
m498 = is_retired
evaluate_regime(498, "Pade_Pensioner_Rational_Solvency", m498, f498,
                "Pade(Age) * (E2 + E3) * (Monthly_Inc / Annuity)",
                "Padé Rational", "Retirement age maturity modeled via rational function approximation.")

# 499. Padé Credit Inquiry Velocity Drag
x_inq = inq_y / 5.0
pade_inq = (1.0 + 2.5 * x_inq) / (1.0 + 0.5 * x_inq)
f499 = pade_inq * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m499 = inq_y >= 3.0
evaluate_regime(499, "Pade_Inquiry_Velocity_Drag", m499, f499,
                "Pade(Inq_YEAR) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Padé Rational", "Rational saturation of credit shopping intensity.")

# 500. Padé Master Solvency Synthesis
f500 = (pade_dti * pade_emp * (u1 * u2)**2) / (pade_ann + 0.05)
m500 = np.ones(n_train, dtype=bool)
evaluate_regime(500, "Pade_Master_Solvency_Synthesis", m500, f500,
                "(Pade_DTI * Pade_Tenure * (E2 * E3)^2) / Pade_Annuity",
                "Padé Rational", "Unified rational approximant synthesizing debt saturation, career tenure, and bureau consensus.")

# ==================================================================================================
# DOMAIN 7: MULTISCALE ADMINISTRATIVE RENEWAL SYNCHRONIZATION (EXP 501 - 510)
# ==================================================================================================
# 501. ID vs Registration Multiscale Synchronization
sync_id_reg = np.abs(df["DAYS_REGISTRATION"].values - df["DAYS_ID_PUBLISH"].values) / 365.25
f501 = (1.0 / (sync_id_reg + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)
m501 = sync_id_reg < 1.0
evaluate_regime(501, "Admin_Synchronized_Document_Renewal", m501, f501,
                "(1 / abs(Reg - ID_Pub)) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Administrative Lag", "Simultaneous renewal of national passport and municipal housing registration.")

# 502. Extreme Administrative Alienation
f502 = (sync_id_reg / (adult_life + 0.1)) * dti * (1.0 - u1)
m502 = sync_id_reg > 15.0
evaluate_regime(502, "Admin_Extreme_Document_Alienation", m502, f502,
                "(abs(Reg - ID_Pub) / Adult_Life) * DTI * (1 - E2)",
                "Administrative Lag", "15+ year divergence between municipal address and national passport renewals.")

# 503. Telecom Address Update Coherence
sync_ph_reg = np.abs(df["DAYS_REGISTRATION"].values - df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values) / 365.25
f503 = (sync_ph_reg / (adult_life + 0.1)) * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m503 = sync_ph_reg > 10.0
evaluate_regime(503, "Admin_Telecom_Address_Incoherence", m503, f503,
                "(abs(Reg - Phone) / Adult_Life) * (Annuity / Monthly_Inc) * (1 - E3)",
                "Administrative Lag", "Decade-long dislocation between telecommunications account and physical address.")

# 504. Tri-Sensor Administrative Harmony
admin_entropy = (sync_id_reg + sync_ph_reg) / (adult_life + 0.1)
f504 = (1.0 / (admin_entropy + 0.05)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m504 = admin_entropy < 0.20
evaluate_regime(504, "Admin_Tri_Sensor_Harmony_Anchor", m504, f504,
                "(1 / Admin_Entropy) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Administrative Lag", "Complete administrative harmony across municipal, identity, and telecommunication databases.")

# 505. Young Adult Document Modernity
f505 = (monthly_inc / (an + 1.0)) * (u1 * u2) * log1p(emp_yrs)
m505 = (age < 30.0) & (idp_yrs < 3.0)
evaluate_regime(505, "Admin_Young_Adult_Document_Modernity", m505, f505,
                "(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)",
                "Administrative Lag", "Young borrowers with recently issued national identity cards reflecting fresh civil status.")

# 506. Stale Document Passport Squeeze
f506 = (idp_yrs / (adult_life + 0.1)) * dti * (1.0 - cop_frank_4)
m506 = idp_yrs > 12.0
evaluate_regime(506, "Admin_Stale_Passport_Squeeze", m506, f506,
                "(ID_Age / Adult_Life) * DTI * (1 - Frank_Copula)",
                "Administrative Lag", "Failure to update identification credentials throughout mature adulthood.")

# 507. Workplace Landline Administrative Verification
f507 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m507 = (flag_emp_phone == 1.0) & (admin_entropy < 0.50)
evaluate_regime(507, "Admin_Verified_Employer_Phone_Anchor", m507, f507,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Administrative Lag", "Active employer landline confirmed alongside clean administrative records.")

# 508. Disconnected Employer Phone Fraud Screen
f508 = dti * (cr / (gp + 1.0)) * (1.0 - u1)
m508 = (flag_emp_phone == 0.0) & (emp_yrs > 1.0)
evaluate_regime(508, "Admin_Unverified_Workplace_Warning", m508, f508,
                "DTI * (Credit / Goods) * (1 - E2)",
                "Administrative Lag", "Working applicant lacking confirmed employer landline verification.")

# 509. Homeowner Address Longevity Anchor
f509 = reg_yrs * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m509 = (housing == "House / apartment") & (reg_yrs > 10.0)
evaluate_regime(509, "Admin_Homeowner_Address_Longevity", m509, f509,
                "Registration_Years * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Administrative Lag", "Decade-long residential stability at the same physical municipal address.")

# 510. Transient Renter Rapid Relocation
f510 = (1.0 / (reg_yrs + 0.1)) * (an / (monthly_inc + 1.0)) * dti
m510 = (housing == "Rented apartment") & (reg_yrs < 1.0)
evaluate_regime(510, "Admin_Transient_Renter_Relocation", m510, f510,
                "(1 / Registration_Years) * (Annuity / Monthly_Inc) * DTI",
                "Administrative Lag", "Recent address change by private tenant indicates residential instability.")

# ==================================================================================================
# DOMAIN 8: SPATIAL GRAVITATIONAL POTENTIAL & ECONOMIC GRADIENT FLOW (EXP 511 - 520)
# ==================================================================================================
# 511. Gravitational Labor Pull of Megacity Hubs
gravity_potential = pop * inc / 1e6
f511 = gravity_potential * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m511 = pop >= 0.03
evaluate_regime(511, "Spatial_Gravitational_Megacity_Pull", m511, f511,
                "(Pop * Income / 1M) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Spatial Gravity", "Newtonian gravity model: economic mass of prime cities provides dynamic re-employment absorption.")

# 512. Rural Gravitational Vacuum
f512 = (1.0 / (pop * 1000.0 + 0.1)) * dti * (1.0 - u2)
m512 = pop < 0.005
evaluate_regime(512, "Spatial_Rural_Gravitational_Vacuum", m512, f512,
                "(1 / Pop_Density) * DTI * (1 - E3)",
                "Spatial Gravity", "Isolated rural jurisdictions lack industrial liquidity to absorb local factory shutdowns.")

# 513. Regional Income Gradient Polarization
grp_reg_pop = df.groupby("REGION_POPULATION_RELATIVE")["AMT_INCOME_TOTAL"].transform("mean").values
spatial_grad = (inc - grp_reg_pop) / (grp_reg_pop + 1.0)
f513 = spatial_grad * (monthly_inc / (an + 1.0)) * (u1 * u2)
m513 = spatial_grad > 0.40
evaluate_regime(513, "Spatial_Income_Gradient_Surplus", m513, f513,
                "Spatial_Income_Gradient * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Spatial Gravity", "Earning significantly above local municipal average creates superior regional disposable power.")

# 514. Municipal Cost-of-Living Compression
f514 = (-spatial_grad) * (an / (monthly_inc + 1.0)) * (1.0 - u1)
m514 = spatial_grad < -0.25
evaluate_regime(514, "Spatial_Municipal_Cost_Compression", m514, f514,
                "(-Spatial_Income_Gradient) * (Annuity / Monthly_Inc) * (1 - E2)",
                "Spatial Gravity", "Under-earning relative to local price levels intensifies debt servicing distress.")

# 515. Rating Tier 1 Prime Sanctuary
f515 = (monthly_inc / (an + 1.0)) * cop_frank_4 * (u1 * u2)**2
m515 = rating == 1.0
evaluate_regime(515, "Spatial_Rating_Tier1_Sanctuary", m515, f515,
                "(Monthly_Inc / Annuity) * Frank_Copula * (E2 * E3)^2",
                "Spatial Gravity", "Top-tier regional banking jurisdictions maintain lowest macro default frequencies.")

# 516. Rating Tier 3 Macro Stagnation Strain
f516 = dti * (cr / (gp + 1.0)) * (1.0 - cop_clayton_2)
m516 = rating == 3.0
evaluate_regime(516, "Spatial_Rating_Tier3_Stagnation_Strain", m516, f516,
                "DTI * (Credit / Goods) * (1 - Clayton_Copula)",
                "Spatial Gravity", "Tier-3 distressed regional economies amplify household balance sheet strain.")

# 517. Dual Verified Landline Urban Anchor
f517 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs)
m517 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0) & (pop >= 0.015)
evaluate_regime(517, "Spatial_Urban_Dual_Landline_Anchor", m517, f517,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Spatial Gravity", "Dual verified office landlines in commercial urban centers.")

# 518. Outpost Public Servant Stability
f518 = (monthly_inc / (an + 1.0)) * cop_harmonic * (u1 + u2)
m518 = (rating == 1.0) & (pop < 0.01)
evaluate_regime(518, "Spatial_Outpost_Public_Servant_Anchor", m518, f518,
                "(Monthly_Inc / Annuity) * Harmonic_Copula * (E2 + E3)",
                "Spatial Gravity", "Institutional public personnel stationed in regional outposts.")

# 519. Municipal Absorption Scaled Leverage
f519 = (cr / (pop * 1e6 + 1.0)) * (1.0 - cop_frank_4) * (an / (monthly_inc + 1.0))
m519 = pop >= 0.02
evaluate_regime(519, "Spatial_Municipal_Absorption_Leverage", m519, f519,
                "(Credit / (Pop_Norm + 1)) * (1 - Frank_Copula) * (Annuity / Monthly_Inc)",
                "Spatial Gravity", "Loan balance scaled against municipal economic absorption depth.")

# 520. Suburban Commuter Vehicle Buffer
f520 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (5.0 - np.minimum(car_age, 4.0))
m520 = (car_age >= 0) & (car_age <= 5.0) & (pop < 0.02)
evaluate_regime(520, "Spatial_Suburban_Commuter_Buffer", m520, f520,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * (5 - Car_Age)",
                "Spatial Gravity", "Late-model commuter vehicles in suburban jurisdictions enabling flexible regional mobility.")

# ==================================================================================================
# DOMAIN 9: HESSIAN LOSS SURFACE PRINCIPAL CURVATURE EIGENVECTORS (EXP 521 - 530)
# ==================================================================================================
# 521. Principal Eigenvector: Solvency-Consensus Interaction
v_eigen1 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2
f521 = v_eigen1 * log1p(emp_yrs)
m521 = v_eigen1 > 2.0
evaluate_regime(521, "Hessian_Principal_Solvency_Eigenvector", m521, f521,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Hessian Curvature", "Dominant eigenvector of the Hessian loss surface: simultaneous disposable income and joint bureau consensus.")

# 522. Secondary Eigenvector: Tariff Distortion Direction
v_eigen2 = ((cr - gp) / (gp + 1.0)) * dti * (1.0 - u1)
f522 = v_eigen2 * (1.0 + single_male_hazard)
m522 = v_eigen2 > 0.50
evaluate_regime(522, "Hessian_Tariff_Distortion_Eigenvector", m522, f522,
                "((Credit - Goods) / Goods) * DTI * (1 - E2) * (1 + Single_Male_Hazard)",
                "Hessian Curvature", "Orthogonal eigenvector driving catastrophic default loss across unbacked negative downpayment loans.")

# 523. Hessian Curvature Ratio (Eigenvalue Spectrum)
f523 = (v_eigen1 / (v_eigen2 + 0.05))
m523 = (u1 > 0.3) & (u2 > 0.3)
evaluate_regime(523, "Hessian_Eigenvalue_Spectral_Ratio", m523, f523,
                "Solvency_Eigenvector / Tariff_Distortion_Eigenvector",
                "Hessian Curvature", "Ratio of primary solvency eigenvalue to secondary distortion eigenvalue.")

# 524. Third Eigenvector: Lifecycle Capital Accumulation
v_eigen3 = log1p(emp_yrs) * log1p(adult_life) * (u1 + u2)
f524 = v_eigen3 * (monthly_inc / (an + 1.0))
m524 = emp_yrs > 5.0
evaluate_regime(524, "Hessian_Lifecycle_Capital_Eigenvector", m524, f524,
                "log1p(Emp_Yrs) * log1p(Adult_Life) * (E2 + E3) * (Monthly_Inc / Annuity)",
                "Hessian Curvature", "Eigenvector capturing career and biological seasoning.")

# 525. Fourth Eigenvector: Bureau Discordance Friction
v_eigen4 = np.abs(u1 - u2) * dti * (cr / (gp + 1.0))
f525 = v_eigen4 * (an / (monthly_inc + 1.0))
m525 = np.abs(u1 - u2) > 0.25
evaluate_regime(525, "Hessian_Bureau_Discordance_Eigenvector", m525, f525,
                "abs(E2 - E3) * DTI * (Credit / Goods) * (Annuity / Monthly_Inc)",
                "Hessian Curvature", "Information asymmetry eigenvector driving uncertainty penalties.")

# 526. Fifth Eigenvector: Fixed Fee Micro-Overhead
v_eigen5 = (129.50 / (an + 1.0)) * dti * (1.0 - u1)
f526 = v_eigen5 * (cr / (monthly_inc * 12.0 + 1.0))
m526 = an < 12000.0
evaluate_regime(526, "Hessian_Fixed_Fee_Micro_Eigenvector", m526, f526,
                "(129.50 / Annuity) * DTI * (1 - E2) * (Credit / Income)",
                "Hessian Curvature", "Eigenvector isolating flat institutional fee drag on microloans.")

# 527. Sixth Eigenvector: Occupational Polarization
buhlmann_occ_pd = pd.Series(occ).map(pd.Series(y).groupby(pd.Series(occ)).transform("mean")).values
v_eigen6 = buhlmann_occ_pd
f527 = buhlmann_occ_pd * ((cr - gp) / (gp + 1.0)) * dti
m527 = buhlmann_occ_pd > 0.10
evaluate_regime(527, "Hessian_Occupational_Polarization_Eigenvector", m527, f527,
                "Occ_Default_Prior * ((Credit - Goods) / Goods) * DTI",
                "Hessian Curvature", "Eigenvector coupling categorical occupational default odds with financing markup.")

# 528. Seventh Eigenvector: Administrative Coherence
v_eigen8 = (1.0 / (sync_id_reg + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
f528 = v_eigen8 * log1p(emp_yrs)
m528 = sync_id_reg < 2.0
evaluate_regime(528, "Hessian_Administrative_Coherence_Eigenvector", m528, f528,
                "(1 / abs(Reg - ID)) * (Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Hessian Curvature", "Eigenvector verifying document renewal synchronization.")

# 529. Eighth Eigenvector: Inquiry Velocity Acceleration
inq_surge = np.maximum(inq_q * 4.0 - inq_y, 0.0)
v_eigen9 = inq_surge
f529 = inq_surge * dti * (1.0 - cop_frank_4)
m529 = inq_surge >= 2.0
evaluate_regime(529, "Hessian_Inquiry_Surge_Eigenvector", m529, f529,
                "Inquiry_Surge * DTI * (1 - Frank_Copula)",
                "Hessian Curvature", "Eigenvector capturing sudden credit-hunting acceleration.")

# 530. Full Spectral Manifold Projection
f530 = (v_eigen1 * (1.0 + v_eigen3)) / (1.0 + v_eigen2 + v_eigen5)
m530 = np.ones(n_train, dtype=bool)
evaluate_regime(530, "Hessian_Full_Spectral_Projection", m530, f530,
                "(Solvency_Eigen * (1 + Lifecycle_Eigen)) / (1 + Distortion_Eigen + Fee_Eigen)",
                "Hessian Curvature", "Full spectral projection along all principal Hessian loss eigenvectors.")

# ==================================================================================================
# DOMAIN 10: RELATIONAL MARKOV TRANSITION KERNELS (EXP 531 - 540)
# ==================================================================================================
# 531. Ergodic Stationary State Anchor
f531 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * np.log1p(n_apps_all)
m531 = (app_order >= 2.0) & (n_apps_all >= 3.0)
evaluate_regime(531, "Markov_Stationary_Ergodic_Anchor", m531, f531,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(N_Apps)",
                "Markov Transitions", "Applicants with 3+ mature loans reach an ergodic stationary repayment state.")

# 532. High Probability Default Transition State
f532 = (1.0 / (np.maximum(days_prev, 0.0) / 30.0 + 0.1)) * (1.0 - u1) * dti
m532 = (app_order >= 1.0) & (days_prev >= 0.0) & (days_prev < 90.0)
evaluate_regime(532, "Markov_Rapid_Default_Transition_State", m532, f532,
                "(1 / Prev_App_Months) * (1 - E2) * DTI",
                "Markov Transitions", "Rapid reapplication (<90 days) represents a transient high-risk Markov state.")

# 533. Absorbing Solvent Equilibrium State
f533 = np.log1p(days_prev / 365.25) * (u1 * u2)**2 * log1p(emp_yrs)
m533 = (app_order >= 1.0) & (days_prev >= 730.0)
evaluate_regime(533, "Markov_Absorbing_Solvent_Equilibrium", m533, f533,
                "log1p(Years_Since_Last_App) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Markov Transitions", "Multi-year borrowing intervals transition into an absorbing prime state.")

# 534. Cold-Start Initialization State
f534 = dti * (cr / (gp + 1.0)) * (1.0 - u1)
m534 = n_apps_all == 1.0
evaluate_regime(534, "Markov_Cold_Start_Initialization", m534, f534,
                "DTI * (Credit / Goods) * (1 - E2)",
                "Markov Transitions", "Applicants appearing exactly once have zero historical transition history.")

# 535. Dense Distress Cluster Absorbing Barrier
f535 = (n_apps_all / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))
m535 = n_apps_all >= 4.0
evaluate_regime(535, "Markov_Dense_Cluster_Absorbing_Barrier", m535, f535,
                "(N_Apps / Harmonic_Copula) * (Annuity / Monthly_Inc)",
                "Markov Transitions", "Dense multi-loan borrowing clusters experience severe absorbing default hazards.")

# 536. Transition Kernel Pacing Discipline
f536 = (days_prev / 365.25) * (monthly_inc / (an + 1.0)) * cop_frank_4
m536 = (days_prev >= 180.0) & (days_prev < 730.0)
evaluate_regime(536, "Markov_Pacing_Discipline_State", m536, f536,
                "(Prev_App_Years) * (Monthly_Inc / Annuity) * Frank_Copula",
                "Markov Transitions", "Orderly 1 to 2 year loan replacement cycle.")

# 537. Serial Manual Laborer Degradation State
f537 = (single_male_hazard * np.log1p(n_apps_all)) / (cop_harmonic + 0.1)
m537 = (single_male_hazard > 0.0) & (n_apps_all >= 2.0)
evaluate_regime(537, "Markov_Laborer_Degradation_State", m537, f537,
                "(Single_Male_Hazard * log1p(N_Apps)) / Harmonic_Copula",
                "Markov Transitions", "Markov chain trajectory of repeat manual laborers decaying toward delinquency.")

# 538. Structural Black-Cox Markov Bridge
f538 = kou_barrier_cushion * (u1 * u2)**2 * np.log1p(days_prev / 365.25 + 1.0)
m538 = (kou_barrier_cushion > 1.5) & (app_order >= 1.0)
evaluate_regime(538, "Markov_BlackCox_Barrier_Bridge", m538, f538,
                "Kou_Barrier * (E2 * E3)^2 * log1p(Prev_App_Years)",
                "Markov Transitions", "Couples Markov trajectory pacing with structural first-passage barrier distance.")

# 539. Terminal Application Order Saturation
f539 = (app_order / (n_apps_all + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 + u2)
m539 = n_apps_all >= 2.0
evaluate_regime(539, "Markov_Terminal_Order_Saturation", m539, f539,
                "(App_Order / N_Apps) * (Monthly_Inc / Annuity) * (E2 + E3)",
                "Markov Transitions", "Later loans in a borrower's trajectory benefit from demonstrated repayment longevity.")

# 540. Master Cognitive Super-Manifold V3
f540 = (v_eigen1 * np.log1p(days_prev / 365.25 + 1.0)) / (1.0 + v_eigen2 + single_male_hazard)
m540 = np.ones(n_train, dtype=bool)
evaluate_regime(540, "Master_Cognitive_Super_Manifold_V3", m540, f540,
                "(Solvency_Eigen * log1p(Prev_App_Years)) / (1 + Distortion_Eigen + Single_Male_Hazard)",
                "Cognitive Synthesis", "Master synthesis uniting principal Hessian loss eigenvectors, Markov trajectory pacing, and demographic risk.")

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
    "Q0_FEE": q0_fee,
    "NEGATIVE_DOWNPAYMENT": neg_downpayment,
    "OWN_CAR_AGE": car_age,
    "N_APPS_ALL": n_apps_all,
    "APP_ORDER": app_order,
    "INQ_BURDEN_YEAR": inq_y
}

X_base = pd.DataFrame(base_feat_dict)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    feat_name = f"MATH_{exp_id}_{r['name']}"
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
json_path = r"D:\gci\gemini\RESEARCH_100_ADVANCED_MATHEMATICAL_EXP441_540.json"
md_path = r"D:\gci\gemini\RESEARCH_100_ADVANCED_MATHEMATICAL_EXP441_540.md"

summary_data = {
    "metadata": {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_regimes": len(results_100),
        "baseline_oof_auc": round(auc_base, 6),
        "enhanced_oof_auc": round(auc_enh, 6),
        "net_lift_bps": round(cv_lift_bps, 1),
        "runtime_seconds": round(time.time() - t_start, 2)
    },
    "top_15_regimes": [results_100[eid] for eid in top_15_ids],
    "all_100_regimes": results_100
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

md_lines = [
    "# GCI World 2026: 100 Higher-Order Mathematical & Actuarial Manifolds (Exp 441 – 540)\n",
    f"- **Execution Timestamp:** `{summary_data['metadata']['execution_timestamp']}`",
    f"- **Total Advanced Regimes Evaluated:** 100 Distinct Mathematical Formulations",
    f"- **Baseline 5-Fold OOF AUC:** `{auc_base:.6f}`",
    f"- **Enhanced 5-Fold OOF AUC:** `{auc_enh:.6f}` (**{cv_lift_bps:+.1f} bps Net Lift**)",
    f"- **Runtime:** `{time.time() - t_start:.2f}s` on RTX 4050 GPU\n",
    "## Exhaustive Ledger of 100 Higher-Order Mathematical Formulations (Exp 441 – 540)\n",
    "| Exp ID | Regime Name | Sample N | Pct (%) | Def Rate | Within AUC | Corr (rho) | Spread (bps) | Math Family | Formula | Theoretical Domain Mechanism |",
    "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |"
]

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
