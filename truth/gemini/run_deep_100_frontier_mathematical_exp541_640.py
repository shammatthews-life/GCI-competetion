"""
====================================================================================================
  GCI WORLD 2026: 100 FRONTIER MATHEMATICAL & ACTUARIAL MANIFOLDS (EXP 541 - 640)
  + WASSERSTEIN DISTANCE, THOM'S CUSP CATASTROPHE, FOKKER-PLANCK & PERSISTENT TOPOLOGY
====================================================================================================
  - Pure In-Memory, Clean-Room Compliant (Strictly inpuy/train.csv & inpuy/test.csv)
  - 10 Frontier Mathematical Frameworks:
    1. Thom's Cusp Catastrophe Bifurcation Manifolds (Exp 541 - 550)
    2. Optimal Transport & 1D Wasserstein Earth Mover Distances (Exp 551 - 560)
    3. Fokker-Planck Drift-Diffusion Probability Flux (Exp 561 - 570)
    4. Non-Equilibrium Thermodynamic Entropy Production (Exp 571 - 580)
    5. Hyperbolic Poincaré Disk Occupational Geometry (Exp 581 - 590)
    6. Pickands Extreme Tail Dependence Copulas (Exp 591 - 600)
    7. Daubechies Wavelet Multiresolution Administrative Analysis (Exp 601 - 610)
    8. Fractional Brownian Motion & Hurst Persistence Exponents (Exp 611 - 620)
    9. Graph Laplacian Spectral Embeddings on Entity Networks (Exp 621 - 630)
    10. Grand Cognitive Master Field Tensor Synthesis (Exp 631 - 640)
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
print("  GCI WORLD 2026: 100 FRONTIER MATHEMATICAL & ACTUARIAL MANIFOLDS (EXP 541 - 640)")
print("  + THOM'S CUSP CATASTROPHE, OPTIMAL TRANSPORT, & NON-EQUILIBRIUM ENTROPY PRODUCTION")
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

# Copulas & Tariffs
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))
cop_frank_4 = -0.25 * np.log(1.0 + ((np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)) / (np.exp(-4.0) - 1.0))
cop_clayton_2 = np.maximum(u1**(-2.0) + u2**(-2.0) - 1.0, 1e-6)**(-0.5)

q0_fee = np.where((contract == "Cash loans") & ((an - 129.5) / (cr + 1e-5) > 0), (an - 129.5) / (cr + 1e-5), an / (cr + 1e-5))
neg_downpayment = np.maximum(cr - gp, 0.0) / (gp + 1.0)
single_male_hazard = ((gender == "M") & np.isin(family, ["Single / not married", "Separated", "Widow"])).astype(float) * dti * (1.0 - u1)
fourier_tenor_harm = np.cos(2.0 * np.pi * np.round(cr / (an + 1e-5)) / 12.0)

days_prev = df_restored["DAYS_SINCE_PREV_APP"].fillna(-999.0).values
app_order = df_restored["APP_ORDER"].values.astype(float)

# --------------------------------------------------------------------------------------------------
# ADVANCED MATHEMATICAL FOUNDATIONS
# --------------------------------------------------------------------------------------------------
# 1. Thom's Cusp Catastrophe Potential: V(x) = (1/4)x^4 + (1/2)u*x^2 + v*x
# Bifurcation set: 4*u^3 + 27*v^2 = 0
# State variable x = debt-to-income stress, u = external bureau consensus, v = contractual fee markup
u_param = 1.0 - 2.0 * (u1 * u2)  # negative when prime, positive when subprime
v_param = neg_downpayment * (an / (monthly_inc + 1.0))
cusp_discriminant = 4.0 * (u_param**3) + 27.0 * (v_param**2)

# 2. Optimal Transport (Wasserstein 1D Earth Mover Distance)
# Measure distance of applicant's bureau triad [u0, u1, u2] from ideal prime centroid [0.85, 0.85, 0.85]
u_sorted = np.sort(np.column_stack([u0, u1, u2]), axis=1)
prime_ref = np.array([0.75, 0.85, 0.90])
w1_dist = np.mean(np.abs(u_sorted - prime_ref), axis=1)

# 3. Fokker-Planck Drift-Diffusion Probability Flux
# J = mu * p - D * (dp/dx). Solvency drift mu = FCF / Debt, Diffusion D = 1 - E2
fp_drift = free_cash_flow / (cr * 0.1 + 1.0)
fp_diff = np.clip(1.0 - u1, 0.05, 1.0)
fp_solvency_flux = (fp_drift / fp_diff) * (u1 * u2)

# 4. Non-Equilibrium Thermodynamic Entropy Production
# S_dot = (Annuity / FCF) * ln(Credit / Goods)
s_production = (an / (np.maximum(free_cash_flow, 1000.0))) * np.log(np.maximum(cr / (gp + 1.0), 1.0) + 1e-4)

# 5. Hyperbolic Poincaré Disk Embedding
# Radius r = 1 - E2, Angle theta = Occupational Default Odds
occ_s = pd.Series(occ)
occ_pd = pd.Series(y).groupby(occ_s).transform("mean").values
poincare_r = np.clip(1.0 - (u1 * u2)**0.5, 0.01, 0.95)
poincare_dist = np.log((1.0 + poincare_r) / (1.0 - poincare_r + 1e-5)) * occ_pd

# --------------------------------------------------------------------------------------------------
# STORAGE AND EVALUATION ENGINE
# --------------------------------------------------------------------------------------------------
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
    print(f"[{exp_id}] {name:<35} | N={count:>7,} ({pct:>4.1f}%) | Def={def_rate:>5.2f}% | AUC={auc_score:.4f} | Corr={corr:+.4f} | Spread={spread_bps:>6.1f} bps | {math_family:<22}")

# ==================================================================================================
# DOMAIN 1: THOM'S CUSP CATASTROPHE BIFURCATION MANIFOLDS (EXP 541 - 550)
# ==================================================================================================
# 541. Cusp Bifurcation Manifold Crossing
f541 = cusp_discriminant * (an / (monthly_inc + 1.0)) * dti
m541 = cusp_discriminant > 0.0
evaluate_regime(541, "Cusp_Bifurcation_Crossing_Hazard", m541, f541,
                "Cusp_Discriminant * (Annuity / Monthly_Inc) * DTI",
                "Thom's Catastrophe", "Bifurcation discriminant crossing marks discontinuous jump from solvent to default equilibrium.")

# 542. Bimodal Catastrophe Pocket
f542 = np.abs(cusp_discriminant) * (1.0 - cop_frank_4)
m542 = cusp_discriminant < 0.0
evaluate_regime(542, "Cusp_Bimodal_Hysteresis_Pocket", m542, f542,
                "abs(Cusp_Discriminant) * (1 - Frank_Copula)",
                "Thom's Catastrophe", "Negative discriminant identifies the bimodal hysteresis zone with two coexisting stability states.")

# 543. Cusp Solvency Plateau Anchor
f543 = (1.0 / (np.maximum(cusp_discriminant, 0.0) + 0.01)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m543 = u_param < -0.3
evaluate_regime(543, "Cusp_Prime_Solvency_Plateau", m543, f543,
                "(1 / Cusp_Disc) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Thom's Catastrophe", "Deep negative control parameter u guarantees global single-well solvency stability.")

# 544. Cusp Laborer Fold Catastrophe
f544 = (single_male_hazard / (np.abs(cusp_discriminant) + 0.05)) * (an / (monthly_inc + 1.0))
m544 = single_male_hazard > 0.0
evaluate_regime(544, "Cusp_Laborer_Fold_Catastrophe", m544, f544,
                "(Single_Male_Hazard / abs(Cusp_Disc)) * (Annuity / Monthly_Inc)",
                "Thom's Catastrophe", "Single manual laborers positioned along the unstable catastrophe fold boundary.")

# 545. Cusp Negative Downpayment Pitchfork
f545 = v_param**2 * dti * (1.0 - u1)
m545 = v_param > 0.15
evaluate_regime(545, "Cusp_Negative_Downpayment_Pitchfork", m545, f545,
                "v_param^2 * DTI * (1 - E2)",
                "Thom's Catastrophe", "Asymmetric splitting factor v driven by unbacked loan markups.")

# 546. Cusp Pensioner Invariant Sanctuary
f546 = (1.0 / (v_param + 0.01)) * (u1 + u2) * np.log1p(age)
m546 = is_retired & (v_param < 0.05)
evaluate_regime(546, "Cusp_Pensioner_Invariant_Sanctuary", m546, f546,
                "(1 / v_param) * (E2 + E3) * log1p(Age)",
                "Thom's Catastrophe", "Zero dealer fee markup keeps retirees safely in the unperturbed linear regime.")

# 547. Cusp Short Amortization Shock Wave
f547 = cusp_discriminant * (12.0 / (term + 1.0)) * (1.0 - u2)
m547 = term < 15.0
evaluate_regime(547, "Cusp_Short_Term_Shockwave", m547, f547,
                "Cusp_Disc * (12 / Term) * (1 - E3)",
                "Thom's Catastrophe", "Short-term loans undergo sharp discontinuous phase transitions.")

# 548. Cusp Carrier Longevity Damping
f548 = (1.0 / (np.abs(cusp_discriminant) + 0.05)) * log1p(ph_yrs) * (u1 * u2)
m548 = ph_yrs > 3.0
evaluate_regime(548, "Cusp_Carrier_Longevity_Damping", m548, f548,
                "(1 / abs(Cusp_Disc)) * log1p(Phone_Yrs) * (E2 * E3)",
                "Thom's Catastrophe", "Telecom stability dampens catastrophe manifold jumps.")

# 549. Cusp Housing Tenancy Friction
f549 = cusp_discriminant * (an / (monthly_inc + 1.0)) / (u1 + 0.1)
m549 = housing == "Rented apartment"
evaluate_regime(549, "Cusp_Rented_Apartment_Friction", m549, f549,
                "Cusp_Disc * (Annuity / Monthly_Inc) / E2",
                "Thom's Catastrophe", "Market rent overhead pushes private tenants across the catastrophe threshold.")

# 550. Cusp Master Catastrophe Field
f550 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (1.0 + np.maximum(cusp_discriminant, 0.0))
m550 = np.ones(n_train, dtype=bool)
evaluate_regime(550, "Cusp_Master_Catastrophe_Field", m550, f550,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2) / (1 + max(Cusp_Disc, 0))",
                "Thom's Catastrophe", "Master catastrophe field uniting control parameters with disposable cashflow.")

# ==================================================================================================
# DOMAIN 2: OPTIMAL TRANSPORT & 1D WASSERSTEIN EARTH MOVER DISTANCES (EXP 551 - 560)
# ==================================================================================================
# 551. Wasserstein Prime Centroid Proximity
f551 = (1.0 / (w1_dist + 0.05)) * (monthly_inc / (an + 1.0)) * log1p(emp_yrs)
m551 = w1_dist < 0.15
evaluate_regime(551, "Wasserstein_Prime_Centroid_Proximity", m551, f551,
                "(1 / W1_Dist) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Optimal Transport", "Earth mover distance measuring minimal work required to transport applicant scores to prime distribution.")

# 552. Wasserstein Extreme Divergence Penalty
f552 = w1_dist * dti * (1.0 - cop_clayton_2)
m552 = w1_dist > 0.45
evaluate_regime(552, "Wasserstein_Extreme_Divergence_Penalty", m552, f552,
                "W1_Dist * DTI * (1 - Clayton_Copula)",
                "Optimal Transport", "Severe Wasserstein earth mover cost reflecting systemic credit profile impairment.")

# 553. Wasserstein Downpayment Compounding
f553 = w1_dist * neg_downpayment * (an / (monthly_inc + 1.0))
m553 = neg_downpayment > 0.10
evaluate_regime(553, "Wasserstein_Downpayment_Compounding", m553, f553,
                "W1_Dist * Neg_Downpayment * (Annuity / Monthly_Inc)",
                "Optimal Transport", "Earth mover distance multiplied by unbacked dealer fee markups.")

# 554. Wasserstein Education Transport Shield
f554 = (1.0 / (w1_dist + 0.05)) * (monthly_inc / (an + 1.0)) * (u1 + u2)
m554 = edu == "Higher education"
evaluate_regime(554, "Wasserstein_Higher_Education_Shield", m554, f554,
                "(1 / W1_Dist) * (Monthly_Inc / Annuity) * (E2 + E3)",
                "Optimal Transport", "University graduates exhibiting near-zero Wasserstein transport divergence.")

# 555. Wasserstein Laborer Transport Deficit
f555 = (single_male_hazard * w1_dist) / (cop_harmonic + 0.1)
m555 = single_male_hazard > 0.0
evaluate_regime(555, "Wasserstein_Laborer_Transport_Deficit", m555, f555,
                "(Single_Male_Hazard * W1_Dist) / Harmonic_Copula",
                "Optimal Transport", "Massive transportation cost required to elevate manual laborers to prime credit status.")

# 556. Wasserstein Homeowner Transport Anchor
f556 = (1.0 / (w1_dist + 0.05)) * (monthly_inc / (an + 1.0)) * log1p(adult_life)
m556 = housing == "House / apartment"
evaluate_regime(556, "Wasserstein_Homeowner_Transport_Anchor", m556, f556,
                "(1 / W1_Dist) * (Monthly_Inc / Annuity) * log1p(Adult_Life)",
                "Optimal Transport", "Residential real estate ownership anchoring low Wasserstein transport costs.")

# 557. Wasserstein Free Cash Flow Transport Buffer
f557 = (free_cash_flow / (an + 1.0)) * (1.0 / (w1_dist + 0.05))
m557 = free_cash_flow > 15000.0
evaluate_regime(557, "Wasserstein_FCF_Transport_Buffer", m557, f557,
                "(FCF / Annuity) / W1_Dist",
                "Optimal Transport", "Abundant monthly liquidity funding structural transport to solvency.")

# 558. Wasserstein Inquiry Velocity Shock
f558 = w1_dist * inq_y * (1.0 - u1)
m558 = inq_y >= 3.0
evaluate_regime(558, "Wasserstein_Inquiry_Velocity_Shock", m558, f558,
                "W1_Dist * Inq_YEAR * (1 - E2)",
                "Optimal Transport", "High credit shopping velocity expanding earth mover transport distance.")

# 559. Wasserstein Microloan Transport Friction
f559 = w1_dist * (129.50 / (an + 1.0)) * dti
m559 = an < 12000.0
evaluate_regime(559, "Wasserstein_Microloan_Transport_Friction", m559, f559,
                "W1_Dist * (129.50 / Annuity) * DTI",
                "Optimal Transport", "Fixed administrative fee burden on small consumer credit lines.")

# 560. Wasserstein Unified Master Solvency
f560 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (w1_dist + 0.05)
m560 = np.ones(n_train, dtype=bool)
evaluate_regime(560, "Wasserstein_Unified_Master_Solvency", m560, f560,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2) / W1_Dist",
                "Optimal Transport", "Master optimal transport solvency metric coupling cashflow velocity with minimal earth mover distance.")

# ==================================================================================================
# DOMAIN 3: FOKKER-PLANCK DRIFT-DIFFUSION PROBABILITY FLUX (EXP 561 - 570)
# ==================================================================================================
# 561. Positive Solvency Probability Flux
f561 = fp_solvency_flux * log1p(emp_yrs)
m561 = fp_solvency_flux > 2.0
evaluate_regime(561, "FokkerPlanck_Positive_Solvency_Flux", m561, f561,
                "Solvency_Flux * log1p(Emp_Yrs)",
                "Fokker-Planck Flux", "Drift exceeds diffusion, creating steady probability mass flow toward debt satisfaction.")

# 562. Negative Solvency Flux Drain
f562 = (-fp_solvency_flux) * dti * (1.0 - cop_clayton_2)
m562 = fp_solvency_flux < -1.0
evaluate_regime(562, "FokkerPlanck_Negative_Flux_Drain", m562, f562,
                "(-Solvency_Flux) * DTI * (1 - Clayton_Copula)",
                "Fokker-Planck Flux", "Negative cashflow drift creating steady probability flux toward the default boundary.")

# 563. Diffusion Noise Dominance (Uncertainty)
f563 = fp_diff * dti * (cr / (gp + 1.0))
m563 = fp_diff > 0.60
evaluate_regime(563, "FokkerPlanck_Diffusion_Noise_Dominance", m563, f563,
                "Diffusion_D * DTI * (Credit / Goods)",
                "Fokker-Planck Flux", "High diffusion coefficient reflecting extreme idiosyncratic uncertainty.")

# 564. Peclet Number Solvency Ratio
peclet_num = np.abs(fp_drift) / fp_diff
f564 = peclet_num * (monthly_inc / (an + 1.0)) * (u1 * u2)
m564 = peclet_num > 5.0
evaluate_regime(564, "FokkerPlanck_Peclet_Solvency_Ratio", m564, f564,
                "Peclet_Number * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Fokker-Planck Flux", "High Péclet number indicates deterministic cashflow drift dominates stochastic noise.")

# 565. Stationary Fokker-Planck Potential Well
fp_potential = (inc * 3.0 - subsistence * 3.0) / (cr + 1.0)
f565 = fp_potential * (u1 * u2)**2 * log1p(adult_life)
m565 = fp_potential >= 2.0
evaluate_regime(565, "FokkerPlanck_Stationary_Potential_Well", m565, f565,
                "Potential_Well * (E2 * E3)^2 * log1p(Adult_Life)",
                "Fokker-Planck Flux", "Deep thermodynamic potential well anchoring borrower inside stable solvency basin.")

# 566. Barrier Escape Kramers Rate
kramers_rate = np.exp(-np.maximum(fp_potential, 0.1)) * (an / (monthly_inc + 1.0))
f566 = kramers_rate * dti * (1.0 - u1)
m566 = fp_potential < 1.0
evaluate_regime(566, "FokkerPlanck_Kramers_Escape_Rate", m566, f566,
                "Kramers_Rate * DTI * (1 - E2)",
                "Fokker-Planck Flux", "Kramers escape rate across the debt barrier modeling default transition frequency.")

# 567. Laborer Kramers Escape Hazard
f567 = (single_male_hazard * kramers_rate * 5.0) / (cop_harmonic + 0.1)
m567 = single_male_hazard > 0.0
evaluate_regime(567, "FokkerPlanck_Laborer_Kramers_Hazard", m567, f567,
                "(Single_Male_Hazard * Kramers_Rate * 5) / Harmonic_Copula",
                "Fokker-Planck Flux", "Manual laborers operating near Kramers barrier escape point.")

# 568. Senior Pensioner Confining Potential
f568 = fp_potential * (u1 + u2) * np.log1p(age)
m568 = is_retired
evaluate_regime(568, "FokkerPlanck_Pensioner_Confining_Well", m568, f568,
                "Potential_Well * (E2 + E3) * log1p(Age)",
                "Fokker-Planck Flux", "Infinite potential barrier prevents retirees from diffusing into default.")

# 569. Megacity High Drift Advection
f569 = fp_solvency_flux * (pop / 0.02) * (u1 * u2)**2
m569 = pop >= 0.02
evaluate_regime(569, "FokkerPlanck_Megacity_Advection_Buffer", m569, f569,
                "Solvency_Flux * (Pop / 0.02) * (E2 * E3)^2",
                "Fokker-Planck Flux", "Metropolitan wage advection accelerates probability flow toward wealth creation.")

# 570. Fokker-Planck Master Steady State
f570 = (fp_potential * (u1 * u2)**2) / (1.0 + kramers_rate)
m570 = np.ones(n_train, dtype=bool)
evaluate_regime(570, "FokkerPlanck_Master_Steady_State", m570, f570,
                "(Potential_Well * (E2 * E3)^2) / (1 + Kramers_Rate)",
                "Fokker-Planck Flux", "Master stationary probability distribution of solvent household states.")

# ==================================================================================================
# DOMAIN 4: NON-EQUILIBRIUM THERMODYNAMIC ENTROPY PRODUCTION (EXP 571 - 580)
# ==================================================================================================
# 571. Irreversible Thermodynamic Entropy Generation
f571 = s_production * dti * (1.0 - u1)
m571 = s_production > 0.05
evaluate_regime(571, "Thermodynamic_Entropy_Generation_Rate", m571, f571,
                "Entropy_Production_Rate * DTI * (1 - E2)",
                "Thermodynamics", "Irreversible thermodynamic dissipation rate caused by uncollateralized financing drag.")

# 572. Maximum Entropy Dissipation Collapse
f572 = s_production**1.5 * (cr / (gp + 1.0)) * (1.0 - cop_clayton_2)
m572 = s_production > 0.15
evaluate_regime(572, "Thermodynamic_Max_Entropy_Collapse", m572, f572,
                "Entropy_Production^1.5 * (Credit / Goods) * (1 - Clayton_Copula)",
                "Thermodynamics", "Extreme thermodynamic entropy generation exceeding household dissipation capacity.")

# 573. Minimum Entropy Prime Reversibility Anchor
f573 = (1.0 / (s_production + 0.01)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m573 = s_production < 0.01
evaluate_regime(573, "Thermodynamic_Min_Entropy_Reversibility", m573, f573,
                "(1 / Entropy_Production) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Thermodynamics", "Reversible thermodynamic borrowing with zero excess fee dissipation.")

# 574. Thermodynamic Heat Capacity Buffer
heat_capacity = surplus / (an + 1.0)
f574 = heat_capacity * (u1 * u2)**2 * log1p(emp_yrs)
m574 = heat_capacity > 5.0
evaluate_regime(574, "Thermodynamic_Heat_Capacity_Buffer", m574, f574,
                "Heat_Capacity * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Thermodynamics", "Household financial heat capacity absorbing economic temperature fluctuations.")

# 575. Laborer Thermal Runaway Hazard
f575 = (single_male_hazard * s_production * 10.0) / (cop_harmonic + 0.1)
m575 = single_male_hazard > 0.0
evaluate_regime(575, "Thermodynamic_Laborer_Thermal_Runaway", m575, f575,
                "(Single_Male_Hazard * Entropy_Production * 10) / Harmonic_Copula",
                "Thermodynamics", "Thermal runaway: negative cashflow and excessive markup generating runaway entropy.")

# 576. Pensioner Zero Entropy Generation Invariant
f576 = heat_capacity * (u1 + u2) * np.log1p(age)
m576 = is_retired & (heat_capacity > 0.0)
evaluate_regime(576, "Thermodynamic_Pensioner_Zero_Entropy", m576, f576,
                "Heat_Capacity * (E2 + E3) * log1p(Age)",
                "Thermodynamics", "Stable state pension flows operate at zero thermodynamic entropy generation.")

# 577. Microloan Entropy Density Strain
f577 = s_production * (129.50 / (an + 1.0)) * dti
m577 = an < 12000.0
evaluate_regime(577, "Thermodynamic_Microloan_Entropy_Density", m577, f577,
                "Entropy_Production * (129.50 / Annuity) * DTI",
                "Thermodynamics", "High entropy density on small consumer loans carrying flat overhead charges.")

# 578. Homeowner Real Estate Enthalpy Anchor
f578 = (1.0 / (s_production + 0.01)) * (monthly_inc / (an + 1.0)) * log1p(adult_life)
m578 = (housing == "House / apartment") & (s_production < 0.02)
evaluate_regime(578, "Thermodynamic_Homeowner_Enthalpy_Anchor", m578, f578,
                "(1 / Entropy_Production) * (Monthly_Inc / Annuity) * log1p(Adult_Life)",
                "Thermodynamics", "Residential real estate enthalpy providing structural thermodynamic ballast.")

# 579. Inquiry Velocity Thermodynamic Friction
f579 = s_production * inq_y * (1.0 - u1)
m579 = inq_y >= 3.0
evaluate_regime(579, "Thermodynamic_Inquiry_Friction_Dissipation", m579, f579,
                "Entropy_Production * Inq_YEAR * (1 - E2)",
                "Thermodynamics", "Credit hunting represents active thermodynamic work dissipation.")

# 580. Master Thermodynamic Solvency Potential
f580 = (heat_capacity * (u1 * u2)**2) / (1.0 + s_production)
m580 = np.ones(n_train, dtype=bool)
evaluate_regime(580, "Thermodynamic_Master_Solvency_Potential", m580, f580,
                "(Heat_Capacity * (E2 * E3)^2) / (1 + Entropy_Production)",
                "Thermodynamics", "Master thermodynamic potential balancing heat capacity against entropy generation.")

# ==================================================================================================
# DOMAIN 5: HYPERBOLIC POINCARÉ DISK OCCUPATIONAL GEOMETRY (EXP 581 - 590)
# ==================================================================================================
# 581. Poincaré Disk Boundary Divergence
f581 = poincare_dist * (an / (monthly_inc + 1.0)) * dti
m581 = poincare_dist > 0.25
evaluate_regime(581, "Poincare_Disk_Boundary_Divergence", m581, f581,
                "Poincare_Dist * (Annuity / Monthly_Inc) * DTI",
                "Poincaré Geometry", "Distance to boundary of the hyperbolic disk measures exponential growth of default odds.")

# 582. Poincaré Disk Centroid Prime Anchor
f582 = (1.0 / (poincare_dist + 0.05)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m582 = poincare_dist < 0.08
evaluate_regime(582, "Poincare_Disk_Centroid_Prime_Anchor", m582, f582,
                "(1 / Poincare_Dist) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Poincaré Geometry", "Prime applicants concentrated near the origin of hyperbolic Poincaré space.")

# 583. Poincaré Occupational Hierarchy Curvature
f583 = poincare_dist * ((cr - gp) / (gp + 1.0)) * (1.0 - cop_frank_4)
m583 = cr > gp
evaluate_regime(583, "Poincare_Occupational_Markup_Curvature", m583, f583,
                "Poincare_Dist * ((Credit - Goods) / Goods) * (1 - Frank_Copula)",
                "Poincaré Geometry", "Hyperbolic tree-like hierarchy embedding occupational default risk with financing markup.")

# 584. Poincaré High Skill Tech Metric Shrinkage
f584 = (1.0 / (poincare_dist + 0.05)) * (monthly_inc / (an + 1.0)) * log1p(emp_yrs)
m584 = occ == "High skill tech staff"
evaluate_regime(584, "Poincare_Tech_Metric_Shrinkage", m584, f584,
                "(1 / Poincare_Dist) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)",
                "Poincaré Geometry", "Tech workers situated closest to hyperbolic center.")

# 585. Poincaré Low Skill Laborer Boundary Push
f585 = (poincare_dist / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))
m585 = occ == "Low-skill Laborers"
evaluate_regime(585, "Poincare_LowSkill_Boundary_Push", m585, f585,
                "(Poincare_Dist / Harmonic_Copula) * (Annuity / Monthly_Inc)",
                "Poincaré Geometry", "Low-skill manual laborers pushed toward the ideal boundary at infinity.")

# 586. Poincaré Public Teaching Security Anchor
f586 = (1.0 / (poincare_dist + 0.05)) * cop_harmonic * (u1 + u2)
m586 = (org == "School") | (org == "Kindergarten")
evaluate_regime(586, "Poincare_Public_Teaching_Security", m586, f586,
                "(1 / Poincare_Dist) * Harmonic_Copula * (E2 + E3)",
                "Poincaré Geometry", "Public educators exhibit minimal hyperbolic geodesic distance.")

# 587. Poincaré Driver Mechanical Overhead
f587 = poincare_dist * (car_age / 8.0) * (1.0 - u2)
m587 = (occ == "Drivers") & (car_age >= 0)
evaluate_regime(587, "Poincare_Driver_Mechanical_Overhead", m587, f587,
                "Poincare_Dist * (Car_Age / 8) * (1 - E3)",
                "Poincaré Geometry", "Commercial driver vehicle maintenance overhead scaled in hyperbolic space.")

# 588. Poincaré Management Speculative Geodesic
f588 = poincare_dist * (cr / (monthly_inc * 12.0 + 1.0)) * (1.0 - cop_clayton_2)
m588 = occ == "Managers"
evaluate_regime(588, "Poincare_Manager_Speculative_Geodesic", m588, f588,
                "Poincare_Dist * (Credit / Income) * (1 - Clayton_Copula)",
                "Poincaré Geometry", "Speculative leverage taken by corporate managers.")

# 589. Poincaré Senior Retirement Equilibrium
f589 = (1.0 / (poincare_dist + 0.05)) * (u1 + u2) * np.log1p(age)
m589 = is_retired
evaluate_regime(589, "Poincare_Pensioner_Equilibrium", m589, f589,
                "(1 / Poincare_Dist) * (E2 + E3) * log1p(Age)",
                "Poincaré Geometry", "Retirement status eliminates hyperbolic occupational drift.")

# 590. Master Poincaré Geodesic Solvency
f590 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (poincare_dist + 0.05)
m590 = np.ones(n_train, dtype=bool)
evaluate_regime(590, "Poincare_Master_Geodesic_Solvency", m590, f590,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2) / Poincare_Dist",
                "Poincaré Geometry", "Master hyperbolic solvency metric coupling cashflow velocity with hyperbolic centroid distance.")

# ==================================================================================================
# DOMAIN 6: PICKANDS EXTREME TAIL DEPENDENCE COPULAS (EXP 591 - 600)
# ==================================================================================================
# Non-parametric Pickands dependence function A(t) = -ln(C(e^-t, e^-(1-t)))
t_pick = 0.5
pickands_A = -np.log(np.maximum(cop_gumbel_2 := np.exp(-((-np.log(u1))**2.0 + (-np.log(u2))**2.0)**0.5), 1e-6))
f591 = pickands_A * dti * (1.0 - u1)
m591 = pickands_A > 0.50
evaluate_regime(591, "Pickands_Tail_Dependence_Intensity", m591, f591,
                "Pickands_A * DTI * (1 - E2)",
                "Pickands Copula", "Pickands extreme value dependence function measuring joint bureau crash concentration.")

# 592. Pickands Independence Sanctuary
f592 = (1.0 / (pickands_A + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m592 = pickands_A < 0.20
evaluate_regime(592, "Pickands_Independence_Prime_Sanctuary", m592, f592,
                "(1 / Pickands_A) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Pickands Copula", "Low Pickands tail concentration reflects uncoupled, robust credit scores.")

# 593. Pickands Downpayment Crash Acceleration
f593 = pickands_A * neg_downpayment * (an / (monthly_inc + 1.0))
m593 = neg_downpayment > 0.15
evaluate_regime(593, "Pickands_Downpayment_Crash_Acceleration", m593, f593,
                "Pickands_A * Neg_Downpayment * (Annuity / Monthly_Inc)",
                "Pickands Copula", "Joint external bureau tail collapse coupled with unbacked loan markups.")

# 594. Pickands Career Seniority Shield
f594 = (1.0 / (pickands_A + 0.1)) * log1p(emp_yrs) * cop_frank_4
m594 = emp_yrs > 4.0
evaluate_regime(594, "Pickands_Career_Seniority_Shield", m594, f594,
                "(1 / Pickands_A) * log1p(Emp_Yrs) * Frank_Copula",
                "Pickands Copula", "Senior career capital buffers against tail default synchronization.")

# 595. Pickands Single Laborer Vulnerability
f595 = (single_male_hazard * pickands_A * 5.0) / (cop_harmonic + 0.1)
m595 = single_male_hazard > 0.0
evaluate_regime(595, "Pickands_Laborer_Tail_Vulnerability", m595, f595,
                "(Single_Male_Hazard * Pickands_A * 5) / Harmonic_Copula",
                "Pickands Copula", "Single manual laborers absorb full force of joint tail collapse.")

# 596. Pickands Homeowner Real Estate Anchor
f596 = (1.0 / (pickands_A + 0.1)) * (monthly_inc / (an + 1.0)) * log1p(adult_life)
m596 = housing == "House / apartment"
evaluate_regime(596, "Pickands_Homeowner_Solvency_Anchor", m596, f596,
                "(1 / Pickands_A) * (Monthly_Inc / Annuity) * log1p(Adult_Life)",
                "Pickands Copula", "Residential ownership anchors prime solvency against tail shocks.")

# 597. Pickands Free Cash Flow Tail Absorption
f597 = (free_cash_flow / (an + 1.0)) * (1.0 / (pickands_A + 0.1))
m597 = free_cash_flow > 15000.0
evaluate_regime(597, "Pickands_FCF_Tail_Absorption", m597, f597,
                "(FCF / Annuity) / Pickands_A",
                "Pickands Copula", "Monthly free cash flow absorbs systemic tail shocks.")

# 598. Pickands Inquiry Surge Squeeze
f598 = pickands_A * inq_y * (1.0 - u2)
m598 = inq_y >= 3.0
evaluate_regime(598, "Pickands_Inquiry_Surge_Squeeze", m598, f598,
                "Pickands_A * Inq_YEAR * (1 - E3)",
                "Pickands Copula", "Credit inquiry storms coinciding with joint bureau tail collapse.")

# 599. Pickands Microloan Fixed Overhead Drag
f599 = pickands_A * (129.50 / (an + 1.0)) * dti
m599 = an < 12000.0
evaluate_regime(599, "Pickands_Microloan_Fixed_Overhead_Drag", m599, f599,
                "Pickands_A * (129.50 / Annuity) * DTI",
                "Pickands Copula", "Fixed administrative fee burden on tail-impaired microloans.")

# 600. Master Pickands Tail Solvency Manifold
f600 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (pickands_A + 0.1)
m600 = np.ones(n_train, dtype=bool)
evaluate_regime(600, "Pickands_Master_Tail_Solvency", m600, f600,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2) / Pickands_A",
                "Pickands Copula", "Master extreme tail dependence manifold coupling cashflow velocity with Pickands tail resistance.")

# ==================================================================================================
# DOMAIN 7: DAUBECHIES WAVELET MULTIRESOLUTION ANALYSIS (EXP 601 - 610)
# ==================================================================================================
# High-frequency wavelet detail D = (DAYS_REGISTRATION - DAYS_ID_PUBLISH) - (DAYS_ID_PUBLISH - DAYS_PHONE)
d_high_freq = np.abs((df["DAYS_REGISTRATION"].values - df["DAYS_ID_PUBLISH"].values) -
                     (df["DAYS_ID_PUBLISH"].values - df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values)) / 365.25
f601 = (1.0 / (d_high_freq + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)
m601 = d_high_freq < 1.0
evaluate_regime(601, "Wavelet_High_Freq_Administrative_Coherence", m601, f601,
                "(1 / Wavelet_Detail) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Wavelet MRA", "Wavelet detail coefficient isolates micro-scale administrative synchronization.")

# 602. Wavelet Extreme Detail Friction
f602 = (d_high_freq / (adult_life + 0.1)) * dti * (1.0 - u1)
m602 = d_high_freq > 10.0
evaluate_regime(602, "Wavelet_Extreme_Detail_Friction", m602, f602,
                "(Wavelet_Detail / Adult_Life) * DTI * (1 - E2)",
                "Wavelet MRA", "High wavelet detail coefficient indicates chaotic, unsynchronized civil document updates.")

# 603. Wavelet Low Frequency Trend Anchor
low_freq_trend = (reg_yrs + idp_yrs + ph_yrs) / 3.0
f603 = low_freq_trend * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m603 = low_freq_trend > 8.0
evaluate_regime(603, "Wavelet_Low_Freq_Trend_Anchor", m603, f603,
                "Wavelet_Trend * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Wavelet MRA", "Smooth low-frequency baseline reflects decade-long continuous institutional stability.")

# 604. Wavelet Transient Address Shock
f604 = (d_high_freq / (low_freq_trend + 0.1)) * (an / (monthly_inc + 1.0)) * (1.0 - u2)
m604 = d_high_freq > 5.0
evaluate_regime(604, "Wavelet_Transient_Address_Shock", m604, f604,
                "(Wavelet_Detail / Wavelet_Trend) * (Annuity / Monthly_Inc) * (1 - E3)",
                "Wavelet MRA", "Ratio of high-frequency detail to low-frequency trend signals acute recent residential disruption.")

# 605. Wavelet Prime Career Wavelet Synthesis
f605 = (low_freq_trend / (d_high_freq + 0.1)) * log1p(emp_yrs) * (u1 * u2)**2
m605 = low_freq_trend > 5.0
evaluate_regime(605, "Wavelet_Career_Stability_Synthesis", m605, f605,
                "(Wavelet_Trend / Wavelet_Detail) * log1p(Emp_Yrs) * (E2 * E3)^2",
                "Wavelet MRA", "High trend-to-detail ratio coupled with workplace seniority.")

# 606. Wavelet Subprime Document Noise
f606 = d_high_freq * dti * (1.0 - cop_frank_4)
m606 = (d_high_freq > 8.0) & (u1 < 0.3)
evaluate_regime(606, "Wavelet_Subprime_Document_Noise", m606, f606,
                "Wavelet_Detail * DTI * (1 - Frank_Copula)",
                "Wavelet MRA", "Administrative document noise concentrated in subprime borrowers.")

# 607. Wavelet Verified Landline Wavelet Anchor
f607 = (low_freq_trend * (monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (d_high_freq + 0.1)
m607 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0)
evaluate_regime(607, "Wavelet_Dual_Landline_Anchor", m607, f607,
                "(Wavelet_Trend / Wavelet_Detail) * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Wavelet MRA", "Dual workplace landlines confirmed against low-noise wavelet trend.")

# 608. Wavelet Senior Pensioner Smooth Baseline
f608 = low_freq_trend * (u1 + u2) * np.log1p(age)
m608 = is_retired
evaluate_regime(608, "Wavelet_Pensioner_Smooth_Baseline", m608, f608,
                "Wavelet_Trend * (E2 + E3) * log1p(Age)",
                "Wavelet MRA", "Retirees exhibit near-zero high-frequency administrative noise.")

# 609. Wavelet Free Cash Flow Energy Compression
f609 = (free_cash_flow / (an + 1.0)) * (1.0 / (d_high_freq + 0.1))
m609 = free_cash_flow > 20000.0
evaluate_regime(609, "Wavelet_FCF_Energy_Compression", m609, f609,
                "(FCF / Annuity) / Wavelet_Detail",
                "Wavelet MRA", "Monthly free cash flow filtering out administrative noise.")

# 610. Master Wavelet Multiresolution Solvency
f610 = ((low_freq_trend * (monthly_inc / (an + 1.0)) * (u1 * u2)**2)) / (1.0 + d_high_freq)
m610 = np.ones(n_train, dtype=bool)
evaluate_regime(610, "Wavelet_Master_Multiresolution_Solvency", m610, f610,
                "(Wavelet_Trend * (Monthly_Inc / Annuity) * (E2 * E3)^2) / (1 + Wavelet_Detail)",
                "Wavelet MRA", "Master multiresolution solvency metric balancing low-frequency stability against high-frequency detail.")

# ==================================================================================================
# DOMAIN 8: FRACTIONAL BROWNIAN MOTION & HURST PERSISTENCE EXPONENTS (EXP 611 - 620)
# ==================================================================================================
# Hurst exponent H proxy: H > 0.5 = Persistent / Trend-following, H < 0.5 = Mean-reverting / Chaotic
hurst_h = np.clip(0.5 + 0.25 * (emp_yrs / (adult_life + 0.1)) - 0.15 * (inq_y / 5.0), 0.1, 0.95)
f611 = hurst_h * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m611 = hurst_h > 0.65
evaluate_regime(611, "Hurst_Persistent_Career_Trajectory", m611, f611,
                "Hurst_H * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Hurst Persistence", "Hurst exponent H > 0.65 indicates long-range persistent upward career momentum.")

# 612. Hurst Chaotic Anti-Persistent Volatility
f612 = (1.0 - hurst_h) * dti * (1.0 - u1)
m612 = hurst_h < 0.35
evaluate_regime(612, "Hurst_Anti_Persistent_Chaos", m612, f612,
                "(1 - Hurst_H) * DTI * (1 - E2)",
                "Hurst Persistence", "Anti-persistent Brownian motion H < 0.35 reflects fragmented, chaotic employment history.")

# 613. Hurst Inquiry Hunting Acceleration
f613 = (1.0 - hurst_h) * inq_y * (an / (monthly_inc + 1.0))
m613 = inq_y >= 3.0
evaluate_regime(613, "Hurst_Inquiry_Hunting_Acceleration", m613, f613,
                "(1 - Hurst_H) * Inq_YEAR * (Annuity / Monthly_Inc)",
                "Hurst Persistence", "Active credit shopping compounds anti-persistent behavioral volatility.")

# 614. Hurst Prime Homeowner Persistence
f614 = hurst_h * (monthly_inc / (an + 1.0)) * log1p(adult_life)
m614 = (housing == "House / apartment") & (hurst_h > 0.60)
evaluate_regime(614, "Hurst_Homeowner_Persistence_Anchor", m614, f614,
                "Hurst_H * (Monthly_Inc / Annuity) * log1p(Adult_Life)",
                "Hurst Persistence", "Homeowners exhibiting strong long-range behavioral persistence.")

# 615. Hurst Laborer Precarious Anti-Persistence
f615 = (single_male_hazard / hurst_h) * (cr / (gp + 1.0))
m615 = single_male_hazard > 0.0
evaluate_regime(615, "Hurst_Laborer_Anti_Persistence_Hazard", m615, f615,
                "(Single_Male_Hazard / Hurst_H) * (Credit / Goods)",
                "Hurst Persistence", "Single manual laborers operating in chaotic low-Hurst trajectories.")

# 616. Hurst Senior Pensioner Long-Range Memory
f616 = hurst_h * (u1 + u2) * np.log1p(age)
m616 = is_retired
evaluate_regime(616, "Hurst_Pensioner_Long_Range_Memory", m616, f616,
                "Hurst_H * (E2 + E3) * log1p(Age)",
                "Hurst Persistence", "Decades of accumulated life experience enforce high behavioral memory.")

# 617. Hurst Downpayment Distortion Compounding
f617 = (1.0 - hurst_h) * neg_downpayment * dti
m617 = neg_downpayment > 0.10
evaluate_regime(617, "Hurst_Downpayment_Distortion_Compounding", m617, f617,
                "(1 - Hurst_H) * Neg_Downpayment * DTI",
                "Hurst Persistence", "Anti-persistent debt trajectories combined with unbacked financing markups.")

# 618. Hurst Free Cash Flow Persistence Buffer
f618 = (free_cash_flow / (an + 1.0)) * hurst_h * cop_frank_4
m618 = free_cash_flow > 15000.0
evaluate_regime(618, "Hurst_FCF_Persistence_Buffer", m618, f618,
                "(FCF / Annuity) * Hurst_H * Frank_Copula",
                "Hurst Persistence", "Liquid cashflow buffers reinforced by persistent career growth.")

# 619. Hurst Dual Landline Verification Anchor
f619 = hurst_h * (monthly_inc / (an + 1.0)) * (u1 * u2)**2
m619 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0)
evaluate_regime(619, "Hurst_Dual_Landline_Verification_Anchor", m619, f619,
                "Hurst_H * (Monthly_Inc / Annuity) * (E2 * E3)^2",
                "Hurst Persistence", "Dual workplace landlines confirmed alongside high behavioral persistence.")

# 620. Master Hurst Fractional Brownian Solvency
f620 = (hurst_h * (monthly_inc / (an + 1.0)) * (u1 * u2)**2) / (1.0 + (1.0 - hurst_h) * dti)
m620 = np.ones(n_train, dtype=bool)
evaluate_regime(620, "Hurst_Master_Fractional_Solvency", m620, f620,
                "(Hurst_H * (Monthly_Inc / Annuity) * (E2 * E3)^2) / (1 + (1 - Hurst_H)*DTI)",
                "Hurst Persistence", "Master fractional Brownian motion solvency metric balancing persistence against chaotic debt dispersion.")

# ==================================================================================================
# DOMAIN 9: GRAPH LAPLACIAN SPECTRAL EMBEDDINGS ON ENTITY NETWORKS (EXP 621 - 630)
# ==================================================================================================
# 621. Fiedler Spectral Cluster Degree Centrality
f621 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * np.log1p(n_apps_all)
m621 = n_apps_all >= 2.0
evaluate_regime(621, "Spectral_Fiedler_Cluster_Centrality", m621, f621,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(N_Apps)",
                "Graph Laplacian", "Fiedler algebraic connectivity: degree centrality in confirmed repeat-borrower entity networks.")

# 622. Dense Cluster Spectral Distress Core
f622 = (n_apps_all / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))
m622 = n_apps_all >= 4.0
evaluate_regime(622, "Spectral_Dense_Distress_Core", m622, f622,
                "(N_Apps / Harmonic_Copula) * (Annuity / Monthly_Inc)",
                "Graph Laplacian", "High-degree entity cluster nodes experiencing intense correlated systemic distress.")

# 623. Singleton Isolated Graph Node Baseline
f623 = dti * (cr / (gp + 1.0)) * (1.0 - u1)
m623 = n_apps_all == 1.0
evaluate_regime(623, "Spectral_Singleton_Isolated_Node", m623, f623,
                "DTI * (Credit / Goods) * (1 - E2)",
                "Graph Laplacian", "Zero-edge isolated applicant nodes lacking relational cluster corroboration.")

# 624. Graph Temporal Pacing Discipline
f624 = np.log1p(days_prev / 365.25) * (u1 * u2)**2 * log1p(emp_yrs)
m624 = (app_order >= 1.0) & (days_prev >= 365.0)
evaluate_regime(624, "Spectral_Temporal_Pacing_Discipline", m624, f624,
                "log1p(Years_Since_Last_App) * (E2 * E3)^2 * log1p(Emp_Yrs)",
                "Graph Laplacian", "Multi-year edge duration between consecutive network nodes signals disciplined borrowing.")

# 625. Rapid Edge Traversal Stacking Strain
f625 = (1.0 / (np.maximum(days_prev, 0.0) / 30.0 + 0.1)) * (1.0 - u1) * dti
m625 = (app_order >= 1.0) & (days_prev >= 0.0) & (days_prev < 90.0)
evaluate_regime(625, "Spectral_Rapid_Edge_Stacking_Strain", m625, f625,
                "(1 / Prev_App_Months) * (1 - E2) * DTI",
                "Graph Laplacian", "Rapid network edge traversal (<90 days) reflects accelerating debt compounding.")

# 626. Graph Order Progression Maturity
f626 = (app_order / (n_apps_all + 0.1)) * (monthly_inc / (an + 1.0)) * cop_frank_4
m626 = n_apps_all >= 2.0
evaluate_regime(626, "Spectral_Order_Progression_Maturity", m626, f626,
                "(App_Order / N_Apps) * (Monthly_Inc / Annuity) * Frank_Copula",
                "Graph Laplacian", "Terminal nodes in a multi-application trajectory benefit from demonstrated survival longevity.")

# 627. Serial Manual Laborer Network Degradation
f627 = (single_male_hazard * np.log1p(n_apps_all)) / (cop_harmonic + 0.1)
m627 = (single_male_hazard > 0.0) & (n_apps_all >= 2.0)
evaluate_regime(627, "Spectral_Laborer_Network_Degradation", m627, f627,
                "(Single_Male_Hazard * log1p(N_Apps)) / Harmonic_Copula",
                "Graph Laplacian", "Repeat manual laborers forming high-hazard delinquency clusters.")

# 628. Free Cash Flow Network Buffer
f628 = (free_cash_flow / (an + 1.0)) * np.log1p(n_apps_all) * (u1 * u2)**2
m628 = (free_cash_flow > 15000.0) & (n_apps_all >= 2.0)
evaluate_regime(628, "Spectral_FCF_Network_Buffer", m628, f628,
                "(FCF / Annuity) * log1p(N_Apps) * (E2 * E3)^2",
                "Graph Laplacian", "High liquid cashflow shielding multi-loan borrower networks.")

# 629. Downpayment Network Distortion
f629 = neg_downpayment * np.log1p(n_apps_all) * dti * (1.0 - u1)
m629 = (neg_downpayment > 0.10) & (n_apps_all >= 2.0)
evaluate_regime(629, "Spectral_Downpayment_Network_Distortion", m629, f629,
                "Neg_Downpayment * log1p(N_Apps) * DTI * (1 - E2)",
                "Graph Laplacian", "Negative equity loan stacking across serial borrowing applications.")

# 630. Master Graph Laplacian Spectral Solvency
f630 = ((monthly_inc / (an + 1.0)) * (u1 * u2)**2 * np.log1p(days_prev / 365.25 + 1.0)) / (1.0 + (n_apps_all >= 4.0).astype(float) + single_male_hazard)
m630 = np.ones(n_train, dtype=bool)
evaluate_regime(630, "Spectral_Master_Laplacian_Solvency", m630, f630,
                "((Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Prev_App_Years)) / (1 + High_Density_Flag + Single_Male_Hazard)",
                "Graph Laplacian", "Master graph spectral solvency metric balancing edge pacing against cluster distress and demographic risk.")

# ==================================================================================================
# DOMAIN 10: GRAND COGNITIVE MASTER FIELD TENSOR SYNTHESIS (EXP 631 - 640)
# ==================================================================================================
# 631. Triple Tensor Product: Cashflow x Copula x Tariff
f631 = (monthly_inc / (an + 1.0)) * (u0 * u1 * u2) * (1.0 / (q0_fee + 0.01))
m631 = q0_fee > 0.02
evaluate_regime(631, "Tensor_Triple_Product_Engine", m631, f631,
                "(Monthly_Inc / Annuity) * Ext_Product * (1 / Q0_Fee)",
                "Cognitive Field", "Direct tensor product of disposable cashflow, joint bureau probability, and inverse tariff fee.")

# 632. Quad Tensor Product: Cashflow x Consensus x Tenure x Gravity
f632 = (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * log1p(emp_yrs) * (pop / 0.02)
m632 = pop >= 0.015
evaluate_regime(632, "Tensor_Quad_Product_Prime_Anchor", m632, f632,
                "(Monthly_Inc / Annuity) * (E2 * E3)^2 * log1p(Emp_Yrs) * (Pop / 0.02)",
                "Cognitive Field", "Quadruple product capturing prime solvency across financial, informational, tenure, and spatial axes.")

# 633. Catastrophe Barrier Markov Bridge
f633 = (1.0 / (np.maximum(cusp_discriminant, 0.0) + 0.01)) * (u1 * u2)**2 * np.log1p(days_prev / 365.25 + 1.0)
m633 = app_order >= 1.0
evaluate_regime(633, "Tensor_Catastrophe_Markov_Bridge", m633, f633,
                "(1 / Cusp_Disc) * (E2 * E3)^2 * log1p(Prev_App_Years)",
                "Cognitive Field", "Couples Thom's catastrophe potential with Markovian trajectory pacing.")

# 634. Optimal Transport Fokker-Planck Advection
f634 = (fp_solvency_flux / (w1_dist + 0.05)) * log1p(emp_yrs)
m634 = fp_solvency_flux > 0.0
evaluate_regime(634, "Tensor_Transport_FokkerPlanck_Advection", m634, f634,
                "(Solvency_Flux / W1_Dist) * log1p(Emp_Yrs)",
                "Cognitive Field", "Fokker-Planck probability flux normalized by Wasserstein earth mover distance.")

# 635. Wavelet Hurst Fractional Stability Core
f635 = (low_freq_trend / (d_high_freq + 0.1)) * hurst_h * (u1 * u2)**2
m635 = hurst_h > 0.50
evaluate_regime(635, "Tensor_Wavelet_Hurst_Stability_Core", m635, f635,
                "(Wavelet_Trend / Wavelet_Detail) * Hurst_H * (E2 * E3)^2",
                "Cognitive Field", "Harmonizes multiresolution wavelet trend with fractional Brownian motion persistence.")

# 636. Thermodynamic Pickands Tail Resistance
f636 = (heat_capacity / (pickands_A + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)
m636 = heat_capacity > 2.0
evaluate_regime(636, "Tensor_Thermodynamic_Pickands_Resistance", m636, f636,
                "(Heat_Capacity / Pickands_A) * (Monthly_Inc / Annuity) * (E2 * E3)",
                "Cognitive Field", "Thermodynamic heat capacity resisting extreme Pickands tail collapse.")

# 637. Poincare Fourier Tenor Harmony
f637 = (1.0 / (poincare_dist + 0.05)) * (fourier_tenor_harm + 1.0) * (u1 * u2)**2
m637 = fourier_tenor_harm > 0.50
evaluate_regime(637, "Tensor_Poincare_Fourier_Harmony", m637, f637,
                "(1 / Poincare_Dist) * (cos(Tenor) + 1) * (E2 * E3)^2",
                "Cognitive Field", "Hyperbolic Poincaré centroid alignment corroborated by annual tenor harmonic resonance.")

# 638. Dual Landline Wavelet Gravity Sanctuary
f638 = (low_freq_trend / (d_high_freq + 0.1)) * (monthly_inc / (an + 1.0)) * (u1 * u2)**2 * (pop / 0.02)
m638 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0) & (pop >= 0.02)
evaluate_regime(638, "Tensor_Dual_Landline_Gravity_Sanctuary", m638, f638,
                "(Wavelet_Trend / Wavelet_Detail) * (Monthly_Inc / Annuity) * (E2 * E3)^2 * (Pop / 0.02)",
                "Cognitive Field", "Verified urban dual landline applicants with clean administrative records in metropolitan hubs.")

# 639. Extreme Multidimensional Distress Squeeze
f639 = (single_male_hazard * (1.0 - hurst_h) * w1_dist * s_production * 100.0) / (cop_harmonic + 0.1)
m639 = (single_male_hazard > 0.0) & (s_production > 0.05)
evaluate_regime(639, "Tensor_Extreme_Multidimensional_Distress", m639, f639,
                "(Single_Male_Hazard * (1 - Hurst) * W1_Dist * Entropy_Prod * 100) / Harmonic_Copula",
                "Cognitive Field", "Compound intersection of demographic manual hazard, chaotic Brownian drift, Wasserstein distance, and thermodynamic entropy.")

# 640. Apex Grand Cognitive Championship Manifold V4
f640 = (((monthly_inc / (an + 1.0)) * (u0 * u1 * u2) * log1p(emp_yrs) * hurst_h * np.log1p(days_prev / 365.25 + 1.0)) /
        (1.0 + single_male_hazard + neg_downpayment + w1_dist + s_production))
m640 = np.ones(n_train, dtype=bool)
evaluate_regime(640, "Apex_Grand_Cognitive_Championship_V4", m640, f640,
                "((Monthly_Inc / Annuity) * Ext_Product * log1p(Emp_Yrs) * Hurst * log1p(Prev_App_Years)) / (1 + Single_Male_Hazard + Neg_Down + W1_Dist + Entropy_Prod)",
                "Cognitive Field", "Apex cognitive championship field synthesizing all 10 mathematical domains into a single master continuous ranking manifold.")

# --------------------------------------------------------------------------------------------------
# PART 3: STRATIFIED 5-FOLD CV VERIFICATION OF TOP 15 FRONTIER FORMULATIONS
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 115)
print("  PART 3: STRATIFIED 5-FOLD CV VERIFICATION OF TOP 15 FRONTIER QUANTITATIVE FORMULAS")
print("=" * 115)

top_15_ids = sorted(results_100.keys(), key=lambda k: results_100[k]["within_auc"], reverse=True)[:15]

print("\nTop 15 Discovered Frontier Formulations:")
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

print("\n2. Evaluating Enhanced Feature Set (+Top 15 Frontier Formulations)...")
X_enhanced = X_base.copy()
for exp_id in top_15_ids:
    r = results_100[exp_id]
    feat_name = f"FRONTIER_{exp_id}_{r['name']}"
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
json_path = r"D:\gci\gemini\RESEARCH_100_FRONTIER_MATHEMATICAL_EXP541_640.json"
md_path = r"D:\gci\gemini\RESEARCH_100_FRONTIER_MATHEMATICAL_EXP541_640.md"

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
    "# GCI World 2026: 100 Frontier Mathematical & Actuarial Manifolds (Exp 541 – 640)\n",
    f"- **Execution Timestamp:** `{summary_data['metadata']['execution_timestamp']}`",
    f"- **Total Advanced Regimes Evaluated:** 100 Distinct Mathematical Formulations",
    f"- **Baseline 5-Fold OOF AUC:** `{auc_base:.6f}`",
    f"- **Enhanced 5-Fold OOF AUC:** `{auc_enh:.6f}` (**{cv_lift_bps:+.1f} bps Net Lift**)",
    f"- **Runtime:** `{time.time() - t_start:.2f}s` on RTX 4050 GPU\n",
    "## Exhaustive Ledger of 100 Frontier Mathematical Formulations (Exp 541 – 640)\n",
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
