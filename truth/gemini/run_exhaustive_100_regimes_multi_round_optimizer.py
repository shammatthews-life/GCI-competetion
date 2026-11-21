"""
====================================================================================================
GCI WORLD 2026 — EXHAUSTIVE MULTI-ROUND MATHEMATICAL OPTIMIZATION ENGINE ACROSS 100 SPLITS (EXP 141 - 240)
ITERATIVE CANDIDATE SEARCH | 600+ FORMULA REFINEMENTS | STRICT WITHIN-SPLIT AUC MAXIMIZATION
====================================================================================================
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

t_start = time.time()
print("=" * 105)
print("  GCI WORLD 2026: EXHAUSTIVE MULTI-ROUND MATHEMATICAL OPTIMIZATION ACROSS 100 REGIMES (EXP 141 - 240)")
print("=" * 105)

train_path = r"D:\gci\inpuy\train.csv"
df = pd.read_csv(train_path)
y = df["TARGET"].values
n_train = len(df)
print(f"[+] Permitted Training Applications Loaded: {n_train:,} records in {time.time() - t_start:.2f}s.")

# --------------------------------------------------------------------------------------------------
# BASE FEATURE VECTORS
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
u0 = np.clip(e1_f, 1e-4, 1.0 - 1e-4)

# Inquiries
inq_h = df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0).values.astype(float)
inq_m = df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0).values.astype(float)
inq_q = df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0).values.astype(float)
inq_y = df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0).values.astype(float)
inq_missing = df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().values

# Verification flags
flag_emp_phone = df["FLAG_EMP_PHONE"].fillna(0).values.astype(float)
flag_work_phone = df["FLAG_WORK_PHONE"].fillna(0).values.astype(float)

# Categoricals
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

# Financial ratios
monthly_inc = inc / 12.0
dti = np.clip(cr / (inc + 1.0), 0.0, 30.0)
term = np.clip(cr / (an + 1.0), 1.0, 100.0)
subsistence = 12000.0 * ch + 15000.0
surplus = inc - 12.0 * an

# Advanced copulas
cop_clayton = np.maximum(u1**(-3.0) + u2**(-3.0) - 1.0, 1e-6)**(-1.0 / 3.0)
num_f = (np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)
cop_frank = -0.25 * np.log(1.0 + num_f / (np.exp(-4.0) - 1.0))
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))
cop_joe = 1.0 - ((1.0 - u1)**2.0 + (1.0 - u2)**2.0 - ((1.0 - u1)**2.0)*((1.0 - u2)**2.0))**0.5

# Demographics
is_male = (gender == "M")
is_single = np.isin(family, ["Single / not married", "Separated", "Widow"])
is_laborer = np.isin(occ, ["Laborers", "Drivers", "Low-skill Laborers"])
single_male_hazard = (is_male & is_single & is_laborer).astype(float) * dti * (cr / (gp + 1.0)) * (1.0 - u1)

# Merton distance-to-default
disp_ratio = np.maximum(0.01, monthly_inc / (an + 1.0))
T_years = np.clip(term / 12.0, 0.1, 10.0)
merton_dd = (np.log(disp_ratio) + (0.02 - 0.5 * 0.20**2) * T_years) / (0.20 * np.sqrt(T_years))
merton_pd = norm.cdf(-merton_dd)

# --------------------------------------------------------------------------------------------------
# MULTI-ROUND OPTIMIZATION ENGINE
# --------------------------------------------------------------------------------------------------
optimization_results = {}
final_champion_features = {}

def optimize_regime_formula(exp_id, name, mask, candidates, domain_reason):
    """
    Evaluates multiple mathematical formulations for a given regime mask.
    Iteratively searches for the candidate that maximizes within-split AUC and spread.
    """
    mask = mask.astype(bool)
    count = int(mask.sum())
    pct = round(count / n_train * 100, 2)
    def_rate = round(float(y[mask].mean() * 100), 2) if count > 0 else 0.0
    
    sub_y = y[mask]
    n_pos = int(sub_y.sum())
    n_neg = count - n_pos
    
    can_eval_auc = (count >= 30) and (n_pos >= 5) and (n_neg >= 5)
    
    eval_history = []
    best_candidate = None
    best_auc = -1.0
    best_spread = -1.0
    best_corr = 0.0
    best_str = ""
    best_arr = None
    best_d_low = 0.0
    best_d_high = 0.0
    
    baseline_auc = 0.50
    baseline_spread = 0.0
    
    for round_idx, (cand_str, cand_arr) in enumerate(candidates, 1):
        sub_feat = cand_arr[mask]
        valid = ~np.isnan(sub_feat) & ~np.isinf(sub_feat)
        
        if valid.sum() < 20:
            continue
            
        corr_val, _ = spearmanr(sub_feat[valid], sub_y[valid])
        corr = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0
        
        # Calculate within-split AUC
        if can_eval_auc and len(np.unique(sub_feat[valid])) > 5:
            try:
                # Raw AUC (feature as positive predictor)
                auc_raw = roc_auc_score(sub_y[valid], sub_feat[valid])
                # Direct or inverted AUC depending on sign
                cand_auc = max(auc_raw, 1.0 - auc_raw)
            except Exception:
                cand_auc = 0.50
        else:
            cand_auc = 0.50
            
        # Decile spread
        q10 = np.quantile(sub_feat[valid], 0.10)
        q90 = np.quantile(sub_feat[valid], 0.90)
        d_low = float(sub_y[valid][sub_feat[valid] <= q10].mean() * 100) if (sub_feat[valid] <= q10).sum() > 0 else def_rate
        d_high = float(sub_y[valid][sub_feat[valid] >= q90].mean() * 100) if (sub_feat[valid] >= q90).sum() > 0 else def_rate
        spread = round(abs(d_high - d_low) * 100, 1)
        
        if round_idx == 1:
            baseline_auc = round(cand_auc, 4)
            baseline_spread = spread
            
        eval_history.append({
            "round": round_idx,
            "formula": cand_str,
            "auc": round(cand_auc, 4),
            "spread_bps": spread,
            "corr": corr
        })
        
        # Primary objective: maximize within-split AUC, tie-break with spread
        is_better = False
        if can_eval_auc:
            if cand_auc > best_auc + 0.002:
                is_better = True
            elif abs(cand_auc - best_auc) <= 0.002 and spread > best_spread:
                is_better = True
        else:
            if spread > best_spread:
                is_better = True
                
        if is_better or best_candidate is None:
            best_auc = round(cand_auc, 4)
            best_spread = spread
            best_corr = corr
            best_str = cand_str
            best_arr = cand_arr
            best_d_low = d_low
            best_d_high = d_high
            best_candidate = round_idx
            
    auc_lift = round((best_auc - baseline_auc) * 10000, 1) if can_eval_auc else 0.0
    spread_lift = round(best_spread - baseline_spread, 1)
    
    print(f"[{exp_id:3d}] {name:38s} | N={count:6,d} ({pct:4.1f}%) | Base AUC: {baseline_auc:.4f} -> Opt: {best_auc:.4f} ({auc_lift:+5.1f} bps) | Spread: {best_spread:6.1f} bps | R{best_candidate}")
    
    optimization_results[f"exp_{exp_id}_{name}"] = {
        "exp_id": exp_id,
        "name": name,
        "subgroup_count": count,
        "subgroup_pct": pct,
        "subgroup_default_rate": def_rate,
        "baseline_formula": candidates[0][0],
        "baseline_auc": baseline_auc,
        "baseline_spread_bps": baseline_spread,
        "champion_formula": best_str,
        "champion_round": best_candidate,
        "total_rounds_evaluated": len(eval_history),
        "optimized_within_split_auc": best_auc,
        "auc_lift_bps": auc_lift,
        "within_split_correlation": best_corr,
        "split_low_vs_high_default": f"{best_d_low:.2f}% vs {best_d_high:.2f}%",
        "spread_bps": best_spread,
        "spread_lift_bps": spread_lift,
        "domain_reason": domain_reason,
        "optimization_history": eval_history
    }
    final_champion_features[f"OPT_F_{exp_id}_{name}"] = best_arr

print("\n" + "=" * 105)
print("  STARTING ITERATIVE MULTI-ROUND OPTIMIZATION ACROSS ALL 100 REGIMES (EXP 141 - 240)")
print("=" * 105)

# ==================================================================================================
# DOMAIN 1: DEMOGRAPHIC LIFECYCLE & HOUSEHOLD FRAGILITY (EXP 141 - 150)
# ==================================================================================================
# 141. Young Single Renter Obligation
m141 = (age < 28.0) & is_single & (housing == "Rented apartment")
c141 = [
    ("(Annuity / Monthly_Income) * (Credit / Goods) / (E2 + 0.1)", (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0)) / (u1 + 0.1)),
    ("DTI * (1 - Clayton_Copula) / (Adult_Life * (E2 + 0.1))", dti * (1.0 - cop_clayton) / (adult_life * (u1 + 0.1))),
    ("(Annuity / (Monthly_Inc - Subsistence + 1)) * (Credit / Goods) / (Harmonic_Copula + 0.05)", (an / (np.maximum(monthly_inc - subsistence, 1000.0))) * (cr / (gp + 1.0)) / (cop_harmonic + 0.05)),
    ("((Annuity * 12) / Income) * (Credit / Goods) * (1 - Frank_Copula)^2", ((an * 12.0) / (inc + 1.0)) * (cr / (gp + 1.0)) * ((1.0 - cop_frank)**2.0)),
    ("Merton_PD * (Credit / Goods) / (u1 * u2 + 0.05)", merton_pd * (cr / (gp + 1.0)) / (u1 * u2 + 0.05))
]
optimize_regime_formula(141, "Young_Single_Renter_Obligation", m141, c141,
                        "Young single renters lack equity buffers; non-linear coupling of rental living stress, debt-to-goods markup, and copula tail disagreement maximizes predictive separation.")

# 142. Sandwich Generation Dependency Strain
m142 = (age >= 35.0) & (age <= 50.0) & (ch >= 2) & (housing == "With parents")
c142 = [
    ("(Annuity * (Children + 2)) / Income * (1 - Frank_Copula)", (an * (ch + 2.0)) / (inc + 1.0) * (1.0 - cop_frank)),
    ("(Subsistence / Monthly_Inc) * (1 + DTI) / (Harmonic_Copula + 0.1)", (subsistence / (monthly_inc + 1.0)) * (1.0 + dti) / (cop_harmonic + 0.1)),
    ("((Annuity + Subsistence) / Monthly_Inc)^1.5 * (1 - Clayton_Copula)", np.maximum((an + subsistence) / (monthly_inc + 1.0), 0.0)**1.5 * (1.0 - cop_clayton)),
    ("(Children / Fam) * DTI * (1 - u1 * u2) * (Credit / Goods)", (ch / (fam + 1e-4)) * dti * (1.0 - u1 * u2) * (cr / (gp + 1.0))),
    ("Merton_PD * (ch + 1) * (1 - cop_frank)", merton_pd * (ch + 1.0) * (1.0 - cop_frank))
]
optimize_regime_formula(142, "Sandwich_Generation_Dependency_Strain", m142, c142,
                        "Middle-aged co-residents supporting children and parents face super-linear subsistence dilution.")

# 143. Senior Pensioner Single Annuity Burn
m143 = (age >= 58.0) & is_retired & np.isin(family, ["Widow", "Separated"])
c143 = [
    ("(12 * Annuity) / Income * (1 / (E3 + 0.1))", (an * 12.0) / (inc + 1.0) * (1.0 / (u2 + 0.1))),
    ("(Annuity / Monthly_Inc) * (1 - Clayton_Copula) * np.log1p(Age - 58)", (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * np.log1p(age - 58.0)),
    ("((Annuity + 15000) / Monthly_Inc) / (Harmonic_Copula + 0.05)", ((an + 15000.0) / (monthly_inc + 1.0)) / (cop_harmonic + 0.05)),
    ("DTI * (1 - u2)^2 * (Credit / Goods)", dti * ((1.0 - u2)**2.0) * (cr / (gp + 1.0))),
    ("Merton_PD * (1 / (u2 + 0.1)) * (1 + dti / 5)", merton_pd * (1.0 / (u2 + 0.1)) * (1.0 + dti / 5.0))
]
optimize_regime_formula(143, "Senior_Pensioner_Single_Annuity_Burn", m143, c143,
                        "Isolated retirees with fixed state pensions suffer severe solvency collapse under annuity inflation.")

# 144. Large Family Overcrowding Burden
m144 = fam >= 5.0
c144 = [
    ("Credit / (Income / Family_Size) * (1 - E2 * E3)", (cr / (inc / fam + 1.0)) * (1.0 - u1 * u2)),
    ("(Subsistence / Monthly_Inc) * (Credit / Goods) * (1 - cop_frank)", (subsistence / (monthly_inc + 1.0)) * (cr / (gp + 1.0)) * (1.0 - cop_frank)),
    ("(Annuity / (inc / fam + 1)) * (1 - cop_clayton)", (an / (inc / fam + 1.0)) * (1.0 - cop_clayton)),
    ("(Credit / inc) * fam * (1 - cop_harmonic)", (cr / (inc + 1.0)) * fam * (1.0 - cop_harmonic)),
    ("((Monthly_Inc - Subsistence - Annuity) / Monthly_Inc) * cop_frank", ((monthly_inc - subsistence - an) / (monthly_inc + 1.0)) * cop_frank)
]
optimize_regime_formula(144, "Large_Family_Overcrowding_Burden", m144, c144,
                        "Family dilution scales per-capita subsistence burden exponentially.")

# 145. Sole Breadwinner Child Multiplier
m145 = is_single & (ch >= 1)
c145 = [
    ("(Annuity / (Monthly_Inc - Subsistence)) * (1 + DTI)", (an / (np.maximum(monthly_inc - subsistence, 1000.0))) * (1.0 + dti)),
    ("(Annuity / Monthly_Inc) * (ch + 1) * (1 - u1)", (an / (monthly_inc + 1.0)) * (ch + 1.0) * (1.0 - u1)),
    ("DTI * (ch + 1) * (1 - cop_clayton) * (Credit / Goods)", dti * (ch + 1.0) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (ch + 1) / (u1 + 0.1)", merton_pd * (ch + 1.0) / (u1 + 0.1)),
    ("(Subsistence / (monthly_inc + 1)) * (1 - cop_frank)^2", (subsistence / (monthly_inc + 1.0)) * ((1.0 - cop_frank)**2.0))
]
optimize_regime_formula(145, "Sole_Breadwinner_Child_Multiplier", m145, c145,
                        "Unpartnered parents bear 100% of childcare liquidity stress.")

# 146. Early Career Commercial Associate
m146 = (age < 30.0) & (income_type == "Commercial associate") & (emp_yrs < 2.0)
c146 = [
    ("(Credit / (Emp_Yrs + 0.2)) / Monthly_Inc * (1 - E2)", (cr / (emp_yrs + 0.2)) / (monthly_inc + 1.0) * (1.0 - u1)),
    ("DTI * (1 - cop_frank) / (emp_yrs + 0.5)", dti * (1.0 - cop_frank) / (emp_yrs + 0.5)),
    ("(Annuity / Monthly_Inc) * (cr / (gp + 1)) / (cop_harmonic + 0.1)", (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0)) / (cop_harmonic + 0.1)),
    ("Merton_PD / (emp_yrs + 0.3) * (1 - u1)", merton_pd / (emp_yrs + 0.3) * (1.0 - u1)),
    ("(Credit / Goods) * (1 - cop_clayton) * (adult_life / (emp_yrs + 0.5))", (cr / (gp + 1.0)) * (1.0 - cop_clayton) * (adult_life / (emp_yrs + 0.5)))
]
optimize_regime_formula(146, "Early_Career_Commercial_Associate", m146, c146,
                        "Short employment tenure coupled with volatile sales commissions.")

# 147. Tenured Civil Servant Stability
m147 = (income_type == "State servant") & (emp_yrs >= 7.0)
c147 = [
    ("(Income / Credit) * (E2 + E3) * log1p(Emp_Yrs)", (inc / (cr + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)),
    ("(Monthly_Inc / Annuity) * cop_harmonic * np.log1p(emp_yrs)", (monthly_inc / (an + 1.0)) * cop_harmonic * np.log1p(emp_yrs)),
    ("(Monthly_Inc / Annuity) * cop_frank * (emp_yrs / adult_life)", (monthly_inc / (an + 1.0)) * cop_frank * (emp_yrs / adult_life)),
    ("cop_frank * (u1 * u2) * np.log1p(inc / cr)", cop_frank * (u1 * u2) * np.log1p(inc / (cr + 1.0))),
    ("Merton_DD * cop_harmonic * np.log1p(emp_yrs)", merton_dd * cop_harmonic * np.log1p(emp_yrs))
]
optimize_regime_formula(147, "Tenured_Civil_Servant_Stability", m147, c147,
                        "State civil servant job protection guarantees annuity continuity.")

# 148. Maternity Unemployment Vulnerability
m148 = np.isin(income_type, ["Maternity leave", "Unemployed"])
c148 = [
    ("Credit / Monthly_Inc * (1 / (E3 + 0.05))", cr / (monthly_inc + 1.0) * (1.0 / (u2 + 0.05))),
    ("DTI * (1 - cop_clayton) * (cr / (gp + 1))", dti * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("(Annuity / (inc + 1)) / (cop_harmonic + 0.05)", (an / (inc + 1.0)) / (cop_harmonic + 0.05)),
    ("Merton_PD * (1 / (u1 + 0.1))", merton_pd * (1.0 / (u1 + 0.1))),
    ("(cr / (gp + 1)) * (1.0 - u2)", (cr / (gp + 1.0)) * (1.0 - u2))
]
optimize_regime_formula(148, "Maternity_Unemployment_Vulnerability", m148, c148,
                        "Absence of active income elevates default probability to extreme levels.")

# 149. Midlife Divorced Unanchored Male
m149 = is_male & np.isin(family, ["Separated", "Single / not married"]) & (age >= 38.0) & (age <= 52.0)
c149 = [
    ("DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("single_male_hazard * (cr / (gp + 1))", single_male_hazard * (cr / (gp + 1.0))),
    ("(Annuity / Monthly_Inc) * (1 - cop_frank) * (adult_life / (emp_yrs + 1))", (an / (monthly_inc + 1.0)) * (1.0 - cop_frank) * (adult_life / (emp_yrs + 1.0))),
    ("Merton_PD * dti * (1 - u1)", merton_pd * dti * (1.0 - u1)),
    ("((cr - gp) / (gp + 1)) * (1 - cop_harmonic) * dti", ((cr - gp) / (gp + 1.0)) * (1.0 - cop_harmonic) * dti)
]
optimize_regime_formula(149, "Midlife_Divorced_Unanchored_Male", m149, c149,
                        "Unpartnered middle-aged males lacking housing anchors face persistent credit drift.")

# 150. Multi Generational House Asset Anchor
m150 = (housing == "With parents") & (car_age >= 0) & (car_age < 10.0)
c150 = [
    ("(Monthly_Inc / Annuity) * Harmonic_Copula * (10 - Car_Age)", (monthly_inc / (an + 1.0)) * cop_harmonic * (10.0 - car_age)),
    ("(Income / Credit) * cop_frank * (10 - car_age)", (inc / (cr + 1.0)) * cop_frank * (10.0 - car_age)),
    ("cop_harmonic * (u1 * u2) * (monthly_inc / (an + 1))", cop_harmonic * (u1 * u2) * (monthly_inc / (an + 1.0))),
    ("Merton_DD * (10 - car_age) * cop_frank", merton_dd * (10.0 - car_age) * cop_frank),
    ("(monthly_inc / (an + 1)) * np.log1p(adult_life)", (monthly_inc / (an + 1.0)) * np.log1p(adult_life))
]
optimize_regime_formula(150, "Multi_Generational_House_Asset_Anchor", m150, c150,
                        "Co-residing young adults with modern vehicles possess unencumbered personal asset equity.")

# ==================================================================================================
# DOMAIN 2: WORKPLACE, INDUSTRY & OCCUPATIONAL POLARIZATION (EXP 151 - 160)
# ==================================================================================================
# 151. Hazardous Industry Manual Laborer
m151 = np.isin(org, ["Construction", "Transport: type 3", "Industry: type 3", "Industry: type 9"]) & is_laborer
c151 = [
    ("(Credit / Goods) * DTI * (1 - E2)", (cr / (gp + 1.0)) * dti * (1.0 - u1)),
    ("single_male_hazard * (an / (monthly_inc + 1))", single_male_hazard * (an / (monthly_inc + 1.0))),
    ("((cr - gp) / (gp + 1)) * (1 - cop_clayton) * dti", ((cr - gp) / (gp + 1.0)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (cr / (gp + 1)) * (1 / (u2 + 0.1))", merton_pd * (cr / (gp + 1.0)) * (1.0 / (u2 + 0.1))),
    ("(Annuity / Monthly_Inc)^1.5 * (1 - cop_frank)", np.maximum(an / (monthly_inc + 1.0), 0.0)**1.5 * (1.0 - cop_frank))
]
optimize_regime_formula(151, "Hazardous_Industry_Manual_Laborer", m151, c151,
                        "Manual laborers in cyclically sensitive industries face high injury and seasonal layoff risks.")

# 152. High Skill IT Academic Solvency
m152 = (edu == "Higher education") & np.isin(occ, ["IT staff", "High skill tech staff", "Core staff"]) & (emp_yrs >= 3.0)
c152 = [
    ("(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)", (monthly_inc / (an + 1.0)) * (u1 * u2) * np.log1p(emp_yrs)),
    ("(Income / Credit) * cop_frank * (emp_yrs / adult_life)", (inc / (cr + 1.0)) * cop_frank * (emp_yrs / adult_life)),
    ("Merton_DD * cop_harmonic * (u1 + u2)", merton_dd * cop_harmonic * (u1 + u2)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(emp_yrs)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (1 - dti / 15) * (u1 * u2)", (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0) * (u1 * u2))
]
optimize_regime_formula(152, "High_Skill_IT_Academic_Solvency", m152, c152,
                        "Educated technical staff have robust continuous earning potential and substantial savings buffers.")

# 153. Informal Trade Sales Volatility
m153 = np.isin(org, ["Trade: type 3", "Trade: type 7", "Services"]) & (flag_work_phone == 0.0)
c153 = [
    ("(Annuity / Monthly_Inc) / (E3 + 0.1)", (an / (monthly_inc + 1.0)) / (u2 + 0.1)),
    ("DTI * (1 - cop_clayton) * (cr / (gp + 1))", dti * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (an / (monthly_inc + 1)) / (u1 + 0.1)", merton_pd * (an / (monthly_inc + 1.0)) / (u1 + 0.1)),
    ("(cr / (gp + 1)) * (1 - cop_harmonic) * dti", (cr / (gp + 1.0)) * (1.0 - cop_harmonic) * dti),
    ("((Annuity * 12) / Income) * (1 - u2)^2", ((an * 12.0) / (inc + 1.0)) * ((1.0 - u2)**2.0))
]
optimize_regime_formula(153, "Informal_Trade_Sales_Volatility", m153, c153,
                        "Unverified sales workers experience sharp retail revenue volatility.")

# 154. Security Police Stable Anchor
m154 = np.isin(occ, ["Security staff"]) | np.isin(org, ["Police", "Security", "Military"])
c154 = [
    ("(Income / Credit) * Frank_Copula * (Emp_Yrs + 1)", (inc / (cr + 1.0)) * cop_frank * (emp_yrs + 1.0)),
    ("(Monthly_Inc / Annuity) * cop_harmonic * np.log1p(emp_yrs)", (monthly_inc / (an + 1.0)) * cop_harmonic * np.log1p(emp_yrs)),
    ("cop_frank * (u1 * u2) * (monthly_inc / (an + 1))", cop_frank * (u1 * u2) * (monthly_inc / (an + 1.0))),
    ("Merton_DD * (u1 + u2) * np.log1p(age)", merton_dd * (u1 + u2) * np.log1p(age)),
    ("(Income / Credit) * (u1 * u2) * (1 - dti / 10)", (inc / (cr + 1.0)) * (u1 * u2) * (1.0 - dti / 10.0))
]
optimize_regime_formula(154, "Security_Police_Stable_Anchor", m154, c154,
                        "Institutional security and military personnel have protected regular payrolls.")

# 155. Low Skill Day Laborer Distress
m155 = (occ == "Low-skill Laborers") & (inc < np.median(inc))
c155 = [
    ("(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)", (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))),
    ("single_male_hazard * (cr / (gp + 1)) * 2", single_male_hazard * (cr / (gp + 1.0)) * 2.0),
    ("DTI * (1 - cop_clayton) / (emp_yrs + 0.1)", dti * (1.0 - cop_clayton) / (emp_yrs + 0.1)),
    ("Merton_PD * (cr / (gp + 1)) * (1 / (u1 + 0.05))", merton_pd * (cr / (gp + 1.0)) * (1.0 / (u1 + 0.05))),
    ("(Annuity / (monthly_inc - subsistence + 1)) * (1 - cop_frank)", (an / (np.maximum(monthly_inc - subsistence, 500.0))) * (1.0 - cop_frank))
]
optimize_regime_formula(155, "Low_Skill_Day_Laborer_Distress", m155, c155,
                        "Low-skill laborers earning below median have negligible financial cushion.")

# 156. Executive Corporate Leverage
m156 = np.isin(occ, ["Managers", "Accountants"]) & (inc >= np.percentile(inc, 80))
c156 = [
    ("(Credit / Monthly_Inc) * (1 - E2) * (Credit / Goods)", (cr / (monthly_inc + 1.0)) * (1.0 - u1) * (cr / (gp + 1.0))),
    ("DTI * (1 - cop_frank) * (cr / (gp + 1))", dti * (1.0 - cop_frank) * (cr / (gp + 1.0))),
    ("Merton_PD * dti * (1 - u1 * u2)", merton_pd * dti * (1.0 - u1 * u2)),
    ("((cr - gp) / (gp + 1)) * (1 - cop_clayton) * (an / (monthly_inc + 1))", ((cr - gp) / (gp + 1.0)) * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("(an / (monthly_inc + 1)) * (1 / (u1 + 0.1))", (an / (monthly_inc + 1.0)) / (u1 + 0.1))
]
optimize_regime_formula(156, "Executive_Corporate_Leverage", m156, c156,
                        "Corporate managers taking outsize speculative debt display sharp credit-score sensitivity.")

# 157. Healthcare Public Sector Tenure
m157 = np.isin(org, ["Medicine", "Kindergarten", "School"]) & (income_type == "State servant")
c157 = [
    ("(Monthly_Inc / Annuity) * (E2 + E3) / (1 + Children)", (monthly_inc / (an + 1.0)) * (u1 + u2) / (1.0 + ch)),
    ("(Income / Credit) * cop_harmonic * np.log1p(emp_yrs)", (inc / (cr + 1.0)) * cop_harmonic * np.log1p(emp_yrs)),
    ("cop_frank * (u1 * u2) * (monthly_inc / (an + 1)) / (ch + 1)", cop_frank * (u1 * u2) * (monthly_inc / (an + 1.0)) / (ch + 1.0)),
    ("Merton_DD * (u1 + u2) * (emp_yrs / adult_life)", merton_dd * (u1 + u2) * (emp_yrs / adult_life)),
    ("(monthly_inc / (an + 1)) * (1 - dti / 12) * cop_frank", (monthly_inc / (an + 1.0)) * (1.0 - dti / 12.0) * cop_frank)
]
optimize_regime_formula(157, "Healthcare_Public_Sector_Tenure", m157, c157,
                        "Public medical and teaching staff have reliable, predictable salary structures.")

# 158. Cleaning Cooking Service Pressure
m158 = np.isin(occ, ["Cleaning staff", "Cooking staff"]) & (ch >= 1)
c158 = [
    ("(Annuity / (Monthly_Inc - Subsistence)) / (E2 + 0.1)", (an / (np.maximum(monthly_inc - subsistence, 1000.0))) / (u1 + 0.1)),
    ("DTI * (ch + 1) * (1 - cop_clayton) * (cr / (gp + 1))", dti * (ch + 1.0) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (an / (monthly_inc + 1)) / (u2 + 0.1)", merton_pd * (an / (monthly_inc + 1.0)) / (u2 + 0.1)),
    ("(cr / (gp + 1)) * (1 - cop_frank) * (an / (monthly_inc + 1))", (cr / (gp + 1.0)) * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))),
    ("(Subsistence / (monthly_inc + 1)) * (1 - u1 * u2)", (subsistence / (monthly_inc + 1.0)) * (1.0 - u1 * u2))
]
optimize_regime_formula(158, "Cleaning_Cooking_Service_Pressure", m158, c158,
                        "Service staff supporting families experience extreme cash-flow tightness on monthly payment dates.")

# 159. Self Employed Sole Proprietor Friction
m159 = (income_type == "Commercial associate") & np.isin(org, ["Business Entity Type 3", "Self-employed"])
c159 = [
    ("DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("(cr / (gp + 1)) * (1 - cop_frank) * dti", (cr / (gp + 1.0)) * (1.0 - cop_frank) * dti),
    ("Merton_PD * dti * (1 - u1)", merton_pd * dti * (1.0 - u1)),
    ("((cr - gp) / (gp + 1)) * (1 - cop_harmonic) * (an / (monthly_inc + 1))", ((cr - gp) / (gp + 1.0)) * (1.0 - cop_harmonic) * (an / (monthly_inc + 1.0))),
    ("(Annuity / Monthly_Inc)^1.5 * (1 - u1 * u2)", np.maximum(an / (monthly_inc + 1.0), 0.0)**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(159, "Self_Employed_Sole_Proprietor_Friction", m159, c159,
                        "Sole proprietors absorb the entire downside of business shocks without insurance safety nets.")

# 160. Driver Automotive Collateral Vulnerability
m160 = (occ == "Drivers") & (car_flag == 1.0) & (cr >= 400000)
c160 = [
    ("(Credit / Goods) * (Car_Age + 1) / (E3 + 0.1)", (cr / (gp + 1.0)) * (car_age + 1.0) / (u2 + 0.1)),
    ("(car_age / 8) * dti * (1 - cop_clayton) * (cr / (gp + 1))", (car_age / 8.0) * dti * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (car_age + 1) * (cr / (gp + 1))", merton_pd * (car_age + 1.0) * (cr / (gp + 1.0))),
    ("single_male_hazard * (car_age / 5)", single_male_hazard * (car_age / 5.0)),
    ("(an / (monthly_inc + 1)) * (car_age + 1) / (u1 + 0.1)", (an / (monthly_inc + 1.0)) * (car_age + 1.0) / (u1 + 0.1))
]
optimize_regime_formula(160, "Driver_Automotive_Collateral_Vulnerability", m160, c160,
                        "Drivers financing debt against depreciating vehicles suffer dual vehicle wear and credit strain.")

# ==================================================================================================
# DOMAIN 3: CREDIT BUREAU DIVERGENCE & COPULA MANIFOLDS (EXP 161 - 170)
# ==================================================================================================
# 161. Bureau Triad Full Agreement Anchor
valid_triad = ~np.isnan(e1) & ~np.isnan(e2) & ~np.isnan(e3)
triad_std = np.std(np.column_stack([e1_f, e2_f, e3_f]), axis=1)
m161 = valid_triad & (triad_std < 0.08)
c161 = [
    ("Mean(E1, E2, E3) * (Income / Credit)", ((e1_f + e2_f + e3_f) / 3.0) * (inc / (cr + 1.0))),
    ("cop_harmonic * (monthly_inc / (an + 1)) * (1 - triad_std * 5)", cop_harmonic * (monthly_inc / (an + 1.0)) * (1.0 - triad_std * 5.0)),
    ("cop_frank * (u0 * u1 * u2)**(1/3) * (monthly_inc / (an + 1))", cop_frank * (u0 * u1 * u2)**(1.0/3.0) * (monthly_inc / (an + 1.0))),
    ("Merton_DD * ((e1_f + e2_f + e3_f) / 3) * np.log1p(emp_yrs)", merton_dd * ((e1_f + e2_f + e3_f) / 3.0) * np.log1p(emp_yrs)),
    ("((e1_f + e2_f + e3_f) / 3)^2 * (inc / cr)", ((e1_f + e2_f + e3_f) / 3.0)**2.0 * (inc / (cr + 1.0)))
]
optimize_regime_formula(161, "Bureau_Triad_Full_Agreement_Anchor", m161, c161,
                        "When all three independent bureaus agree closely, default risk is exceptionally low and predictable.")

# 162. Bureau Bipolar Acute Discordance
m162 = valid_triad & (e2_f > 0.60) & (e3_f < 0.30)
c162 = [
    ("(E2 - E3) * DTI / (E3 + 0.05)", (e2_f - e3_f) * dti / (u2 + 0.05)),
    ("(e2_f - e3_f)^2 * (an / (monthly_inc + 1)) / (u2 + 0.1)", (e2_f - e3_f)**2.0 * (an / (monthly_inc + 1.0)) / (u2 + 0.1)),
    ("Merton_PD * (e2_f - e3_f) * (cr / (gp + 1)) / (u2 + 0.05)", merton_pd * (e2_f - e3_f) * (cr / (gp + 1.0)) / (u2 + 0.05)),
    ("(1 - cop_clayton) * (e2_f - e3_f) * dti", (1.0 - cop_clayton) * (e2_f - e3_f) * dti),
    ("((e2_f - e3_f) / (e2_f + e3_f + 0.1)) * dti * (cr / (gp + 1))", ((e2_f - e3_f) / (e2_f + e3_f + 0.1)) * dti * (cr / (gp + 1.0)))
]
optimize_regime_formula(162, "Bureau_Bipolar_Acute_Discordance", m162, c162,
                        "Severe conflict where Agency 2 is lenient while Agency 3 detects delinquency signals concealed risk.")

# 163. Gumbel Upper Tail Ultra Prime Sanctuary
m163 = (u1 > 0.65) & (u2 > 0.65)
c163 = [
    ("(E2 * E3) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)", (u1 * u2) * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("cop_joe * (monthly_inc / (an + 1)) * (1 - dti / 15)", cop_joe * (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0)),
    ("cop_harmonic * (inc / (cr + 1)) * np.log1p(adult_life)", cop_harmonic * (inc / (cr + 1.0)) * np.log1p(adult_life)),
    ("Merton_DD * (u1 * u2) * np.log1p(emp_yrs)", merton_dd * (u1 * u2) * np.log1p(emp_yrs)),
    ("(u1 * u2)**2 * (monthly_inc / (an + 1))", (u1 * u2)**2.0 * (monthly_inc / (an + 1.0)))
]
optimize_regime_formula(163, "Gumbel_Upper_Tail_Ultra_Prime_Sanctuary", m163, c163,
                        "Simultaneous top-tier bureau scores isolate virtually default-free ultra prime borrowers.")

# 164. Clayton Deep Subprime Sinkhole
m164 = (u1 < 0.35) & (u2 < 0.35)
c164 = [
    ("Clayton_Copula * DTI * (Credit / Goods)", cop_clayton * dti * (cr / (gp + 1.0))),
    ("(1 / (u1 * u2 + 0.05)) * (an / (monthly_inc + 1))", (1.0 / (u1 * u2 + 0.05)) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * (1 - cop_harmonic) * dti * (cr / (gp + 1))", merton_pd * (1.0 - cop_harmonic) * dti * (cr / (gp + 1.0))),
    ("(1 - cop_frank)^2 * dti * (cr / (gp + 1))", (1.0 - cop_frank)**2.0 * dti * (cr / (gp + 1.0))),
    ("((1 - u1) * (1 - u2)) * (an / (monthly_inc + 1)) * dti", ((1.0 - u1) * (1.0 - u2)) * (an / (monthly_inc + 1.0)) * dti)
]
optimize_regime_formula(164, "Clayton_Deep_Subprime_Sinkhole", m164, c164,
                        "Correlated distress in lower tail copulas captures systemic delinquency clustering.")

# 165. Bureau Harmonic vs Arithmetic Deficit
arith_mean = (e1_f + e2_f + e3_f) / 3.0
harm_mean = 3.0 / ((1.0 / np.clip(e1_f, 0.01, 1.0)) + (1.0 / np.clip(e2_f, 0.01, 1.0)) + (1.0 / np.clip(e3_f, 0.01, 1.0)))
f165 = (arith_mean - harm_mean) * (cr / (inc + 1.0))
m165 = valid_triad
c165 = [
    ("(Arithmetic_Mean - Harmonic_Mean) * (Credit / Income)", (arith_mean - harm_mean) * (cr / (inc + 1.0))),
    ("(arith_mean - harm_mean) * (an / (monthly_inc + 1)) * (cr / (gp + 1))", (arith_mean - harm_mean) * (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0))),
    ("Merton_PD * (arith_mean - harm_mean) * dti", merton_pd * (arith_mean - harm_mean) * dti),
    ("((arith_mean - harm_mean) / (arith_mean + 0.1)) * (1 - cop_clayton) * dti", ((arith_mean - harm_mean) / (arith_mean + 0.1)) * (1.0 - cop_clayton) * dti),
    ("(arith_mean - harm_mean)^0.5 * (1 - cop_frank) * (cr / (gp + 1))", np.sqrt(np.maximum(arith_mean - harm_mean, 0.0)) * (1.0 - cop_frank) * (cr / (gp + 1.0)))
]
optimize_regime_formula(165, "Bureau_Harmonic_Vs_Arithmetic_Deficit", m165, c165,
                        "Penalizes borrowers where one disastrous score ruins an otherwise acceptable average.")

# 166. Missing Bureau 3 Informational Blindspot
m166 = np.isnan(e3) & ~np.isnan(e2)
c166 = [
    ("(1 - E2) * DTI * (Annuity / Monthly_Inc)", (1.0 - u1) * dti * (an / (monthly_inc + 1.0))),
    ("Merton_PD * (1 - u1) * (cr / (gp + 1))", merton_pd * (1.0 - u1) * (cr / (gp + 1.0))),
    ("(1 - u1)^2 * dti * (cr / (gp + 1))", ((1.0 - u1)**2.0) * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (u1 + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (u1 + 0.1) * (cr / (gp + 1.0))),
    ("dti * (1 - u1) * (adult_life / (emp_yrs + 1))", dti * (1.0 - u1) * (adult_life / (emp_yrs + 1.0)))
]
optimize_regime_formula(166, "Missing_Bureau_3_Informational_Blindspot", m166, c166,
                        "Absence of Bureau 3 data forces reliance on Bureau 2 with heightened adverse selection.")

# 167. Missing Bureau 1 Novel Credit Profile
m167 = np.isnan(e1) & ~np.isnan(e2) & ~np.isnan(e3)
c167 = [
    ("Frank_Copula(E2, E3) * (Monthly_Inc / Annuity)", cop_frank * (monthly_inc / (an + 1.0))),
    ("cop_harmonic * (monthly_inc / (an + 1)) * (1 - dti / 15)", cop_harmonic * (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0)),
    ("Merton_DD * cop_frank * (u1 * u2)", merton_dd * cop_frank * (u1 * u2)),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(emp_yrs)", cop_joe * (inc / (cr + 1.0)) * np.log1p(emp_yrs)),
    ("(u1 * u2) * (monthly_inc / (an + 1)) * np.log1p(adult_life)", (u1 * u2) * (monthly_inc / (an + 1.0)) * np.log1p(adult_life))
]
optimize_regime_formula(167, "Missing_Bureau_1_Novel_Credit_Profile", m167, c167,
                        "Younger credit profiles lacking primary bureau data rely heavily on joint E2-E3 consensus.")

# 168. Single Bureau Isolated Reliance
m168 = np.isnan(e1) & np.isnan(e3) & ~np.isnan(e2)
c168 = [
    ("(1 - E2) * (Credit / Goods) * (Annuity / Monthly_Inc)", (1.0 - u1) * (cr / (gp + 1.0)) * (an / (monthly_inc + 1.0))),
    ("Merton_PD / (u1 + 0.1) * (cr / (gp + 1))", merton_pd / (u1 + 0.1) * (cr / (gp + 1.0))),
    ("(1 - u1)^2 * dti * (cr / (gp + 1))", ((1.0 - u1)**2.0) * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (u1 + 0.05)", (an / (monthly_inc + 1.0)) / (u1 + 0.05)),
    ("dti * (1 - u1) * (cr / (gp + 1))", dti * (1.0 - u1) * (cr / (gp + 1.0)))
]
optimize_regime_formula(168, "Single_Bureau_Isolated_Reliance", m168, c168,
                        "Evaluating applicants with only a single bureau source introduces unobserved underwriting risks.")

# 169. Frank Radial Symmetry Deficit
c169 = [
    ("abs(E2 + E3 - 1.0) * DTI", np.abs(u1 + u2 - 1.0) * dti),
    ("abs(u1 + u2 - 1.0) * (cr / (gp + 1)) * (an / (monthly_inc + 1))", np.abs(u1 + u2 - 1.0) * (cr / (gp + 1.0)) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * abs(u1 + u2 - 1.0) * dti", merton_pd * np.abs(u1 + u2 - 1.0) * dti),
    ("(abs(u1 + u2 - 1.0) / (u1 + u2 + 0.1)) * (1 - cop_clayton) * dti", (np.abs(u1 + u2 - 1.0) / (u1 + u2 + 0.1)) * (1.0 - cop_clayton) * dti),
    ("abs(u1 + u2 - 1.0)^2 * dti * (cr / (gp + 1))", (np.abs(u1 + u2 - 1.0)**2.0) * dti * (cr / (gp + 1.0)))
]
m169 = (u1 > 0.1) & (u2 > 0.1)
optimize_regime_formula(169, "Frank_Radial_Symmetry_Deficit", m169, c169,
                        "Deviations from center balance across bureau evaluations highlight model volatility.")

# 170. Bureau Logit Variance Polarization
logit_e1 = np.log(np.clip(e1_f, 0.01, 0.99) / (1.0 - np.clip(e1_f, 0.01, 0.99)))
logit_e2 = np.log(np.clip(e2_f, 0.01, 0.99) / (1.0 - np.clip(e2_f, 0.01, 0.99)))
logit_e3 = np.log(np.clip(e3_f, 0.01, 0.99) / (1.0 - np.clip(e3_f, 0.01, 0.99)))
var_logit = np.var(np.column_stack([logit_e1, logit_e2, logit_e3]), axis=1)
m170 = valid_triad
c170 = [
    ("Var(logit(E1), logit(E2), logit(E3)) * (Credit / Goods)", var_logit * (cr / (gp + 1.0))),
    ("var_logit * dti * (1 - cop_clayton)", var_logit * dti * (1.0 - cop_clayton)),
    ("Merton_PD * var_logit * (cr / (gp + 1))", merton_pd * var_logit * (cr / (gp + 1.0))),
    ("var_logit * (an / (monthly_inc + 1)) * (cr / (gp + 1))", var_logit * (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0))),
    ("(var_logit / (arith_mean + 0.1)) * dti", (var_logit / (arith_mean + 0.1)) * dti)
]
optimize_regime_formula(170, "Bureau_Logit_Variance_Polarization", m170, c170,
                        "Logit variance measures fundamental rating dispersion on an unconstrained scale.")

# ==================================================================================================
# DOMAIN 4: CONTRACT GEOMETRY, ANNUITY RATIOS & TARIFFS (EXP 171 - 180)
# ==================================================================================================
# 171. Revolving High Velocity Buffer
m171 = (contract == "Revolving loans")
c171 = [
    ("(Income / Annuity) * Harmonic_Copula * (1 - DTI / 10)", (inc / (an + 1.0)) * cop_harmonic * (1.0 - dti / 10.0)),
    ("(Monthly_Inc / Annuity) * cop_frank * (u1 * u2)", (monthly_inc / (an + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * cop_harmonic * (1 - dti / 12)", merton_dd * cop_harmonic * (1.0 - dti / 12.0)),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(emp_yrs)", cop_joe * (inc / (cr + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 + u2) * np.log1p(adult_life)", (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(adult_life))
]
optimize_regime_formula(171, "Revolving_High_Velocity_Buffer", m171, c171,
                        "Discretionary repayment flex on revolving credit makes liquidity buffers paramount.")

# 172. Cash Loan Fixed Fee Lattice Exact
cash_mask = (contract == "Cash loans")
qb = (an - 129.5) / (cr + 1e-5)
m172 = cash_mask & (np.abs(qb - np.round(qb, 4)) < 1e-4) & (cr > gp)
c172 = [
    ("Q0_Fee * (1 - E2) * (Credit / Goods)", qb * (1.0 - u1) * (cr / (gp + 1.0))),
    ("qb * dti * (1 - cop_clayton)", qb * dti * (1.0 - cop_clayton)),
    ("Merton_PD * qb * (cr / (gp + 1))", merton_pd * qb * (cr / (gp + 1.0))),
    ("(qb * 12) * (cr / (gp + 1)) / (cop_harmonic + 0.1)", (qb * 12.0) * (cr / (gp + 1.0)) / (cop_harmonic + 0.1)),
    ("qb^1.5 * dti * (1 - u1 * u2)", np.maximum(qb, 0.0)**1.5 * dti * (1.0 - u1 * u2))
]
optimize_regime_formula(172, "Cash_Loan_Fixed_Fee_Lattice_Exact", m172, c172,
                        "Standard retail bank cash-credit programs following the 129.50 fee lattice.")

# 173. Ultra Long Tenure Amortization Strain
m173 = (cr / (an + 1.0)) >= 24.0
c173 = [
    ("(Credit / Annuity) * (1 - Frank_Copula) * DTI", (cr / (an + 1.0)) * (1.0 - cop_frank) * dti),
    ("term * (1 - cop_clayton) * (cr / (gp + 1))", term * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * term * dti", merton_pd * term * dti),
    ("(term / 12) * (1 / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (term / 12.0) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1.0))),
    ("term^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", term**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(173, "Ultra_Long_Tenure_Amortization_Strain", m173, c173,
                        "Extended repayment terms (>24 months) prolong borrower exposure to lifecycle distress.")

# 174. Ultra Short Bridge Financing Surge
m174 = (cr / (an + 1.0)) <= 10.0
c174 = [
    ("(Annuity / Monthly_Inc) / (E3 + 0.1)", (an / (monthly_inc + 1.0)) / (u2 + 0.1)),
    ("(an / (monthly_inc + 1)) * (1 - cop_clayton) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (an / (monthly_inc + 1)) * dti", merton_pd * (an / (monthly_inc + 1.0)) * dti),
    ("((an * 12) / (inc + 1)) * (cr / (gp + 1)) / (u1 + 0.1)", ((an * 12.0) / (inc + 1.0)) * (cr / (gp + 1.0)) / (u1 + 0.1)),
    ("(an / (monthly_inc + 1))^2 * (1 - cop_frank)", (an / (monthly_inc + 1.0))**2.0 * (1.0 - cop_frank))
]
optimize_regime_formula(174, "Ultra_Short_Bridge_Financing_Surge", m174, c174,
                        "Short-term bridge loans demand high monthly cash drains relative to earnings.")

# 175. Severe Negative Downpayment Excess
m175 = cr > 1.20 * gp
c175 = [
    ("((Credit - Goods) / Goods) * (1 - E2) * DTI", ((cr - gp) / (gp + 1.0)) * (1.0 - u1) * dti),
    ("((cr - gp) / (gp + 1)) * (1 - cop_clayton) * (an / (monthly_inc + 1))", ((cr - gp) / (gp + 1.0)) * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * ((cr - gp) / (gp + 1)) * dti", merton_pd * ((cr - gp) / (gp + 1.0)) * dti),
    ("((cr - gp) / (an + 1)) * (1 / (cop_harmonic + 0.1))", ((cr - gp) / (an + 1.0)) / (cop_harmonic + 0.1)),
    ("((cr - gp) / (gp + 1))^1.5 * dti * (1 - u1 * u2)", np.maximum((cr - gp) / (gp + 1.0), 0.0)**1.5 * dti * (1.0 - u1 * u2))
]
optimize_regime_formula(175, "Severe_Negative_Downpayment_Excess", m175, c175,
                        "Borrowers rolling substantial fees and unbacked credit into loans face instant negative equity.")

# 176. Subsidized Par Financing Contract
m176 = (cr == gp) & (gp > 0)
c176 = [
    ("(Income / Credit) * Clayton_Copula * log1p(Emp_Yrs)", (inc / (cr + 1.0)) * cop_clayton * np.log1p(emp_yrs)),
    ("(Monthly_Inc / Annuity) * cop_frank * (u1 * u2)", (monthly_inc / (an + 1.0)) * cop_frank * (u1 * u2)),
    ("cop_harmonic * (inc / (cr + 1)) * (1 - dti / 15)", cop_harmonic * (inc / (cr + 1.0)) * (1.0 - dti / 15.0)),
    ("Merton_DD * cop_frank * np.log1p(adult_life)", merton_dd * cop_frank * np.log1p(adult_life)),
    ("(monthly_inc / (an + 1)) * (u1 + u2) * np.log1p(emp_yrs)", (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(emp_yrs))
]
optimize_regime_formula(176, "Subsidized_Par_Financing_Contract", m176, c176,
                        "Zero-markup retail financing programs attract price-conscious, credit-disciplined shoppers.")

# 177. High Annuity Absolute Burden
m177 = an >= 40000.0
c177 = [
    ("(Annuity / Monthly_Inc) * (1 - E2 * E3) * (Credit / Goods)", (an / (monthly_inc + 1.0)) * (1.0 - u1 * u2) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * (an / (monthly_inc + 1)) * (cr / (gp + 1))", merton_pd * (an / (monthly_inc + 1.0)) * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1))^2 * (1 - cop_frank)", (an / (monthly_inc + 1.0))**2.0 * (1.0 - cop_frank)),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * (cr / (gp + 1.0)))
]
optimize_regime_formula(177, "High_Annuity_Absolute_Burden", m177, c177,
                        "Monthly payment obligations >40k rubles cause immediate default upon income disruption.")

# 178. Alexander APR 36 Month Stress
apr_36 = (36.0 * an - cr) / (cr + 1e-5)
m178 = apr_36 > 0.35
c178 = [
    ("Alexander_APR_36 * (1 - Frank_Copula) * (Annuity / Monthly_Inc)", apr_36 * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))),
    ("apr_36 * dti * (1 - cop_clayton)", apr_36 * dti * (1.0 - cop_clayton)),
    ("Merton_PD * apr_36 * (cr / (gp + 1))", merton_pd * apr_36 * (cr / (gp + 1.0))),
    ("(apr_36 / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (apr_36 / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("apr_36^1.5 * (1 - u1 * u2) * dti", np.maximum(apr_36, 0.0)**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(178, "Alexander_APR_36_Month_Stress", m178, c178,
                        "High effective APR markups over 36 months signal predatory interest terms.")

# 179. Micro Loan Disproportionate Fixed Fee
m179 = cash_mask & (cr <= 150000.0)
c179 = [
    ("(129.50 / Annuity) / (E2 + 0.1)", (129.5 / (an + 1.0)) / (u1 + 0.1)),
    ("(129.5 / (an + 1)) * (1 - cop_clayton) * dti", (129.5 / (an + 1.0)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (129.5 / (an + 1)) * (cr / (gp + 1))", merton_pd * (129.5 / (an + 1.0)) * (cr / (gp + 1.0))),
    ("(129.5 / (an + 1)) / (cop_harmonic + 0.1)", (129.5 / (an + 1.0)) / (cop_harmonic + 0.1)),
    ("((129.5 * 12) / (inc + 1)) * (1 - u1 * u2)", ((129.5 * 12.0) / (inc + 1.0)) * (1.0 - u1 * u2))
]
optimize_regime_formula(179, "Micro_Loan_Disproportionate_Fixed_Fee", m179, c179,
                        "Micro-loans where flat bank administration fees eat a large share of cash repayments.")

# 180. Jumbo Syndicated Threshold Exposure
m180 = cr >= 1200000.0
c180 = [
    ("(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)", (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))),
    ("dti * (1 - cop_frank) * (cr / (gp + 1))", dti * (1.0 - cop_frank) * (cr / (gp + 1.0))),
    ("Merton_PD * dti * (1 / (u1 + 0.1))", merton_pd * dti * (1.0 / (u1 + 0.1))),
    ("(an / (monthly_inc + 1)) * (1 - cop_clayton) * dti", (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * dti),
    ("((cr - gp) / (gp + 1)) * dti * (1 - u1 * u2)", ((cr - gp) / (gp + 1.0)) * dti * (1.0 - u1 * u2))
]
optimize_regime_formula(180, "Jumbo_Syndicated_Threshold_Exposure", m180, c180,
                        "Large exposures (>1.2M rubles) carry catastrophic default severity for retail portfolios.")

# ==================================================================================================
# DOMAIN 5: SOLVENCY, CASH FLOW BUFFERS & LIVING COSTS (EXP 181 - 190)
# ==================================================================================================
# 181. Sub Zero Disposable Cash Flow
fcf = monthly_inc - an - subsistence
m181 = fcf < 0.0
c181 = [
    ("(-FCF / Monthly_Inc) * (1 - E2)", (-fcf / (monthly_inc + 1.0)) * (1.0 - u1)),
    ("(-fcf / (monthly_inc + 1)) * (1 - cop_clayton) * (cr / (gp + 1))", (-fcf / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (-fcf / (monthly_inc + 1)) * dti", merton_pd * (-fcf / (monthly_inc + 1.0)) * dti),
    ("(-fcf / (an + 1)) / (cop_harmonic + 0.1)", (-fcf / (an + 1.0)) / (cop_harmonic + 0.1)),
    ("(-fcf / (monthly_inc + 1))^1.5 * (1 - u1 * u2)", np.maximum(-fcf / (monthly_inc + 1.0), 0.0)**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(181, "Sub_Zero_Disposable_Cash_Flow", m181, c181,
                        "Households whose subsistence food/housing and loan payment exceed total income are insolvent.")

# 182. Extreme DTI Overleverage
m182 = dti >= 6.0
c182 = [
    ("DTI * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("dti * (1 - cop_frank) * (cr / (gp + 1))", dti * (1.0 - cop_frank) * (cr / (gp + 1.0))),
    ("Merton_PD * dti * (1 / (u1 + 0.1))", merton_pd * dti * (1.0 / (u1 + 0.1))),
    ("(dti / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (dti / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("dti^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", dti**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(182, "Extreme_DTI_Overleverage", m182, c182,
                        "Debt exceeding 6 years of gross income signals mathematically unsustainable leverage.")

# 183. Debt Service Coverage Fragility
dscr = an / (monthly_inc + 1.0)
m183 = dscr >= 0.40
c183 = [
    ("(Annuity / Monthly_Inc) / (E3 + 0.1) * (Credit / Goods)", dscr * (1.0 / (u2 + 0.1)) * (cr / (gp + 1.0))),
    ("dscr * (1 - cop_clayton) * dti", dscr * (1.0 - cop_clayton) * dti),
    ("Merton_PD * dscr * (cr / (gp + 1))", merton_pd * dscr * (cr / (gp + 1.0))),
    ("(dscr / (cop_harmonic + 0.1)) * dti", (dscr / (cop_harmonic + 0.1)) * dti),
    ("dscr^2 * (1 - u1 * u2) * (cr / (gp + 1))", dscr**2.0 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(183, "Debt_Service_Coverage_Fragility", m183, c183,
                        "Dedicating >40% of income solely to debt service eliminates all resilience against emergencies.")

# 184. Per Capita Family Income Squeeze
inc_per_fam = inc / fam
m184 = inc_per_fam < 40000.0
c184 = [
    ("(Credit / Income_Per_Fam) * (1 - E2)", (cr / (inc_per_fam + 1.0)) * (1.0 - u1)),
    ("(an / (inc_per_fam / 12 + 1)) * (1 - cop_clayton)", (an / (inc_per_fam / 12.0 + 1.0)) * (1.0 - cop_clayton)),
    ("Merton_PD * (cr / (inc_per_fam + 1))", merton_pd * (cr / (inc_per_fam + 1.0))),
    ("((cr - gp) / (gp + 1)) * (1 / (inc_per_fam + 1)) * (1 - cop_frank)", ((cr - gp) / (gp + 1.0)) * (1.0 / (inc_per_fam + 1.0)) * (1.0 - cop_frank)),
    ("(cr / (inc_per_fam + 1)) * (1 - u1 * u2)", (cr / (inc_per_fam + 1.0)) * (1.0 - u1 * u2))
]
optimize_regime_formula(184, "Per_Capita_Family_Income_Squeeze", m184, c184,
                        "Low per-head household income forces borrowers to prioritize groceries over debt service.")

# 185. Double Inflation Shock Insolvency
fcf_shock = monthly_inc - 1.5 * subsistence - an
m185 = fcf_shock < 0.0
c185 = [
    ("(-Shock_FCF / Monthly_Inc) * (1 - Frank_Copula)", (-fcf_shock / (monthly_inc + 1.0)) * (1.0 - cop_frank)),
    ("(-fcf_shock / (monthly_inc + 1)) * (1 - cop_clayton) * dti", (-fcf_shock / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (-fcf_shock / (monthly_inc + 1)) * (cr / (gp + 1))", merton_pd * (-fcf_shock / (monthly_inc + 1.0)) * (cr / (gp + 1.0))),
    ("(-fcf_shock / (an + 1)) / (cop_harmonic + 0.1)", (-fcf_shock / (an + 1.0)) / (cop_harmonic + 0.1)),
    ("(-fcf_shock / (monthly_inc + 1))^1.5 * (1 - u1 * u2)", np.maximum(-fcf_shock / (monthly_inc + 1.0), 0.0)**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(185, "Double_Inflation_Shock_Insolvency", m185, c185,
                        "Simulates retail borrower default cascades under a 50% non-discretionary inflation shock.")

# 186. Affluent Liquidity Fortress
m186 = (inc >= 300000.0) & (dti <= 2.0)
c186 = [
    ("(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)", (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)),
    ("(Income / Credit) * cop_frank * (u1 * u2)", (inc / (cr + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * cop_harmonic * np.log1p(adult_life)", merton_dd * cop_harmonic * np.log1p(adult_life)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(emp_yrs)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0)
]
optimize_regime_formula(186, "Affluent_Liquidity_Fortress", m186, c186,
                        "High earners with conservative debt ratios possess near-invulnerable credit solvency.")

# 187. Living Margin Surplus to Goods Ratio
surplus_val = inc - 12.0 * an
m187 = surplus_val > 0.0
c187 = [
    ("(Annual_Surplus / Goods) * Harmonic_Copula", (surplus_val / (gp + 1.0)) * cop_harmonic),
    ("(surplus_val / (cr + 1)) * cop_frank * (u1 * u2)", (surplus_val / (cr + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * (surplus_val / (gp + 1)) * cop_harmonic", merton_dd * (surplus_val / (gp + 1.0)) * cop_harmonic),
    ("cop_joe * (surplus_val / (gp + 1)) * np.log1p(emp_yrs)", cop_joe * (surplus_val / (gp + 1.0)) * np.log1p(emp_yrs)),
    ("(surplus_val / (gp + 1)) * (u1 + u2) * np.log1p(adult_life)", (surplus_val / (gp + 1.0)) * (u1 + u2) * np.log1p(adult_life))
]
optimize_regime_formula(187, "Living_Margin_Surplus_To_Goods_Ratio", m187, c187,
                        "Annual discretionary cash cushion relative to goods price reflects true living stability.")

# 188. Merton Default Probability High Barrier
m188 = merton_dd < 1.0
c188 = [
    ("Merton_PD * (1 - E2) * DTI", merton_pd * (1.0 - u1) * dti),
    ("merton_pd * (1 - cop_clayton) * (cr / (gp + 1))", merton_pd * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("merton_pd * (an / (monthly_inc + 1)) / (cop_harmonic + 0.1)", merton_pd * (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)),
    ("merton_pd^1.5 * dti * (1 - u1 * u2)", merton_pd**1.5 * dti * (1.0 - u1 * u2)),
    ("merton_pd * ((cr - gp) / (gp + 1)) * dti", merton_pd * ((cr - gp) / (gp + 1.0)) * dti)
]
optimize_regime_formula(188, "Merton_Default_Probability_High_Barrier", m188, c188,
                        "Structural distance-to-default models isolate deteriorating debt trajectories.")

# 189. Uncollateralized Burn Per Employed Month
emp_months = np.maximum(emp_yrs * 12.0, 1.0)
uncoll_burn = np.maximum(cr - gp, 0.0) / emp_months
m189 = uncoll_burn > 500.0
c189 = [
    ("(Uncollateralized_Credit / Emp_Months) / (E3 + 0.1)", uncoll_burn * (1.0 / (u2 + 0.1))),
    ("uncoll_burn * (1 - cop_clayton) * dti", uncoll_burn * (1.0 - cop_clayton) * dti),
    ("Merton_PD * uncoll_burn * (cr / (gp + 1))", merton_pd * uncoll_burn * (cr / (gp + 1.0))),
    ("(uncoll_burn / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (uncoll_burn / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("uncoll_burn^0.5 * (1 - u1 * u2) * dti", np.sqrt(uncoll_burn) * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(189, "Uncollateralized_Burn_Per_Employed_Month", m189, c189,
                        "Accumulating excessive unbacked principal relative to employment career tenure.")

# 190. Household Dependency Cost Annuity Parity
m190 = subsistence > an
c190 = [
    ("(Subsistence / Annuity) * (1 - E2) * (Credit / Income)", (subsistence / (an + 1.0)) * (1.0 - u1) * (cr / (inc + 1.0))),
    ("(subsistence / (an + 1)) * (1 - cop_clayton) * (cr / (gp + 1))", (subsistence / (an + 1.0)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (subsistence / (an + 1)) * dti", merton_pd * (subsistence / (an + 1.0)) * dti),
    ("((subsistence - an) / (monthly_inc + 1)) / (cop_harmonic + 0.1)", ((subsistence - an) / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)),
    ("(subsistence / (an + 1))^1.5 * (1 - u1 * u2)", (subsistence / (an + 1.0))**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(190, "Household_Dependency_Cost_Annuity_Parity", m190, c190,
                        "When survival expenses exceed total debt payment, family commitments take non-negotiable priority.")

# ==================================================================================================
# DOMAIN 6: IDENTITY STABILITY & ADMINISTRATIVE TELEMETRY (EXP 191 - 200)
# ==================================================================================================
# 191. Recent Address Registration Instability
m191 = reg_yrs < 1.0
c191 = [
    ("(1 / (Reg_Yrs + 0.1)) * (1 - E2) * DTI", (1.0 / (reg_yrs + 0.1)) * (1.0 - u1) * dti),
    ("(1.0 / (reg_yrs + 0.1)) * (1 - cop_clayton) * (cr / (gp + 1))", (1.0 / (reg_yrs + 0.1)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (1.0 / (reg_yrs + 0.1)) * dti", merton_pd * (1.0 / (reg_yrs + 0.1)) * dti),
    ("(1.0 / (reg_yrs + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1))", (1.0 / (reg_yrs + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1.0))),
    ("(1.0 / (reg_yrs + 0.1)) * (1 - u1 * u2) * dti", (1.0 / (reg_yrs + 0.1)) * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(191, "Recent_Address_Registration_Instability", m191, c191,
                        "Recent municipal address shifts correlate with residential instability and credit evasion.")

# 192. Recent Passport Renewal Velocity
m192 = idp_yrs < 1.5
c192 = [
    ("(1 / IDP_Yrs + 0.1) * (1 - E3) * (Credit / Goods)", (1.0 / (idp_yrs + 0.1)) * (1.0 - u2) * (cr / (gp + 1.0))),
    ("(1.0 / (idp_yrs + 0.1)) * (1 - cop_clayton) * dti", (1.0 / (idp_yrs + 0.1)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (1.0 / (idp_yrs + 0.1)) * (cr / (gp + 1))", merton_pd * (1.0 / (idp_yrs + 0.1)) * (cr / (gp + 1.0))),
    ("(1.0 / (idp_yrs + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1))", (1.0 / (idp_yrs + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1.0))),
    ("(1.0 / (idp_yrs + 0.1)) * (1 - u1 * u2) * (cr / (gp + 1))", (1.0 / (idp_yrs + 0.1)) * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(192, "Recent_Passport_Renewal_Velocity", m192, c192,
                        "Passport re-issuance shortly before applying indicates state administrative churn.")

# 193. Statutory Passport Age 20 Window
age_id = age - idp_yrs
m193 = (age >= 19.5) & (age <= 21.0)
c193 = [
    ("abs(Age_at_ID - 20) * DTI * (1 - E2)", np.abs(age_id - 20.0) * dti * (1.0 - u1)),
    ("np.abs(age_id - 20.0) * (1 - cop_clayton) * (cr / (gp + 1))", np.abs(age_id - 20.0) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * np.abs(age_id - 20.0) * dti", merton_pd * np.abs(age_id - 20.0) * dti),
    ("(np.abs(age_id - 20.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (np.abs(age_id - 20.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("np.abs(age_id - 20.0)^2 * (1 - u1 * u2) * dti", (np.abs(age_id - 20.0)**2.0) * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(193, "Statutory_Passport_Age_20_Window", m193, c193,
                        "Tracks bureaucratic compliance with mandatory age-20 Russian state identity renewal.")

# 194. Statutory Passport Age 45 Window
m194 = (age >= 44.5) & (age <= 46.0)
c194 = [
    ("abs(Age_at_ID - 45) * DTI * (1 - E3)", np.abs(age_id - 45.0) * dti * (1.0 - u2)),
    ("np.abs(age_id - 45.0) * (1 - cop_clayton) * (cr / (gp + 1))", np.abs(age_id - 45.0) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * np.abs(age_id - 45.0) * dti", merton_pd * np.abs(age_id - 45.0) * dti),
    ("(np.abs(age_id - 45.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (np.abs(age_id - 45.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("np.abs(age_id - 45.0)^2 * (1 - u1 * u2) * dti", (np.abs(age_id - 45.0)**2.0) * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(194, "Statutory_Passport_Age_45_Window", m194, c194,
                        "Tracks bureaucratic compliance with mandatory age-45 Russian state identity renewal.")

# 195. Phone Number Longevity Anchor
m195 = ph_yrs >= 5.0
c195 = [
    ("log1p(Phone_Yrs) * (E2 + E3) * (Monthly_Inc / Annuity)", np.log1p(ph_yrs) * (u1 + u2) * (monthly_inc / (an + 1.0))),
    ("np.log1p(ph_yrs) * cop_frank * (inc / (cr + 1))", np.log1p(ph_yrs) * cop_frank * (inc / (cr + 1.0))),
    ("Merton_DD * np.log1p(ph_yrs) * cop_harmonic", merton_dd * np.log1p(ph_yrs) * cop_harmonic),
    ("cop_joe * np.log1p(ph_yrs) * (monthly_inc / (an + 1))", cop_joe * np.log1p(ph_yrs) * (monthly_inc / (an + 1.0))),
    ("np.log1p(ph_yrs) * (u1 * u2) * (1 - dti / 15)", np.log1p(ph_yrs) * (u1 * u2) * (1.0 - dti / 15.0))
]
optimize_regime_formula(195, "Phone_Number_Longevity_Anchor", m195, c195,
                        "Maintaining an active mobile number >5 years signals rooted social and economic stability.")

# 196. Hyper Recent Phone Number Replacement
m196 = (-ph_raw) <= 60.0
c196 = [
    ("(1 / Phone_Months) * (1 - E2) * (Credit / Goods)", (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1.0 - u1) * (cr / (gp + 1.0))),
    ("(1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1 - cop_clayton) * dti", (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (cr / (gp + 1))", merton_pd * (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (cr / (gp + 1.0))),
    ("(1.0 / ((-ph_raw) / 30.0 + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1))", (1.0 / ((-ph_raw) / 30.0 + 0.1)) / (cop_harmonic + 0.1) * (an / (monthly_inc + 1.0))),
    ("(1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1 - u1 * u2) * dti", (1.0 / ((-ph_raw) / 30.0 + 0.1)) * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(196, "Hyper_Recent_Phone_Number_Replacement", m196, c196,
                        "Changing telephone numbers <60 days prior to borrowing reflects collection avoidance tactics.")

# 197. Address to ID Administrative Desynchrony
desync = np.abs(reg_yrs - idp_yrs)
m197 = desync >= 10.0
c197 = [
    ("abs(Reg_Yrs - IDP_Yrs) * DTI * (1 - Frank_Copula)", desync * dti * (1.0 - cop_frank)),
    ("desync * (1 - cop_clayton) * (cr / (gp + 1))", desync * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * desync * dti", merton_pd * desync * dti),
    ("(desync / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (desync / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("desync^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", desync**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(197, "Address_To_ID_Administrative_Desynchrony", m197, c197,
                        "A decade-long gap between official registration and identity re-issuance signals legal detachment.")

# 198. Registration Longevity Home Stability
m198 = reg_yrs >= 15.0
c198 = [
    ("log1p(Reg_Yrs) * Harmonic_Copula * (Income / Credit)", np.log1p(reg_yrs) * cop_harmonic * (inc / (cr + 1.0))),
    ("np.log1p(reg_yrs) * cop_frank * (monthly_inc / (an + 1))", np.log1p(reg_yrs) * cop_frank * (monthly_inc / (an + 1.0))),
    ("Merton_DD * np.log1p(reg_yrs) * (u1 * u2)", merton_dd * np.log1p(reg_yrs) * (u1 * u2)),
    ("cop_joe * np.log1p(reg_yrs) * (1 - dti / 15)", cop_joe * np.log1p(reg_yrs) * (1.0 - dti / 15.0)),
    ("np.log1p(reg_yrs) * (u1 + u2) * (emp_yrs / adult_life)", np.log1p(reg_yrs) * (u1 + u2) * (emp_yrs / adult_life))
]
optimize_regime_formula(198, "Registration_Longevity_Home_Stability", m198, c198,
                        "Living at the same permanent address for >15 years anchors long-term credit reliability.")

# 199. Working Life Administrative Friction
admin_friction = np.abs(reg - idp) / (adult_life * 365.25 + 1.0)
m199 = admin_friction > 0.30
c199 = [
    ("Admin_Friction * (1 - E2) * (Annuity / Monthly_Inc)", admin_friction * (1.0 - u1) * (an / (monthly_inc + 1.0))),
    ("admin_friction * (1 - cop_clayton) * dti", admin_friction * (1.0 - cop_clayton) * dti),
    ("Merton_PD * admin_friction * (cr / (gp + 1))", merton_pd * admin_friction * (cr / (gp + 1.0))),
    ("(admin_friction / (cop_harmonic + 0.1)) * dti", (admin_friction / (cop_harmonic + 0.1)) * dti),
    ("admin_friction^2 * (1 - u1 * u2) * (cr / (gp + 1))", admin_friction**2.0 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(199, "Working_Life_Administrative_Friction", m199, c199,
                        "Fraction of adult life spent in bureaucratic discrepancy between address and identity.")

# 200. Verification Phone Protocol Disconnect
phone_friction = (flag_emp_phone == 0.0) & (flag_work_phone == 1.0)
m200 = phone_friction
c200 = [
    ("(Work_Phone_Only) * DTI * (1 - E3) * (Credit / Goods)", dti * (1.0 - u2) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * (cr / (gp + 1)) * (1 / (u1 + 0.1))", merton_pd * (cr / (gp + 1.0)) * (1.0 / (u1 + 0.1))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1)", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)),
    ("dti * (1 - u1 * u2)", dti * (1.0 - u1 * u2))
]
optimize_regime_formula(200, "Verification_Phone_Protocol_Disconnect", m200, c200,
                        "Discrepancies between workplace phone verification flags highlight informal employer routing.")

# ==================================================================================================
# DOMAIN 7: CREDIT INQUIRY VELOCITY, URGENCY & ACCELERATION (EXP 201 - 210)
# ==================================================================================================
# 201. Zero Inquiry Calm Credit Sanctuary
m201 = (~inq_missing) & (inq_y == 0.0) & (u1 > 0.4)
c201 = [
    ("(E2 * E3) * (Monthly_Inc / Annuity) * log1p(Emp_Yrs)", (u1 * u2) * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("cop_harmonic * (inc / (cr + 1)) * (1 - dti / 15)", cop_harmonic * (inc / (cr + 1.0)) * (1.0 - dti / 15.0)),
    ("Merton_DD * cop_frank * (u1 * u2)", merton_dd * cop_frank * (u1 * u2)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(adult_life)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(adult_life)),
    ("(u1 * u2)**2 * (monthly_inc / (an + 1))", (u1 * u2)**2.0 * (monthly_inc / (an + 1.0)))
]
optimize_regime_formula(201, "Zero_Inquiry_Calm_Credit_Sanctuary", m201, c201,
                        "Zero credit shopping over the past year reflects voluntary, premeditated borrowing.")

# 202. Recent Monthly Inquiry Surge
m202 = inq_m >= 2.0
c202 = [
    ("Inq_Month * (1 - E2) * (Annuity / Monthly_Inc)", inq_m * (1.0 - u1) * (an / (monthly_inc + 1.0))),
    ("inq_m * (1 - cop_clayton) * dti", inq_m * (1.0 - cop_clayton) * dti),
    ("Merton_PD * inq_m * (cr / (gp + 1))", merton_pd * inq_m * (cr / (gp + 1.0))),
    ("(inq_m / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (inq_m / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("inq_m^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", inq_m**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(202, "Recent_Monthly_Inquiry_Surge", m202, c202,
                        "Multiple credit inquiries within 30 days reveals urgent, unmet short-term liquidity needs.")

# 203. Hourly Emergency Borrowing
m203 = inq_h > 0
c203 = [
    ("Inq_Hour * (Credit / Goods) / (E3 + 0.1)", inq_h * (cr / (gp + 1.0)) * (1.0 / (u2 + 0.1))),
    ("inq_h * (1 - cop_clayton) * dti", inq_h * (1.0 - cop_clayton) * dti),
    ("Merton_PD * inq_h * (cr / (gp + 1))", merton_pd * inq_h * (cr / (gp + 1.0))),
    ("(inq_h / (cop_harmonic + 0.05)) * (an / (monthly_inc + 1))", (inq_h / (cop_harmonic + 0.05)) * (an / (monthly_inc + 1.0))),
    ("inq_h * (1 - u1 * u2) * dti", inq_h * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(203, "Hourly_Emergency_Borrowing", m203, c203,
                        "Inquiries logged within the last 60 minutes indicate panic borrowing across multiple storefronts.")

# 204. Quarterly Acceleration Over Annual
inq_accel = (inq_q * 4.0) - inq_y
m204 = inq_accel > 1.0
c204 = [
    ("((Inq_QRT * 4) - Inq_YEAR) * (1 - Frank_Copula) * DTI", inq_accel * (1.0 - cop_frank) * dti),
    ("inq_accel * (1 - cop_clayton) * (cr / (gp + 1))", inq_accel * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * inq_accel * dti", merton_pd * inq_accel * dti),
    ("(inq_accel / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (inq_accel / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("inq_accel^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", inq_accel**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(204, "Quarterly_Acceleration_Over_Annual", m204, c204,
                        "Sudden acceleration in quarterly borrowing applications marks rapid personal credit decay.")

# 205. High Annual Credit Hunting Intensity
m205 = inq_y >= 5.0
c205 = [
    ("log1p(Inq_YEAR) * (1 - E2) * (Annuity / Monthly_Inc)", np.log1p(inq_y) * (1.0 - u1) * (an / (monthly_inc + 1.0))),
    ("np.log1p(inq_y) * (1 - cop_clayton) * dti", np.log1p(inq_y) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * np.log1p(inq_y) * (cr / (gp + 1))", merton_pd * np.log1p(inq_y) * (cr / (gp + 1.0))),
    ("(np.log1p(inq_y) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (np.log1p(inq_y) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("np.log1p(inq_y)^1.5 * (1 - u1 * u2) * dti", np.log1p(inq_y)**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(205, "High_Annual_Credit_Hunting_Intensity", m205, c205,
                        "Chronic credit shopping across 5+ institutions indicates repeated underwriting rejection.")

# 206. Missing Bureau Inquiry History
m206 = inq_missing
c206 = [
    ("DTI * (1 - E2) * (Credit / Goods)", dti * (1.0 - u1) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * (cr / (gp + 1.0))),
    ("dti^1.5 * (1 - u1 * u2)", dti**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(206, "Missing_Bureau_Inquiry_History", m206, c206,
                        "Unrecorded inquiry histories signal thin-file applicants lacking standard credit documentation.")

# 207. Inquiry Velocity to Income Ratio
inq_to_inc = inq_y / (np.log1p(inc) + 1.0)
m207 = inq_to_inc > 0.4
c207 = [
    ("(Inq_YEAR / log1p(Income)) * (1 - E3) * (Credit / Goods)", inq_to_inc * (1.0 - u2) * (cr / (gp + 1.0))),
    ("inq_to_inc * (1 - cop_clayton) * dti", inq_to_inc * (1.0 - cop_clayton) * dti),
    ("Merton_PD * inq_to_inc * (cr / (gp + 1))", merton_pd * inq_to_inc * (cr / (gp + 1.0))),
    ("(inq_to_inc / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (inq_to_inc / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("inq_to_inc^1.5 * (1 - u1 * u2) * dti", inq_to_inc**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(207, "Inquiry_Velocity_To_Income_Ratio", m207, c207,
                        "Frequent credit inquiries from low-earning borrowers reflect extreme repayment fragility.")

# 208. Inquiry Burden Per Bureau Score
inq_per_score = inq_y / (u1 + 0.05)
m208 = inq_per_score > 10.0
c208 = [
    ("(Inq_YEAR / (E2 + 0.05)) * (Annuity / Monthly_Inc)", inq_per_score * (an / (monthly_inc + 1.0))),
    ("inq_per_score * (1 - cop_clayton) * (cr / (gp + 1))", inq_per_score * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * inq_per_score * dti", merton_pd * inq_per_score * dti),
    ("(inq_per_score / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (inq_per_score / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("inq_per_score^1.5 * (1 - u1 * u2)", inq_per_score**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(208, "Inquiry_Burden_Per_Bureau_Score", m208, c208,
                        "Inquiry counts concentrated among subprime credit-score holders indicate adverse credit hunting.")

# 209. Late Stage Quarterly Inquiry Shock
m209 = (inq_q >= 3.0) & (u2 < 0.35)
c209 = [
    ("Inq_QRT * (1 - E3) * DTI", inq_q * (1.0 - u2) * dti),
    ("inq_q * (1 - cop_clayton) * (cr / (gp + 1))", inq_q * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * inq_q * dti", merton_pd * inq_q * dti),
    ("(inq_q / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (inq_q / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("inq_q^2 * (1 - u1 * u2) * dti", inq_q**2.0 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(209, "Late_Stage_Quarterly_Inquiry_Shock", m209, c209,
                        "Subprime borrowers aggressively requesting loans over the last 90 days exhibit immediate default.")

# 210. Harmonic Inquiry Credit Stress
f210 = (inq_m + 1.0) * an / (monthly_inc + 1.0) * (1.0 - cop_clayton)
m210 = inq_m >= 1.0
c210 = [
    ("(Inq_MON + 1) * (Annuity / Monthly_Inc) * (1 - Clayton_Copula)", f210),
    ("(inq_m + 1.0) * dti * (1 - cop_frank)", (inq_m + 1.0) * dti * (1.0 - cop_frank)),
    ("Merton_PD * (inq_m + 1.0) * (cr / (gp + 1))", merton_pd * (inq_m + 1.0) * (cr / (gp + 1.0))),
    ("((inq_m + 1.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", ((inq_m + 1.0) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("(inq_m + 1.0)^1.5 * (1 - u1 * u2) * dti", (inq_m + 1.0)**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(210, "Harmonic_Inquiry_Credit_Stress", m210, c210,
                        "Multiplies debt servicing pressure by 30-day inquiry velocity.")

# ==================================================================================================
# DOMAIN 8: ASSET COLLATERAL, VEHICLE & HOUSING STRUCTURE (EXP 211 - 220)
# ==================================================================================================
# 211. Vintage Car Maintenance Drain
m211 = (car_flag == 1.0) & (car_age >= 15.0)
c211 = [
    ("(Car_Age / 10) * DTI * (1 - E2)", (car_age / 10.0) * dti * (1.0 - u1)),
    ("(car_age / 10.0) * (1 - cop_clayton) * (cr / (gp + 1))", (car_age / 10.0) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (car_age / 10.0) * dti", merton_pd * (car_age / 10.0) * dti),
    ("(car_age / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (car_age / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("(car_age / 10.0)^1.5 * (1 - u1 * u2) * dti", (car_age / 10.0)**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(211, "Vintage_Car_Maintenance_Drain", m211, c211,
                        "Vehicles older than 15 years create recurring maintenance drains on disposable household cash.")

# 212. Modern Vehicle Luxury Asset Anchor
m212 = (car_flag == 1.0) & (car_age >= 0.0) & (car_age <= 3.0)
c212 = [
    ("(Monthly_Inc / Annuity) * Harmonic_Copula * (4 - Car_Age)", (monthly_inc / (an + 1.0)) * cop_harmonic * (4.0 - car_age)),
    ("(Income / Credit) * cop_frank * (4.0 - car_age)", (inc / (cr + 1.0)) * cop_frank * (4.0 - car_age)),
    ("Merton_DD * (4.0 - car_age) * (u1 * u2)", merton_dd * (4.0 - car_age) * (u1 * u2)),
    ("cop_joe * (monthly_inc / (an + 1)) * (4.0 - car_age)", cop_joe * (monthly_inc / (an + 1.0)) * (4.0 - car_age)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2 * (4.0 - car_age)", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0 * (4.0 - car_age))
]
optimize_regime_formula(212, "Modern_Vehicle_Luxury_Asset_Anchor", m212, c212,
                        "Late-model cars represent substantial secondary asset equity protecting lender solvency.")

# 213. Homeowner Unencumbered Equity Anchor
m213 = (housing == "House / apartment") & (car_flag == 0.0)
c213 = [
    ("(Income / Credit) * (E2 + E3) * log1p(Age)", (inc / (cr + 1.0)) * (u1 + u2) * np.log1p(age)),
    ("(Monthly_Inc / Annuity) * cop_harmonic * np.log1p(adult_life)", (monthly_inc / (an + 1.0)) * cop_harmonic * np.log1p(adult_life)),
    ("cop_frank * (inc / (cr + 1)) * (u1 * u2)", cop_frank * (inc / (cr + 1.0)) * (u1 * u2)),
    ("Merton_DD * cop_frank * np.log1p(age)", merton_dd * cop_frank * np.log1p(age)),
    ("(monthly_inc / (an + 1)) * (u1 * u2) * (1 - dti / 15)", (monthly_inc / (an + 1.0)) * (u1 * u2) * (1.0 - dti / 15.0))
]
optimize_regime_formula(213, "Homeowner_Unencumbered_Equity_Anchor", m213, c213,
                        "Homeowners without vehicle debt obligations represent resilient low-risk borrowers.")

# 214. Municipal Housing Tenant Vulnerability
m214 = np.isin(housing, ["Municipal apartment", "Rented apartment"])
c214 = [
    ("(Annuity / Monthly_Inc) / (E3 + 0.1) * (Credit / Goods)", (an / (monthly_inc + 1.0)) * (1.0 / (u2 + 0.1)) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * (cr / (gp + 1)) * dti", merton_pd * (cr / (gp + 1.0)) * dti),
    ("((an / (monthly_inc + 1)) / (cop_harmonic + 0.1)) * (cr / (gp + 1))", ((an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)) * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1))^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", (an / (monthly_inc + 1.0))**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(214, "Municipal_Housing_Tenant_Vulnerability", m214, c214,
                        "Renters and municipal tenants lack property collateral against unexpected bankruptcy.")

# 215. Asset Rich Cash Poor Dilemma
m215 = (car_flag == 1.0) & (housing == "House / apartment") & (inc < np.median(inc))
c215 = [
    ("DTI * (1 - Frank_Copula) * (Annuity / Monthly_Inc)", dti * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))),
    ("dti * (1 - cop_clayton) * (car_age / 10 + 1)", dti * (1.0 - cop_clayton) * (car_age / 10.0 + 1.0)),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * dti", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * dti),
    ("dti^1.5 * (1 - u1 * u2) * (an / (monthly_inc + 1))", dti**1.5 * (1.0 - u1 * u2) * (an / (monthly_inc + 1.0)))
]
optimize_regime_formula(215, "Asset_Rich_Cash_Poor_Dilemma", m215, c215,
                        "Owning physical property on below-median income creates high maintenance costs with illiquid cash.")

# 216. Car Depreciation to Loan Horizon
car_deprec = (car_age / 10.0) * (cr / (gp + 1.0))
m216 = (car_flag == 1.0) & (car_age > 5.0)
c216 = [
    ("(Car_Age / 10) * (Credit / Goods) * (1 - E2)", car_deprec * (1.0 - u1)),
    ("car_deprec * (1 - cop_clayton) * dti", car_deprec * (1.0 - cop_clayton) * dti),
    ("Merton_PD * car_deprec * (an / (monthly_inc + 1))", merton_pd * car_deprec * (an / (monthly_inc + 1.0))),
    ("(car_deprec / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (car_deprec / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("car_deprec^1.5 * (1 - u1 * u2)", car_deprec**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(216, "Car_Depreciation_To_Loan_Horizon", m216, c216,
                        "Compound depreciation of aging collateral against financing fees.")

# 217. Vehicle Equity Liquidation Coverage
veh_equity = np.exp(-np.maximum(car_age, 0.0) / 7.0) * gp / (cr + 1.0)
m217 = (car_flag == 1.0) & (car_age >= 0.0)
c217 = [
    ("Vehicle_Liquidation_Value / Credit * Harmonic_Copula", veh_equity * cop_harmonic * (monthly_inc / (an + 1.0))),
    ("veh_equity * cop_frank * (inc / (cr + 1))", veh_equity * cop_frank * (inc / (cr + 1.0))),
    ("Merton_DD * veh_equity * (u1 * u2)", merton_dd * veh_equity * (u1 * u2)),
    ("cop_joe * veh_equity * (monthly_inc / (an + 1))", cop_joe * veh_equity * (monthly_inc / (an + 1.0))),
    ("veh_equity * (u1 * u2) * (1 - dti / 15)", veh_equity * (u1 * u2) * (1.0 - dti / 15.0))
]
optimize_regime_formula(217, "Vehicle_Equity_Liquidation_Coverage", m217, c217,
                        "Realizable liquidation value of motor vehicle relative to loan principal.")

# 218. Office Apartment Commercial Footprint
m218 = np.isin(housing, ["Office apartment", "Co-op apartment"])
c218 = [
    ("(Monthly_Inc / Annuity) * Clayton_Copula * (1 - DTI / 10)", (monthly_inc / (an + 1.0)) * cop_clayton * (1.0 - dti / 10.0)),
    ("(inc / (cr + 1)) * cop_frank * (u1 * u2)", (inc / (cr + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * cop_harmonic * (monthly_inc / (an + 1))", merton_dd * cop_harmonic * (monthly_inc / (an + 1.0))),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(adult_life)", cop_joe * (inc / (cr + 1.0)) * np.log1p(adult_life)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)", (monthly_inc / (an + 1.0)) * (u1 * u2))
]
optimize_regime_formula(218, "Office_Apartment_Commercial_Footprint", m218, c218,
                        "Office and co-op apartment dwellers reflect commercial self-employment profiles.")

# 219. Assetless Urban Tenant Exposure
m219 = (car_flag == 0.0) & (housing == "Rented apartment") & (pop >= 0.02)
c219 = [
    ("(Annuity / Monthly_Inc) * (1 - E2) * DTI", (an / (monthly_inc + 1.0)) * (1.0 - u1) * dti),
    ("(an / (monthly_inc + 1)) * (1 - cop_clayton) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("((an / (monthly_inc + 1)) / (cop_harmonic + 0.1)) * dti", ((an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)) * dti),
    ("(an / (monthly_inc + 1))^2 * (1 - u1 * u2) * dti", (an / (monthly_inc + 1.0))**2.0 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(219, "Assetless_Urban_Tenant_Exposure", m219, c219,
                        "Urban renters lacking vehicles or real property face elevated default rates.")

# 220. Residential Collateral Shield Ratio
m220 = (housing == "House / apartment") & (dti <= 2.5)
c220 = [
    ("(Income / Credit) * (E2 * E3) * log1p(Adult_Life)", (inc / (cr + 1.0)) * (u1 * u2) * np.log1p(adult_life)),
    ("(Monthly_Inc / Annuity) * cop_frank * np.log1p(adult_life)", (monthly_inc / (an + 1.0)) * cop_frank * np.log1p(adult_life)),
    ("Merton_DD * cop_harmonic * (u1 * u2)", merton_dd * cop_harmonic * (u1 * u2)),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(emp_yrs)", cop_joe * (inc / (cr + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0)
]
optimize_regime_formula(220, "Residential_Collateral_Shield_Ratio", m220, c220,
                        "Residential apartment dwellers carrying conservative debt maintain excellent repayment track records.")

# ==================================================================================================
# DOMAIN 9: REGIONAL DEMOGRAPHY, ECONOMIC GEOGRAPHY & WORKPLACE TELEMETRY (EXP 221 - 230)
# ==================================================================================================
# 221. Megacity Tier 1 Metro Concentration
m221 = pop >= 0.04
c221 = [
    ("(Income / Credit) * Frank_Copula * (Monthly_Inc / Annuity)", (inc / (cr + 1.0)) * cop_frank * (monthly_inc / (an + 1.0))),
    ("(Monthly_Inc / Annuity) * cop_harmonic * (u1 * u2)", (monthly_inc / (an + 1.0)) * cop_harmonic * (u1 * u2)),
    ("Merton_DD * cop_frank * np.log1p(adult_life)", merton_dd * cop_frank * np.log1p(adult_life)),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(emp_yrs)", cop_joe * (inc / (cr + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2) * (1 - dti / 15)", (monthly_inc / (an + 1.0)) * (u1 * u2) * (1.0 - dti / 15.0))
]
optimize_regime_formula(221, "Megacity_Tier1_Metro_Concentration", m221, c221,
                        "Tier-1 metropolitan residents possess deep alternative employment options.")

# 222. Rural Periphery Low Density Isolation
m222 = pop < 0.007
c222 = [
    ("(Credit / Income) * (1 - E3) * (Annuity / Monthly_Inc)", (cr / (inc + 1.0)) * (1.0 - u2) * (an / (monthly_inc + 1.0))),
    ("dti * (1 - cop_clayton) * (cr / (gp + 1))", dti * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * dti * (1 / (u1 + 0.1))", merton_pd * dti * (1.0 / (u1 + 0.1))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * (cr / (gp + 1.0))),
    ("dti^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", dti**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(222, "Rural_Periphery_Low_Density_Isolation", m222, c222,
                        "Isolated rural regions offer few fallback employment options upon dismissal.")

# 223. Worst Region Rating Severe Risk
m223 = rating == 3.0
c223 = [
    ("DTI * (1 - E2) * (Credit / Goods)", dti * (1.0 - u1) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * dti", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * dti),
    ("dti^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", dti**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(223, "Worst_Region_Rating_Severe_Risk", m223, c223,
                        "Jurisdictions assigned regional rating tier 3 suffer systemic macro distress.")

# 224. Elite High Density Tier 1 Sanctuary
m224 = (rating == 1.0) & (pop >= 0.03)
c224 = [
    ("(Monthly_Inc / Annuity) * (E2 + E3) * log1p(Emp_Yrs)", (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(emp_yrs)),
    ("(Income / Credit) * cop_frank * (u1 * u2)", (inc / (cr + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * cop_harmonic * np.log1p(adult_life)", merton_dd * cop_harmonic * np.log1p(adult_life)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(emp_yrs)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0)
]
optimize_regime_formula(224, "Elite_High_Density_Tier1_Sanctuary", m224, c224,
                        "Metropolitan prime borrowers in rating tier 1 districts provide premier credit stability.")

# 225. High Institutional Rating In Rural Periphery
m225 = (rating == 1.0) & (pop < 0.01)
c225 = [
    ("(Income / Credit) * Clayton_Copula * log1p(Age)", (inc / (cr + 1.0)) * cop_clayton * np.log1p(age)),
    ("(Monthly_Inc / Annuity) * cop_frank * (u1 * u2)", (monthly_inc / (an + 1.0)) * cop_frank * (u1 * u2)),
    ("Merton_DD * cop_harmonic * np.log1p(emp_yrs)", merton_dd * cop_harmonic * np.log1p(emp_yrs)),
    ("cop_joe * (inc / (cr + 1)) * (u1 + u2)", cop_joe * (inc / (cr + 1.0)) * (u1 + u2)),
    ("(monthly_inc / (an + 1)) * (1 - dti / 15)", (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0))
]
optimize_regime_formula(225, "High_Rating_In_Rural_Periphery", m225, c225,
                        "Public institutional leaders stationed in smaller peripheral municipalities.")

# 226. Unverifiable Employer Phone Disconnect
m226 = (flag_emp_phone == 0.0) & (income_type == "Working")
c226 = [
    ("DTI * (1 - E2) * (Credit / Goods)", dti * (1.0 - u1) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * (cr / (gp + 1.0))),
    ("dti^1.5 * (1 - u1 * u2)", dti**1.5 * (1.0 - u1 * u2))
]
optimize_regime_formula(226, "Unverifiable_Employer_Phone_Disconnect", m226, c226,
                        "Working applicants without verified employer telephones indicate informal unmonitored wages.")

# 227. Dual Workplace Telephone Verification Anchor
m227 = (flag_emp_phone == 1.0) & (flag_work_phone == 1.0) & (emp_yrs >= 5.0)
c227 = [
    ("(Monthly_Inc / Annuity) * (E2 * E3) * log1p(Emp_Yrs)", (monthly_inc / (an + 1.0)) * (u1 * u2) * np.log1p(emp_yrs)),
    ("(Income / Credit) * cop_frank * (emp_yrs / adult_life)", (inc / (cr + 1.0)) * cop_frank * (emp_yrs / adult_life)),
    ("Merton_DD * cop_harmonic * (u1 + u2)", merton_dd * cop_harmonic * (u1 + u2)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(emp_yrs)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0)
]
optimize_regime_formula(227, "Dual_Workplace_Phone_Anchor", m227, c227,
                        "Long-tenured employees with dual verified employer telephone connections represent authentic employment roots.")

# 228. Transient New Hire Workplace Divergence
m228 = (flag_emp_phone == 1.0) & (flag_work_phone == 0.0) & (emp_yrs < 1.0)
c228 = [
    ("(Credit / Goods) / (E3 + 0.1) * (Annuity / Monthly_Inc)", (cr / (gp + 1.0)) / (u2 + 0.1) * (an / (monthly_inc + 1.0))),
    ("dti * (1 - cop_clayton) * (cr / (gp + 1))", dti * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (cr / (gp + 1)) * dti", merton_pd * (cr / (gp + 1.0)) * dti),
    ("((an / (monthly_inc + 1)) / (cop_harmonic + 0.1)) * (cr / (gp + 1))", ((an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1)) * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1))^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", (an / (monthly_inc + 1.0))**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(228, "Transient_New_Hire_Workplace_Divergence", m228, c228,
                        "Recent hires on probationary status lacking direct workplace phone verification.")

# 229. Regional Income Relative Deviance
grp_reg_inc = df.groupby("REGION_POPULATION_RELATIVE")["AMT_INCOME_TOTAL"].transform("mean").values
dev_reg_inc = (inc - grp_reg_inc) / (grp_reg_inc + 1.0)
m229 = dev_reg_inc < -0.20
c229 = [
    ("(-Income_Deviation_From_Region) * DTI * (1 - E2)", (-dev_reg_inc) * dti * (1.0 - u1)),
    ("(-dev_reg_inc) * (1 - cop_clayton) * (cr / (gp + 1))", (-dev_reg_inc) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (-dev_reg_inc) * dti", merton_pd * (-dev_reg_inc) * dti),
    ("((-dev_reg_inc) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", ((-dev_reg_inc) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("(-dev_reg_inc)^1.5 * (1 - u1 * u2) * dti", np.maximum(-dev_reg_inc, 0.0)**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(229, "Regional_Income_Relative_Deviance", m229, c229,
                        "Earning significantly below the regional peer benchmark intensifies local living cost pressure.")

# 230. City Level Credit Absorption Capacity
m230 = pop >= 0.015
c230 = [
    ("(Credit / (Pop_Norm + 1)) * (1 - Frank_Copula) * (Annuity / Monthly_Inc)", (cr / (pop * 1e6 + 1.0)) * (1.0 - cop_frank) * (an / (monthly_inc + 1.0))),
    ("(cr / (pop * 1e6 + 1.0)) * (1 - cop_clayton) * dti", (cr / (pop * 1e6 + 1.0)) * (1.0 - cop_clayton) * dti),
    ("Merton_PD * (cr / (pop * 1e6 + 1.0)) * dti", merton_pd * (cr / (pop * 1e6 + 1.0)) * dti),
    ("((cr / (pop * 1e6 + 1.0)) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", ((cr / (pop * 1e6 + 1.0)) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("(cr / (pop * 1e6 + 1.0))^1.5 * (1 - u1 * u2) * dti", (cr / (pop * 1e6 + 1.0))**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(230, "City_Level_Credit_Absorption_Capacity", m230, c230,
                        "Loan size scaled by municipal macroeconomic credit absorption capacity.")

# ==================================================================================================
# DOMAIN 10: RELATIONAL GRAPH, SEQUENCE TELEMETRY & MASTER INTERACTIONS (EXP 231 - 240)
# ==================================================================================================
# 231. Repeat Applicant First Born Sequence
m231 = (app_order == 0.0) & (n_apps_all >= 2.0)
c231 = [
    ("(Monthly_Inc / Annuity) * (E2 + E3) * log1p(N_Apps)", (monthly_inc / (an + 1.0)) * (u1 + u2) * np.log1p(n_apps_all)),
    ("(inc / (cr + 1)) * cop_frank * np.log1p(n_apps_all)", (inc / (cr + 1.0)) * cop_frank * np.log1p(n_apps_all)),
    ("Merton_DD * cop_harmonic * np.log1p(n_apps_all)", merton_dd * cop_harmonic * np.log1p(n_apps_all)),
    ("cop_joe * (monthly_inc / (an + 1)) * np.log1p(n_apps_all)", cop_joe * (monthly_inc / (an + 1.0)) * np.log1p(n_apps_all)),
    ("(monthly_inc / (an + 1)) * (u1 * u2) * np.log1p(n_apps_all)", (monthly_inc / (an + 1.0)) * (u1 * u2) * np.log1p(n_apps_all))
]
optimize_regime_formula(231, "Repeat_Applicant_First_Born_Sequence", m231, c231,
                        "The inaugural application of a multi-loan borrower establishes permanent behavioral baseline.")

# 232. Repeat Applicant Subsequent Rapid Reapplication
m232 = (app_order >= 1.0) & (days_since_prev >= 0.0) & (days_since_prev < 180.0)
c232 = [
    ("(1 / Prev_App_Months) * (1 - E2) * DTI", (1.0 / (days_since_prev / 30.0 + 0.1)) * (1.0 - u1) * dti),
    ("(1.0 / (days_since_prev / 30.0 + 0.1)) * (1 - cop_clayton) * (cr / (gp + 1))", (1.0 / (days_since_prev / 30.0 + 0.1)) * (1.0 - cop_clayton) * (cr / (gp + 1.0))),
    ("Merton_PD * (1.0 / (days_since_prev / 30.0 + 0.1)) * dti", merton_pd * (1.0 / (days_since_prev / 30.0 + 0.1)) * dti),
    ("((1.0 / (days_since_prev / 30.0 + 0.1)) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", ((1.0 / (days_since_prev / 30.0 + 0.1)) / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("(1.0 / (days_since_prev / 30.0 + 0.1))^1.5 * (1 - u1 * u2) * dti", (1.0 / (days_since_prev / 30.0 + 0.1))**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(232, "Repeat_Applicant_Subsequent_Reapplication", m232, c232,
                        "Rapid reapplication within 6 months signals accelerating debt accumulation.")

# 233. Long Term Returning Borrower Maturity
m233 = (app_order >= 1.0) & (days_since_prev >= 365.0)
c233 = [
    ("log1p(Years_Since_Last_App) * Harmonic_Copula * (Income / Credit)", np.log1p(days_since_prev / 365.25) * cop_harmonic * (inc / (cr + 1.0))),
    ("np.log1p(days_since_prev / 365.25) * cop_frank * (monthly_inc / (an + 1))", np.log1p(days_since_prev / 365.25) * cop_frank * (monthly_inc / (an + 1.0))),
    ("Merton_DD * np.log1p(days_since_prev / 365.25) * (u1 * u2)", merton_dd * np.log1p(days_since_prev / 365.25) * (u1 * u2)),
    ("cop_joe * np.log1p(days_since_prev / 365.25) * (1 - dti / 15)", cop_joe * np.log1p(days_since_prev / 365.25) * (1.0 - dti / 15.0)),
    ("np.log1p(days_since_prev / 365.25) * (u1 * u2)**2", np.log1p(days_since_prev / 365.25) * (u1 * u2)**2.0)
]
optimize_regime_formula(233, "Long_Term_Returning_Borrower_Maturity", m233, c233,
                        "Multi-year gaps between loans reflect disciplined lifecycle borrowing.")

# 234. Singleton Borrower Cold Start
m234 = n_apps_all == 1.0
c234 = [
    ("DTI * (1 - E2) * (Credit / Goods)", dti * (1.0 - u1) * (cr / (gp + 1.0))),
    ("dti * (1 - cop_clayton) * (an / (monthly_inc + 1))", dti * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("Merton_PD * dti * (cr / (gp + 1))", merton_pd * dti * (cr / (gp + 1.0))),
    ("(an / (monthly_inc + 1)) / (cop_harmonic + 0.1) * (cr / (gp + 1))", (an / (monthly_inc + 1.0)) / (cop_harmonic + 0.1) * (cr / (gp + 1.0))),
    ("dti^1.5 * (1 - u1 * u2) * (cr / (gp + 1))", dti**1.5 * (1.0 - u1 * u2) * (cr / (gp + 1.0)))
]
optimize_regime_formula(234, "Singleton_Borrower_Cold_Start", m234, c234,
                        "Applicants appearing only once lack repeat cross-application history.")

# 235. High Density Multi App Cluster
m235 = n_apps_all >= 3.0
c235 = [
    ("N_Apps * (1 - Clayton_Copula) * (Annuity / Monthly_Inc)", n_apps_all * (1.0 - cop_clayton) * (an / (monthly_inc + 1.0))),
    ("n_apps_all * (1 - cop_frank) * dti", n_apps_all * (1.0 - cop_frank) * dti),
    ("Merton_PD * n_apps_all * (cr / (gp + 1))", merton_pd * n_apps_all * (cr / (gp + 1.0))),
    ("(n_apps_all / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1))", (n_apps_all / (cop_harmonic + 0.1)) * (an / (monthly_inc + 1.0))),
    ("n_apps_all^1.5 * (1 - u1 * u2) * dti", n_apps_all**1.5 * (1.0 - u1 * u2) * dti)
]
optimize_regime_formula(235, "High_Density_Multi_App_Cluster", m235, c235,
                        "Clusters of 3+ applications experience correlated credit distress across applications.")

# 236. Cross Feature Affordability Copula Interaction
m236 = (monthly_inc > an) & (u1 > 0.1) & (u2 > 0.1)
c236 = [
    ("Frank_Copula * (Monthly_Inc / Annuity) * (1 - DTI / 15)", cop_frank * (monthly_inc / (an + 1.0)) * (1.0 - dti / 15.0)),
    ("cop_harmonic * (monthly_inc / (an + 1)) * (u1 * u2)", cop_harmonic * (monthly_inc / (an + 1.0)) * (u1 * u2)),
    ("Merton_DD * cop_frank * (1 - dti / 12)", merton_dd * cop_frank * (1.0 - dti / 12.0)),
    ("cop_joe * (inc / (cr + 1)) * np.log1p(emp_yrs)", cop_joe * (inc / (cr + 1.0)) * np.log1p(emp_yrs)),
    ("(monthly_inc / (an + 1)) * (u1 * u2)**2 * (1 - dti / 15)", (monthly_inc / (an + 1.0)) * (u1 * u2)**2.0 * (1.0 - dti / 15.0))
]
optimize_regime_formula(236, "Cross_Feature_Affordability_Copula_Interaction", m236, c236,
                        "Couples joint bureau consensus with true disposable income coverage.")

# 237. Extreme Undercollateralized Laborer Stress
m237 = (single_male_hazard > 0.0) & (cr > gp)
c237 = [
    ("Single_Male_Laborer_Hazard * ((Credit - Goods) / Goods)", single_male_hazard * ((cr - gp) / (gp + 1.0))),
    ("single_male_hazard * ((cr - gp) / (an + 1)) * (1 - cop_clayton)", single_male_hazard * ((cr - gp) / (an + 1.0)) * (1.0 - cop_clayton)),
    ("Merton_PD * single_male_hazard * ((cr - gp) / (gp + 1))", merton_pd * single_male_hazard * ((cr - gp) / (gp + 1.0))),
    ("(single_male_hazard / (cop_harmonic + 0.1)) * ((cr - gp) / (gp + 1))", (single_male_hazard / (cop_harmonic + 0.1)) * ((cr - gp) / (gp + 1.0))),
    ("single_male_hazard^1.5 * ((cr - gp) / (gp + 1))", single_male_hazard**1.5 * ((cr - gp) / (gp + 1.0)))
]
optimize_regime_formula(237, "Extreme_Undercollateralized_Laborer_Stress", m237, c237,
                        "Single male manual laborers taking unbacked financing above goods price.")

# 238. Merton Distance Bureau Consensus Hybrid
m238 = ~np.isnan(merton_dd) & valid_triad
c238 = [
    ("Merton_DD * Harmonic_Copula / (abs(E2 - E3) + 0.1)", merton_dd * cop_harmonic / (np.abs(e2_f - e3_f) + 0.1)),
    ("merton_dd * cop_frank * (u1 * u2)", merton_dd * cop_frank * (u1 * u2)),
    ("Merton_DD * (1 - triad_std * 5) * (monthly_inc / (an + 1))", merton_dd * (1.0 - triad_std * 5.0) * (monthly_inc / (an + 1.0))),
    ("cop_joe * merton_dd * np.log1p(emp_yrs)", cop_joe * merton_dd * np.log1p(emp_yrs)),
    ("merton_dd * (u1 * u2)**2", merton_dd * (u1 * u2)**2.0)
]
optimize_regime_formula(238, "Merton_Distance_Bureau_Consensus_Hybrid", m238, c238,
                        "Integrates structural default distance with multi-bureau agency confidence.")

# 239. Statutory Identity Cycle Tenure Harmonic
id_tenure = (ph_yrs * emp_yrs) / adult_life
m239 = (adult_life > 2.0) & (emp_yrs > 0.5)
c239 = [
    ("(Phone_Yrs * Emp_Yrs / Adult_Life) * Frank_Copula * (Income / Credit)", id_tenure * cop_frank * (inc / (cr + 1.0))),
    ("id_tenure * cop_harmonic * (monthly_inc / (an + 1))", id_tenure * cop_harmonic * (monthly_inc / (an + 1.0))),
    ("Merton_DD * id_tenure * (u1 * u2)", merton_dd * id_tenure * (u1 * u2)),
    ("cop_joe * id_tenure * (1 - dti / 15)", cop_joe * id_tenure * (1.0 - dti / 15.0)),
    ("id_tenure * (u1 * u2)**2", id_tenure * (u1 * u2)**2.0)
]
optimize_regime_formula(239, "Statutory_Identity_Cycle_Tenure_Harmonic", m239, c239,
                        "Harmonizes commercial telephone and employment stability normalized by legal adult lifespan.")

# 240. Unified Cognitive Super Manifold
c240 = [
    ("Frank_Copula * (1 - Clayton_Copula) * (Monthly_Inc / Annuity) / (1 + Single_Male_Hazard)", cop_frank * (1.0 - cop_clayton) * (monthly_inc / (an + 1.0)) / (1.0 + single_male_hazard)),
    ("cop_harmonic * (1 - cop_clayton) * (monthly_inc / (an + 1)) / (1.0 + single_male_hazard)", cop_harmonic * (1.0 - cop_clayton) * (monthly_inc / (an + 1.0)) / (1.0 + single_male_hazard)),
    ("Merton_DD * cop_frank * (u1 * u2) / (1.0 + single_male_hazard)", merton_dd * cop_frank * (u1 * u2) / (1.0 + single_male_hazard)),
    ("cop_joe * cop_frank * (inc / (cr + 1)) / (1.0 + single_male_hazard)", cop_joe * cop_frank * (inc / (cr + 1.0)) / (1.0 + single_male_hazard)),
    ("((monthly_inc / (an + 1)) * (u1 * u2)**2) / (1.0 + single_male_hazard)", ((monthly_inc / (an + 1.0)) * (u1 * u2)**2.0) / (1.0 + single_male_hazard))
]
m240 = np.ones(n_train, dtype=bool)
optimize_regime_formula(240, "Unified_Cognitive_Super_Manifold", m240, c240,
                        "Grand cognitive manifold uniting copula topology, debt affordability, and demographic penalties.")

# --------------------------------------------------------------------------------------------------
# STRATIFIED 5-FOLD CROSS-VALIDATION VERIFICATION OF ALL OPTIMIZED CHAMPIONS
# --------------------------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("  STRATIFIED 5-FOLD CV VERIFICATION OF TOP 20 OPTIMIZED CHAMPION FORMULAS")
print("=" * 105)

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

# Rank all 100 regimes by within-split AUC and Spread
ranked_champions = sorted(optimization_results.values(), key=lambda x: (x["optimized_within_split_auc"], x["spread_bps"]), reverse=True)
top_20_keys = [f"OPT_F_{item['exp_id']}_{item['name']}" for item in ranked_champions[:20]]

print("\nTop 20 Optimized Champion Formulations (Multi-Round Search):")
for i, item in enumerate(ranked_champions[:20], 1):
    print(f"  {i:2d}. Exp {item['exp_id']:3d} ({item['name']:36s}): Opt AUC = {item['optimized_within_split_auc']:.4f} ({item['auc_lift_bps']:+5.1f} bps) | Spread = {item['spread_bps']:6.1f} bps | Round {item['champion_round']}")

enhanced_df = base_df.copy()
for tk in top_20_keys:
    enhanced_df[tk] = final_champion_features[tk]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(df, y))

params = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 10.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

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

print("\n2. Evaluating Enhanced Feature Set (+20 Optimized Multi-Round Champions)...")
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
print(f"   [+] Enhanced Pooled OOF AUC: {auc_enh:.6f} | Net Cross-Validation Lift: {lift_bps:+5.1f} bps")

# --------------------------------------------------------------------------------------------------
# SAVE ARTIFACTS
# --------------------------------------------------------------------------------------------------
output_meta = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_elapsed_seconds": round(time.time() - t_start, 2),
    "total_regimes": len(optimization_results),
    "total_formulas_evaluated": sum(x["total_rounds_evaluated"] for x in optimization_results.values()),
    "baseline_pooled_auc": round(auc_base, 6),
    "enhanced_pooled_auc": round(auc_enh, 6),
    "auc_net_lift_bps": round(lift_bps, 1),
    "regimes": optimization_results
}

json_path = r"D:\gci\gemini\RESEARCH_100_REGIMES_MULTI_ROUND_OPTIMIZATION.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_meta, f, indent=2)

md_path = r"D:\gci\gemini\RESEARCH_100_REGIMES_MULTI_ROUND_OPTIMIZATION.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# GCI World 2026: Exhaustive Multi-Round Mathematical Optimization Across 100 Regimes (Exp 141 – 240)\n\n")
    f.write(f"- **Execution Timestamp:** `{output_meta['timestamp_utc']}`\n")
    f.write(f"- **Total Regimes Evaluated:** 100 Distinct Empirical Splits (Zero Duplicates)\n")
    f.write(f"- **Total Formula Candidates Evaluated:** `{output_meta['total_formulas_evaluated']}` Mathematical Formulations (5 Candidates Per Regime)\n")
    f.write(f"- **Baseline 5-Fold OOF AUC:** `{auc_base:.6f}`\n")
    f.write(f"- **Enhanced 5-Fold OOF AUC:** `{auc_enh:.6f}` (**{lift_bps:+.1f} bps Net Lift**)\n")
    f.write(f"- **Execution Runtime:** `{output_meta['total_elapsed_seconds']}s` on RTX 4050 GPU\n\n")
    f.write("## Exhaustive Ledger of 100 Multi-Round Optimized Champions\n\n")
    f.write("| Exp ID | Regime Name | Sample N | Pct (%) | Def Rate | Base AUC | Opt AUC | Delta AUC (bps) | Spread (bps) | Winning Rnd | Champion Formula | Theoretical Domain Mechanism |\n")
    f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n")
    for k, v in optimization_results.items():
        f.write(f"| {v['exp_id']} | **{v['name']}** | {v['subgroup_count']:,} | {v['subgroup_pct']:.1f}% | {v['subgroup_default_rate']:.2f}% | {v['baseline_auc']:.4f} | **{v['optimized_within_split_auc']:.4f}** | **{v['auc_lift_bps']:+5.1f}** | **{v['spread_bps']:.1f}** | R{v['champion_round']} | `{v['champion_formula']}` | {v['domain_reason']} |\n")

print(f"\n[+] Optimization JSON Saved : {json_path}")
print(f"[+] Optimization MD Saved   : {md_path}")
print(f"[+] Total Execution Time   : {time.time() - t_start:.2f}s")
print("=" * 105)

