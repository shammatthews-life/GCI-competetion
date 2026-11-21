"""
====================================================================================================
GCI WORLD 2026: DEEP MATHEMATICAL ATTRIBUTION & CATEGORICAL AUC SENSITIVITY ENGINE
QUANTIFYING THE EXACT MECHANISMS, INFORMATION VALUES (IV), AND MARGINAL AUC LIFTS DRIVING 0.77+
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
print("=" * 105)
print("  GCI WORLD 2026: DEEP MATHEMATICAL ATTRIBUTION & CATEGORICAL AUC SENSITIVITY ENGINE")
print("=" * 105)

train_path = r"D:\gci\inpuy\train.csv"
df = pd.read_csv(train_path)
y = df["TARGET"].values
n_train = len(df)
overall_prior = float(y.mean())
print(f"[+] Loaded {n_train:,} training records (Prior Default Rate = {overall_prior*100:.2f}%) in {time.time() - t_start:.2f}s.")

# ==================================================================================================
# 1. MATHEMATICAL WEIGHT OF EVIDENCE (WoE) & INFORMATION VALUE (IV) ANALYSIS FOR ALL CATEGORIES
# ==================================================================================================
print("\n" + "=" * 105)
print("  PHASE 1: INFORMATION VALUE (IV) & WOE ENTROPY ANALYSIS FOR ALL CATEGORICAL VARIABLES")
print("=" * 105)

CATS = [
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "CODE_GENDER",
    "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE"
]

cat_metrics = {}

print(f"{'Category Column':25s} | {'Unique':6s} | {'Total IV':9s} | {'Predictive Power Class':25s} | {'Top Risk Subgroup (Default %)':35s}")
print("-" * 115)

for col in CATS:
    s = df[col].fillna("MISSING").astype(str)
    counts = s.value_counts()
    
    # Calculate WoE and IV
    sub_df = pd.DataFrame({"val": s, "y": y})
    grp = sub_df.groupby("val")["y"].agg(["count", "sum"])
    grp["non_events"] = grp["count"] - grp["sum"]
    grp["events"] = grp["sum"]
    
    total_events = grp["events"].sum()
    total_non_events = grp["non_events"].sum()
    
    # Avoid zero division
    grp["dist_events"] = np.clip(grp["events"] / (total_events + 1e-6), 1e-5, 1.0)
    grp["dist_non_events"] = np.clip(grp["non_events"] / (total_non_events + 1e-6), 1e-5, 1.0)
    
    grp["woe"] = np.log(grp["dist_non_events"] / grp["dist_events"])
    grp["iv"] = (grp["dist_non_events"] - grp["dist_events"]) * grp["woe"]
    total_iv = float(grp["iv"].sum())
    
    if total_iv < 0.02:
        power = "Unpredictive (<0.02)"
    elif total_iv < 0.10:
        power = "Weak (0.02 - 0.10)"
    elif total_iv < 0.30:
        power = "Medium (0.10 - 0.30)"
    elif total_iv < 0.50:
        power = "Strong (0.30 - 0.50)"
    else:
        power = "Suspiciously High (>0.50)"
        
    grp["default_rate"] = grp["sum"] / grp["count"]
    top_risk = grp.sort_values("count", ascending=False)
    # Filter groups with at least 500 records
    sig_risk = top_risk[top_risk["count"] >= 500].sort_values("default_rate", ascending=False)
    if len(sig_risk) > 0:
        top_name = sig_risk.index[0]
        top_def = sig_risk["default_rate"].iloc[0] * 100
        top_str = f"{top_name[:20]} ({top_def:.1f}%)"
    else:
        top_str = "N/A"
        
    print(f"{col:25s} | {len(counts):6d} | {total_iv:9.4f} | {power:25s} | {top_str:35s}")
    
    cat_metrics[col] = {
        "unique_categories": len(counts),
        "information_value": round(total_iv, 4),
        "predictive_power": power,
        "highest_risk_subgroup": top_str,
        "distribution": {
            str(k): {
                "count": int(v["count"]),
                "default_rate": round(float(v["default_rate"] * 100), 2),
                "woe": round(float(v["woe"]), 4),
                "iv": round(float(v["iv"]), 4)
            }
            for k, v in grp.iterrows()
        }
    }

# ==================================================================================================
# 2. FEATURE GROUP ABLATION & INCREMENTAL AUC SENSITIVITY (5-FOLD CV)
# ==================================================================================================
print("\n" + "=" * 105)
print("  PHASE 2: INCREMENTAL FEATURE BLOCKS & ABLATION STUDY (STRATIFIED 5-FOLD CV)")
print("=" * 105)

# Prepare all feature groups from scratch
inc = df["AMT_INCOME_TOTAL"].values.astype(float)
cr = df["AMT_CREDIT"].values.astype(float)
an = df["AMT_ANNUITY"].values.astype(float)
gp = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"]).values.astype(float)
birth = df["DAYS_BIRTH"].values.astype(float)
emp_raw = df["DAYS_EMPLOYED"].values.astype(float)
days_emp = np.where(emp_raw == 365243.0, np.nan, -emp_raw)
emp_yrs = np.nan_to_num(days_emp, nan=0.0) / 365.25
age = -birth / 365.25
adult_life = np.maximum(age - 18.0, 1.0)
reg = -df["DAYS_REGISTRATION"].values.astype(float) / 365.25
idp = -df["DAYS_ID_PUBLISH"].values.astype(float) / 365.25
ph_yrs = -df["DAYS_LAST_PHONE_CHANGE"].fillna(0).values.astype(float) / 365.25
monthly_inc = inc / 12.0
dti = np.clip(cr / (inc + 1.0), 0.0, 30.0)
term = np.clip(cr / (an + 1.0), 1.0, 100.0)

e1 = df["EXT_SOURCE_1"].values.astype(float)
e2 = df["EXT_SOURCE_2"].values.astype(float)
e3 = df["EXT_SOURCE_3"].values.astype(float)
u1 = np.clip(np.nan_to_num(e2, nan=float(np.nanmedian(e2))), 1e-4, 1.0 - 1e-4)
u2 = np.clip(np.nan_to_num(e3, nan=float(np.nanmedian(e3))), 1e-4, 1.0 - 1e-4)
u0 = np.clip(np.nan_to_num(e1, nan=float(np.nanmedian(e1))), 1e-4, 1.0 - 1e-4)

# Invariant identity entity key
entity_key = (
    (df["DAYS_BIRTH"] - df["DAYS_REGISTRATION"]).round(0).astype(int).astype(str) + "|" +
    (df["DAYS_BIRTH"] - df["DAYS_ID_PUBLISH"]).round(0).astype(int).astype(str) + "|" +
    df["CODE_GENDER"].fillna("").astype(str) + "|" +
    df["NAME_EDUCATION_TYPE"].fillna("").astype(str)
)
cnt_ent = entity_key.value_counts()
n_apps_all = entity_key.map(cnt_ent).values.astype(float)

# Copulas
cop_clayton = np.maximum(u1**(-3.0) + u2**(-3.0) - 1.0, 1e-6)**(-1.0 / 3.0)
num_f = (np.exp(-4.0 * u1) - 1.0) * (np.exp(-4.0 * u2) - 1.0)
cop_frank = -0.25 * np.log(1.0 + num_f / (np.exp(-4.0) - 1.0))
cop_harmonic = 2.0 / ((1.0 / u1) + (1.0 / u2))

# Hazard
is_male = (df["CODE_GENDER"].values == "M")
is_single = np.isin(df["NAME_FAMILY_STATUS"].fillna("").values, ["Single / not married", "Separated", "Widow"])
is_laborer = np.isin(df["OCCUPATION_TYPE"].fillna("").values, ["Laborers", "Drivers", "Low-skill Laborers"])
single_male_hazard = (is_male & is_single & is_laborer).astype(float) * dti * (cr / (gp + 1.0)) * (1.0 - u1)

# Tariff keys
cash_mask = (df["NAME_CONTRACT_TYPE"] == "Cash loans").values
qb = (an - 129.5) / (cr + 1e-5)
q0_fee = np.where(cash_mask & (qb > 0), qb, an / (cr + 1e-5))
q3 = np.round(q0_fee, 3)
q4 = np.round(q0_fee, 4)
p3 = np.round(cr / (gp + 1e-5), 3)
ct1 = np.round(cr / (an + 1e-5), 1)

feature_blocks = {
    "Block_01_Raw_Bureaus": pd.DataFrame({
        "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3
    }),
    "Block_02_Bureau_Manifolds": pd.DataFrame({
        "EXT_MEAN": np.nanmean(np.column_stack([e1, e2, e3]), axis=1),
        "EXT_STD": np.nan_to_num(np.nanstd(np.column_stack([e1, e2, e3]), axis=1), nan=0.0),
        "EXT_MIN": np.nanmin(np.column_stack([e1, e2, e3]), axis=1),
        "COPULA_CLAYTON": cop_clayton,
        "COPULA_FRANK": cop_frank,
        "COPULA_HARMONIC": cop_harmonic
    }),
    "Block_03_Loan_Terms_Tariffs": pd.DataFrame({
        "AMT_CREDIT": cr, "AMT_ANNUITY": an, "AMT_GOODS_PRICE": gp,
        "CREDIT_TERM": term, "Q0_FEE": q0_fee, "Q3": q3, "Q4": q4, "P3": p3, "CT1": ct1,
        "NEGATIVE_DOWNPAYMENT": np.maximum(cr - gp, 0.0) / (gp + 1.0)
    }),
    "Block_04_Income_Solvency": pd.DataFrame({
        "AMT_INCOME_TOTAL": inc, "DTI": dti, "MONTHLY_INC": monthly_inc,
        "DISPOSABLE_BURDEN": an / (monthly_inc + 1.0),
        "SURPLUS": inc - 12.0 * an,
        "SUBSISTENCE_FCF": (monthly_inc - an - (12000.0 * df["CNT_CHILDREN"].values + 15000.0)) / (monthly_inc + 1.0)
    }),
    "Block_05_Demographics_Lifespan": pd.DataFrame({
        "DAYS_BIRTH": birth, "AGE_YEARS": age, "EMP_YEARS": emp_yrs,
        "ADULT_LIFE": adult_life, "EMP_TO_AGE": emp_yrs / (adult_life + 1.0),
        "IS_RETIRED": (emp_raw == 365243.0).astype(float),
        "SINGLE_MALE_HAZARD": single_male_hazard
    }),
    "Block_06_Administrative_Tenure": pd.DataFrame({
        "DAYS_REGISTRATION": reg, "DAYS_ID_PUBLISH": idp, "DAYS_LAST_PHONE_CHANGE": ph_yrs,
        "REG_TO_ID_LAG": np.abs(reg - idp),
        "PHONE_TO_EMP_RATIO": ph_yrs / (emp_yrs + 0.1)
    }),
    "Block_07_Relational_Entity_Graph": pd.DataFrame({
        "N_APPS_ALL": n_apps_all,
        "INV_REG_BIRTH": (df["DAYS_REGISTRATION"] - birth).values,
        "INV_IDP_BIRTH": (df["DAYS_ID_PUBLISH"] - birth).values
    }),
    "Block_08_Credit_Inquiries": pd.DataFrame({
        "INQ_HOUR": df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0).values,
        "INQ_MON": df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0).values,
        "INQ_QRT": df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0).values,
        "INQ_YEAR": df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0).values,
        "INQ_MISSING": df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(float).values
    })
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(df, y))

lgb_params = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 10.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

# 1. Evaluate Cumulative Block Additions
cumulative_df = pd.DataFrame(index=np.arange(n_train))
cumulative_results = {}
prev_auc = 0.50

print(f"{'Feature Block Added':35s} | {'Block Cols':10s} | {'Cumulative Cols':15s} | {'5-Fold Pooled AUC':18s} | {'Marginal Lift':15s}")
print("-" * 115)

for block_name, block_data in feature_blocks.items():
    cumulative_df = pd.concat([cumulative_df, block_data], axis=1)
    oof_pred = np.zeros(n_train)
    
    for trn_idx, val_idx in splits:
        X_tr, y_tr = cumulative_df.iloc[trn_idx], y[trn_idx]
        X_va, y_va = cumulative_df.iloc[val_idx], y[val_idx]
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        m = lgb.train(lgb_params, trn_data, num_boost_round=450, valid_sets=[val_data], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_pred[val_idx] = m.predict(X_va, num_iteration=m.best_iteration)
        
    auc_score = roc_auc_score(y, oof_pred)
    lift = (auc_score - prev_auc) * 10000 if prev_auc > 0.50 else 0.0
    print(f"{block_name:35s} | {block_data.shape[1]:10d} | {cumulative_df.shape[1]:15d} | {auc_score:18.6f} | {lift:+15.1f} bps")
    
    cumulative_results[block_name] = {
        "columns_in_block": block_data.shape[1],
        "cumulative_columns": cumulative_df.shape[1],
        "pooled_oof_auc": round(auc_score, 6),
        "marginal_lift_bps": round(lift, 1)
    }
    prev_auc = auc_score

# ==================================================================================================
# 3. EVALUATING CATEGORICAL TARGET ENCODING BOOST
# ==================================================================================================
print("\n" + "=" * 105)
print("  PHASE 3: MARGINAL AUC GAIN FROM HERMETIC IN-FOLD NESTED TARGET ENCODINGS")
print("=" * 105)

# Add Target Encodings of top categories to the cumulative feature set
TE_CATS = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "Q3", "CT1"]

oof_te = np.zeros(n_train)
for trn_idx, val_idx in splits:
    X_tr = cumulative_df.iloc[trn_idx].copy()
    y_tr = y[trn_idx]
    X_va = cumulative_df.iloc[val_idx].copy()
    y_va = y[val_idx]
    
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prior = float(y_tr.mean())
    
    for c in TE_CATS:
        col_series = df.iloc[trn_idx][c] if c in df.columns else cumulative_df.iloc[trn_idx][c]
        val_series = df.iloc[val_idx][c] if c in df.columns else cumulative_df.iloc[val_idx][c]
        
        # Out-of-inner-fold target encoding for train
        te_trn = np.full(len(X_tr), prior, dtype=float)
        for in_trn, in_val in inner_cv.split(X_tr, y_tr):
            grp = pd.DataFrame({"k": col_series.iloc[in_trn].values, "y": y_tr[in_trn]}).groupby("k")["y"].agg(["count", "mean"])
            k_val = col_series.iloc[in_val]
            cnt = k_val.map(grp["count"]).fillna(0.0).values
            mean_y = k_val.map(grp["mean"]).fillna(prior).values
            te_trn[in_val] = (cnt * mean_y + 50.0 * prior) / (cnt + 50.0)
        X_tr[f"TE_{c}"] = te_trn
        
        # Map to validation from full outer trn_idx
        grp_full = pd.DataFrame({"k": col_series.values, "y": y_tr}).groupby("k")["y"].agg(["count", "mean"])
        cnt_val = val_series.map(grp_full["count"]).fillna(0.0).values
        mean_val = val_series.map(grp_full["mean"]).fillna(prior).values
        X_va[f"TE_{c}"] = (cnt_val * mean_val + 50.0 * prior) / (cnt_val + 50.0)
        
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    m = lgb.train(lgb_params, trn_data, num_boost_round=450, valid_sets=[val_data], callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_te[val_idx] = m.predict(X_va, num_iteration=m.best_iteration)

auc_te = roc_auc_score(y, oof_te)
te_lift = (auc_te - prev_auc) * 10000
print(f"[+] With In-Fold Nested Target Encodings (+6 Core Keys): {auc_te:.6f} ({te_lift:+5.1f} bps lift over raw blocks)")

# ==================================================================================================
# 4. SAVE COMPREHENSIVE RESEARCH REPORT & MANIFEST
# ==================================================================================================
report_meta = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_elapsed_seconds": round(time.time() - t_start, 2),
    "categorical_iv_metrics": cat_metrics,
    "feature_block_ablation": cumulative_results,
    "target_encoding_boost": {
        "final_auc_with_te": round(auc_te, 6),
        "net_te_lift_bps": round(te_lift, 1),
        "keys_encoded": TE_CATS
    }
}

json_path = r"D:\gci\gemini\RESEARCH_DEEP_MATHEMATICAL_ATTRIBUTION.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report_meta, f, indent=2)

md_path = r"D:\gci\gemini\RESEARCH_DEEP_MATHEMATICAL_ATTRIBUTION.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# GCI World 2026: Deep Mathematical Feature Attribution & Categorical Sensitivity Analysis\n\n")
    f.write(f"- **Execution Timestamp:** `{report_meta['timestamp_utc']}`\n")
    f.write(f"- **Runtime:** `{report_meta['total_elapsed_seconds']}s` on RTX 4050 GPU\n")
    f.write(f"- **Final 5-Fold Pooled OOF AUC:** `{auc_te:.6f}`\n\n")
    
    f.write("## 1. Information Value (IV) & Entropy Hierarchy of All Categories\n\n")
    f.write("| Category Feature | Unique Classes | Information Value (IV) | Predictive Power Class | Highest-Risk Subgroup (Default %) |\n")
    f.write("| :--- | :---: | :---: | :--- | :--- |\n")
    for k, v in cat_metrics.items():
        f.write(f"| **`{k}`** | {v['unique_categories']} | **{v['information_value']:.4f}** | {v['predictive_power']} | {v['highest_risk_subgroup']} |\n")
        
    f.write("\n## 2. Feature Block Incremental AUC Sensitivity (5-Fold CV Ablation)\n\n")
    f.write("| Feature Block Name | Block Features | Cumulative Features | 5-Fold Pooled OOF AUC | Marginal Lift (bps) | Underlying Mathematical Mechanism |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
    f.write(f"| **Raw External Bureaus** | 3 | 3 | {cumulative_results['Block_01_Raw_Bureaus']['pooled_oof_auc']:.6f} | — | Base credit bureau agency scores (E1, E2, E3). |\n")
    f.write(f"| **Bureau Copula Manifolds** | 6 | 9 | {cumulative_results['Block_02_Bureau_Manifolds']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_02_Bureau_Manifolds']['marginal_lift_bps']:+.1f}** | Archimedean Clayton/Frank/Harmonic lower-tail non-linear distress. |\n")
    f.write(f"| **Loan Terms & Tariff Decoders** | 10 | 19 | {cumulative_results['Block_03_Loan_Terms_Tariffs']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_03_Loan_Terms_Tariffs']['marginal_lift_bps']:+.1f}** | Standard retail bank pricing lattices (Q3, Q4, CT1, Negative Downpayment). |\n")
    f.write(f"| **Income & Solvency Buffers** | 6 | 25 | {cumulative_results['Block_04_Income_Solvency']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_04_Income_Solvency']['marginal_lift_bps']:+.1f}** | Debt-to-income (DTI) and household subsistence cash flow stress. |\n")
    f.write(f"| **Demographics & Lifespan** | 7 | 32 | {cumulative_results['Block_05_Demographics_Lifespan']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_05_Demographics_Lifespan']['marginal_lift_bps']:+.1f}** | Career velocity, retirement flags, and Single Male Laborer Hazard. |\n")
    f.write(f"| **Administrative Tenure** | 5 | 37 | {cumulative_results['Block_06_Administrative_Tenure']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_06_Administrative_Tenure']['marginal_lift_bps']:+.1f}** | Address/ID registration lags and phone number tenure longevity. |\n")
    f.write(f"| **Relational Entity Graph** | 3 | 40 | {cumulative_results['Block_07_Relational_Entity_Graph']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_07_Relational_Entity_Graph']['marginal_lift_bps']:+.1f}** | Invariant multi-application clustering (`entity_key`, `N_APPS_ALL`). |\n")
    f.write(f"| **Credit Bureau Inquiries** | 5 | 45 | {cumulative_results['Block_08_Credit_Inquiries']['pooled_oof_auc']:.6f} | **{cumulative_results['Block_08_Credit_Inquiries']['marginal_lift_bps']:+.1f}** | Trailing inquiry velocity, quarterly acceleration, and missingness flags. |\n")
    f.write(f"| **Hermetic Nested Target Encodings** | 6 | 51 | **{auc_te:.6f}** | **{te_lift:+.1f}** | In-fold inner cross-fitted Bayesian target encodings on top categorical keys. |\n")

print(f"\n[+] Analysis JSON Saved : {json_path}")
print(f"[+] Analysis MD Saved   : {md_path}")
print(f"[+] Total Execution Time: {time.time() - t_start:.2f}s")
print("=" * 105)

