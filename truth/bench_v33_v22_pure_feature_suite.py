import time
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
print("[*] Diagnostic Part 3: Full V22 Feature Reconstruction (Pure Covariates, Zero Leakage)")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

# --- V28 Standard Features ---
cmaps = v28.fit_category_maps(df_ia)
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), res_ia], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)

m_base = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_f, y_ia)
p_base = m_base.predict_proba(Xib_f)[:, 1]
print(f"[V28 Base Single Model] AUC = {roc_auc_score(y_ib, p_base):.6f}")

# --- Extract Authentic V22 Pure Covariate Features Fitted Strictly on df_ref ---
def extract_v22_pure_covariates(df_src, df_ref):
    """
    Extracts authentic V22 features (clean ratios, external interactions, bureau ratios,
    frequency counts, group deviations, and demographic neighbor match statistics)
    fitted strictly on df_ref (df_ia) without ever seeing validation or test labels.
    """
    d = {}
    # 1. Authentic V22 External Products & Ratios
    e1, e2, e3 = df_src["EXT_SOURCE_1"], df_src["EXT_SOURCE_2"], df_src["EXT_SOURCE_3"]
    d["V22_EXT_W"] = (e1 * 2.0 + e2 * 3.0 + e3 * 4.0) / 9.0
    d["V22_EXT_RANGE"] = e3 - e1
    d["V22_EXT_PROD"] = e1 * e2 * e3
    d["V22_E12"] = e1 * e2
    d["V22_E13"] = e1 * e3
    d["V22_E23"] = e2 * e3
    d["V22_E1_MIN_E2"] = e1 - e2
    d["V22_E1_MIN_E3"] = e1 - e3

    # 2. Authentic V22 Financial & Bureau Terms
    c, g, a, inc = df_src["AMT_CREDIT"], df_src["AMT_GOODS_PRICE"], df_src["AMT_ANNUITY"], df_src["AMT_INCOME_TOTAL"]
    d["V22_ANN_TO_INC"] = a / inc
    d["V22_CR_TO_INC"] = c / inc
    d["V22_GP_TO_INC"] = g / inc
    d["V22_CR_TO_ANN"] = c / a
    d["V22_BUR_RECENT"] = df_src["AMT_REQ_CREDIT_BUREAU_HOUR"] + df_src["AMT_REQ_CREDIT_BUREAU_MON"]
    d["V22_BUR_MON_YEAR"] = df_src["AMT_REQ_CREDIT_BUREAU_MON"] / (df_src["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1.0)
    
    # 3. Authentic V22 Group Means Fitted STRICTLY on df_ref
    for grp in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE"]:
        ref_means = df_ref.groupby(grp)["AMT_CREDIT"].mean()
        d[f"V22_G_{grp[:6]}_CR_DIFF"] = (c - df_src[grp].map(ref_means)).astype(np.float32)
        ref_inc_means = df_ref.groupby(grp)["AMT_INCOME_TOTAL"].mean()
        d[f"V22_G_{grp[:6]}_INC_DIFF"] = (inc - df_src[grp].map(ref_inc_means)).astype(np.float32)

    # 4. Authentic V22 Clean Applicant-Link Demographics (Zero Target Access)
    ref_keys = (
        df_ref["CODE_GENDER"].astype(str) + "|" +
        df_ref["NAME_EDUCATION_TYPE"].astype(str) + "|" +
        df_ref["REGION_POPULATION_RELATIVE"].astype(str) + "|" +
        (df_ref["DAYS_REGISTRATION"] - df_ref["DAYS_BIRTH"]).astype(str) + "|" +
        (df_ref["DAYS_ID_PUBLISH"] - df_ref["DAYS_BIRTH"]).astype(str)
    )
    src_keys = (
        df_src["CODE_GENDER"].astype(str) + "|" +
        df_src["NAME_EDUCATION_TYPE"].astype(str) + "|" +
        df_src["REGION_POPULATION_RELATIVE"].astype(str) + "|" +
        (df_src["DAYS_REGISTRATION"] - df_src["DAYS_BIRTH"]).astype(str) + "|" +
        (df_src["DAYS_ID_PUBLISH"] - df_src["DAYS_BIRTH"]).astype(str)
    )
    ref_counts = ref_keys.value_counts()
    ref_ext2 = df_ref.groupby(ref_keys)["EXT_SOURCE_2"].mean()
    ref_cr = df_ref.groupby(ref_keys)["AMT_CREDIT"].mean()
    
    d["V22_APP_COUNT"] = src_keys.map(ref_counts).fillna(0).values.astype(np.float32)
    d["V22_APP_HAS_PARTNER"] = (d["V22_APP_COUNT"] > 1).astype(np.float32)
    d["V22_APP_EXT2_DIFF"] = (df_src["EXT_SOURCE_2"].values - src_keys.map(ref_ext2)).astype(np.float32)
    d["V22_APP_CR_RATIO"] = (c.values / np.maximum(src_keys.map(ref_cr), 1.0)).astype(np.float32)

    return pd.DataFrame(d, index=df_src.index)

f22_ia = extract_v22_pure_covariates(df_ia, df_ia).reset_index(drop=True)
f22_ib = extract_v22_pure_covariates(df_ib, df_ia).reset_index(drop=True)

Xia_full = pd.concat([Xia_f.reset_index(drop=True), f22_ia], axis=1)
Xib_full = pd.concat([Xib_f.reset_index(drop=True), f22_ib], axis=1)

m_full = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.45, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_full, y_ia)
p_full = m_full.predict_proba(Xib_full)[:, 1]
print(f"[V22 Enhanced Full Model]   AUC = {roc_auc_score(y_ib, p_full):.6f} (Lift: {10000*(roc_auc_score(y_ib, p_full) - roc_auc_score(y_ib, p_base)):+.2f} bps)")

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
blend_full = rk(0.50 * rk(p_base) + 0.50 * rk(p_full))
print(f"[Blend Base + V22-Enhanced] AUC = {roc_auc_score(y_ib, blend_full):.6f} (Lift: {10000*(roc_auc_score(y_ib, blend_full) - roc_auc_score(y_ib, p_base)):+.2f} bps)")

