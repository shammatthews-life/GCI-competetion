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
print("[*] Diagnostic Part 4: Individual Feature Isolation (Finding the Pure Gold)")

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
auc_base = roc_auc_score(y_ib, p_base)
print(f"[V28 Base Single Model] AUC = {auc_base:.6f}")

# Candidate 1: Demographic Applicant Partner Link (from bench_v32)
def f_applicant_links(df_src, df_ref):
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
    cnt = src_keys.map(ref_keys.value_counts()).fillna(0).values.astype(np.float32)
    ext2_ref = df_ref.groupby(ref_keys)["EXT_SOURCE_2"].mean()
    diff_ext2 = (df_src["EXT_SOURCE_2"].values - src_keys.map(ext2_ref)).astype(np.float32)
    return pd.DataFrame({"APP_COUNT": cnt, "APP_DIFF_EXT2": diff_ext2}, index=df_src.index)

# Candidate 2: Bureau Mon/Year Ratio
def f_bureau_ratio(df_src):
    bm = df_src["AMT_REQ_CREDIT_BUREAU_MON"]
    by = df_src["AMT_REQ_CREDIT_BUREAU_YEAR"]
    return pd.DataFrame({"BUR_MON_TO_YEAR": (bm / (by + 1.0)).astype(np.float32)}, index=df_src.index)

# Candidate 3: Credit to Annuity Term Approximation
def f_credit_term(df_src):
    c, a = df_src["AMT_CREDIT"], df_src["AMT_ANNUITY"]
    return pd.DataFrame({"CR_TO_ANN_TERM": (c / a).astype(np.float32)}, index=df_src.index)

candidates = {
    "APP_LINKS": (f_applicant_links(df_ia, df_ia), f_applicant_links(df_ib, df_ia)),
    "BUR_RATIO": (f_bureau_ratio(df_ia), f_bureau_ratio(df_ib)),
    "CR_TERM":   (f_credit_term(df_ia), f_credit_term(df_ib)),
}

for name, (fa, fb) in candidates.items():
    xa = pd.concat([Xia_f.reset_index(drop=True), fa.reset_index(drop=True)], axis=1)
    xb = pd.concat([Xib_f.reset_index(drop=True), fb.reset_index(drop=True)], axis=1)
    m = lgb.LGBMClassifier(
        n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
        subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
        reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
    ).fit(xa, y_ia)
    p = m.predict_proba(xb)[:, 1]
    auc = roc_auc_score(y_ib, p)
    print(f"  [+ {name:10s}] AUC = {auc:.6f} (Lift: {10000*(auc - auc_base):+.2f} bps)")

