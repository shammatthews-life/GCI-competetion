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
print("[*] Diagnostic Part 9: Feature Interaction Synthesis (Finding Pure Additive Lift)")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

# Current V28 standard features
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
print(f"[V28 Base Model] AUC = {auc_base:.6f}")

# Novel Pure-Covariate Interactions (100% legal, zero external data, zero target access)
def add_novel_interactions(df_src):
    d = {}
    d["PAYMENT_BURDEN_TO_GOODS_NOVEL"] = (df_src["AMT_ANNUITY"] / (df_src["AMT_GOODS_PRICE"] + 1.0)).astype(np.float32)
    e1 = np.maximum(df_src["EXT_SOURCE_1"].fillna(0.5).values, 1e-3)
    e2 = np.maximum(df_src["EXT_SOURCE_2"].fillna(0.5).values, 1e-3)
    e3 = np.maximum(df_src["EXT_SOURCE_3"].fillna(0.5).values, 1e-3)
    d["EXT_HARMONIC_MEAN"] = (3.0 / (1.0/e1 + 1.0/e2 + 1.0/e3)).astype(np.float32)
    d["AGE_TO_EMPLOYED_RATIO"] = (df_src["DAYS_BIRTH"] / np.minimum(df_src["DAYS_EMPLOYED"], -1.0)).astype(np.float32)
    return pd.DataFrame(d, index=df_src.index)

f_ia_nov = add_novel_interactions(df_ia).reset_index(drop=True)
f_ib_nov = add_novel_interactions(df_ib).reset_index(drop=True)

Xia_nov = pd.concat([Xia_f.reset_index(drop=True), f_ia_nov], axis=1)
Xib_nov = pd.concat([Xib_f.reset_index(drop=True), f_ib_nov], axis=1)

m_nov = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_nov, y_ia)
p_nov = m_nov.predict_proba(Xib_nov)[:, 1]
auc_nov = roc_auc_score(y_ib, p_nov)
print(f"[With 4 Novel Interactions] AUC = {auc_nov:.6f} (Lift: {10000*(auc_nov - auc_base):+.2f} bps)")

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
blend_nov = rk(0.50 * rk(p_base) + 0.50 * rk(p_nov))
print(f"[Blend Base + Novel Interactions] AUC = {roc_auc_score(y_ib, blend_nov):.6f} (Lift: {10000*(roc_auc_score(y_ib, blend_nov) - auc_base):+.2f} bps)")

