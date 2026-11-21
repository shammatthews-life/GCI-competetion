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
print("[*] Diagnostic Part 6: Multi-Seed GBDT Ensembling (Variance Reduction)")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

# Fit standard V28 feature pipeline
cmaps = v28.fit_category_maps(df_ia)
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), res_ia], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)

# Base single seed model (LGB_Mono_ET_6View)
p_s1 = lgb.LGBMClassifier(
    n_estimators=1200, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1,
    monotone_constraints=[-1 if "EXT_SOURCE" in c else (1 if "TE_" in c else 0) for c in Xia_f.columns]
).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
print(f"[Seed 1 LGB_Mono_ET_6View] AUC = {roc_auc_score(y_ib, p_s1):.6f}")

# Seed 2
p_s2 = lgb.LGBMClassifier(
    n_estimators=1200, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=2024, verbose=-1,
    monotone_constraints=[-1 if "EXT_SOURCE" in c else (1 if "TE_" in c else 0) for c in Xia_f.columns]
).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
print(f"[Seed 2 LGB_Mono_ET_6View] AUC = {roc_auc_score(y_ib, p_s2):.6f}")

# Seed 3
p_s3 = lgb.LGBMClassifier(
    n_estimators=1200, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=99, verbose=-1,
    monotone_constraints=[-1 if "EXT_SOURCE" in c else (1 if "TE_" in c else 0) for c in Xia_f.columns]
).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
print(f"[Seed 3 LGB_Mono_ET_6View] AUC = {roc_auc_score(y_ib, p_s3):.6f}")

p_seed_avg = (p_s1 + p_s2 + p_s3) / 3.0
print(f"[3-Seed Bagged LGB_Mono_ET] AUC = {roc_auc_score(y_ib, p_seed_avg):.6f} (Lift vs Seed 1: {10000*(roc_auc_score(y_ib, p_seed_avg) - roc_auc_score(y_ib, p_s1)):+.2f} bps)")

