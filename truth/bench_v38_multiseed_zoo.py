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
print("[*] Diagnostic Part 8: Multi-Seed 3-Bagged Full 10-Model Zoo")

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

def to_z(p):
    r = pd.Series(p).rank(method="average").values / (len(p) + 1.0)
    return norm.ppf(np.clip(r, 1e-6, 1.0 - 1e-6))

# Single-seed (seed=42) baseline run
t0 = time.time()
preds_s1 = v28.fit_predict_all_10_models(
    Xia_res, Xib_res, df_ia, df_ib, y_ia, te_ia, te_ib, v_ia, v_ib, k_ia, k_ib,
    cb_task="GPU", xgb_device="cuda", seed=42
)
z_s1 = np.column_stack([to_z(preds_s1[m]) for m in preds_s1])
auc_s1 = roc_auc_score(y_ib, z_s1.mean(axis=1))
print(f"[Seed 42 Single Run 10-Model Zoo] AUC = {auc_s1:.6f} ({time.time()-t0:.1f}s)")

# Run Seed 2 (seed=1042)
t0 = time.time()
preds_s2 = v28.fit_predict_all_10_models(
    Xia_res, Xib_res, df_ia, df_ib, y_ia, te_ia, te_ib, v_ia, v_ib, k_ia, k_ib,
    cb_task="GPU", xgb_device="cuda", seed=1042
)
z_s2 = np.column_stack([to_z(preds_s2[m]) for m in preds_s2])
auc_s2 = roc_auc_score(y_ib, z_s2.mean(axis=1))
print(f"[Seed 1042 Single Run 10-Model Zoo] AUC = {auc_s2:.6f} ({time.time()-t0:.1f}s)")

# Bagged 2-Seed Blend
z_bag2 = 0.5 * z_s1 + 0.5 * z_s2
auc_bag2 = roc_auc_score(y_ib, z_bag2.mean(axis=1))
print(f"\n>>> [2-Seed Bagged 10-Model Zoo] AUC = {auc_bag2:.6f} (Lift vs Seed 42: {10000*(auc_bag2 - auc_s1):+.2f} bps) <<<")

