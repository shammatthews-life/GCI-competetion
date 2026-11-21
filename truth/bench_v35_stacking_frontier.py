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
print("[*] Diagnostic Part 5: The Stacking & Blending Frontier")

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

preds_b = v28.fit_predict_all_10_models(
    Xia_res, Xib_res, df_ia, df_ib, y_ia, te_ia, te_ib, v_ia, v_ib, k_ia, k_ib,
    cb_task="GPU", xgb_device="cuda", seed=42
)

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
def to_z(p):
    r = pd.Series(p).rank(method="average").values / (len(p) + 1.0)
    return norm.ppf(np.clip(r, 1e-6, 1.0 - 1e-6))

print("\n--- 10 Individual Model Scores on Development Test Set ---")
aucs = {}
for m, p in preds_b.items():
    aucs[m] = roc_auc_score(y_ib, p)
    print(f"  {m:24s}: AUC = {aucs[m]:.6f}")

z_mat = np.column_stack([to_z(preds_b[m]) for m in preds_b])
r_mat = np.column_stack([rk(preds_b[m]) for m in preds_b])

# 1. Simple Equal-Weight Probit
auc_eq_probit = roc_auc_score(y_ib, z_mat.mean(axis=1))
# 2. Simple Equal-Weight Rank Average
auc_eq_rank = roc_auc_score(y_ib, r_mat.mean(axis=1))
# 3. Top-Tier Weighted Probit (giving higher weight to high-performing models: ET, Mono, Neural)
weights_custom = np.array([0.10, 0.08, 0.16, 0.06, 0.16, 0.05, 0.05, 0.06, 0.14, 0.14])
auc_custom = roc_auc_score(y_ib, z_mat @ weights_custom)
# 4. Pure Tree Zoo (Top 5 GBDTs)
z_trees = z_mat[:, :8]
w_trees = np.ones(8) / 8.0
auc_trees = roc_auc_score(y_ib, z_trees @ w_trees)
# 5. Pure Neural Zoo (Top 2 Neural)
z_nn = z_mat[:, 8:]
w_nn = np.ones(2) / 2.0
auc_nn = roc_auc_score(y_ib, z_nn @ w_nn)
# 6. Convex Blend of Pure Tree Zoo (75%) + Pure Neural Zoo (25%)
auc_tree_nn_blend = roc_auc_score(y_ib, 0.75 * (z_trees @ w_trees) + 0.25 * (z_nn @ w_nn))

print("\n--- Blending & Stacking Ensembling Comparison ---")
print(f"  Equal-Weight Probit:             AUC = {auc_eq_probit:.6f}")
print(f"  Equal-Weight Rank Average:       AUC = {auc_eq_rank:.6f}")
print(f"  Custom-Weighted Probit:          AUC = {auc_custom:.6f}")
print(f"  GBDT Sub-Ensemble (8 models):    AUC = {auc_trees:.6f}")
print(f"  Neural Sub-Ensemble (2 models):  AUC = {auc_nn:.6f}")
print(f"  Convex 75% GBDT + 25% Neural:    AUC = {auc_tree_nn_blend:.6f} (Lift over Equal Probit: {10000*(auc_tree_nn_blend - auc_eq_probit):+.2f} bps)")

