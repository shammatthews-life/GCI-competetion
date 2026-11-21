import time
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import solo_champion_v28_certified as v28

v28.set_all_seeds(42)
print("[*] Diagnostic: Rebuilding V22 Pure Signal under 100% Fold-Local Rules")

df_train = pd.read_csv(r"D:\gci\inpuy\train.csv")
y_train = df_train["TARGET"].values.astype(np.int32)
dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(df_train)), y_train))[0][0]
ia, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(np.zeros(len(dev_outer)), y_train[dev_outer]))[0]
df_ia, df_ib = df_train.iloc[dev_outer[ia]], df_train.iloc[dev_outer[ib]]
y_ia, y_ib = y_train[dev_outer[ia]], y_train[dev_outer[ib]]

# --- 1. Current V28 Pure Inductive Baseline ---
cmaps = v28.fit_category_maps(df_ia)
res_ia, res_ib = v28.compute_fold_residuals(df_ia, df_ib, None, cmaps, seed=42)
Xia_res = pd.concat([v28.build_row_features(df_ia, cmaps), res_ia], axis=1)
Xib_res = pd.concat([v28.build_row_features(df_ib, cmaps), res_ib], axis=1)
te_ia, te_ib, k_ia, k_ib = v28.compute_local_te(df_ia, df_ib, y_ia, n_inner=4, seed=42)
v_ia, v_ib = v28.compute_local_views(Xia_res, Xib_res, df_ia, df_ib, y_ia, n_inner=2, seed=42)
Xia_f = pd.concat([Xia_res, te_ia, v_ia], axis=1)
Xib_f = pd.concat([Xib_res, te_ib, v_ib], axis=1)

m_v28 = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_f, y_ia)
p_v28 = m_v28.predict_proba(Xib_f)[:, 1]
print(f"[Baseline V28 Single Model] AUC = {roc_auc_score(y_ib, p_v28):.6f}")

# --- 2. V22 Multi-Spec 5-Model Diversity (qf_shallow, qf_reg, v1_allfam_oh, v2_allfam_oh, v2_mono) ---
# One-hot encode strictly on df_ia categories
cats = v28.CATS
u_cats = {c: df_ia[c].dropna().unique() for c in cats}
def make_oh(df_src):
    d_oh = {}
    for c in cats:
        u = u_cats[c]
        cat_s = pd.Categorical(df_src[c], categories=u)
        oh = pd.get_dummies(cat_s, dummy_na=True, dtype=np.float32)
        for i_col, col in enumerate(oh.columns):
            safe_c = "".join(ch if ch.isalnum() else "_" for ch in str(col))
            d_oh[f"OH_{c}_{i_col}_{safe_c}"] = oh[col].values
    return pd.DataFrame(d_oh, index=df_src.index)

oh_ia = make_oh(df_ia).reset_index(drop=True)
oh_ib = make_oh(df_ib).reset_index(drop=True)

Xia_oh = pd.concat([Xia_f.reset_index(drop=True), oh_ia], axis=1)
Xib_oh = pd.concat([Xib_f.reset_index(drop=True), oh_ib], axis=1)

specs = {
    "qf_shallow": (Xia_f, Xib_f, dict(n_estimators=1300, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=150, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20.0)),
    "qf_reg":     (Xia_f, Xib_f, dict(n_estimators=1800, learning_rate=0.025, num_leaves=11, min_child_samples=300, colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=40.0)),
    "v1_oh":      (Xia_oh, Xib_oh, dict(n_estimators=900, learning_rate=0.035, num_leaves=15, min_child_samples=150, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=20.0)),
    "v2_et":      (Xia_f, Xib_f, dict(n_estimators=1200, learning_rate=0.022, num_leaves=23, min_child_samples=160, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_alpha=3.0, reg_lambda=10.0, extra_trees=True)),
}

preds = {}
for sn, (xa, xb, prms) in specs.items():
    m = lgb.LGBMClassifier(**prms, n_jobs=8, random_state=42, verbose=-1).fit(xa, y_ia)
    preds[sn] = m.predict_proba(xb)[:, 1]
    print(f"  [Spec {sn:12s}] AUC = {roc_auc_score(y_ib, preds[sn]):.6f}")

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
blend_lgb = rk(0.25 * rk(preds["qf_shallow"]) + 0.15 * rk(preds["qf_reg"]) + 0.30 * rk(preds["v1_oh"]) + 0.30 * rk(preds["v2_et"]))
print(f"[V22 4-Spec Pure LGB Blend]   AUC = {roc_auc_score(y_ib, blend_lgb):.6f} (Lift vs Single V28: {10000*(roc_auc_score(y_ib, blend_lgb) - roc_auc_score(y_ib, p_v28)):+.2f} bps)")

