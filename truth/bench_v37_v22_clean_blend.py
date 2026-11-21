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
print("[*] Diagnostic Part 7: Reconstructing V22's Exact Non-Leaking Components")

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

# Baseline single model
m_base = lgb.LGBMClassifier(
    n_estimators=1250, learning_rate=0.022, num_leaves=23, min_child_samples=160,
    subsample_for_bin=100000, colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1
).fit(Xia_f, y_ia)
p_base = m_base.predict_proba(Xib_f)[:, 1]
print(f"[V28 Base Single Model] AUC = {roc_auc_score(y_ib, p_base):.6f}")

# Train V22 5-Spec Zoo strictly on Xia_f
# Spec 1: qf_shallow (leaves=7, depth=3)
p_shallow = lgb.LGBMClassifier(n_estimators=1300, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=150, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20.0, n_jobs=8, random_state=42, verbose=-1).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
# Spec 2: qf_reg (leaves=11, min_child=300)
p_reg = lgb.LGBMClassifier(n_estimators=1800, learning_rate=0.025, num_leaves=11, min_child_samples=300, colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=40.0, n_jobs=8, random_state=42, verbose=-1).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
# Spec 3: v2_et (leaves=23, extra_trees=True)
p_et = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.022, num_leaves=23, min_child_samples=160, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_alpha=3.0, reg_lambda=10.0, extra_trees=True, n_jobs=8, random_state=42, verbose=-1).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
# Spec 4: v2_mono (leaves=15, monotone_constraints)
mono_k = [-1 if "EXT_SOURCE" in c else (1 if "TE_" in c else 0) for c in Xia_f.columns]
p_mono = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.020, num_leaves=15, min_child_samples=150, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20.0, monotone_constraints=mono_k, n_jobs=8, random_state=42, verbose=-1).fit(Xia_f, y_ia).predict_proba(Xib_f)[:, 1]
# Spec 5: CatBoost Product Specialist
cb_base_cats = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_CONTRACT_TYPE"]
cb_keys = ["Q3", "Q4", "P4_1285", "CT1", "CT1_GP", "K066_P4", "GENDER_EDU", "REG_HOUS", "INC_EDU"]
cb_cats = cb_base_cats + [f"K_{k}" for k in cb_keys]
def make_cb_df(X_num_base, df_src, k_dict, v_df):
    c_str = pd.DataFrame({
        **{c: df_src[c].fillna("NA").astype(str).reset_index(drop=True) for c in cb_base_cats},
        **{f"K_{k}": k_dict[k][0].astype(str).reset_index(drop=True) for k in cb_keys}
    })
    return pd.concat([X_num_base.drop(columns=v28.CATS).reset_index(drop=True), v_df.reset_index(drop=True), c_str], axis=1)

ca_cb = make_cb_df(Xia_res, df_ia, k_ia, v_ia)
cb_cb = make_cb_df(Xib_res, df_ib, k_ib, v_ib)
mc = CatBoostClassifier(iterations=850, depth=7, learning_rate=0.04, l2_leaf_reg=10.0, border_count=128, max_ctr_complexity=2, loss_function="Logloss", random_seed=42, task_type="GPU", verbose=False)
mc.fit(ca_cb, y_ia, cat_features=cb_cats, verbose=False)
p_cb = mc.predict_proba(cb_cb)[:, 1]

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)
v22_clean_blend = rk(0.24 * rk(p_shallow) + 0.12 * rk(p_reg) + 0.28 * rk(p_et) + 0.20 * rk(p_mono) + 0.16 * rk(p_cb))
print(f"[V22 Authentic Clean-Room Blend] AUC = {roc_auc_score(y_ib, v22_clean_blend):.6f} (Lift vs Single V28: {10000*(roc_auc_score(y_ib, v22_clean_blend) - roc_auc_score(y_ib, p_base)):+.2f} bps)")

