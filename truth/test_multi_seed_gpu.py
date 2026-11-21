"""
================================================================================
GCI WORLD 2026: MULTI-SEED GPU BOOSTING VARIANCE REDUCTION EXPERIMENT
================================================================================
Tests whether 3-seed GPU bagging on CatBoost and XGBoost accelerates convergence
and reduces residual variance on out-of-fold predictions.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier
import xgboost as xgb

# 1. Load data
df_train = pd.read_csv("d:/gci/inpuy/train.csv")
y_train = df_train["TARGET"].to_numpy("int32")

# Quick feature construction
c = pd.to_numeric(df_train["AMT_CREDIT"], errors="coerce").to_numpy("float64")
a = pd.to_numeric(df_train["AMT_ANNUITY"], errors="coerce").to_numpy("float64")
inc = pd.to_numeric(df_train["AMT_INCOME_TOTAL"], errors="coerce").to_numpy("float64")
g = pd.to_numeric(df_train["AMT_GOODS_PRICE"], errors="coerce").to_numpy("float64")
e1 = pd.to_numeric(df_train["EXT_SOURCE_1"], errors="coerce").to_numpy("float64")
e2 = pd.to_numeric(df_train["EXT_SOURCE_2"], errors="coerce").to_numpy("float64")
e3 = pd.to_numeric(df_train["EXT_SOURCE_3"], errors="coerce").to_numpy("float64")

ext_matrix = np.column_stack([e1, e2, e3])
ext_nan_count = np.isnan(ext_matrix).sum(axis=1)
ext_prod = np.nanprod(ext_matrix, axis=1)
ext_count = np.maximum(3 - ext_nan_count, 1.0)
ext_geo_mean = np.power(ext_prod, 1.0 / ext_count)
ext_min = np.nanmin(ext_matrix, axis=1)

features = {
    "EXT_SOURCE_1": e1,
    "EXT_SOURCE_2": e2,
    "EXT_SOURCE_3": e3,
    "EXT_GEO_MEAN": np.where(ext_nan_count < 3, ext_geo_mean, np.nan),
    "SOLVENCY_STRESS": (a / np.maximum(inc, 1.0)) / np.maximum(ext_min, 0.01),
    "CREDIT_TO_ANNUITY": c / np.maximum(a, 1.0),
    "GOODS_TO_CREDIT": g / np.maximum(c, 1.0),
    "DAYS_BIRTH": pd.to_numeric(df_train["DAYS_BIRTH"], errors="coerce").to_numpy("float64"),
    "DAYS_EMPLOYED": pd.to_numeric(df_train["DAYS_EMPLOYED"], errors="coerce").to_numpy("float64"),
    "DAYS_ID_PUBLISH": pd.to_numeric(df_train["DAYS_ID_PUBLISH"], errors="coerce").to_numpy("float64"),
    "OWN_CAR_AGE": pd.to_numeric(df_train["OWN_CAR_AGE"], errors="coerce").to_numpy("float64"),
    "REGION_RATING_CLIENT": pd.to_numeric(df_train["REGION_RATING_CLIENT"], errors="coerce").to_numpy("float64"),
}

X_df = pd.DataFrame(features)

# Stratified 3-fold split
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

seeds = [42, 2026, 777]
print("=" * 80)
print(f"MULTI-SEED GPU EXPERIMENT: Testing XGBoost GPU with seeds {seeds}")
print("=" * 80)

single_seed_oof = np.zeros(len(y_train))
multi_seed_oof = np.zeros(len(y_train))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_df, y_train)):
    X_tr, y_tr = X_df.iloc[trn_idx], y_train[trn_idx]
    X_val, y_val = X_df.iloc[val_idx], y_train[val_idx]

    fold_preds = []
    for s in seeds:
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=s
        )
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_val)[:, 1]
        fold_preds.append(p)
        if s == 42:
            single_seed_oof[val_idx] = p

    multi_seed_oof[val_idx] = np.mean(fold_preds, axis=0)
    print(f"  Fold {fold+1}: Seed 42 AUC: {roc_auc_score(y_val, single_seed_oof[val_idx]):.5f} | 3-Seed Bagged AUC: {roc_auc_score(y_val, multi_seed_oof[val_idx]):.5f}")

auc_single = roc_auc_score(y_train, single_seed_oof)
auc_multi = roc_auc_score(y_train, multi_seed_oof)

print("\n" + "=" * 80)
print(f"Single-Seed (42) XGBoost GPU OOF AUC:   {auc_single:.6f}")
print(f"3-Seed Bagged (42,2026,777) OOF AUC:    {auc_multi:.6f}")
print(f"Variance Reduction Delta:              +{auc_multi - auc_single:+.6f}")
print("=" * 80)
