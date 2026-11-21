"""
================================================================================
GCI WORLD 2026: DEEP MATHEMATICAL F1, CONFUSION MATRIX & SUBGROUP ERROR AUDIT
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from scipy.stats import spearmanr

# 1. Load Data
df_train = pd.read_csv("d:/gci/inpuy/train.csv")
y_true = df_train["TARGET"].to_numpy("int32")

print("=" * 80)
print("DEEP MATHEMATICAL AUDIT: DEFAULT RISK STRUCTURE (N = 171,202)")
print("=" * 80)

n_pos = np.sum(y_true == 1)
n_neg = np.sum(y_true == 0)
base_rate = np.mean(y_true)
print(f"Total Applicants: {len(y_true):,}")
print(f"Defaults (Y=1):   {n_pos:,} ({base_rate*100:.2f}%)")
print(f"Non-Defaults (0): {n_neg:,} ({(1-base_rate)*100:.2f}%)")
print(f"Class Imbalance Ratio: {n_neg / n_pos:.2f} : 1")

# Analyze Key Covariate Signal Strengths (Univariate AUCS)
print("\n" + "=" * 80)
print("UNIVARIATE PREDICTIVE POWER (INDIVIDUAL COVARIATE AUCS)")
print("=" * 80)

num_cols = [c for c in df_train.columns if c not in ["SK_ID_CURR", "TARGET"] and df_train[c].dtype != "object"]
feature_aucs = []

for col in num_cols:
    vals = pd.to_numeric(df_train[col], errors="coerce").to_numpy("float64")
    valid_mask = ~np.isnan(vals)
    if valid_mask.sum() > 1000:
        auc = roc_auc_score(y_true[valid_mask], vals[valid_mask])
        if auc < 0.5:
            auc = 1.0 - auc
        feature_aucs.append((col, auc, valid_mask.mean()))

feature_aucs.sort(key=lambda x: x[1], reverse=True)
print(f"{'Feature':35s} | {'Univariate AUC':14s} | {'Coverage':10s}")
print("-" * 65)
for col, auc, cov in feature_aucs[:15]:
    print(f"{col:35s} | {auc:14.5f} | {cov*100:8.1f}%")

# Key Interaction Discovery: Multiplicative Debt-Burden & Solvency Ratios
print("\n" + "=" * 80)
print("HIGH-ORDER MATHEMATICAL INTERACTION SCREENING")
print("=" * 80)

c = pd.to_numeric(df_train["AMT_CREDIT"], errors="coerce").to_numpy("float64")
a = pd.to_numeric(df_train["AMT_ANNUITY"], errors="coerce").to_numpy("float64")
inc = pd.to_numeric(df_train["AMT_INCOME_TOTAL"], errors="coerce").to_numpy("float64")
g = pd.to_numeric(df_train["AMT_GOODS_PRICE"], errors="coerce").to_numpy("float64")
e1 = pd.to_numeric(df_train["EXT_SOURCE_1"], errors="coerce").to_numpy("float64")
e2 = pd.to_numeric(df_train["EXT_SOURCE_2"], errors="coerce").to_numpy("float64")
e3 = pd.to_numeric(df_train["EXT_SOURCE_3"], errors="coerce").to_numpy("float64")

interactions = {}
with np.errstate(all="ignore"):
    interactions["ANNUITY_TO_INCOME"] = a / np.maximum(inc, 1.0)
    interactions["CREDIT_TO_INCOME"] = c / np.maximum(inc, 1.0)
    interactions["CREDIT_TO_ANNUITY"] = c / np.maximum(a, 1.0)
    interactions["GOODS_TO_CREDIT"] = g / np.maximum(c, 1.0)
    interactions["BURDEN_PER_DEPENDENT"] = (a / np.maximum(inc, 1.0)) * (1.0 + pd.to_numeric(df_train["CNT_FAM_MEMBERS"], errors="coerce").fillna(1.0).to_numpy("float64"))

    # Non-linear Geometric Mean of Available External Sources
    ext_matrix = np.column_stack([e1, e2, e3])
    ext_nan_count = np.isnan(ext_matrix).sum(axis=1)
    ext_prod = np.nanprod(ext_matrix, axis=1)
    ext_count = np.maximum(3 - ext_nan_count, 1.0)
    ext_geo_mean = np.power(ext_prod, 1.0 / ext_count)
    interactions["EXT_GEO_MEAN"] = np.where(ext_nan_count < 3, ext_geo_mean, np.nan)

    # Debt-Service Stress under Low Bureau Signal
    ext_min = np.nanmin(ext_matrix, axis=1)
    interactions["SOLVENCY_STRESS_INDEX"] = (a / np.maximum(inc, 1.0)) / np.maximum(ext_min, 0.01)

    # Annuity to Goods Price (True Loan Cost Multiple)
    interactions["ANNUITY_TO_GOODS"] = a / np.maximum(g, 1.0)

    # Income Residual after Annual Annuity
    interactions["DISPOSABLE_INCOME_BUFFER"] = (inc - a) / np.maximum(inc, 1.0)

for name, feat in interactions.items():
    valid = ~np.isnan(feat) & np.isfinite(feat)
    if valid.sum() > 1000:
        auc = roc_auc_score(y_true[valid], feat[valid])
        if auc < 0.5:
            auc = 1.0 - auc
        print(f"  {name:30s} | AUC: {auc:.5f} (Coverage: {valid.mean()*100:.1f}%)")
