import os
import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.stats import rankdata, norm

print("[*] Testing V51 Mathematical Feature Extraction & Multi-Engine Suite...")

# Load small sample of train
tr_raw = pd.read_csv("d:/gci/inpuy/train.csv", nrows=5000)
te_raw = pd.read_csv("d:/gci/inpuy/test.csv", nrows=1000)
y = tr_raw["TARGET"].values

print(f"Loaded train sample: {tr_raw.shape}, test sample: {te_raw.shape}")
print(f"Target mean: {y.mean():.4f}")

# Test CatBoost with native cat_features
CATS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

cb_tr = tr_raw[CATS].fillna("MISS").astype(str)
cb_te = te_raw[CATS].fillna("MISS").astype(str)

cb = CatBoostClassifier(iterations=30, depth=4, task_type='GPU', verbose=0)
cb.fit(cb_tr, y, cat_features=CATS)
p_cb = cb.predict_proba(cb_te)[:, 1]
print(f"CatBoost native GPU test OK, preds shape: {p_cb.shape}")

# Test XGBoost with GPU Hist and NaN handling
X_num = tr_raw[['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].values.astype(np.float32)
dtr = xgb.DMatrix(X_num, label=y)
p_xgb = dict(tree_method="hist", device="cuda", max_depth=4, objective="binary:logistic")
m_xgb = xgb.train(p_xgb, dtr, num_boost_round=30)
p_xgb_out = m_xgb.predict(dtr)
print(f"XGBoost GPU Hist OK, AUC: {roc_auc_score(y, p_xgb_out):.4f}")

# Test Nelder-Mead Simplex optimization on AUC
rk_norm = lambda s: (rankdata(s) - 1.0) / np.maximum(float(len(s) - 1.0), 1.0)
R1 = rk_norm(p_xgb_out)
R2 = rk_norm(tr_raw['EXT_SOURCE_2'].fillna(0.5).values)
R_mat = np.column_stack([R1, R2])

def obj(w):
    w = np.maximum(w, 0.0)
    if w.sum() == 0: return 0.0
    w = w / w.sum()
    return -roc_auc_score(y, R_mat @ w)

res = minimize(obj, [0.5, 0.5], method='Nelder-Mead', options={'maxiter': 50})
w_opt = np.maximum(res.x, 0.0)
w_opt /= w_opt.sum()
print(f"Nelder-Mead Simplex OK, optimal weights: {w_opt}, Blended AUC: {-res.fun:.4f}")

print("[PASS] All component prototypes verified successfully!")
