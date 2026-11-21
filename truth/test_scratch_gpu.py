"""
Fast Prototype & Verification for V31 Pure Scratch Architecture
Tests learned expectation residuals, multi-view stacking, and GPU models.
"""

import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print("[*] Testing Pure Scratch Feature Pipeline & GPU Models...")
t0 = time.time()

tr = pd.read_csv("d:/gci/inpuy/train.csv")
te = pd.read_csv("d:/gci/inpuy/test.csv")
y = tr["TARGET"].values
ntr = len(tr)
nte = len(te)

print(f"Data Loaded: Train {tr.shape}, Test {te.shape} in {time.time()-t0:.2f}s")

# Test GPU XGBoost Hist
X_dummy = tr[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]].fillna(0).values.astype(np.float32)
dtrain = xgb.DMatrix(X_dummy, label=y)
param = {"tree_method": "hist", "device": "cuda", "max_depth": 5, "objective": "binary:logistic"}
bst = xgb.train(param, dtrain, num_boost_round=50)
print(f"XGBoost GPU Hist works! Output shape: {bst.predict(dtrain).shape}")

# Test CatBoost GPU
cb = CatBoostClassifier(iterations=50, depth=5, task_type="GPU", verbose=0)
cb.fit(X_dummy, y)
print(f"CatBoost GPU works! Output shape: {cb.predict_proba(X_dummy).shape}")

print("[+] All GPU engines operational!")
