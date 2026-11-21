import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

# Check XGBoost GPU availability
print("Testing XGBoost GPU...")
try:
    clf = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=50, max_depth=6)
    X_dummy = np.random.randn(100, 10).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 100)
    clf.fit(X_dummy, y_dummy)
    print("[+] XGBoost CUDA device is active and functional!")
except Exception as e:
    print(f"[-] XGBoost CUDA error: {e}")

