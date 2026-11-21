import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

# Load train data
df_train = pd.read_csv('d:/gci/inpuy/train.csv')
y = df_train['TARGET'].values

# Quick fold 1 test
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf.split(df_train, y)):
    if fold > 0: break

# Extract key numeric features
cols_num = [c for c in df_train.columns if df_train[c].dtype in ['float64', 'int64'] and c not in ['SK_ID_CURR', 'TARGET']]
X = df_train[cols_num].copy()
X['EXT_MEAN'] = df_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
X['PAY'] = df_train['AMT_ANNUITY'] / (df_train['AMT_CREDIT'] + 1e-5)
X['CREDIT_INC'] = df_train['AMT_CREDIT'] / (df_train['AMT_INCOME_TOTAL'] + 1e-5)

X_tr = X.iloc[tr_idx].values.astype(np.float32)
y_tr = y[tr_idx]
X_va = X.iloc[val_idx].values.astype(np.float32)
y_va = y[val_idx]

print(f"Testing Fold 1 on {X_tr.shape[1]} features...")

# 1. LightGBM
t0 = time.time()
m_lgb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
m_lgb.fit(X_tr, y_tr)
p_lgb = m_lgb.predict_proba(X_va)[:, 1]
print(f"LightGBM AUC: {roc_auc_score(y_va, p_lgb):.5f} ({time.time()-t0:.1f}s)")

# 2. CatBoost GPU
t0 = time.time()
m_cb = CatBoostClassifier(iterations=600, learning_rate=0.04, depth=6, task_type="GPU", random_seed=42, verbose=False)
m_cb.fit(X_tr, y_tr)
p_cb = m_cb.predict_proba(X_va)[:, 1]
print(f"CatBoost GPU AUC: {roc_auc_score(y_va, p_cb):.5f} ({time.time()-t0:.1f}s)")

# 3. XGBoost GPU
t0 = time.time()
m_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, tree_method="hist", device="cuda", random_state=42)
m_xgb.fit(X_tr, y_tr)
p_xgb = m_xgb.predict_proba(X_va)[:, 1]
print(f"XGBoost GPU AUC: {roc_auc_score(y_va, p_xgb):.5f} ({time.time()-t0:.1f}s)")

# Tri-Engine Ensemble
for w_lgb, w_cb, w_xgb in [(0.5, 0.25, 0.25), (0.6, 0.2, 0.2), (0.4, 0.3, 0.3)]:
    p_ens = rk(w_lgb * rk(p_lgb) + w_cb * rk(p_cb) + w_xgb * rk(p_xgb))
    print(f"Tri-Engine ({w_lgb:.2f} LGB + {w_cb:.2f} CB + {w_xgb:.2f} XGB) -> AUC: {roc_auc_score(y_va, p_ens):.5f}")

