import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('d:/gci/inpuy/train.csv')
y = df_train['TARGET'].values

# Base Features
ext = df_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
em = ext.mean(axis=1)
cr = df_train['AMT_CREDIT']
an = df_train['AMT_ANNUITY']
inc = df_train['AMT_INCOME_TOTAL']
pay = an / (cr + 1e-5)
demp = df_train['DAYS_EMPLOYED'].where(df_train['DAYS_EMPLOYED'] != 365243, np.nan)
age = -df_train['DAYS_BIRTH'] / 365.25
emp_yr = -demp / 365.25

# Novel Manifold A: Inquiry-Stress & Bureau Dispersion
inq_yr = df_train['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)
id_yr = -df_train['DAYS_ID_PUBLISH'] / 365.25
inq_velocity = (inq_yr + 1.0) / np.maximum(1.0, id_yr)
ext_asym = ext.max(axis=1) - ext.min(axis=1)
vulnerability = inq_velocity * (1.0 + ext_asym.fillna(0)) * (an / (inc / 12.0 + 1.0))

# Novel Manifold B: Stability & Solvency Buffer
work_ph = df_train['FLAG_WORK_PHONE'].fillna(0)
stability_buf = np.nan_to_num(emp_yr, nan=0) * (1.0 + work_ph) * (inc / (an + 1.0))
ext2_resilience = df_train['EXT_SOURCE_2'].fillna(0) * stability_buf

X_base = pd.DataFrame({
    'EXT_MEAN': em, 'EXT_1': ext['EXT_SOURCE_1'], 'EXT_2': ext['EXT_SOURCE_2'], 'EXT_3': ext['EXT_SOURCE_3'],
    'PAY': pay, 'CREDIT_INC': cr / inc, 'ANN_INC': an / inc, 'AGE': age, 'EMP_YR': emp_yr
})

X_enhanced = X_base.copy()
X_enhanced['INQ_VELOCITY'] = inq_velocity
X_enhanced['EXT_ASYM'] = ext_asym
X_enhanced['VULNERABILITY'] = vulnerability
X_enhanced['STABILITY_BUF'] = stability_buf
X_enhanced['EXT2_RESILIENCE'] = ext2_resilience

print("Running 5-Fold Stratified CV to evaluate AUC lift...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, X in [("Base Features", X_base), ("Enhanced with Manifolds A & B", X_enhanced)]:
    oof = np.zeros(len(df_train))
    for tr_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name:<35} | OOF AUC: {auc:.5f}")

