import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import os

train = pd.read_csv('d:/gci/inpuy/train.csv')
y = train['TARGET'].values

CATS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

X = train.drop(columns=['SK_ID_CURR', 'TARGET']).copy()
for c in CATS:
    X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)

# Base financial features
X['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
X['CTI'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
X['ATI'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
X['EXT_MEAN'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
ext_nobs = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1)
X['EXT_NOBS'] = ext_nobs

# BASELINE EVALUATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_base = np.zeros(len(train))

for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
    dtrain = lgb.Dataset(X.iloc[trn_idx], y[trn_idx])
    m = lgb.train(
        {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'seed': 42},
        dtrain, num_boost_round=300
    )
    oof_base[val_idx] = m.predict(X.iloc[val_idx])

auc_base = roc_auc_score(y, oof_base)
print(f"Base AUC: {auc_base:.5f}")

# NOW ADD REMEDIES FOR PENSIONER & THIN-FILE ERROR REGIONS
is_pen = (train['DAYS_EMPLOYED'] == 365243).astype(float)
inc_monthly = train['AMT_INCOME_TOTAL'] / 12.0
ann = train['AMT_ANNUITY']

X['IS_PENSIONER'] = is_pen
X['PENSION_DILUTION'] = train['AMT_INCOME_TOTAL'] / np.maximum(train['CNT_FAM_MEMBERS'].fillna(1), 1.0)
X['PENSION_BURDEN'] = is_pen * (ann / np.maximum(inc_monthly, 100.0))
X['PENSION_BUFFER'] = is_pen * (inc_monthly - ann)
X['PENSIONER_RISK_INTERACTION'] = is_pen * (train['AMT_CREDIT'] / (train['AMT_INCOME_TOTAL'] + 1.0))

# THIN-FILE ENHANCEMENTS
doc_cols = [c for c in train.columns if c.startswith('FLAG_DOCUMENT_')]
X['DOC_COUNT'] = train[doc_cols].sum(axis=1)
X['EXT_MISSING_COUNT'] = 3 - ext_nobs
X['THIN_FILE_STRESS'] = (3 - ext_nobs) * (train['AMT_CREDIT'] / (train['AMT_INCOME_TOTAL'] + 1.0))
X['REG_TO_AGE'] = train['DAYS_REGISTRATION'] / (train['DAYS_BIRTH'] - 1.0)
X['PHONE_TO_AGE'] = train['DAYS_LAST_PHONE_CHANGE'].fillna(0) / (train['DAYS_BIRTH'] - 1.0)
X['GOODS_TO_CREDIT'] = train['AMT_GOODS_PRICE'] / (train['AMT_CREDIT'] + 1.0)

# EVALUATE WITH REMEDIES
oof_new = np.zeros(len(train))
fold_aucs = []

for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
    dtrain = lgb.Dataset(X.iloc[trn_idx], y[trn_idx])
    m = lgb.train(
        {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'seed': 42},
        dtrain, num_boost_round=300
    )
    p_val = m.predict(X.iloc[val_idx])
    oof_new[val_idx] = p_val
    f_auc = roc_auc_score(y[val_idx], p_val)
    fold_aucs.append(f_auc)
    print(f"  Fold {fold+1}: AUC = {f_auc:.5f}")

auc_new = roc_auc_score(y, oof_new)
print(f"\nRemediated OOF AUC: {auc_new:.5f} (Delta: +{(auc_new - auc_base)*10000:.1f} bps)")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f}")

# Check error breakdown on pensioners
train['oof_new'] = oof_new
train['is_pensioner'] = is_pen
for pen, grp in train.groupby('is_pensioner'):
    label = 'Pensioner' if pen == 1 else 'Employed'
    old_auc = roc_auc_score(grp['TARGET'], train.loc[grp.index, 'oof']) if 'oof' in train.columns else np.nan
    new_auc = roc_auc_score(grp['TARGET'], grp['oof_new'])
    print(f"  {label}: New AUC = {new_auc:.5f}")
