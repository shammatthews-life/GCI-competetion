import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv('d:/gci/inpuy/train.csv')
y = train['TARGET'].values
ntr = len(train)

# Base features
X = pd.DataFrame(index=train.index)
for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE', 'CNT_FAM_MEMBERS']:
    X[col] = train[col]

CATS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
for c in CATS:
    X[c] = train[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)

# Compute the Top 7 Pure-Covariate Formulas
E1, E2, E3 = train['EXT_SOURCE_1'], train['EXT_SOURCE_2'], train['EXT_SOURCE_3']
INC, CR, AN, GP = train['AMT_INCOME_TOTAL'], train['AMT_CREDIT'], train['AMT_ANNUITY'], train['AMT_GOODS_PRICE']
AGE = -train['DAYS_BIRTH'] / 365.25
EMP = np.where(train['DAYS_EMPLOYED'] == 365243, 0.0, -train['DAYS_EMPLOYED'] / 365.25)
PHONE_YRS = -train['DAYS_LAST_PHONE_CHANGE'].fillna(0) / 365.25
FAM = np.maximum(train['CNT_FAM_MEMBERS'].fillna(1), 1.0)
REG_POP = train['REGION_POPULATION_RELATIVE']

# 1. F_69: Subprime Collapse Leverage Spike
X['F69_SUBPRIME_LEVERAGE'] = (1.0 / (E2.fillna(0.3) * E3.fillna(0.3) + 0.01)) * (CR / (INC + 1.0))

# 2. F_67: Contested Bureau Safety Floor
min_e23 = np.minimum(E2.fillna(0.5), E3.fillna(0.5))
X['F67_CONTESTED_FLOOR'] = (min_e23 ** 2) * (INC / (AN + 1.0))

# 3. F_71: Thin File Internal Capacity
X['F71_THIN_FILE_CAPACITY'] = E2.fillna(0.5) * (INC / (AN + 1.0)) * (AGE / 40.0)

# 4. F_74: Lagging Region Solvency Index
X['F74_LAGGING_REGION_SOLV'] = (INC * REG_POP / (AN + 1.0)) * E2.fillna(0.5)

# 5. F_89: Five Year Phone Veteran Anchor
X['F89_PHONE_VETERAN_ANCHOR'] = np.log1p(PHONE_YRS) * (E2.fillna(0.5) + E3.fillna(0.5))

# 6. F_77: Married Female Household Solvency
is_fem_marr = ((train['CODE_GENDER'] == 'F') & (train['NAME_FAMILY_STATUS'] == 'Married')).astype(float)
X['F77_MARRIED_FEMALE_SOLV'] = is_fem_marr * (INC / FAM / (AN + 1.0)) * np.sqrt(np.maximum(E2.fillna(0.5) * E3.fillna(0.5), 0.0))

# 7. F_81: Laborer Income Volatility Strain
is_laborer = train['OCCUPATION_TYPE'].isin(['Laborers', 'Drivers', 'Low-skill Laborers']).astype(float)
X['F81_LABORER_STRAIN'] = is_laborer * (AN / (INC / 12.0 + 1.0)) / (EMP + 1.0)

# 8. Merton Distance to Default and Clayton Copula
implied_term = np.clip((CR / (AN + 1.0)) / 12.0, 0.1, 10.0)
dd = (np.log(np.maximum(INC / 12.0, 1.0) / (AN + 1.0)) + (0.02 - 0.5 * 0.04) * implied_term) / (0.20 * np.sqrt(implied_term))
X['MERTON_DD'] = dd

# Fast 5-fold CV evaluation
oof = np.zeros(ntr)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
    dtrain = lgb.Dataset(X.iloc[trn_idx], y[trn_idx])
    m = lgb.train(
        {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1, 'seed': 42},
        dtrain, num_boost_round=400
    )
    preds = m.predict(X.iloc[val_idx])
    oof[val_idx] = preds
    f_auc = roc_auc_score(y[val_idx], preds)
    aucs.append(f_auc)
    print(f"  Fold {fold+1} AUC: {f_auc:.5f}")

pooled_auc = roc_auc_score(y, oof)
print(f"\nPooled OOF AUC with Top 7 Interaction Formulas: {pooled_auc:.5f}")
print(f"Mean Fold AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")
