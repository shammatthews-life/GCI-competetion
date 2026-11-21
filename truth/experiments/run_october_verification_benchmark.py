import os, sys, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

print("=== RUNNING OCTOBER VERIFICATION BENCHMARK ===")
t0 = time.time()

TRAIN_PATH = r'D:\gci\inpuy\train.csv'
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = r'D:\gci\rules and info\04. Competition (due Nov 20)\Competition\input\train.csv'

df = pd.read_csv(TRAIN_PATH)
y = df['TARGET'].values
n = len(df)
print(f"Loaded {n} rows, target mean = {y.mean():.5f}")

# 1. Base clean features
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]
X_base = pd.DataFrame(index=df.index)
for c in CATS:
    X_base[c] = df[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)

# Numeric clean
for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "REGION_POPULATION_RELATIVE"]:
    if c in df.columns:
        X_base[c] = df[c].values

# Sentinel & missingness
X_base['DE_SENT'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
X_base['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED'] != 365243)
X_base['MISS_CNT'] = df.isna().sum(axis=1)
for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    X_base[c] = df[c]
    X_base[c + '_NA'] = df[c].isna().astype(int)

# 2. Ext features
e1 = df['EXT_SOURCE_1'].fillna(0.5)
e2 = df['EXT_SOURCE_2'].fillna(0.5)
e3 = df['EXT_SOURCE_3'].fillna(0.5)
X_base['EXT_PRODUCT'] = e1 * e2 * e3
X_base['EXT_MEAN'] = (e1 + e2 + e3) / 3.0
X_base['EXT_HARMONIC'] = 3.0 / (1.0/np.clip(e1, 1e-4, 1.0) + 1.0/np.clip(e2, 1e-4, 1.0) + 1.0/np.clip(e3, 1e-4, 1.0))
X_base['EXT_MIN'] = np.minimum(np.minimum(e1, e2), e3)
X_base['EXT_MAX'] = np.maximum(np.maximum(e1, e2), e3)
X_base['EXT_STD'] = np.std([e1, e2, e3], axis=0)

# 3. Loan features
X_base['CREDIT_TERM'] = df['AMT_CREDIT'] / np.clip(df['AMT_ANNUITY'], 1.0, None)
X_base['GOODS_TO_CREDIT'] = df['AMT_GOODS_PRICE'] / np.clip(df['AMT_CREDIT'], 1.0, None)
X_base['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / np.clip(df['AMT_INCOME_TOTAL'], 1.0, None)
X_base['CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / np.clip(df['AMT_INCOME_TOTAL'], 1.0, None)

# 4. Decoded Product Keys
annuity = df['AMT_ANNUITY'].values
credit = df['AMT_CREDIT'].values
goods = df['AMT_GOODS_PRICE'].values
q_raw = annuity / np.clip(credit, 1.0, None)
q0_fee = (annuity - 129.5) / np.clip(credit, 1.0, None)

q3 = np.round(q0_fee, 3)
q4 = np.round(q0_fee, 4)
p3 = np.round(q_raw, 3)
p4 = np.round(q_raw, 4)
pf4 = np.round(goods / np.clip(credit, 1.0, None), 4)

keys_df = pd.DataFrame({
    'q3': q3, 'q4': q4, 'p3': p3, 'p4': p4, 'pf4': pf4
})

print(f"Features prepared in {time.time()-t0:.2f}s. Running 5-fold CV to test October architecture...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_standard = np.zeros(n)
oof_et = np.zeros(n)

lgb_params_standard = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 600,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42
}

lgb_params_et = {
    **lgb_params_standard,
    'extra_trees': True
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_base, y)):
    f_t0 = time.time()
    # In-fold target encoding for keys
    X_trn = X_base.iloc[trn_idx].copy()
    X_val = X_base.iloc[val_idx].copy()
    y_trn = y[trn_idx]
    y_val = y[val_idx]
    prior = y_trn.mean()

    for col in ['q3', 'q4', 'p3', 'p4', 'pf4']:
        # Smooth TE with M0=30
        means = y_trn.groupby(keys_df.iloc[trn_idx][col]).mean() if hasattr(y_trn, 'groupby') else pd.Series(y_trn).groupby(keys_df.iloc[trn_idx][col].values).mean()
        counts = keys_df.iloc[trn_idx][col].value_counts()
        smoothed = (means * counts + 30.0 * prior) / (counts + 30.0)
        
        te_name = f'TE_{col}_M30'
        X_trn[te_name] = keys_df.iloc[trn_idx][col].map(smoothed).fillna(prior).values
        X_val[te_name] = keys_df.iloc[val_idx][col].map(smoothed).fillna(prior).values

    # Train Standard LightGBM
    clf_std = lgb.LGBMClassifier(**lgb_params_standard)
    clf_std.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)])
    val_pred_std = clf_std.predict_proba(X_val)[:, 1]
    oof_standard[val_idx] = val_pred_std
    auc_std = roc_auc_score(y_val, val_pred_std)

    # Train ExtraTrees LightGBM
    clf_et = lgb.LGBMClassifier(**lgb_params_et)
    clf_et.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)])
    val_pred_et = clf_et.predict_proba(X_val)[:, 1]
    oof_et[val_idx] = val_pred_et
    auc_et = roc_auc_score(y_val, val_pred_et)

    print(f"Fold {fold+1}: Standard AUC = {auc_std:.5f} | ExtraTrees AUC = {auc_et:.5f} (diff: {auc_et - auc_std:+.5f}) [{time.time()-f_t0:.1f}s]")

auc_total_std = roc_auc_score(y, oof_standard)
auc_total_et = roc_auc_score(y, oof_et)
rank_blend = (rankdata(oof_standard) + rankdata(oof_et)) / 2.0
auc_blend = roc_auc_score(y, rank_blend)

print("-" * 60)
print(f"POOLED 5-FOLD RESULTS:")
print(f"  Standard LightGBM OOF AUC : {auc_total_std:.5f}")
print(f"  ExtraTrees LightGBM OOF AUC: {auc_total_et:.5f} ({auc_total_et - auc_total_std:+.5f})")
print(f"  Rank Blend (Std + ET)     : {auc_blend:.5f} ({auc_blend - auc_total_std:+.5f})")
print(f"Total benchmark time: {time.time()-t0:.1f}s")
