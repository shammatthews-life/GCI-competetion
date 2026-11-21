import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Starting Deep Research & Diagnostic Autopsy...")
t0 = time.time()
df_tr = pd.read_csv('d:/gci/inpuy/train.csv')
y = df_tr['TARGET'].values

print(f"Loaded train.csv: {df_tr.shape}, target rate: {y.mean():.4f}")

# Quick feature extraction from V43 base
ext1 = df_tr['EXT_SOURCE_1'].values
ext2 = df_tr['EXT_SOURCE_2'].values
ext3 = df_tr['EXT_SOURCE_3'].values
ext_mean = np.nanmean(np.column_stack([ext1, ext2, ext3]), axis=1)

inc = df_tr['AMT_INCOME_TOTAL'].values
cred = df_tr['AMT_CREDIT'].values
ann = df_tr['AMT_ANNUITY'].values
goods = df_tr['AMT_GOODS_PRICE'].fillna(df_tr['AMT_CREDIT']).values

ltv = cred / (goods + 1.0)
dti = cred / (inc + 1.0)
term = cred / (ann + 1.0)
ann_inc = ann / (inc + 1.0)
ear = 24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0)

# Build a compact diagnostic feature set
feats = pd.DataFrame({
    'EXT_SOURCE_1': ext1,
    'EXT_SOURCE_2': ext2,
    'EXT_SOURCE_3': ext3,
    'EXT_MEAN': ext_mean,
    'EXT_NA_COUNT': np.isnan(ext1).astype(int) + np.isnan(ext2).astype(int) + np.isnan(ext3).astype(int),
    'AMT_CREDIT': cred,
    'AMT_ANNUITY': ann,
    'AMT_INCOME_TOTAL': inc,
    'AMT_GOODS_PRICE': goods,
    'LTV': ltv,
    'DTI': dti,
    'CREDIT_TERM': term,
    'ANN_INC': ann_inc,
    'EAR': ear,
    'DAYS_BIRTH': df_tr['DAYS_BIRTH'].values,
    'DAYS_EMPLOYED': df_tr['DAYS_EMPLOYED'].values,
    'DAYS_EMPLOYED_ANOM': (df_tr['DAYS_EMPLOYED'].values == 365243).astype(int),
    'DAYS_ID_PUBLISH': df_tr['DAYS_ID_PUBLISH'].values,
    'DAYS_REGISTRATION': df_tr['DAYS_REGISTRATION'].values,
    'DAYS_LAST_PHONE_CHANGE': df_tr['DAYS_LAST_PHONE_CHANGE'].values,
    'REGION_RATING_CLIENT': df_tr['REGION_RATING_CLIENT'].values,
    'CNT_FAM_MEMBERS': df_tr['CNT_FAM_MEMBERS'].fillna(1).values,
    'CNT_CHILDREN': df_tr['CNT_CHILDREN'].values,
    'IS_REVOLVING': (df_tr['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(int),
    'CODE_GENDER_M': (df_tr['CODE_GENDER'] == 'M').astype(int),
    'RATIO_EXT_MEAN_LTV': ext_mean / (ltv + 1.0),
    'DIFF_EXT_MEAN_LTV': ext_mean - ltv,
})

# Run 5-fold CV with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'n_jobs': 8,
    'verbose': -1,
    'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(feats, y)):
    X_tr, y_tr = feats.iloc[trn_idx], y[trn_idx]
    X_va, y_va = feats.iloc[val_idx], y[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=600,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof[val_idx] = model.predict(X_va)

auc = roc_auc_score(y, oof)
pr_auc = average_precision_score(y, oof)
print(f"Compact Baseline OOF ROC-AUC: {auc:.6f}, PR-AUC: {pr_auc:.6f} in {time.time()-t0:.1f}s")

# Operational Risk Analysis
for pct in [5, 10, 15, 20]:
    cutoff = np.percentile(oof, 100 - pct)
    preds = (oof >= cutoff).astype(int)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    f1 = f1_score(y, preds)
    print(f"\n--- Cutoff Top {pct}% (threshold={cutoff:.4f}) ---")
    print(f"Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    print(f"Recall: {rec*100:.2f}%, Precision: {prec*100:.2f}%, F1: {f1:.4f}")

# Slice error characteristics at 10% cutoff
cutoff_10 = np.percentile(oof, 90)
preds_10 = (oof >= cutoff_10).astype(int)
tp_mask = (y == 1) & (preds_10 == 1)
fn_mask = (y == 1) & (preds_10 == 0)
fp_mask = (y == 0) & (preds_10 == 1)
tn_mask = (y == 0) & (preds_10 == 0)

print("\n" + "="*80)
print("DEEP SLICE FORENSIC: TP (Caught Defaulters) vs FN (Missed Defaulters)")
print("="*80)

def print_stat(col_name, series):
    print(f"{col_name:25s}: TP={series[tp_mask].mean():.4f} | FN={series[fn_mask].mean():.4f} | FP={series[fp_mask].mean():.4f} | TN={series[tn_mask].mean():.4f}")

print_stat("EXT_MEAN", feats['EXT_MEAN'])
print_stat("EXT_NA_COUNT", feats['EXT_NA_COUNT'])
print_stat("IS_REVOLVING", feats['IS_REVOLVING'])
print_stat("CODE_GENDER_M", feats['CODE_GENDER_M'])
print_stat("DAYS_EMPLOYED_ANOM", feats['DAYS_EMPLOYED_ANOM'])
print_stat("LTV", feats['LTV'])
print_stat("DTI", feats['DTI'])
print_stat("CREDIT_TERM", feats['CREDIT_TERM'])
print_stat("EAR", feats['EAR'])
print_stat("AMT_CREDIT", feats['AMT_CREDIT'])
print_stat("AMT_ANNUITY", feats['AMT_ANNUITY'])
print_stat("AMT_INCOME_TOTAL", feats['AMT_INCOME_TOTAL'])
