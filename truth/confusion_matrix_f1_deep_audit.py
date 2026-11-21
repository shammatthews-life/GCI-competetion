import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Load training data
train = pd.read_csv('d:/gci/inpuy/train.csv')
y = train['TARGET'].values

print(f"Total training applicants: {len(train):,}")
print(f"Defaults (y=1): {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"Non-defaults (y=0): {(1-y).sum():,} ({(1-y).mean()*100:.2f}%)")
print(f"Imbalance ratio: {(1-y).sum() / y.sum():.2f} : 1")

CATS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

X = train.drop(columns=['SK_ID_CURR', 'TARGET']).copy()
for c in CATS:
    X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)

# Add basic key ratios
X['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
X['CTI'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
X['ATI'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
X['EXT_MEAN'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
train['EXT_NOBS'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1)
X['EXT_NOBS'] = train['EXT_NOBS']

oof_preds = np.zeros(len(train))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
    dtrain = lgb.Dataset(X.iloc[trn_idx], y[trn_idx])
    m = lgb.train(
        {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'seed': 42},
        dtrain, num_boost_round=400
    )
    oof_preds[val_idx] = m.predict(X.iloc[val_idx])

auc = roc_auc_score(y, oof_preds)
print(f"\nBaseline OOF AUC: {auc:.5f}")

# Threshold scan for F1 score
thresholds = np.linspace(0.05, 0.50, 46)
f1_scores = []
precisions = []
recalls = []

best_f1 = 0.0
best_tau = 0.0
for tau in thresholds:
    pred_bin = (oof_preds >= tau).astype(int)
    f1 = f1_score(y, pred_bin)
    p = precision_score(y, pred_bin, zero_division=0)
    r = recall_score(y, pred_bin, zero_division=0)
    f1_scores.append(f1)
    precisions.append(p)
    recalls.append(r)
    if f1 > best_f1:
        best_f1 = f1
        best_tau = tau

print(f"\nOptimal Operating Threshold tau*: {best_tau:.3f}")
print(f"Optimal F1 Score: {best_f1:.4f} (Precision: {precisions[np.argmax(f1_scores)]:.4f}, Recall: {recalls[np.argmax(f1_scores)]:.4f})")

# Compute Confusion Matrix at tau*
pred_opt = (oof_preds >= best_tau).astype(int)
cm = confusion_matrix(y, pred_opt)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix at tau* = {best_tau:.3f}:")
print(f"  True Negatives (TN):  {tn:,} ({tn/len(y)*100:.2f}%)")
print(f"  False Positives (FP): {fp:,} ({fp/len(y)*100:.2f}%)")
print(f"  False Negatives (FN): {fn:,} ({fn/len(y)*100:.2f}%)")
print(f"  True Positives (TP):  {tp:,} ({tp/len(y)*100:.2f}%)")

# Forensic Error Region Analysis
train['oof'] = oof_preds
train['pred_bin'] = pred_opt
train['error_type'] = np.where((y==1) & (pred_opt==0), 'FN',
                      np.where((y==0) & (pred_opt==1), 'FP',
                      np.where((y==1) & (pred_opt==1), 'TP', 'TN')))

print("\n--- Error Breakdown by External Score Missingness (EXT_NOBS) ---")
for nobs, grp in train.groupby('EXT_NOBS'):
    fn_rate = (grp['error_type'] == 'FN').sum() / max(1, (grp['TARGET'] == 1).sum())
    fp_rate = (grp['error_type'] == 'FP').sum() / max(1, (grp['TARGET'] == 0).sum())
    auc_grp = roc_auc_score(grp['TARGET'], grp['oof']) if len(grp['TARGET'].unique()) > 1 else np.nan
    print(f"  EXT_NOBS={nobs} (N={len(grp):,}): AUC={auc_grp:.5f}, FN Rate on Defaults={fn_rate*100:.1f}%, FP Rate on Non-defaults={fp_rate*100:.1f}%")

print("\n--- Error Breakdown by Pensioner Status ---")
train['is_pensioner'] = (train['DAYS_EMPLOYED'] == 365243).astype(int)
for pen, grp in train.groupby('is_pensioner'):
    fn_rate = (grp['error_type'] == 'FN').sum() / max(1, (grp['TARGET'] == 1).sum())
    fp_rate = (grp['error_type'] == 'FP').sum() / max(1, (grp['TARGET'] == 0).sum())
    auc_grp = roc_auc_score(grp['TARGET'], grp['oof']) if len(grp['TARGET'].unique()) > 1 else np.nan
    label = 'Pensioner' if pen == 1 else 'Employed'
    print(f"  {label} (N={len(grp):,}): AUC={auc_grp:.5f}, FN Rate on Defaults={fn_rate*100:.1f}%, FP Rate on Non-defaults={fp_rate*100:.1f}%")

print("\n--- Error Breakdown by Gender ---")
for gen, grp in train.groupby('CODE_GENDER'):
    fn_rate = (grp['error_type'] == 'FN').sum() / max(1, (grp['TARGET'] == 1).sum())
    fp_rate = (grp['error_type'] == 'FP').sum() / max(1, (grp['TARGET'] == 0).sum())
    auc_grp = roc_auc_score(grp['TARGET'], grp['oof']) if len(grp['TARGET'].unique()) > 1 else np.nan
    print(f"  Gender {gen} (N={len(grp):,}): AUC={auc_grp:.5f}, FN Rate on Defaults={fn_rate*100:.1f}%, FP Rate on Non-defaults={fp_rate*100:.1f}%")
