import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score

oof = pd.read_csv('d:/gci/oof_predictions_v10_conservative_grade.csv')
y_true = oof['TARGET'].values
y_pred = oof['oof_ensemble'].values

auc = roc_auc_score(y_true, y_pred)
print(f"Overall OOF AUC: {auc:.6f}")

print("\nThreshold Search for F1 Score:")
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<7} | {'FP':<7} | {'FN':<7} | {'TN':<7}")
print("-" * 80)

best_f1 = 0
best_th = 0
for th in np.arange(0.05, 0.45, 0.025):
    preds_binary = (y_pred >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_binary).ravel()
    p = precision_score(y_true, preds_binary, zero_division=0)
    r = recall_score(y_true, preds_binary, zero_division=0)
    f1 = f1_score(y_true, preds_binary, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
    print(f"{th:<10.3f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {tp:<7d} | {fp:<7d} | {fn:<7d} | {tn:<7d}")

print(f"\nBest F1: {best_f1:.4f} at threshold {best_th:.3f}")

# Deep Dive into Error Regions
df_train = pd.read_csv('d:/gci/inpuy/train.csv')
merged = pd.concat([df_train, oof[['oof_ensemble', 'fold']]], axis=1)

# Zone 1: False Negatives (TARGET=1, but predicted < 0.08) - Defaulters deemed completely safe!
fn_mask = (merged['TARGET'] == 1) & (merged['oof_ensemble'] < 0.08)
tp_mask = (merged['TARGET'] == 1) & (merged['oof_ensemble'] >= 0.15)
tn_mask = (merged['TARGET'] == 0) & (merged['oof_ensemble'] < 0.08)
fp_mask = (merged['TARGET'] == 0) & (merged['oof_ensemble'] >= 0.20)

print(f"\nTotal Severe False Negatives (Defaulted with P < 0.08): {fn_mask.sum():,}")
print(f"Total Severe False Positives (Safe with P >= 0.20): {fp_mask.sum():,}")

cols_to_compare = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL',
    'DAYS_BIRTH', 'DAYS_EMPLOYED'
]

print("\n--- Feature Profiles across Error Cohorts ---")
print(f"{'Feature':<25} | {'Severe FN (Default <0.08)':<25} | {'True Positives (>=0.15)':<25} | {'True Negatives (<0.08)':<25}")
print("-" * 105)
for col in cols_to_compare:
    fn_val = merged.loc[fn_mask, col].median()
    tp_val = merged.loc[tp_mask, col].median()
    tn_val = merged.loc[tn_mask, col].median()
    print(f"{col:<25} | {fn_val:<25.4f} | {tp_val:<25.4f} | {tn_val:<25.4f}")

