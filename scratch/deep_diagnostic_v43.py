import sys, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, f1_score, precision_score, recall_score
)

print("=" * 80)
print("COMPREHENSIVE ERROR DIAGNOSTIC: CONFUSION MATRIX, PRECISION & RECALL ANALYSIS")
print("=" * 80)

# Check if V43 OOF predictions are available or can be reconstructed quickly
tr_path = "inpuy/train.csv"
df_tr = pd.read_csv(tr_path)
y_true = df_tr['TARGET'].values.astype(int)
n_train = len(df_tr)
pos_prev = np.mean(y_true)

print(f"Total training applicants: {n_train:,}")
print(f"Class imbalance: {np.sum(y_true == 0):,} non-defaulters ({100*(1-pos_prev):.2f}%) vs {np.sum(y_true == 1):,} defaulters ({100*pos_prev:.2f}%)")
print(f"Natural default prevalence rate: {pos_prev:.5f} (~{pos_prev*100:.2f}%)")
print("=" * 80)
