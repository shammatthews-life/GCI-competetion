import sys, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)

print("=" * 80)
print("ERROR DIAGNOSTIC: CONFUSION MATRIX, PRECISION, RECALL AT MULTIPLE THRESHOLDS")
print("=" * 80)

def evaluate_risk_thresholds(y_true, y_prob, model_name="Ensemble"):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    pos_prev = np.mean(y_true)

    print(f"\n==================== {model_name} EVALUATION ====================")
    print(f"  * ROC-AUC : {auc:.6f}")
    print(f"  * PR-AUC (Average Precision) : {ap:.6f} (Baseline: {pos_prev:.4f})")

    # Define operational evaluation thresholds
    # 1. Natural prevalence threshold
    # 2. Top 5% risk percentile (Strict Subprime Reject)
    # 3. Top 10% risk percentile (Standard Bank Reject Cutoff)
    # 4. Top 15% risk percentile (Conservative Underwriting)
    # 5. Top 20% risk percentile (High Coverage Reject)

    percentiles = [5, 10, 15, 20]
    thresh_prev = np.quantile(y_prob, 1.0 - pos_prev)

    print("\n  OPERATIONAL THRESHOLD PERFORMANCE TABLE:")
    print("  " + "-" * 75)
    print(f"  {'Operating Cutoff':<22} | {'Thresh':<7} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'FP / FN Count':<16}")
    print("  " + "-" * 75)

    # Natural prevalence cutoff
    y_pred_prev = (y_prob >= thresh_prev).astype(int)
    cm_prev = confusion_matrix(y_true, y_pred_prev)
    tn, fp, fn, tp = cm_prev.ravel()
    prec_prev = precision_score(y_true, y_pred_prev)
    rec_prev = recall_score(y_true, y_pred_prev)
    f1_prev = f1_score(y_true, y_pred_prev)
    print(f"  {'Prevalence (~8.07%)':<22} | {thresh_prev:.4f}  | {prec_prev*100:6.2f}%   | {rec_prev*100:5.2f}%  | {f1_prev:.4f} | FP:{fp:,} FN:{fn:,}")

    for pct in percentiles:
        tau = np.quantile(y_prob, 1.0 - pct / 100.0)
        y_pred = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"  {f'Top {pct}% Risk Cutoff':<22} | {tau:.4f}  | {prec*100:6.2f}%   | {rec*100:5.2f}%  | {f1:.4f} | FP:{fp:,} FN:{fn:,}")
    print("  " + "-" * 75)

    # Print Confusion Matrix at Top 10% Operational Reject Cutoff
    tau_10 = np.quantile(y_prob, 0.90)
    y_pred_10 = (y_prob >= tau_10).astype(int)
    cm_10 = confusion_matrix(y_true, y_pred_10)
    print(f"\n  CONFUSION MATRIX at Top 10% Reject Cutoff (tau = {tau_10:.4f}):")
    print(f"               Predicted Non-Default (0)   Predicted Default (1)")
    print(f"  Actual 0 :         {cm_10[0, 0]:>10,}               {cm_10[0, 1]:>10,}  (False Positives)")
    print(f"  Actual 1 :         {cm_10[1, 0]:>10,}               {cm_10[1, 1]:>10,}  (True Positives)")
    print(f"  --> At Top 10% Cutoff, Model catches {cm_10[1,1]:,} of {np.sum(y_true):,} defaulters ({100*cm_10[1,1]/np.sum(y_true):.2f}% Recall)!")

# Test on synthetic data
np.random.seed(42)
y_dummy = (np.random.rand(10000) < 0.0807).astype(int)
p_dummy = np.clip(y_dummy * 0.3 + np.random.beta(2, 5, 10000), 0, 1)
evaluate_risk_thresholds(y_dummy, p_dummy, "Test Benchmark")
