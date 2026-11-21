import pandas as pd
import numpy as np

oof = pd.read_csv('d:/gci/oof_predictions_v10_conservative_grade.csv')
df_train = pd.read_csv('d:/gci/inpuy/train.csv')
merged = pd.concat([df_train, oof[['oof_ensemble']]], axis=1)

fp_mask = (merged['TARGET'] == 0) & (merged['oof_ensemble'] >= 0.20)
tp_mask = (merged['TARGET'] == 1) & (merged['oof_ensemble'] >= 0.20)

fp_df = merged[fp_mask]
tp_df = merged[tp_mask]

print(f"Comparing {len(fp_df)} False Positives vs {len(tp_df)} True Positives (both predicted >= 0.20):")

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.drop(['SK_ID_CURR', 'TARGET'])

diffs = []
for c in numeric_cols:
    fp_m = fp_df[c].median()
    tp_m = tp_df[c].median()
    fp_mean = fp_df[c].mean()
    tp_mean = tp_df[c].mean()
    std = df_train[c].std()
    if std > 0:
        d = (fp_mean - tp_mean) / std
        diffs.append((c, d, fp_m, tp_m))

diffs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Top 20 Features Separating False Positives from True Positives':<70}")
print(f"{'Feature':<30} | {'Cohen d':<10} | {'FP Median':<15} | {'TP Median':<15}")
print("-" * 75)
for c, d, fp_m, tp_m in diffs[:20]:
    print(f"{c:<30} | {d:<10.4f} | {fp_m:<15.4f} | {tp_m:<15.4f}")

