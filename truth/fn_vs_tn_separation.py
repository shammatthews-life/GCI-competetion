import pandas as pd
import numpy as np

oof = pd.read_csv('d:/gci/oof_predictions_v10_conservative_grade.csv')
df_train = pd.read_csv('d:/gci/inpuy/train.csv')
merged = pd.concat([df_train, oof[['oof_ensemble']]], axis=1)

fn_mask = (merged['TARGET'] == 1) & (merged['oof_ensemble'] < 0.08)
tn_mask = (merged['TARGET'] == 0) & (merged['oof_ensemble'] < 0.08)

fn_df = merged[fn_mask]
tn_df = merged[tn_mask]

print(f"Comparing {len(fn_df)} Severe FN vs {len(tn_df)} True Negatives:")

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.drop(['SK_ID_CURR', 'TARGET'])

diffs = []
for c in numeric_cols:
    fn_m = fn_df[c].median()
    tn_m = tn_df[c].median()
    fn_mean = fn_df[c].mean()
    tn_mean = tn_df[c].mean()
    std = df_train[c].std()
    if std > 0:
        d = (fn_mean - tn_mean) / std
        diffs.append((c, d, fn_m, tn_m))

diffs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Top 20 Features Separating Severe FN from TN (Standardized Mean Difference)':<70}")
print(f"{'Feature':<30} | {'Cohen d':<10} | {'FN Median':<15} | {'TN Median':<15}")
print("-" * 75)
for c, d, fn_m, tn_m in diffs[:20]:
    print(f"{c:<30} | {d:<10.4f} | {fn_m:<15.4f} | {tn_m:<15.4f}")

