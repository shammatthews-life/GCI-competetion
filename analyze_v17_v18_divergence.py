import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

# Load test data and predictions
test = pd.read_csv(r"d:\gci\inpuy\test.csv")
v17 = pd.read_csv(r"d:\gci\pre info\oct22\GCI_submission_v17.csv")
v18 = pd.read_csv(r"d:\gci\pre info\oct22\GCI_submission_v18.csv")
v22 = pd.read_csv(r"d:\gci\gemini\v_22_pure_champion_ensemble.csv")

rk = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)
r17 = rk(v17['TARGET'])
r18 = rk(v18['TARGET'])
r22 = rk(v22['TARGET'])

diff = r17 - r18
abs_diff = np.abs(diff)

print("Correlation v17 vs v18:", spearmanr(r17, r18).statistic)
print(f"Max rank difference: {abs_diff.max():.6f}")
print(f"Mean absolute rank difference: {abs_diff.mean():.6f}")
print(f"95th percentile rank difference: {np.percentile(abs_diff, 95):.6f}")
print(f"Number of rows with |rank diff| > 0.01: {(abs_diff > 0.01).sum()} ({ (abs_diff > 0.01).mean()*100:.2f}%)")

# Add divergence to test dataframe
df = test.copy()
df['diff'] = diff
df['abs_diff'] = abs_diff
df['v17_higher'] = (diff > 0).astype(int)

# Check divergence by key segments
print("\n--- Divergence by NAME_CONTRACT_TYPE ---")
for val, grp in df.groupby('NAME_CONTRACT_TYPE'):
    print(f"  {val} (N={len(grp)}): Mean diff={grp['diff'].mean():.6f}, Mean |diff|={grp['abs_diff'].mean():.6f}")

print("\n--- Divergence by EXT_SOURCE Missingness Regime ---")
ext_nobs = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1)
df['ext_nobs'] = ext_nobs
for val, grp in df.groupby('ext_nobs'):
    print(f"  ext_nobs={val} (N={len(grp)}): Mean diff={grp['diff'].mean():.6f}, Mean |diff|={grp['abs_diff'].mean():.6f}")

print("\n--- Divergence by CODE_GENDER ---")
for val, grp in df.groupby('CODE_GENDER'):
    print(f"  {val} (N={len(grp)}): Mean diff={grp['diff'].mean():.6f}, Mean |diff|={grp['abs_diff'].mean():.6f}")

print("\n--- Divergence by DAYS_EMPLOYED Sentinel ---")
df['is_pensioner'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
for val, grp in df.groupby('is_pensioner'):
    print(f"  is_pensioner={val} (N={len(grp)}): Mean diff={grp['diff'].mean():.6f}, Mean |diff|={grp['abs_diff'].mean():.6f}")

