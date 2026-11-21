import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

test = pd.read_csv('d:/gci/inpuy/test.csv')
train = pd.read_csv('d:/gci/inpuy/train.csv')
orig = pd.read_csv('d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv')['TARGET'].values
r_orig = rk_norm(orig)

v11 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v11.csv')['TARGET'].values)
v17 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v17.csv')['TARGET'].values)
v18 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v18.csv')['TARGET'].values)
v20 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v20.csv')['TARGET'].values)
v22 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v22.csv')['TARGET'].values)
v23 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v23.csv')['TARGET'].values)
v24 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v24.csv')['TARGET'].values)

# Diagnostic: where does V17 diverge most from V18?
diff_17_18 = v17 - v18
test['diff_17_18'] = diff_17_18
test['abs_diff'] = np.abs(diff_17_18)

print("Correlation v17 vs v18 on Test:", spearmanr(v17, v18).statistic)
print("Correlation v17 vs Orig 0.782:", spearmanr(v17, r_orig).statistic)
print("Correlation v18 vs Orig 0.782:", spearmanr(v18, r_orig).statistic)
print("Correlation v24 vs Orig 0.782:", spearmanr(v24, r_orig).statistic)

# Multi-segment analysis
test['ext_nobs'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1)
test['is_pensioner'] = (test['DAYS_EMPLOYED'] == 365243).astype(int)
test['is_revolving'] = (test['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(int)
test['leverage'] = test['AMT_CREDIT'] / (test['AMT_INCOME_TOTAL'] + 1.0)
test['high_leverage'] = (test['leverage'] > 4.0).astype(int)
test['is_male'] = (test['CODE_GENDER'] == 'M').astype(int)

# Segment-level correlation against 0.782 benchmark
segments = {
    'Complete Bureau (EXT_NOBS=3)': test['ext_nobs'] == 3,
    'Duo Bureau (EXT_NOBS=2)': test['ext_nobs'] == 2,
    'Thin Bureau (EXT_NOBS<=1)': test['ext_nobs'] <= 1,
    'Pensioners': test['is_pensioner'] == 1,
    'Employed': test['is_pensioner'] == 0,
    'Revolving Loans': test['is_revolving'] == 1,
    'Cash Loans': test['is_revolving'] == 0,
    'High Leverage (C/I > 4)': test['high_leverage'] == 1,
    'Low Leverage (C/I <= 4)': test['high_leverage'] == 0,
    'Males': test['is_male'] == 1,
    'Females': test['is_male'] == 0,
}

print("\n" + "=" * 80)
print(f"{'Segment Name':32s} | {'N Rows':7s} | {'v17 vs 0.782':12s} | {'v18 vs 0.782':12s} | {'Winner':8s} | {'Spread (bps)':12s}")
print("=" * 80)

v17_wins = 0
v18_wins = 0
segment_weights = {}

for name, mask in segments.items():
    n_rows = mask.sum()
    r_sub_orig = rk_norm(r_orig[mask])
    r_sub_17 = rk_norm(v17[mask])
    r_sub_18 = rk_norm(v18[mask])
    
    rho_17 = spearmanr(r_sub_17, r_sub_orig).statistic
    rho_18 = spearmanr(r_sub_18, r_sub_orig).statistic
    diff_bps = (rho_18 - rho_17) * 10000
    
    winner = 'V18' if rho_18 > rho_17 else 'V17'
    if winner == 'V18': v18_wins += 1
    else: v17_wins += 1
    segment_weights[name] = (rho_17, rho_18, winner)
    
    print(f"{name:32s} | {n_rows:7,d} | {rho_17:.6f}     | {rho_18:.6f}     | {winner:8s} | {diff_bps:+7.1f} bps")

print("=" * 80)
print(f"Summary: V17 wins {v17_wins} segments, V18 wins {v18_wins} segments.")
