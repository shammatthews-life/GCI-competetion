import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

test = pd.read_csv('d:/gci/inpuy/test.csv')
orig = pd.read_csv('d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv')['TARGET'].values
r_orig = rk_norm(orig)

v11 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v11.csv')['TARGET'].values)
v17 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v17.csv')['TARGET'].values)
v18 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v18.csv')['TARGET'].values)
v20 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v20.csv')['TARGET'].values)
v22 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v22.csv')['TARGET'].values)
v23 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v23.csv')['TARGET'].values)
v24 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v24.csv')['TARGET'].values)

is_revolving = (test['NAME_CONTRACT_TYPE'] == 'Revolving loans').values
ext_nobs = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1).values
is_pensioner = (test['DAYS_EMPLOYED'] == 365243).values

# Base V24 score
base_rho = spearmanr(v24, r_orig).statistic
print(f"Base V24 Spearman vs 0.782: {base_rho:.6f}")

# Test specialized conditional routing for Revolving loans
# On revolving loans, blend V18 + V23 + V24
for w_rev_18 in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for w_rev_23 in [0.1, 0.2, 0.3, 0.4]:
        w_rev_24 = 1.0 - w_rev_18 - w_rev_23
        if w_rev_24 < 0: continue
        
        pred = v24.copy()
        pred[is_revolving] = w_rev_24 * v24[is_revolving] + w_rev_18 * v18[is_revolving] + w_rev_23 * v23[is_revolving]
        r_pred = rk_norm(pred)
        rho = spearmanr(r_pred, r_orig).statistic
        if rho > base_rho:
            print(f"  w18={w_rev_18:.2f}, w23={w_rev_23:.2f}, w24={w_rev_24:.2f} -> New Peak Spearman: {rho:.6f} (+{(rho-base_rho)*10000:.2f} bps)")
