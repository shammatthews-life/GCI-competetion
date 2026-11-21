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

test['is_revolving'] = (test['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(int)
test['ext_nobs'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1)

print("Segment Evaluation across V11, V17, V18, V20, V22, V23, V24 vs Orig 0.782:")
models = {'V11': v11, 'V17': v17, 'V18': v18, 'V20': v20, 'V22': v22, 'V23': v23, 'V24': v24}

for m_name, m_vals in models.items():
    rho_all = spearmanr(m_vals, r_orig).statistic
    rho_rev = spearmanr(rk_norm(m_vals[test['is_revolving']==1]), rk_norm(r_orig[test['is_revolving']==1])).statistic
    rho_cash = spearmanr(rk_norm(m_vals[test['is_revolving']==0]), rk_norm(r_orig[test['is_revolving']==0])).statistic
    rho_ext3 = spearmanr(rk_norm(m_vals[test['ext_nobs']==3]), rk_norm(r_orig[test['ext_nobs']==3])).statistic
    rho_thin = spearmanr(rk_norm(m_vals[test['ext_nobs']<=1]), rk_norm(r_orig[test['ext_nobs']<=1])).statistic
    print(f"  {m_name:5s} | Overall: {rho_all:.6f} | Revolving: {rho_rev:.6f} | Cash: {rho_cash:.6f} | Complete: {rho_ext3:.6f} | Thin: {rho_thin:.6f}")
