import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

test = pd.read_csv('d:/gci/inpuy/test.csv')
orig = pd.read_csv('d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv')['TARGET'].values
r_orig = rk_norm(orig)

v25 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v25.csv')['TARGET'].values)

base_rho = spearmanr(v25, r_orig).statistic
print(f"Base V25 Spearman vs 0.782: {base_rho:.6f}")

is_revolving = (test['NAME_CONTRACT_TYPE'] == 'Revolving loans').values
ext_nobs = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1).values
is_pensioner = (test['DAYS_EMPLOYED'] == 365243).values
is_thin = (ext_nobs <= 1)

part_rev = is_revolving
part_distress = (~is_revolving) & (is_thin | is_pensioner)
part_dense = (~is_revolving) & (~(is_thin | is_pensioner))

# What if we apply smooth probit quantile alignment?
# Instead of hard partition boundaries, let's test a continuous multi-regime convex blend:
# P = w_base * v25 + w_solo_v23 * (correlated components)
for alpha in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
    # test power transform to match skewness of original
    # original target has skewness
    p_trans = v25 ** (1.0 + alpha)
    r_trans = rk_norm(p_trans)
    # spearman of monotonic transform is identical, but check pearson
    pr = np.corrcoef(r_trans, r_orig)[0, 1]
    # print
print(f"Base V25 Pearson: {np.corrcoef(v25, r_orig)[0, 1]:.6f}")

# Check correlation with LEADERBOARD_SOLO_CHAMPION_V23.csv
solo_v23 = rk_norm(pd.read_csv('d:/gci/LEADERBOARD_SOLO_CHAMPION_V23.csv')['TARGET'].values)
print(f"V25 vs LEADERBOARD_SOLO_CHAMPION_V23: Spearman = {spearmanr(v25, solo_v23).statistic:.6f}")
