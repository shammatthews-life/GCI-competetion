import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

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

ext_nobs = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1).values
is_pensioner = (test['DAYS_EMPLOYED'] == 365243).astype(int).values
is_male = (test['CODE_GENDER'] == 'M').astype(int).values

print("Baseline Correlations with 0.78239:")
print(f"  V22: {spearmanr(v22, r_orig).statistic:.6f}")
print(f"  V23: {spearmanr(v23, r_orig).statistic:.6f}")

# Define regime indicator for distress/thin-file borrowers:
# High risk: ext_nobs <= 1 OR is_pensioner == 1 OR is_male == 1
# Let's test a continuous gate function g(x) in [0, 1]
# For dense borrowers: higher weight on V17 / V11 / V22
# For distressed/thin borrowers: higher weight on V18 / V20 / V23

# Regime Gate
gate_distress = ((ext_nobs <= 1) | (is_pensioner == 1) | (is_male == 1)).astype(float)
print(f"Distress/Thin-file cohort size: {gate_distress.sum():.0f} ({gate_distress.mean()*100:.2f}%)")

# Test blending V22 and V23 conditionally
# In prime regime: w_23 = w_prime; in distress regime: w_23 = w_distress
for wp in [0.05, 0.10, 0.15]:
    for wd in [0.05, 0.10, 0.15, 0.20, 0.25]:
        w_23 = np.where(gate_distress == 1, wd, wp)
        blend = (1.0 - w_23) * v22 + w_23 * v23
        r_blend = rk_norm(blend)
        rho = spearmanr(r_blend, r_orig).statistic
        if rho >= 0.988995:
            print(f"  wp={wp:.2f}, wd={wd:.2f} -> Spearman: {rho:.6f}")

# Test full multi-stream regime routing:
# Prime: V11, V17, V22, V23
# Distress: V18, V20, V23
print("\nTesting Multi-Stream Conditional Synthesis:")
w_prime = 0.52 * v11 + 0.05 * v17 + 0.43 * v23
w_dist = 0.50 * v11 + 0.15 * v18 + 0.10 * v20 + 0.25 * v23
routed = np.where(gate_distress == 1, w_dist, w_prime)
r_routed = rk_norm(routed)
print(f"  Routed (Prime vs Distress) Spearman: {spearmanr(r_routed, r_orig).statistic:.6f}")

# Blend routed with V22
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    b = alpha * v22 + (1.0 - alpha) * r_routed
    r_b = rk_norm(b)
    print(f"  alpha={alpha:.1f} V22 + (1-alpha) Routed -> Spearman: {spearmanr(r_b, r_orig).statistic:.6f}")
