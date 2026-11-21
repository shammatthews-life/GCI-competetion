import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

orig = pd.read_csv('d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv')['TARGET'].values
r_orig = rk_norm(orig)

v11 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v11.csv')['TARGET'].values)
v17 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v17.csv')['TARGET'].values)
v18 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v18.csv')['TARGET'].values)
v20 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v20.csv')['TARGET'].values)
v22 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v22.csv')['TARGET'].values)
v23 = rk_norm(pd.read_csv('d:/gci/v_22_pure_champion_ensemble_certified_v23.csv')['TARGET'].values)

streams = {'v11': v11, 'v17': v17, 'v18': v18, 'v20': v20, 'v22': v22, 'v23': v23}

print('Pairwise Spearman Correlation with Orig 0.78239:')
for k, s in streams.items():
    rho, _ = spearmanr(s, r_orig)
    print(f'  {k:5s}: {rho:.6f}')

names = ['v11', 'v17', 'v18', 'v20', 'v23']
X = np.column_stack([v11, v17, v18, v20, v23])

best_rho = 0.0
best_w = None
np.random.seed(42)
for _ in range(10000):
    w = np.random.dirichlet(np.ones(5))
    blend = X @ w
    rho = np.corrcoef(blend, r_orig)[0, 1]
    if rho > best_rho:
        best_rho = rho
        best_w = w

print('\nBest Random Search Simplex Weights:')
for n, wt in zip(names, best_w):
    print(f'  {n}: {wt:.4f}')

blend_best = rk_norm(X @ best_w)
rho_s, _ = spearmanr(blend_best, r_orig)
print(f'Actual Spearman Rank Correlation: {rho_s:.6f}')

# Now also test (v22, v23) convex blend
print('\nTesting V22 + V23 convex combinations:')
for w_23 in np.linspace(0.0, 0.5, 11):
    w_22 = 1.0 - w_23
    b = w_22 * v22 + w_23 * v23
    rho_b, _ = spearmanr(b, r_orig)
    print(f'  w_v22={w_22:.2f}, w_v23={w_23:.2f} -> Spearman={rho_b:.6f}')
