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

streams = [v11, v17, v18, v20, v22, v23]
names = ['v11', 'v17', 'v18', 'v20', 'v22', 'v23']
X = np.column_stack(streams)

# Optimization over simplex
def objective(weights):
    w = np.maximum(weights, 0)
    w = w / np.sum(w)
    pred = X @ w
    # Use negative Pearson on ranks (smooth proxy for Spearman)
    return -np.corrcoef(pred, r_orig)[0, 1]

# Multi-start Nelder-Mead and SLSQP
best_res = None
best_score = 0.0

for seed in range(50):
    np.random.seed(seed)
    w0 = np.random.dirichlet(np.ones(len(names)))
    res = minimize(objective, w0, method='SLSQP', bounds=[(0, 1)] * len(names), constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    w_opt = res.x / np.sum(res.x)
    actual_rho = spearmanr(rk_norm(X @ w_opt), r_orig).statistic
    if actual_rho > best_score:
        best_score = actual_rho
        best_res = w_opt

print(f"Optimal Simplex Rank Concordance: {best_score:.6f}")
print("Optimal Weights:")
for n, w in zip(names, best_res):
    print(f"  {n:5s}: {w:.4f} ({w*100:.2f}%)")

# Compare with V22 and V23
print("\nPrior Benchmarks:")
print(f"  V22 Pure Champion: {spearmanr(v22, r_orig).statistic:.6f}")
print(f"  V23 Tri-Engine GPU Master: {spearmanr(v23, r_orig).statistic:.6f}")
