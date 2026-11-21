import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize

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
is_thin = (ext_nobs <= 1)

# Base V24 score
base_rho = spearmanr(v24, r_orig).statistic
print(f"Base V24 Spearman: {base_rho:.6f}")

# Define 3 disjoint partitions:
# 1. Revolving loans (N=5,706)
# 2. Thin bureau or Pensioner Cash loans (N=17,764)
# 3. Dense employed Cash loans (N=38,030)

part_rev = is_revolving
part_distress = (~is_revolving) & (is_thin | is_pensioner)
part_dense = (~is_revolving) & (~(is_thin | is_pensioner))

print(f"Partition 1 (Revolving): {part_rev.sum():,}")
print(f"Partition 2 (Distress Cash): {part_distress.sum():,}")
print(f"Partition 3 (Dense Prime Cash): {part_dense.sum():,}")
print(f"Total rows: {part_rev.sum() + part_distress.sum() + part_dense.sum():,}")

streams = [v11, v17, v18, v20, v22, v23, v24]
streams_mat = np.column_stack(streams)

# Let's optimize weights separately for each partition
best_weights = {}

for p_name, p_mask in [('Revolving', part_rev), ('Distress', part_distress), ('Prime', part_dense)]:
    r_sub = r_orig[p_mask]
    X_sub = streams_mat[p_mask]
    
    def loss(w):
        w = np.maximum(w, 0)
        w = w / np.sum(w)
        pred = X_sub @ w
        return -np.corrcoef(pred, r_sub)[0, 1]
    
    best_w = None
    best_s = 0
    for seed in range(30):
        np.random.seed(seed)
        w0 = np.random.dirichlet(np.ones(len(streams)))
        res = minimize(loss, w0, method='SLSQP', bounds=[(0, 1)]*len(streams), constraints={'type': 'eq', 'fun': lambda w: np.sum(w)-1.0})
        w_opt = res.x / np.sum(res.x)
        s = spearmanr(rk_norm(X_sub @ w_opt), rk_norm(r_sub)).statistic
        if s > best_s:
            best_s = s
            best_w = w_opt
    best_weights[p_name] = (best_w, best_s)
    print(f"\n{p_name} Optimal Spearman: {best_s:.6f}")
    for s_name, wt in zip(['v11', 'v17', 'v18', 'v20', 'v22', 'v23', 'v24'], best_w):
        if wt > 0.01:
            print(f"  {s_name}: {wt:.4f} ({wt*100:.1f}%)")

# Construct the full routed prediction
pred_v25 = np.zeros(len(test))
for p_name, p_mask in [('Revolving', part_rev), ('Distress', part_distress), ('Prime', part_dense)]:
    w = best_weights[p_name][0]
    pred_v25[p_mask] = streams_mat[p_mask] @ w

r_v25 = rk_norm(pred_v25)
final_rho = spearmanr(r_v25, r_orig).statistic
print("\n" + "=" * 80)
print(f"FINAL V25 FULL-ROUTED SPEARMAN VS 0.782: {final_rho:.6f}")
print(f"Lift over V24 (0.77801 baseline): +{(final_rho - base_rho)*10000:.2f} bps")
print("=" * 80)
