import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr
import glob, os

rk_norm = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

orig = pd.read_csv('d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv')
r_orig = rk_norm(orig['TARGET'].values)

print("Scanning all candidate submission files in d:/gci:")
results = []
for f in sorted(glob.glob('d:/gci/*pure_champion*.csv') + glob.glob('d:/gci/*submission*.csv')):
    try:
        df = pd.read_csv(f)
        if len(df) == 61500 and 'TARGET' in df.columns:
            r = rk_norm(df['TARGET'].values)
            rho = spearmanr(r, r_orig).statistic
            results.append((f, rho))
    except Exception as e:
        pass

results.sort(key=lambda x: x[1], reverse=True)
for f, rho in results[:25]:
    print(f"  {os.path.basename(f):50s} -> Spearman vs 0.782: {rho:.6f} ({rho*100:.3f}%)")
