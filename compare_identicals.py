import pandas as pd
import numpy as np
from scipy.stats import spearmanr

orig_p = r"d:/gci/gemini/v_22_pure_champion_ensemble.csv"
scratch_p = r"d:/gci/v_22_pure_champion_ensemble_scratch.csv"
blend_p = r"d:/gci/submission_4_v22_champion_078239.csv"

orig = pd.read_csv(orig_p)
scratch = pd.read_csv(scratch_p)
blend = pd.read_csv(blend_p)

corr_scratch, _ = spearmanr(orig['TARGET'], scratch['TARGET'])
max_diff_scratch = np.max(np.abs(orig['TARGET'] - scratch['TARGET']))

corr_blend, _ = spearmanr(orig['TARGET'], blend['TARGET'])
max_diff_blend = np.max(np.abs(orig['TARGET'] - blend['TARGET']))

print(f"1. Fast Scratch vs Original Champion:")
print(f"   Spearman Correlation: {corr_scratch:.8f}")
print(f"   Max Absolute Difference: {max_diff_scratch:.8f}")
print(f"   100% Identical: {corr_scratch == 1.0}")

print(f"\n2. Blend Script (v17 + v18) vs Original Champion:")
print(f"   Spearman Correlation: {corr_blend:.8f}")
print(f"   Max Absolute Difference: {max_diff_blend:.8f}")
print(f"   100% Identical: {corr_blend == 1.0}")

