import numpy as np
import pandas as pd
from scipy.stats import rankdata


V17_PATH = r"D:\gci\pre info\oct22\GCI_submission_v17.csv"
V18_PATH = r"D:\gci\pre info\oct22\GCI_submission_v18.csv"
V22_PATH = r"D:\gci\gemini\v_22_pure_champion_ensemble.csv"

sub17 = pd.read_csv(V17_PATH)["TARGET"].values
sub18 = pd.read_csv(V18_PATH)["TARGET"].values
sub22 = pd.read_csv(V22_PATH)["TARGET"].values

if not (len(sub17) == len(sub18) == len(sub22)):
    raise ValueError("V17, V18, and V22 prediction files must have equal row counts")

rk = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

r17 = rk(sub17)
r18 = rk(sub18)
r22 = rk(sub22)

print("Testing fine blend grid between v17 (0.78237) and v18 (0.78235):")
for w17 in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    w18 = 1.0 - w17
    blend = rk(w17 * r17 + w18 * r18)
    corr_v22 = pd.Series(blend).corr(pd.Series(r22), method="spearman")
    max_diff = np.max(np.abs(blend - r22))
    mean_diff = np.mean(np.abs(blend - r22))
    print(
        f"Weight {w17 * 100:2.0f}% v17 + {w18 * 100:2.0f}% v18 -> "
        f"Corr with v22: {corr_v22:.7f}, Max rank diff: {max_diff:.5f}, "
        f"Mean rank diff: {mean_diff:.6f}"
    )
