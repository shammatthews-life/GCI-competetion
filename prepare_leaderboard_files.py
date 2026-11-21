import shutil
import os
import pandas as pd
from scipy.stats import spearmanr

sample = pd.read_csv(r"d:\gci\inpuy\sample_submission.csv")
print(f"Sample submission: shape={sample.shape}, columns={list(sample.columns)}")

# 1. Pure Champion Master v22
src_v22 = r"d:\gci\gemini\v_22_pure_champion_ensemble.csv"
dst_v22 = r"d:\gci\LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"
shutil.copyfile(src_v22, dst_v22)

# 2. V10 Conservative Clean-Room (Average Ranks)
src_v10_rank = r"d:\gci\submission_audit_certified_v10_conservative_average_ranks.csv"
dst_v10 = r"d:\gci\LEADERBOARD_SUBMISSION_2_V10_CONSERVATIVE.csv"
shutil.copyfile(src_v10_rank, dst_v10)

# 3. V10 Conservative (Probabilities)
src_v10_prob = r"d:\gci\submission_audit_certified_v10_conservative_blended_probabilities.csv"
dst_v10_prob = r"d:\gci\LEADERBOARD_SUBMISSION_2B_V10_PROBABILITIES.csv"
shutil.copyfile(src_v10_prob, dst_v10_prob)

# Verify each destination file
for name, p in [
    ("Pure Champion Master (v22)", dst_v22),
    ("V10 Conservative (Average Ranks)", dst_v10),
    ("V10 Conservative (Probabilities)", dst_v10_prob)
]:
    df = pd.read_csv(p)
    assert len(df) == 61500, f"Error: row count is {len(df)}"
    assert list(df.columns) == ["SK_ID_CURR", "TARGET"], f"Error: columns are {list(df.columns)}"
    assert (df["SK_ID_CURR"].values == sample["SK_ID_CURR"].values).all(), "Error: ID mismatch"
    assert df["TARGET"].isna().sum() == 0, "Error: nulls found"
    print(f"[VERIFIED] {name}:")
    print(f"  Path: {p}")
    print(f"  Rows: {len(df)}, Nulls: 0, Match sample IDs: True")
    print(f"  Target range: [{df['TARGET'].min():.5f}, {df['TARGET'].max():.5f}]")
    print(f"  First 3 rows:\n{df.head(3)}\n")

