"""
================================================================================
GCI All-Time Record Champion: Submission v22 (Public LB: 0.78239)
================================================================================
This script reproduces `v_22_pure_champion_ensemble.csv` (LB 0.78239)
via convex rank fusion of the two 20-member multi-view stacking champions:
    v22 = 0.55 * rank(v17) + 0.45 * rank(v18)

Where:
  - v17 (LB 0.78237): 20-member multi-view pipeline (5 streams x 4-tier folds)
  - v18 (LB 0.78235): 20-member evidence-driven multi-view pipeline
================================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def generate_v22():
    path_v17 = r"D:/gci/pre info/oct22/GCI_submission_v17.csv"
    path_v18 = r"D:/gci/pre info/oct22/GCI_submission_v18.csv"
    output_path = r"D:/gci/v_22_pure_champion_ensemble_reproduced.csv"

    if not os.path.exists(path_v17) or not os.path.exists(path_v18):
        raise FileNotFoundError("Missing v17 or v18 base submission files!")

    print(f"Loading v17 from: {path_v17}")
    sub17 = pd.read_csv(path_v17)

    print(f"Loading v18 from: {path_v18}")
    sub18 = pd.read_csv(path_v18)

    # Min-max normalized rank transform: [0.0, 1.0]
    rk = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)
    
    r17 = rk(sub17['TARGET'].values)
    r18 = rk(sub18['TARGET'].values)

    # Convex rank blend
    v22_rank = rk(0.55 * r17 + 0.45 * r18)

    submission = pd.DataFrame({
        'SK_ID_CURR': sub17['SK_ID_CURR'],
        'TARGET': v22_rank
    })

    submission.to_csv(output_path, index=False)
    print(f"Successfully generated: {output_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Target sample:\n{submission.head()}")

if __name__ == '__main__':
    generate_v22()

