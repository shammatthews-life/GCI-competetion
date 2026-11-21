import os
import pandas as pd

submissions = [
    ("Model 1 (Conservative Rules-First)", r"d:/gci/submission_1_conservative_rules_first.csv"),
    ("Model 2 (Tariff & Bureau Package)", r"d:/gci/submission_2_tariff_bureau_package.csv"),
    ("Model 3 (V17 Multi-View Suite)", r"d:/gci/submission_3_v17_multiview.csv"),
    ("Model 4 (V22 Champion Ensemble 0.78239)", r"d:/gci/submission_4_v22_champion_078239.csv")
]

sample = pd.read_csv(r"d:/gci/inpuy/sample_submission.csv")

for label, p in submissions:
    print("=" * 60)
    print(f"{label}:")
    if os.path.exists(p):
        df = pd.read_csv(p)
        size_kb = os.path.getsize(p) / 1024.0
        print(f"  Path: {p}")
        print(f"  File size: {size_kb:.1f} KB")
        print(f"  Shape: {df.shape}")
        print(f"  Null count: {df.isna().sum().to_dict()}")
        id_match = (df['SK_ID_CURR'].values == sample['SK_ID_CURR'].values).all()
        print(f"  SK_ID_CURR matches sample_submission.csv exactly: {id_match}")
        print(f"  Target range: [{df['TARGET'].min():.5f}, {df['TARGET'].max():.5f}]")
    else:
        print("  FILE NOT FOUND!")

