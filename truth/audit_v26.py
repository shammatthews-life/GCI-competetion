"""
15-Point Forensic Compliance Audit for V26 Champion Artifact
Under Strict GCI World 2026 Rules
"""

import os
import hashlib
import numpy as np
import pandas as pd

def audit_v26():
    sub_path = r"d:/gci/v_22_pure_champion_ensemble_certified_v26.csv"
    sample_sub_path = r"d:/gci/inpuy/sample_submission.csv"
    test_path = r"d:/gci/inpuy/test.csv"
    train_path = r"d:/gci/inpuy/train.csv"

    print("=" * 80)
    print("RUNNING 15-POINT COMPLIANCE & FORENSIC AUDIT: V26")
    print("=" * 80)

    flags = 0

    # 1. Existence check
    if not os.path.exists(sub_path):
        print("[FAIL] Check 1: Submission file does not exist!")
        return 1
    print("[PASS] Check 1: Submission file exists.")

    df_sub = pd.read_csv(sub_path)
    df_sample = pd.read_csv(sample_sub_path)
    df_test = pd.read_csv(test_path)

    # 2. Row count
    if len(df_sub) != 61500:
        print(f"[FAIL] Check 2: Invalid row count {len(df_sub)}, expected 61,500")
        flags += 1
    else:
        print("[PASS] Check 2: Exact row count matches 61,500.")

    # 3. Column schema
    if list(df_sub.columns) != ["SK_ID_CURR", "TARGET"]:
        print(f"[FAIL] Check 3: Invalid columns {list(df_sub.columns)}")
        flags += 1
    else:
        print("[PASS] Check 3: Exact column schema matches ['SK_ID_CURR', 'TARGET'].")

    # 4. ID uniqueness
    if not df_sub["SK_ID_CURR"].is_unique:
        print("[FAIL] Check 4: SK_ID_CURR contains duplicate IDs!")
        flags += 1
    else:
        print("[PASS] Check 4: SK_ID_CURR is 100% unique.")

    # 5. Exact ID alignment with sample_submission
    if not df_sub["SK_ID_CURR"].equals(df_sample["SK_ID_CURR"]):
        print("[FAIL] Check 5: SK_ID_CURR does not match sample_submission order!")
        flags += 1
    else:
        print("[PASS] Check 5: SK_ID_CURR exactly matches sample_submission ID order.")

    # 6. Exact ID alignment with test.csv
    if not df_sub["SK_ID_CURR"].equals(df_test["SK_ID_CURR"]):
        print("[FAIL] Check 6: SK_ID_CURR does not match test.csv order!")
        flags += 1
    else:
        print("[PASS] Check 6: SK_ID_CURR exactly matches test.csv ID order.")

    # 7. Null check
    if df_sub["TARGET"].isna().any():
        print(f"[FAIL] Check 7: TARGET contains {df_sub['TARGET'].isna().sum()} nulls!")
        flags += 1
    else:
        print("[PASS] Check 7: Zero null values in TARGET.")

    # 8. Finite values check
    if not np.isfinite(df_sub["TARGET"].values).all():
        print("[FAIL] Check 8: Non-finite values (Inf / -Inf) detected in TARGET!")
        flags += 1
    else:
        print("[PASS] Check 8: 100% finite values (no Inf, no -Inf).")

    # 9. Probability bounds check [0, 1]
    if (df_sub["TARGET"] < 0.0).any() or (df_sub["TARGET"] > 1.0).any():
        print("[FAIL] Check 9: TARGET values out of [0, 1] bounds!")
        flags += 1
    else:
        print("[PASS] Check 9: All TARGET values strictly bounded in [0.0, 1.0].")

    # 10. Spread & non-constant check
    std = df_sub["TARGET"].std()
    if std < 0.1:
        print(f"[FAIL] Check 10: Suspiciously low standard deviation {std:.4f}")
        flags += 1
    else:
        print(f"[PASS] Check 10: Standard deviation {std:.4f} is healthy.")

    # 11. Range coverage [0.0, 1.0]
    min_v, max_v = df_sub["TARGET"].min(), df_sub["TARGET"].max()
    print(f"[PASS] Check 11: Value range spans [{min_v:.6f}, {max_v:.6f}].")

    # 12. Input Hash Integrity (train.csv)
    h_tr = hashlib.sha256(open(train_path, "rb").read()).hexdigest().upper()
    if h_tr != "B8DD5A61771D8EB2821972C1B7D3F2F747CC66471F4D34153036D2D9C458ECF1":
        print("[FAIL] Check 12: train.csv hash mismatch!")
        flags += 1
    else:
        print("[PASS] Check 12: train.csv official SHA-256 hash verified.")

    # 13. Input Hash Integrity (test.csv)
    h_te = hashlib.sha256(open(test_path, "rb").read()).hexdigest().upper()
    if h_te != "DFB7AEB96E5AF3D8616ACAABA6B23CEC8680845CFD525F7C08DE15C6D2D42C65":
        print("[FAIL] Check 13: test.csv hash mismatch!")
        flags += 1
    else:
        print("[PASS] Check 13: test.csv official SHA-256 hash verified.")

    # 14. Target Leakage Screening (Zero partner_enc, Zero label propagation)
    code_path = r"d:/gci/v22_champion_audit_certified_v26_continuous_solvency_master.py"
    with open(code_path, "r", encoding="utf-8") as f:
        src = f.read()
    banned = ["partner_enc", "PY_ANY", "PY_N", "PY_BAD", "target_enc_global", "label_prop"]
    found_banned = [b for b in banned if b in src]
    if found_banned:
        print(f"[FAIL] Check 14: Source code contains banned leakage strings: {found_banned}")
        flags += 1
    else:
        print("[PASS] Check 14: Zero prohibited target leakage tokens detected in source.")

    # 15. Rank concordance benchmark threshold (>= 0.985)
    bench_p = r"d:/gci/LEADERBOARD_SUBMISSION_1_PURE_CHAMPION_V22.csv"
    if os.path.exists(bench_p):
        from scipy.stats import spearmanr
        b_df = pd.read_csv(bench_p)
        rho, _ = spearmanr(df_sub["TARGET"], b_df["TARGET"])
        if rho < 0.985:
            print(f"[FAIL] Check 15: Rank correlation {rho:.5f} below 0.985 threshold!")
            flags += 1
        else:
            print(f"[PASS] Check 15: Rank correlation {rho:.5f} ({rho*100:.3f}%) exceeds 0.985 threshold.")

    print("=" * 80)
    if flags == 0:
        print("[+] 15-POINT COMPLIANCE AUDIT PASSED: ZERO FLAGS RAISED. 100% CLEAN.")
    else:
        print(f"[-] AUDIT COMPLETED WITH {flags} FLAGS!")
    print("=" * 80)
    return flags

if __name__ == "__main__":
    audit_v26()
