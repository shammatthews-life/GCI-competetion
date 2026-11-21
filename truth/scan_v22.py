import re
from pathlib import Path

code = Path("D:/gci/generate_v22_pure_champion_complete_scratch.py").read_text(encoding="utf-8")
checks = [
    ("External CSV Read", r"(GCI_submission|pre info|\.npz|\.pkl|\.npy)"),
    ("Target Propagation", r"(partner_enc|link_key|PY_ANY|PY_NEAR180)"),
    ("Person-Link Pooling", r"\bPL_S\b"),
    ("Hard-Coded Test Subset Weights", r"(w17\[|ext_nobs_te|is_pensioner_te|is_revolving_te)"),
    ("Train+Test Concatenation Pooling", r"pd\.concat\(\[df_train|pd\.concat\(\[tr,\s*te\]"),
    ("Early-Stopping Peeking", r"early_stopping_rounds"),
]
clean = True
print("=== AUDIT SCAN: generate_v22_pure_champion_complete_scratch.py ===")
for label, pat in checks:
    hits = [ln for ln in code.splitlines() if re.search(pat, ln) and not ln.strip().startswith("#") and not ln.strip().startswith('"""')]
    if hits:
        clean = False
        print(f"  [FLAG] {label}: {len(hits)} hits")
        for h in hits:
            print(f"    -> {h.strip()[:100]}")
    else:
        print(f"  [PASS] {label}: 0 hits")

if clean:
    print("\n>>> ZERO AUDIT FLAGS: 100% CLEAN-ROOM COMPLIANT <<<")

