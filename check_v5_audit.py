import re
from pathlib import Path

CHECKS = [
    ("External/Historical CSV or Cache Read", r"(GCI_submission|pre info|\.npz|\.pkl|\.npy)"),
    ("Target Propagation (partner_enc/link_key)", r"(partner_enc|link_key|PY_ANY|PY_NEAR180)"),
    ("Legacy Covariate Person-Link Pooling", r"\bPL_S\b"),
    ("Hard-Coded Test Subset Weights", r"(w17\[|ext_nobs_te|is_pensioner_te|is_revolving_te)"),
    ("Train+Test Concatenation Pooling", r"pd\.concat\(\[df_train|pd\.concat\(\[tr,\s*te\]"),
    ("Scrambled View Concatenation Bug", r"np\.concatenate\(pv\)"),
    ("Validation Fold Early-Stopping Peeking", r"early_stopping_rounds"),
    ("Incomplete Cache Key Collision Risk", r"\(len\(fit\),\s*len\(qi\)|\(\'tr\',\s*len\(a\)"),
]

for fname in ["v22_champion_audit_certified_v5_high_loc.py", "v22_champion_audit_certified_v6_ultra_loc.py"]:
    p = Path(f"d:/gci/{fname}")
    code_lines = []
    in_doc = False
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#") or s.startswith("print("):
            continue
        code_lines.append((f"L{i}", line))

    clean = True
    for label, pat in CHECKS:
        hits = [(loc, ln.strip()[:100]) for loc, ln in code_lines if re.search(pat, ln)]
        if hits:
            clean = False
            print(f"  [{fname}] [FLAG] {label}: {len(hits)} hit(s) -> first at {hits[0][0]}: {hits[0][1]}")

    if clean:
        print(f"  [PASS] ZERO rule or audit flags on {fname}!")

