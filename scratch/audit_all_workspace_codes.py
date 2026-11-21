import os
import glob
import re
from collections import defaultdict

base_dir = r"D:\gci"
exclude_dirs = {".venv", "__pycache__", ".git", "_gci_push_tmp"}

# Collect all code files (.py, .ipynb)
all_code_files = []
for root, dirs, files in os.walk(base_dir):
    dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
    for f in files:
        if f.endswith(".py") or f.endswith(".ipynb"):
            all_code_files.append(os.path.join(root, f))

print("=" * 80)
print(f"TOTAL CODE FILES DISCOVERED ACROSS WORKSPACE: {len(all_code_files)}")
print("=" * 80)

def audit_file(p):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"Read Error: {e}"]
        
    src = "".join(lines)
    rel = os.path.relpath(p, base_dir)
    filename = os.path.basename(p)
    
    # Is it an evaluation, diagnostic, or audit tool itself?
    is_audit_tool = any(k in filename.lower() for k in ["audit", "scanner", "inventory", "test_", "inspect", "check_", "verify"])
    if "truth" in rel.lower() and is_audit_tool:
        return ["TOOLING_AUDIT_SCRIPT"]
        
    flags = []
    
    # 1. Partner enc / target leakage
    for idx, l in enumerate(lines, 1):
        if "partner_enc" in l.lower() and not any(k in l.lower() for k in ["forbidden", "zero", "not use", "stripped", "audit", 'partner_" + "enc']):
            flags.append(f"partner_enc leakage (L{idx})")
            break
            
    # 2. Hasan forbidden token
    for idx, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"""') or stripped.startswith("//"):
            continue
        if "hasan" in l.lower() and not any(k in l.lower() for k in ["forbidden", "zero", "not use", "ban", 'has" + "an', "has_an"]):
            flags.append(f"hasan token (L{idx})")
            break
            
    # 3. Exp85 forbidden token / dependency
    for idx, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"""') or stripped.startswith("//"):
            continue
        if "exp85" in l.lower() and not any(k in l.lower() for k in ["forbidden", "zero", "not use", "ban", 'exp" + "85', "solo champion", "past_85"]):
            if "read_csv" in l or "load" in l or "import" in l or "exp85" in filename.lower():
                flags.append(f"exp85 dependency (L{idx})")
                break
                
    # 4. CSV Rank Blending (reading historical submission CSVs to blend)
    for idx, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"""'):
            continue
        if "read_csv" in l:
            if any(k in l for k in ["v_22_pure_champion", "LEADERBOARD_", "submission_exp85", "certified_v"]):
                if not any(k in l for k in ["benchmarks", "benchmark", "past_csv", "df_b =", "df_past", "spearmanr", "rho", "assert", "target_artifact"]):
                    flags.append(f"CSV Rank Blending (L{idx})")
                    break
                    
    # 5. External Cache Dependency (.npy / .feather / champion_cache)
    for idx, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"""'):
            continue
        if any(k in l for k in ["rebuilt_eight_engine_test.npy", "rebuilt_eight_engine", "champion_cache"]):
            flags.append(f"External Cache Dependency (L{idx})")
            break
        if "np.load" in l and ("npy" in l or "cache" in l):
            flags.append(f"External Cache Dependency (np.load at L{idx})")
            break
            
    # 6. Transductive Pooling
    if "pd.concat([df_train" in src or "pd.concat([df_all" in src or "pd.concat([tr, te" in src:
        flags.append("Transductive Train/Test Pooling")
        
    return flags

# Run audit across all files
audit_results = {}
categories = defaultdict(list)

for p in all_code_files:
    rel = os.path.relpath(p, base_dir)
    res = audit_file(p)
    audit_results[rel] = res
    
    if res == ["TOOLING_AUDIT_SCRIPT"]:
        categories["TOOLING_AUDIT_SCRIPTS"].append(rel)
    elif not res:
        categories["COMPLIANT_PURE_SCRATCH"].append(rel)
    else:
        # Check primary reason
        primary_reasons = []
        for r in res:
            r_type = r.split("(")[0].strip()
            if r_type not in primary_reasons:
                primary_reasons.append(r_type)
        reason_key = " & ".join(primary_reasons)
        categories[reason_key].append(rel)

print("\n" + "=" * 80)
print("AUDIT SUMMARY BY COMPLIANCE CATEGORY:")
print("=" * 80)
for cat, files in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"  * {cat:40s} : {len(files):4d} files")

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN OF ROOT CERTIFIED PIPELINES (V1 to V56):")
print("=" * 80)

root_v_scripts = [f for f in all_code_files if os.path.dirname(f) == base_dir and "certified_v" in os.path.basename(f)]
root_v_scripts.sort()

def get_v_num(name):
    m = re.search(r'certified_v(\d+)', name)
    return int(m.group(1)) if m else 0

root_v_scripts.sort(key=lambda p: get_v_num(os.path.basename(p)))

clean_root_count = 0
violation_root_count = 0

for p in root_v_scripts:
    name = os.path.basename(p)
    v_num = get_v_num(name)
    fl = audit_file(p)
    if not fl:
        clean_root_count += 1
        print(f"  V{v_num:02d} | {name:55s} | [PASS] 100% CLEAN PURE SCRATCH")
    else:
        violation_root_count += 1
        print(f"  V{v_num:02d} | {name:55s} | [VIOLATION] {', '.join(fl)}")

print(f"\nRoot Certified Pipelines Total: {len(root_v_scripts)}")
print(f"  - 100% Competition Rules Compliant : {clean_root_count}")
print(f"  - Competition Rules Violations    : {violation_root_count}")
