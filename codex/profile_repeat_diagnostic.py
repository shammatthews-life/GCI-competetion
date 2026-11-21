from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "inpuy" / "train.csv")
profiles = {
    "A_exact_work_family": ["CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "REGION_RATING_CLIENT"],
    "B_general_work_family": ["CODE_GENDER", "DAYS_BIRTH", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "REGION_RATING_CLIENT"],
    "C_occupation_family": ["CODE_GENDER", "DAYS_BIRTH", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT"],
    "D_exact_financial": ["CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "REGION_RATING_CLIENT", "AMT_INCOME_TOTAL"],
}
out = {}
for name, cols in profiles.items():
    group_sizes = df.groupby(cols, dropna=False, observed=True)["TARGET"].size()
    sizes = df.groupby(cols, dropna=False, observed=True)["TARGET"].transform("size")
    stats = {"rows": int(len(df)), "unique_profiles": int(len(group_sizes)),
             "repeated_profile_groups": int((group_sizes >= 2).sum()),
             "maximum_profile_size": int(group_sizes.max())}
    for threshold in (2, 3, 5, 10):
        stats[f"rows_in_groups_ge_{threshold}"] = int((sizes >= threshold).sum())
        stats[f"fraction_in_groups_ge_{threshold}"] = float((sizes >= threshold).mean())
    dup = sizes >= 2
    stats["dup_rows"] = int(dup.sum())
    stats["dup_target_mean"] = float(df.loc[dup, "TARGET"].mean()) if dup.any() else None
    stats["all_target_mean"] = float(df["TARGET"].mean())
    out[name] = stats
    print(name, json.dumps(stats))
path = ROOT / "codex" / "profile_repeat_diagnostic.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved", path)
