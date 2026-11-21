import re

def extract_features(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find feature list or dataframe assignments
    cols_assigned = set(re.findall(r"df\[['\"]([A-Za-z0-9_]+)['\"]\]\s*=", content))
    return cols_assigned

f43 = extract_features('d:/gci/v22_champion_audit_certified_v43_grand_apex_soVEREIGN_master.py')
f48 = extract_features('d:/gci/v22_champion_audit_certified_v48_grand_apex_soVEREIGN_master.py')

print(f"Features assigned in V43: {len(f43)}")
print(f"Features assigned in V48: {len(f48)}")
in_48_not_43 = f48 - f43
in_43_not_48 = f43 - f48
print(f"Features in V48 but not V43: {len(in_48_not_43)}")
print(f"Sample of new in V48: {list(in_48_not_43)[:20]}")
print(f"Features in V43 but not V48: {len(in_43_not_48)}")
print(f"Sample of missing in V48: {list(in_43_not_48)}")
