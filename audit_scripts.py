import re

scripts = [
    r"d:/gci/scratch_model_1_conservative_rules_first.py",
    r"d:/gci/scratch_model_2_tariff_bureau_package.py",
    r"d:/gci/scratch_model_3_v17_multiview.py",
    r"d:/gci/scratch_model_4_v22_champion_ensemble.py"
]

for s in scripts:
    print("=" * 60)
    print(f"Auditing {s}:")
    with open(s, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all file reads (pd.read_csv, np.load, open)
    reads = re.findall(r"(?:pd\.read_csv|np\.load|open)\((.*?)\)", content)
    for r in reads:
        print(f"  File Read: {r.strip()}")

