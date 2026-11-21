with open(r"d:\gci\reports\NEW_MATERIALS_FORENSIC_AUDIT.md", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "0.765968" in line or "Seed 20261029" in line or "tariff/bureau" in line or "Shares fold/cache lineage" in line:
        print(f"Line {i+1}: {line.strip()[:140]}")

