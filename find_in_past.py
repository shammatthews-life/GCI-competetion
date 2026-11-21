with open(r"d:\gci\past.md", "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "0.765968" in line or "tariff/bureau" in line or "Seed `20261029`" in line or "Shares fold/cache lineage" in line:
            print(f"past.md:{i+1}: {line.strip()[:140]}")

