import json

p = r"d:\gci\truth\zip_extracted\zip\GCI_v18_evidence_driven.ipynb"
with open(p, "r", encoding="utf-8", errors="ignore") as f:
    nb = json.load(f)

for idx in [3, 6, 8, 10]:
    src = "".join(nb["cells"][idx]["source"])
    print(f"=== CELL {idx} ===")
    print(src[:2000])

