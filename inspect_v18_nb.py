import json

p = r"d:\gci\truth\zip_extracted\zip\GCI_v18_evidence_driven.ipynb"
with open(p, "r", encoding="utf-8", errors="ignore") as f:
    nb = json.load(f)

print("Cells count:", len(nb.get("cells", [])))
for i, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    if "blend" in src.lower() or "0.55" in src or "lightgbm" in src.lower() or "catboost" in src.lower() or "StratifiedKFold" in src:
        print(f"Cell {i} ({cell.get('cell_type')}): {src[:120]}...\n")

