import json

def inspect_nb(path):
    print("=" * 80)
    print(f"Inspecting {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb.get('cells', []))}")
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    print(f"Total code cells: {len(code_cells)}")
    for i, c in enumerate(code_cells):
        src = "".join(c.get("source", []))
        first_line = src.strip().split("\n")[0] if src.strip() else "(empty)"
        print(f"  Code cell {i+1} ({len(src.splitlines())} lines): {first_line[:90]}")

inspect_nb(r"D:\gci\pre info\oct22\GCI_v17_optimised_O 78237 score.ipynb")
inspect_nb(r"D:\gci\truth\zip_extracted\zip\GCI_v18_evidence_driven.ipynb")

