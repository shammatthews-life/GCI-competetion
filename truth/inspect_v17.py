import json

path = "d:/gci/pre info/oct22/GCI_v17_optimised_O 78237 score.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']

print("=== CELL 7 ===")
print("".join(code_cells[7].get('source', [])))

print("\n=== CELL 2 HEAD ===")
src2 = "".join(code_cells[2].get('source', []))
print(src2[:3000])
