import json

with open(r"d:\gci\GCI_v22_pure_champion_master.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    print(f"Cell {i} ({c['cell_type']}): exec_count={c.get('execution_count')}, outputs_len={len(c.get('outputs', []))}")

