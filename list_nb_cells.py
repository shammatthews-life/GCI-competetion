import json

with open(r"D:\gci\pre info\oct22\GCI_v17_optimised_O 78237 score.ipynb", "r", encoding="utf-8") as f:
    nb17 = json.load(f)

with open(r"D:\gci\truth\zip_extracted\zip\GCI_v18_evidence_driven.ipynb", "r", encoding="utf-8") as f:
    nb18 = json.load(f)

print("nb17 cells:", len(nb17["cells"]))
for i, c in enumerate(nb17["cells"]):
    print(f"nb17 cell {i} ({c['cell_type']}):", "".join(c.get("source", []))[:80].replace("\n", " "))

print("\nnb18 cells:", len(nb18["cells"]))
for i, c in enumerate(nb18["cells"]):
    print(f"nb18 cell {i} ({c['cell_type']}):", "".join(c.get("source", []))[:80].replace("\n", " "))

