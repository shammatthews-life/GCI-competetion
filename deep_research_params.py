import json

def extract_cells(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            code.append("".join(c["source"]))
    return "\n".join(code)

c17 = extract_cells(r"d:\gci\pre info\oct22\GCI_v17_optimised_O 78237 score.ipynb")
c18 = extract_cells(r"d:\gci\pre info\oct22\GCI_v18_evidence_driven_o 78235 score.ipynb")

print("V17 code length:", len(c17.splitlines()))
print("V18 code length:", len(c18.splitlines()))

# Search for hyperparameters in v17
for line in c17.splitlines():
    if "SPEC1 =" in line or "SPEC0 =" in line or "champ =" in line or "CBP =" in line or "W =" in line:
        print("V17:", line[:120])

print("---")
for line in c18.splitlines():
    if "SPEC1 =" in line or "SPEC0 =" in line or "champ =" in line or "CBP =" in line or "W =" in line:
        print("V18:", line[:120])

