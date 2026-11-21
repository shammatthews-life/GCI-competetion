import json

def dump_nb_to_py(nb_path, py_path):
    with open(nb_path, "r", encoding="utf-8", errors="ignore") as f:
        nb = json.load(f)
    with open(py_path, "w", encoding="utf-8") as out:
        out.write(f'""" Extracted from {nb_path} """\n\n')
        for i, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") == "code":
                out.write(f"\n# {'='*60}\n# CODE CELL {i}\n# {'='*60}\n")
                out.write("".join(c.get("source", [])))
                out.write("\n")

dump_nb_to_py(r"D:\gci\pre info\oct22\GCI_v17_optimised_O 78237 score.ipynb", r"d:\gci\v17_extracted.py")
dump_nb_to_py(r"D:\gci\truth\zip_extracted\zip\GCI_v18_evidence_driven.ipynb", r"d:\gci\v18_extracted.py")
print("Extracted v17 and v18 code successfully.")

