import difflib

with open(r"d:\gci\v17_extracted.py", "r", encoding="utf-8") as f:
    v17_lines = f.readlines()

with open(r"d:\gci\v18_extracted.py", "r", encoding="utf-8") as f:
    v18_lines = f.readlines()

diff = list(difflib.unified_diff(v17_lines, v18_lines, fromfile="v17", tofile="v18", n=2))
print("Diff lines count:", len(diff))
for line in diff[:120]:
    print(line, end="")

