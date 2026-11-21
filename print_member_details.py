with open(r"d:\gci\v17_extracted.py", "r", encoding="utf-8") as f:
    text = f.read()

# find build_members
start = text.find("def build_members(f):")
print("=== build_members ===")
print(text[start:start+1200])

start2 = text.find("def run_member(")
print("=== run_member ===")
print(text[start2:start2+1400])

