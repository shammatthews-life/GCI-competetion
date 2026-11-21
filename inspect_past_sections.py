with open(r"D:\gci\truth\past.md", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
sections = re.findall(r"##\s+([^\n]+)", text)
print("Sections in past.md (first 30):")
for s in sections[:30]:
    print(" -", s)

