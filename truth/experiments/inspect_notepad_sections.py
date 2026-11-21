import re

with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines in notepade.txt: {len(lines)}")

# Scan for distinct script boundaries
sections = []
current_start = 0

for i, line in enumerate(lines):
    # Detect distinct code starts
    if any(sig in line for sig in [
        "# cell 1 - setup",
        "import pytabkit",
        "GCI World 2026 - q-fingerprint",
        "# GCI World 2026 - FINAL reproduction notebook",
        "#!/usr/bin/env python3",
        "class ",
        "def main(",
        "# cell 1",
    ]):
        sections.append((i+1, line[:100]))

print(f"Found {len(sections)} distinct section/header markers:")
for s in sections:
    print(f"Line {s[0]}: {s[1]}")

