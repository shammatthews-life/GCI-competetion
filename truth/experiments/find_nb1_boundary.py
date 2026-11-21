with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print("Line 0:", lines[0][:80])
# Find where the first notebook JSON ends
for i in range(400, 480):
    if lines[i].strip() in ['}', '},', ']'] or 'cells' in lines[i] or 'nbformat' in lines[i]:
        print(f"Line {i}: {lines[i].rstrip()[:80]}")

