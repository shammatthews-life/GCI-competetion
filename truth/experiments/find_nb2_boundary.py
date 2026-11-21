with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print("Line 424-432:")
for i in range(424, 435):
    print(f"Line {i}: {lines[i].rstrip()[:80]}")

print("\nLine 835-842:")
for i in range(835, 843):
    print(f"Line {i}: {lines[i].rstrip()[:80]}")

