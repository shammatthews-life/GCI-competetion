import json

with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Let's check line 1 to 10
print("--- Lines 1-20 ---")
for l in lines[:20]:
    print(l.rstrip()[:120])

# Let's check lines around 440-470
print("\n--- Lines 440-465 ---")
for l in lines[439:465]:
    print(l.rstrip()[:120])

# Let's check lines 840-845
print("\n--- Lines 840-845 ---")
for l in lines[839:845]:
    print(l.rstrip()[:120])

# Let's check lines 935-939
print("\n--- Lines 935-939 ---")
for l in lines[934:939]:
    print(l.rstrip()[:120])

