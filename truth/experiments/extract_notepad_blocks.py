import json
import os

with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines: {len(lines)}")

# Let's see what lines start with "{" or have "{"cells""
json_starts = []
for i, l in enumerate(lines):
    if '{"cells"' in l or l.strip().startswith('{"cells"') or l.strip().startswith('0.77283 -> {'):
        json_starts.append(i)

print("JSON starts at lines:", json_starts)

# Let's inspect each line in json_starts
for idx in json_starts:
    print(f"\n--- Line {idx} (length {len(lines[idx])}) ---")
    line_preview = lines[idx][:150]
    print(line_preview)
    # Check if this line itself is valid JSON
    target_str = lines[idx]
    if target_str.startswith('0.77283 -> {'):
        # Multi-line json
        pass
    else:
        try:
            nb_obj = json.loads(target_str)
            cells = nb_obj.get('cells', [])
            first_src = ''.join(cells[0].get('source', [])) if cells else ''
            print(f"  Valid single-line Notebook JSON! Total cells: {len(cells)}")
            print(f"  First cell source: {first_src[:100]}")
        except Exception as e:
            print(f"  Not single-line JSON: {e}")

