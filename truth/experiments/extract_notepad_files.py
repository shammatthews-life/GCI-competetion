import json
import re

with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

print("Length of text:", len(text))

# Let's inspect line by line and find JSON objects or code scripts
lines = text.splitlines()

embedded_files = []

# Block 1: Lines 1 to ~445 (Starts with "0.77283 -> {")
if lines[0].startswith("0.77283 -> {"):
    json_str = "{" + "\n".join(lines[1:440]) # wait, let's find the closing brace
    # Actually line 0 has "0.77283 -> {"
    full_first_block = "{" + "\n".join(lines[1:])
    # Let's find where first valid json ends
    # Or let's parse with a custom search
    print("Line 0 starts with 0.77283 -> {")

# Let's write a script to find all JSON notebooks and python scripts inside notepade.txt

