import re

with open(r'D:\gci\truth\process.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

print(f"Total length: {len(text)} characters, {len(text.splitlines())} lines")

# Find occurrences of scores
scores = re.findall(r'(0\.[78]\d{2,5})', text)
print(f"Total score mentions: {len(scores)}")
from collections import Counter
print("Most frequent scores:", Counter(scores).most_common(20))

# Find occurrences of prompt/section headers
sections = re.findall(r'(###.*|##.*|#.*|can u.*|what.*|how.*)', text, re.IGNORECASE)
print(f"Total headers/questions: {len(sections)}")
print("\nSample headers:")
for s in sections[:30]:
    if len(s.strip()) > 5:
        print(" ", s.strip()[:100])

