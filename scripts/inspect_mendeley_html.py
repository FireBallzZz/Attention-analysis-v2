import re
from pathlib import Path

path = Path(r"C:\Users\User\.gemini\antigravity-ide\brain\2fb50f86-a373-46ba-a40b-527d488a9f1f\.system_generated\steps\2155\content.md")
text = path.read_text(encoding="utf-8", errors="ignore")

# Find all hrefs
links = re.findall(r'href=["\']([^"\']+)["\']', text)
data_links = [l for l in links if any(k in l.lower() for k in ['download', 'file', 'csv', 'zip', 'mendeley'])]
print("Found data-related links in Mendeley page:")
for l in sorted(set(data_links)):
    print(" -", l)

# Also check for files data or json embedded in page
json_snippets = re.findall(r'(\{[^{}]*"downloadUrl"[^{}]*\})', text)
print(f"JSON snippets with downloadUrl: {len(json_snippets)}")
for s in json_snippets[:3]:
    print("  ", s)
