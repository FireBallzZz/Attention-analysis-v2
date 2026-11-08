import re
from pathlib import Path

path = Path(r"C:\Users\User\.gemini\antigravity-ide\brain\2fb50f86-a373-46ba-a40b-527d488a9f1f\.system_generated\steps\2166\content.md")
text = path.read_text(encoding="utf-8", errors="ignore")

# Find blocks containing both aria-label and data-id
blocks = re.findall(r'<div[^>]+aria-label="([^"]+)"[^>]*>.*?data-id="([^"]+)"', text, flags=re.DOTALL)
print(f"Matched blocks: {len(blocks)}")
for label, did in blocks:
    print(f"  {label:55s} -> ID: {did}")
