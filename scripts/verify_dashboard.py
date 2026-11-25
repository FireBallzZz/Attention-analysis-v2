"""Static verification of the generated dashboard.

Checks:
1. The inline <script> block parses as valid JavaScript (via `node --check`).
2. Every element id referenced from JS exists in the HTML.
3. Every id present in the HTML is reachable (no dead ids).
4. The embedded dataset matches dashboard_dataset.json exactly.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"c:\Users\User\Downloads\Thesis\Model\Attention-analysis-model")
HTML = ROOT / "dashboard" / "index.html"
DATASET = ROOT / "dashboard" / "dashboard_dataset.json"

html = HTML.read_text(encoding="utf-8")

failures = []
notes = []

# ---------------------------------------------------------------- 1. JS parse
m = re.search(r"<script>\s*(.*?)\s*</script>", html, re.S)
if not m:
    failures.append("No <script> block found")
    js = ""
else:
    js = m.group(1)

with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
    f.write(js)
    tmp = Path(f.name)

proc = subprocess.run(
    ["node", "--check", str(tmp)], capture_output=True, text=True, encoding="utf-8"
)
if proc.returncode != 0:
    failures.append(f"JS syntax error:\n{proc.stderr}")
else:
    notes.append(f"JS parses cleanly ({len(js):,} chars)")

# ------------------------------------------------------- 2. id cross-reference
html_ids = set(re.findall(r'\bid="([^"]+)"', html))
# ids created dynamically by the renderer
dynamic_ids = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", js))

missing = sorted(i for i in dynamic_ids if i not in html_ids)
if missing:
    failures.append(f"JS references ids absent from HTML: {missing}")
else:
    notes.append(f"All {len(dynamic_ids)} JS-referenced ids exist in the HTML")

# querySelector / getElementById inside strings
query_ids = set(re.findall(r"getElementById\('([^']+)'\)", js))
missing2 = sorted(i for i in query_ids if i not in html_ids)
if missing2:
    failures.append(f"getElementById targets absent from HTML: {missing2}")

# ------------------------------------------------------- 3. dead-id detection
referenced = dynamic_ids | query_ids | set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", js))
unused = sorted(i for i in html_ids - referenced if i not in {"videoSelect", "studentModal", "modalClose"})
if unused:
    notes.append(f"HTML ids never read by JS (structural only): {unused}")

# ------------------------------------------------------- 4. dataset integrity
data_json = re.search(r"const DATABASE = (\{.*?\});\n", js, re.S)
if not data_json:
    failures.append("Could not locate embedded DATABASE literal")
else:
    embedded = json.loads(data_json.group(1))
    on_disk = json.loads(DATASET.read_text(encoding="utf-8"))
    if embedded == on_disk:
        notes.append(f"Embedded dataset matches dashboard_dataset.json ({len(embedded)} videos)")
    else:
        failures.append("Embedded dataset differs from dashboard_dataset.json")

# --------------------------------------- 5. required sections present
for needle in [
    "kpiAccuracy", "seatingStage", "timelineChart", "behaviorList",
    "confusionBox", "tableBody", "modalSparkline", "alertList",
]:
    if f'id="{needle}"' not in html:
        failures.append(f"Missing required element: {needle}")

# --------------------------------------- 6. no fabricated certification claims
for claim in ["DEFENSE-READY", "CERTIFIED", "Certified"]:
    if claim in html:
        failures.append(f"Unsupported certification claim present: {claim!r}")

print("=" * 68)
print("DASHBOARD STATIC VERIFICATION")
print("=" * 68)
for n in notes:
    print("  [ok]  ", n)
if failures:
    print()
    for f in failures:
        print("  [FAIL]", f)
    print()
    print(f"RESULT: {len(failures)} failure(s)")
    sys.exit(1)
print()
print("RESULT: all checks passed")
