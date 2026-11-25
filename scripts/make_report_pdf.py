"""Render DEFENCE_REPORT.md to DEFENCE_REPORT.pdf with headless Microsoft Edge.

    python scripts/make_report_pdf.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]
CSS = """@page{size:A4;margin:16mm 15mm}
body{font:10.5pt/1.5 "Segoe UI",system-ui,sans-serif;color:#14171c}
h1{font-size:18pt;margin:0 0 6pt}
h2{font-size:13pt;margin:16pt 0 6pt;color:#3b5bdb;border-bottom:1px solid #e2e5ea;padding-bottom:3pt}
h3{font-size:11pt;margin:12pt 0 4pt}
h1,h2,h3,p:has(+table),p:has(+ul){break-after:avoid-page}
table{border-collapse:collapse;width:100%;margin:6pt 0 10pt;font-size:9.5pt}
tr{break-inside:avoid}
th,td{border-bottom:1px solid #e2e5ea;padding:4pt 6pt;text-align:left;vertical-align:top}
th{background:#f0f2f5}
code{font-family:Consolas,monospace;font-size:9pt;background:#f0f2f5;padding:0 3px;border-radius:3px}
pre{background:#f5f6f8;padding:8pt;border-radius:6px;font-size:8.5pt;white-space:pre-wrap}
pre code{background:none}em{color:#5d6672}li{margin:2pt 0}"""


def main() -> None:
    src = ROOT / "DEFENCE_REPORT.md"
    html = ROOT / "DEFENCE_REPORT.html"
    pdf = ROOT / "DEFENCE_REPORT.pdf"
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Corrected Technical Report</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    edge = next((e for e in EDGE if e.exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge not found; open DEFENCE_REPORT.html and print to PDF instead.")
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    html.unlink(missing_ok=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
