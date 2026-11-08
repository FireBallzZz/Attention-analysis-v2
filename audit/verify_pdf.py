"""Verify generated PDF with pypdf."""

from pathlib import Path
from pypdf import PdfReader

WORKSPACE_ROOT = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model")
pdf_path = WORKSPACE_ROOT / "runs" / "reports" / "Student_Attention_Model_Technical_Audit_and_Corrected_Report.pdf"

reader = PdfReader(str(pdf_path))
print(f"Total Pages: {len(reader.pages)}")

for idx, page in enumerate(reader.pages, 1):
    text = page.extract_text()
    first_line = text.splitlines()[0] if text.splitlines() else "<EMPTY>"
    last_line = text.splitlines()[-1] if text.splitlines() else "<EMPTY>"
    print(f"Page {idx}: Length={len(text)} chars | First: {first_line[:60]}... | Last: {last_line[-50:]}")

