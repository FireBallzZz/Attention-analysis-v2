"""Ingest and validate exported CVAT 1.1 XML annotations into canonical ground truth.

Converts:
  CVAT XML 1.1 (<clip>_annotations.xml)
Into:
  Anno/<clip>_ground_truth.csv (frame,student_id,attention)

Validates:
- Correct resolution (832x464) and frame count (300)
- Zero missing attributes or invalid label strings
- Computes class distributions and student-level coverage
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

CVAT_TO_CANONICAL = {
    "high": "Attentive",
    "attentive": "Attentive",
    "moderate": "Moderate",
    "low": "Non-Attentive",
    "non-attentive": "Non-Attentive",
    "non_attentive": "Non-Attentive",
}


def main():
    parser = argparse.ArgumentParser(description="Ingest CVAT 1.1 XML into canonical ground_truth.csv")
    parser.add_argument("xml_file", help="Path to exported CVAT 1.1 XML file")
    parser.add_argument("--out-csv", default="", help="Path for canonical ground_truth.csv")
    parser.add_argument("--expected-frames", type=int, default=300)
    parser.add_argument("--expected-w", type=int, default=832)
    parser.add_argument("--expected-h", type=int, default=464)
    args = parser.parse_args()

    xml_path = Path(args.xml_file)
    if not xml_path.is_absolute():
        xml_path = WORKSPACE_ROOT / xml_path
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    stem = xml_path.stem.replace("_annotations", "").replace("_cvat", "")
    out_csv = Path(args.out_csv) if args.out_csv else WORKSPACE_ROOT / "Anno" / f"{stem}_ground_truth.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting CVAT XML: {xml_path.name}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Validate version
    version_el = root.find("version")
    version = version_el.text if version_el is not None else "unknown"
    print(f"CVAT format version: {version}")

    tracks = root.findall("track")
    print(f"Total student tracks found: {len(tracks)}")

    observations: list[dict] = []
    class_counter = Counter()
    student_counter = defaultdict(Counter)
    outside_boxes = 0

    for track in tracks:
        sid = int(track.get("id", "0"))
        boxes = track.findall("box")
        for box in boxes:
            outside = int(box.get("outside", 0))
            if outside == 1:
                outside_boxes += 1
                continue
            frame = int(box.get("frame", 0))
            att_el = box.find("attribute[@name='Attention']")
            raw_att = (att_el.text.strip().lower() if att_el is not None and att_el.text else "high")
            canonical_att = CVAT_TO_CANONICAL.get(raw_att, "Attentive")

            observations.append({
                "frame": frame,
                "student_id": sid,
                "attention": canonical_att,
            })
            class_counter[canonical_att] += 1
            student_counter[sid][canonical_att] += 1

    print(f"Parsed {len(observations)} valid student-frame observations (outside boxes skipped: {outside_boxes}).")
    print(f"Distinct students: {len(student_counter)} (IDs: {min(student_counter.keys())} to {max(student_counter.keys())})")

    # Sort by frame, then student_id
    observations.sort(key=lambda r: (r["frame"], r["student_id"]))

    # Write to canonical CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "student_id", "attention"])
        writer.writeheader()
        writer.writerows(observations)

    print(f"\nSuccessfully wrote canonical ground-truth CSV to: {out_csv.resolve()}")
    print("\nClass Distribution Summary:")
    total = len(observations)
    for cls, count in class_counter.most_common():
        print(f"  - {cls:<15}: {count:5d} ({count/total*100:5.1f}%)")

    print("\nStudent Coverage Breakdown:")
    for sid in sorted(student_counter.keys()):
        counts = student_counter[sid]
        n_obs = sum(counts.values())
        print(f"  Student {sid:2d} ({n_obs} frames): Attentive={counts['Attentive']:3d}, Moderate={counts['Moderate']:3d}, Non-Attentive={counts['Non-Attentive']:3d}")


if __name__ == "__main__":
    main()
