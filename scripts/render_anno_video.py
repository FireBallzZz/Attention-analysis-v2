"""Draw Anno/ground_truth.csv attention labels onto Anno/clip_004.mp4.

Boxes come from Anno/annotations.xml (CVAT). Labels come from the CSV
(frame, student_id, attention), matched by track id = student_id.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

# CSV labels -> display band + BGR color
LABEL_MAP = {
    "attentive": ("HIGH", (0, 200, 0)),
    "high": ("HIGH", (0, 200, 0)),
    "moderate": ("MODERATE", (0, 165, 255)),
    "non-attentive": ("LOW", (0, 0, 220)),
    "non_attentive": ("LOW", (0, 0, 220)),
    "low": ("LOW", (0, 0, 220)),
}


def load_csv_labels(path: Path) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            student_id = int(row["student_id"])
            labels[(frame, student_id)] = str(row["attention"]).strip()
    return labels


def load_boxes(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """frame -> list of (student_id, x1, y1, x2, y2)."""
    root = ET.parse(path).getroot()
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for track in root.findall("track"):
        student_id = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", "0"))
            by_frame[frame].append(
                (
                    student_id,
                    float(box.get("xtl", 0)),
                    float(box.get("ytl", 0)),
                    float(box.get("xbr", 0)),
                    float(box.get("ybr", 0)),
                )
            )
    return by_frame


def band_color(label: str) -> tuple[str, tuple[int, int, int]]:
    key = label.strip().lower().replace(" ", "-")
    return LABEL_MAP.get(key, ("MODERATE", (0, 165, 255)))


def draw_legend(frame) -> None:
    lines = [
        ("HIGH / Attentive", (0, 200, 0)),
        ("MODERATE", (0, 165, 255)),
        ("LOW / Non-Attentive", (0, 0, 220)),
    ]
    x, y = 8, 18
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        y += 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotated video from Anno CSV + XML")
    parser.add_argument("--video", default=str(ROOT / "Anno" / "clip_004.mp4"))
    parser.add_argument("--csv", default=str(ROOT / "Anno" / "ground_truth.csv"))
    parser.add_argument("--xml", default=str(ROOT / "Anno" / "annotations.xml"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "Anno" / "clip_004_gt_annotated.mp4"),
    )
    args = parser.parse_args()

    video = Path(args.video)
    csv_path = Path(args.csv)
    xml_path = Path(args.xml)
    out = Path(args.out)
    for path in (video, csv_path, xml_path):
        if not path.exists():
            raise SystemExit(f"Missing {path.resolve()}")

    labels = load_csv_labels(csv_path)
    boxes = load_boxes(xml_path)
    print(f"CSV labels: {len(labels)}")
    print(f"XML frames with boxes: {len(boxes)}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    drawn = 0
    missing_label = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for student_id, x1, y1, x2, y2 in boxes.get(frame_idx, []):
            label = labels.get((frame_idx, student_id))
            if label is None:
                missing_label += 1
                label = "Moderate"
            band, color = band_color(label)
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            tag = f"id{student_id}:{band}"
            ty = max(14, pt1[1] - 6)
            cv2.putText(
                frame,
                tag,
                (pt1[0], ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            drawn += 1
        draw_legend(frame)
        writer.write(frame)
        if frame_idx % 50 == 0 or frame_idx + 1 == total:
            print(f"  frame {frame_idx + 1}/{total}  boxes={len(boxes.get(frame_idx, []))}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Drew {drawn} boxes  missing CSV labels={missing_label}")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
