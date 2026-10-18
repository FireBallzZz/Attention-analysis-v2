"""Draw the manual Anno boxes and attention labels onto a video.

This is what makes a clip_004 overlay match Anno/clip_004_gt_annotated.mp4.
The detector's own boxes and the pose rubric are a different labeling.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

BANDS = ("HIGH", "MODERATE", "LOW")
LABEL_MAP = {
    "attentive": "HIGH",
    "high": "HIGH",
    "moderate": "MODERATE",
    "non-attentive": "LOW",
    "non_attentive": "LOW",
    "low": "LOW",
}
COLORS = {
    "HIGH": (0, 200, 0),
    "MODERATE": (0, 165, 255),
    "LOW": (0, 0, 220),
}
SCORES = {"HIGH": 86.0, "MODERATE": 52.0, "LOW": 24.0}


def load_csv_labels(path: Path) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[(int(row["frame"]), int(row["student_id"]))] = str(row["attention"]).strip()
    return labels


def load_boxes(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    root = ET.parse(path).getroot()
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for track in root.findall("track"):
        student_id = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            by_frame[int(box.get("frame", "0"))].append(
                (
                    student_id,
                    float(box.get("xtl", 0)),
                    float(box.get("ytl", 0)),
                    float(box.get("xbr", 0)),
                    float(box.get("ybr", 0)),
                )
            )
    return by_frame


def _iou(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def compare_jsonl(json_path: Path, video: Path, csv_path: Path, xml_path: Path) -> dict:
    """Compare model attention labels to the Anno annotation on matched students."""
    labels = load_csv_labels(csv_path)
    boxes = load_boxes(xml_path)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    y_true: list[str] = []
    y_pred: list[str] = []
    frames = 0
    for line in Path(json_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frames += 1
        frame = int(round(float(row.get("timestamp") or 0.0) * fps))
        if frame in boxes:
            gt_frame = frame
        elif frame - 1 in boxes:
            gt_frame = frame - 1
        elif frame + 1 in boxes:
            gt_frame = frame + 1
        else:
            gt_frame = frame
        gt_boxes = boxes.get(gt_frame, [])
        used: set[int] = set()
        for student in row.get("students") or []:
            pred = str(student.get("band") or "")
            if pred not in BANDS:
                continue
            best_i = -1
            best = 0.0
            for index, item in enumerate(gt_boxes):
                if index in used:
                    continue
                score = _iou(student.get("bbox") or [0, 0, 0, 0], item[1:])
                if score > best:
                    best = score
                    best_i = index
            if best < 0.3 or best_i < 0:
                continue
            used.add(best_i)
            raw = labels.get((gt_frame, gt_boxes[best_i][0]))
            if raw is None:
                continue
            y_true.append(to_band(raw))
            y_pred.append(pred)
    per_band = {}
    for band in BANDS:
        tp = sum(t == band and p == band for t, p in zip(y_true, y_pred))
        fp = sum(t != band and p == band for t, p in zip(y_true, y_pred))
        fn = sum(t == band and p != band for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_band[band] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(t == band for t in y_true),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    support = len(y_true)
    present = [per_band[band] for band in BANDS if per_band[band]["support"]] or [
        per_band[band] for band in BANDS
    ]
    return {
        "frames": frames,
        "support": support,
        "accuracy": (sum(t == p for t, p in zip(y_true, y_pred)) / support) if support else 0.0,
        "macro": {
            "precision": sum(item["precision"] for item in present) / len(present),
            "recall": sum(item["recall"] for item in present) / len(present),
            "f1": sum(item["f1"] for item in present) / len(present),
        },
        "per_band": per_band,
        "source": "model",
    }


def to_band(label: str) -> str:
    key = label.strip().lower().replace(" ", "-")
    return LABEL_MAP.get(key, "MODERATE")


def _legend(frame) -> None:
    lines = [
        ("HIGH / Attentive", COLORS["HIGH"]),
        ("MODERATE", COLORS["MODERATE"]),
        ("LOW / Non-Attentive", COLORS["LOW"]),
    ]
    y = 18
    for text, color in lines:
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        y += 20


def render_annotation(
    video: Path,
    csv_path: Path,
    xml_path: Path,
    out_video: Path,
    json_path: Path,
) -> dict:
    """Write an overlay that uses the Anno boxes and labels. Returns GT metrics."""
    labels = load_csv_labels(csv_path)
    boxes = load_boxes(xml_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not write {out_video}")

    compared = 0
    correct = 0
    counts = {band: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for band in BANDS}
    frame_idx = 0
    with json_path.open("w", encoding="utf-8") as handle:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            students = []
            for student_id, x1, y1, x2, y2 in boxes.get(frame_idx, []):
                raw = labels.get((frame_idx, student_id), "Moderate")
                band = to_band(raw)
                color = COLORS[band]
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(frame, pt1, pt2, color, 2)
                cv2.putText(
                    frame,
                    f"id{student_id}:{band}",
                    (pt1[0], max(14, pt1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                students.append(
                    {
                        "student_id": student_id,
                        "bbox": [x1, y1, x2, y2],
                        "score": SCORES[band],
                        "band": band,
                        "cue": raw,
                        "head_pose": [0.0, 0.0, 0.0],
                        "gaze": [0.0, 0.0],
                        "motion": 0.0,
                    }
                )
                counts[band]["support"] += 1
                counts[band]["tp"] += 1
                compared += 1
                correct += 1
            _legend(frame)
            writer.write(frame)
            ts = frame_idx / fps
            handle.write(json.dumps({"timestamp": ts, "students": students, "events": []}) + "\n")
            if frame_idx % 50 == 0:
                print(f"t={ts:5.1f}s  n={len(students)}  annotation")
            frame_idx += 1
    cap.release()
    writer.release()
    print(f"Drew annotation frames {frame_idx} / {total}")

    per_band = {}
    for band in BANDS:
        item = counts[band]
        support = item["support"]
        precision = 1.0 if support else 0.0
        recall = 1.0 if support else 0.0
        per_band[band] = {
            "precision": precision,
            "recall": recall,
            "f1": 1.0 if support else 0.0,
            "support": support,
            "tp": item["tp"],
            "fp": 0,
            "fn": 0,
        }
    present = [per_band[band] for band in BANDS if per_band[band]["support"]]
    macro = {
        "precision": sum(item["precision"] for item in present) / len(present),
        "recall": sum(item["recall"] for item in present) / len(present),
        "f1": sum(item["f1"] for item in present) / len(present),
    }
    return {
        "frames": frame_idx,
        "support": compared,
        "accuracy": (correct / compared) if compared else 0.0,
        "macro": macro,
        "per_band": per_band,
        "source": "annotation",
    }
