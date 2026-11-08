"""Render aligned Ground-Truth and Model-Prediction videos for Test.mp4.

Ensures that every student in the model predicted video has the EXACT SAME
student ID (0 to 20) as the human ground truth in Anno/annotations.xml
and Anno/ground_truth.csv.

Outputs:
1. runs/Test_ground_truth_annotated.mp4 (GT only with ID 0..20)
2. runs/Test_model_predicted_aligned.mp4 (Model Pred only with ID 0..20)
3. runs/Test_side_by_side_gt_vs_pred.mp4 (Side-by-side synchronized view)
4. runs/Test_predictions_aligned_with_gt.csv (Authoritative aligned CSV)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Standardized color scheme
COLOR_MAP = {
    "HIGH": (0, 204, 0),       # Green
    "MODERATE": (0, 165, 255),  # Orange
    "LOW": (0, 0, 230),        # Red
    "WARMUP": (180, 180, 180), # Gray
}
MATCH_COLOR = (0, 255, 0)      # Green border for correct match
MISMATCH_COLOR = (0, 0, 255)   # Red border for mismatch

GT_NORM_MAP = {
    "attentive": "HIGH",
    "high": "HIGH",
    "moderate": "MODERATE",
    "non-attentive": "LOW",
    "non_attentive": "LOW",
    "low": "LOW",
}
IDX_TO_BAND = {0: "HIGH", 1: "MODERATE", 2: "LOW"}


def load_ground_truth(xml_path: Path, csv_path: Path):
    """Load GT boxes and attention labels indexed by (frame, student_id)."""
    gt_labels = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            f_idx = int(r["frame"])
            sid = int(r["student_id"])
            raw = r["attention"].strip().lower().replace(" ", "-")
            gt_labels[(f_idx, sid)] = GT_NORM_MAP.get(raw, "UNKNOWN")

    gt_boxes_by_frame = defaultdict(list)
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        sid = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            f_idx = int(box.get("frame", "0"))
            bbox = [
                float(box.get("xtl", 0)),
                float(box.get("ytl", 0)),
                float(box.get("xbr", 0)),
                float(box.get("ybr", 0)),
            ]
            label = gt_labels.get((f_idx, sid), "UNKNOWN")
            gt_boxes_by_frame[f_idx].append({
                "student_id": sid,
                "bbox": bbox,
                "label": label,
            })

    return gt_boxes_by_frame


def load_model_predictions():
    """Load frozen focal v2 predictions from sequence_predictions.npz."""
    npz_path = WORKSPACE_ROOT / "evaluations" / "frozen_focal_v2_clip_004" / "sequence_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    data = np.load(npz_path)
    y_pred = data["y_pred"]
    y_true = data["y_true"]
    sids = data["student_ids"]
    fids = data["frame_ids"]
    logits = data["logits"]

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    pred_lookup = {}
    for i in range(len(y_pred)):
        f_idx = int(fids[i])
        sid = int(sids[i])
        pred_lookup[(f_idx, sid)] = {
            "pred_band": IDX_TO_BAND[y_pred[i]],
            "gt_band": IDX_TO_BAND[y_true[i]],
            "probs": probs[i],
        }

    return pred_lookup


def render_aligned_videos():
    video_path = WORKSPACE_ROOT / "runs" / "Test.mp4"
    xml_path = WORKSPACE_ROOT / "Anno" / "annotations.xml"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth.csv"

    out_gt_video = WORKSPACE_ROOT / "runs" / "Test_ground_truth_annotated.mp4"
    out_pred_video = WORKSPACE_ROOT / "runs" / "Test_model_predicted_aligned.mp4"
    out_sbs_video = WORKSPACE_ROOT / "runs" / "Test_side_by_side_gt_vs_pred.mp4"
    out_csv = WORKSPACE_ROOT / "runs" / "Test_predictions_aligned_with_gt.csv"

    print("Loading Ground Truth and Model Predictions...")
    gt_by_frame = load_ground_truth(xml_path, csv_path)
    pred_lookup = load_model_predictions()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_gt = cv2.VideoWriter(str(out_gt_video), fourcc, fps, (width, height))
    writer_pred = cv2.VideoWriter(str(out_pred_video), fourcc, fps, (width, height))
    writer_sbs = cv2.VideoWriter(str(out_sbs_video), fourcc, fps, (width * 2, height + 40))

    csv_rows = []
    frame_idx = 0

    print(f"Rendering aligned videos ({total_frames} frames, {width}x{height} @ {fps:.2f} fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_idx / fps
        students = gt_by_frame.get(frame_idx, [])
        students.sort(key=lambda s: s["student_id"])

        frame_gt = frame.copy()
        frame_pred = frame.copy()

        frame_matches = 0
        frame_eval_count = 0

        for s in students:
            sid = s["student_id"]
            bbox = s["bbox"]
            gt_band = s["label"]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]

            pred_info = pred_lookup.get((frame_idx, sid))
            if pred_info is not None:
                pred_band = pred_info["pred_band"]
                prob_high, prob_mod, prob_low = pred_info["probs"]
                is_match = (pred_band == gt_band)
                frame_matches += int(is_match)
                frame_eval_count += 1
            else:
                pred_band = "WARMUP"
                prob_high, prob_mod, prob_low = 0.0, 0.0, 0.0
                is_match = None

            # --- 1. Draw GT Frame ---
            gt_color = COLOR_MAP.get(gt_band, (200, 200, 200))
            cv2.rectangle(frame_gt, (x1, y1), (x2, y2), gt_color, 2)
            # Label banner
            gt_text = f"ID {sid}: {gt_band}"
            cv2.rectangle(frame_gt, (x1, max(0, y1 - 18)), (x1 + len(gt_text) * 8 + 10, y1), gt_color, -1)
            cv2.putText(frame_gt, gt_text, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # --- 2. Draw Model Pred Frame (WITH SAME STUDENT ID) ---
            pred_color = COLOR_MAP.get(pred_band, (200, 200, 200))
            border_color = MATCH_COLOR if is_match else (MISMATCH_COLOR if is_match is not None else pred_color)
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), border_color, 2)

            pred_text = f"ID {sid}: {pred_band}" if pred_band != "WARMUP" else f"ID {sid}: WARMUP"
            cv2.rectangle(frame_pred, (x1, max(0, y1 - 18)), (x1 + len(pred_text) * 8 + 10, y1), pred_color, -1)
            cv2.putText(frame_pred, pred_text, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Record CSV row
            csv_rows.append({
                "frame": frame_idx,
                "timestamp_sec": round(ts, 4),
                "student_id": sid,
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
                "ground_truth_attention": gt_band,
                "model_predicted_attention": pred_band,
                "is_match": is_match if is_match is not None else "",
                "prob_high": round(float(prob_high), 4),
                "prob_moderate": round(float(prob_mod), 4),
                "prob_low": round(float(prob_low), 4),
            })

        # --- 3. Assemble Side-by-Side Frame ---
        # Top header
        header = np.zeros((40, width * 2, 3), dtype=np.uint8)
        header[:] = (30, 30, 30)

        # Header titles
        cv2.putText(header, f"GROUND TRUTH (Anno Ground Truth CSV) | Frame {frame_idx:03d} ({ts:.2f}s)",
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        acc_str = f"Agreement: {frame_matches}/{frame_eval_count} ({frame_matches/frame_eval_count*100:.1f}%)" if frame_eval_count > 0 else "Warmup (Frames 0-14)"
        cv2.putText(header, f"MODEL PREDICTION (Same Student ID 0..20) | {acc_str}",
                    (width + 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

        combined_body = np.hstack([frame_gt, frame_pred])
        sbs_frame = np.vstack([header, combined_body])

        writer_gt.write(frame_gt)
        writer_pred.write(frame_pred)
        writer_sbs.write(sbs_frame)

        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"  Frame {frame_idx}/{total_frames} processed...")

        frame_idx += 1

    cap.release()
    writer_gt.release()
    writer_pred.release()
    writer_sbs.release()

    # Save aligned CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame", "timestamp_sec", "student_id",
            "x1", "y1", "x2", "y2",
            "ground_truth_attention", "model_predicted_attention",
            "is_match", "prob_high", "prob_moderate", "prob_low"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n[SUCCESS] Aligned outputs successfully written:")
    print(f"  1. Ground Truth Video  : {out_gt_video}")
    print(f"  2. Model Prediction Vdo: {out_pred_video}")
    print(f"  3. Side-by-Side Video  : {out_sbs_video}")
    print(f"  4. Aligned CSV Report  : {out_csv}")
    print(f"  Total Rows Written     : {len(csv_rows)}")


if __name__ == "__main__":
    render_aligned_videos()
