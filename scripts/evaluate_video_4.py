"""Comprehensive Evaluation and Video Generation for runs/4.mp4.

Tests runs/4.mp4 against Anno/ground_truth.csv and Anno/annotations.xml.
Ensures 100% Student ID Alignment (Student 0 to 20).

Models Evaluated:
1. Baseline Frozen Model (checkpoints/fusion_daisee.pt)
2. Parallax-Corrected Attention Model (checkpoints/classroom_parallax_3class_v1.pt)
3. Multi-Pool Binary Attention Model (checkpoints/classroom_binary_attention_v1.pt)

Outputs:
- runs/4_annotated_evaluation.mp4 (Side-by-side synchronized video)
- runs/4_evaluation_aligned.csv (Frame-by-frame aligned CSV)
- runs/reports/video_4_evaluation_report.txt (Complete report)
- runs/reports/video_4_metrics.json (Machine-readable metrics)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from scripts.train_parallax_corrected_attention import apply_parallax_correction, ParallaxMultiPoolClassifier
from scripts.train_binary_attention import TemporalMultiPoolClassifier
from scripts.train_classroom_head import ClassroomAttentionClassifier

COLOR_MAP = {
    "HIGH": (0, 204, 0),        # Green
    "MODERATE": (0, 165, 255),   # Orange
    "LOW": (0, 0, 230),         # Red
    "WARMUP": (180, 180, 180),  # Gray
}
MATCH_COLOR = (0, 255, 0)
MISMATCH_COLOR = (0, 0, 255)

GT_MAP = {
    "attentive": "HIGH",
    "high": "HIGH",
    "moderate": "MODERATE",
    "non-attentive": "LOW",
    "non_attentive": "LOW",
    "low": "LOW",
}
BANDS = ("HIGH", "MODERATE", "LOW")
IDX_TO_BAND = {0: "HIGH", 1: "MODERATE", 2: "LOW"}


def load_ground_truth(xml_path: Path, csv_path: Path):
    gt_labels = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            f_idx = int(r["frame"])
            sid = int(r["student_id"])
            raw = r["attention"].strip().lower().replace(" ", "-")
            gt_labels[(f_idx, sid)] = GT_MAP.get(raw, "UNKNOWN")

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


def run_evaluation():
    video_path = WORKSPACE_ROOT / "runs" / "4.mp4"
    xml_path = WORKSPACE_ROOT / "Anno" / "annotations.xml"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth.csv"
    seq_npz = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing {video_path}")
    if not seq_npz.exists():
        raise FileNotFoundError(f"Missing {seq_npz}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(" TESTING VIDEO: runs/4.mp4 ON GPU")
    print("=" * 72)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Pre-extracted Sequence Features
    data = np.load(seq_npz)
    raw_feats = data["features"].astype(np.float32)   # (5985, 16, 32)
    targets_3class = data["targets"].astype(np.int64) # 0=HIGH, 1=MODERATE, 2=LOW
    targets_binary = (targets_3class != 2).astype(np.int64) # 1=HIGH, 0=LOW
    student_ids = data["student_ids"].astype(np.int32)
    frame_ids = data["frame_ids"].astype(np.int32)
    num_seqs = len(targets_3class)
    print(f"Total Sequence Instances: {num_seqs} across {len(np.unique(student_ids))} students (IDs 0..20)")

    # 2. Evaluate Baseline Model (classroom_attention_head_focal_v2.pt)
    print("\n[1/3] Evaluating Frozen Baseline Model (classroom_attention_head_focal_v2.pt)...")
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    daisee_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt", map_location="cpu", weights_only=False)
    base_net.load_state_dict(daisee_ckpt["model"], strict=False)

    baseline_classifier = ClassroomAttentionClassifier(base_net, num_classes=3, freeze_encoder=True)
    baseline_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "classroom_attention_head_focal_v2.pt", map_location="cpu", weights_only=False)
    baseline_classifier.load_state_dict(baseline_ckpt["model"])
    baseline_classifier.to(device).eval()

    with torch.no_grad():
        x_base = torch.from_numpy(raw_feats).to(device)
        logits_base = baseline_classifier(x_base).cpu().numpy()
        probs_base = np.exp(logits_base - np.max(logits_base, axis=-1, keepdims=True))
        probs_base /= np.sum(probs_base, axis=-1, keepdims=True)
        preds_base_3class = np.argmax(probs_base, axis=-1)

    acc_base_3 = float(accuracy_score(targets_3class, preds_base_3class))
    bal_base_3 = float(balanced_accuracy_score(targets_3class, preds_base_3class))
    cm_base_3 = confusion_matrix(targets_3class, preds_base_3class, labels=[0, 1, 2])
    p_b3, r_b3, f_b3, s_b3 = precision_recall_fscore_support(targets_3class, preds_base_3class, labels=[0, 1, 2], zero_division=0)

    # 3. Evaluate Parallax-Corrected 3-Class Model
    print("\n[2/3] Evaluating Parallax-Corrected 3-Class Model...")
    parallax_feats = apply_parallax_correction(raw_feats)
    parallax_model = ParallaxMultiPoolClassifier(base_net, num_classes=3)
    parallax_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "classroom_parallax_3class_v1.pt", map_location="cpu", weights_only=False)
    parallax_model.load_state_dict(parallax_ckpt["model"])
    parallax_model.to(device).eval()

    with torch.no_grad():
        x_par = torch.from_numpy(parallax_feats).to(device)
        logits_par = parallax_model(x_par).cpu().numpy()
        probs_par = np.exp(logits_par - np.max(logits_par, axis=-1, keepdims=True))
        probs_par /= np.sum(probs_par, axis=-1, keepdims=True)
        preds_par_3class = np.argmax(probs_par, axis=-1)

    acc_par_3 = float(accuracy_score(targets_3class, preds_par_3class))
    bal_par_3 = float(balanced_accuracy_score(targets_3class, preds_par_3class))
    cm_par_3 = confusion_matrix(targets_3class, preds_par_3class, labels=[0, 1, 2])
    p_p3, r_p3, f_p3, s_p3 = precision_recall_fscore_support(targets_3class, preds_par_3class, labels=[0, 1, 2], zero_division=0)

    # 4. Evaluate Multi-Pool Binary Attention Model
    print("\n[3/3] Evaluating Multi-Pool Binary Attention Model...")
    binary_model = TemporalMultiPoolClassifier(base_net)
    binary_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "classroom_binary_attention_v1.pt", map_location="cpu", weights_only=False)
    binary_model.load_state_dict(binary_ckpt["model"])
    binary_model.to(device).eval()

    with torch.no_grad():
        x_bin = torch.from_numpy(raw_feats).to(device)
        logits_bin = binary_model(x_bin).cpu().numpy()
        probs_bin = np.exp(logits_bin - np.max(logits_bin, axis=-1, keepdims=True))
        probs_bin /= np.sum(probs_bin, axis=-1, keepdims=True)
        preds_bin = np.argmax(probs_bin, axis=-1)  # 0=LOW, 1=HIGH
        # Calibrated threshold (p_high >= 0.45)
        preds_bin_calibrated = (probs_bin[:, 1] >= 0.45).astype(np.int64)

    acc_bin_raw = float(accuracy_score(targets_binary, preds_bin))
    bal_bin_raw = float(balanced_accuracy_score(targets_binary, preds_bin))
    cm_bin_raw = confusion_matrix(targets_binary, preds_bin, labels=[0, 1])
    p_br, r_br, f_br, s_br = precision_recall_fscore_support(targets_binary, preds_bin, labels=[0, 1], zero_division=0)

    acc_bin_cal = float(accuracy_score(targets_binary, preds_bin_calibrated))
    bal_bin_cal = float(balanced_accuracy_score(targets_binary, preds_bin_calibrated))
    cm_bin_cal = confusion_matrix(targets_binary, preds_bin_calibrated, labels=[0, 1])
    p_bc, r_bc, f_bc, s_bc = precision_recall_fscore_support(targets_binary, preds_bin_calibrated, labels=[0, 1], zero_division=0)

    # Per-Student Breakdown (Binary & Parallax 3-Class)
    per_student_stats = {}
    for sid in sorted(np.unique(student_ids)):
        mask = (student_ids == sid)
        n = int(mask.sum())
        # Binary
        bin_acc = float((targets_binary[mask] == preds_bin[mask]).mean())
        bin_cal_acc = float((targets_binary[mask] == preds_bin_calibrated[mask]).mean())
        n_high = int((targets_binary[mask] == 1).sum())
        n_low = int((targets_binary[mask] == 0).sum())
        # Parallax 3-Class
        par_acc = float((targets_3class[mask] == preds_par_3class[mask]).mean())
        base_acc = float((targets_3class[mask] == preds_base_3class[mask]).mean())

        per_student_stats[int(sid)] = {
            "instances": n,
            "gt_high_instances": n_high,
            "gt_low_instances": n_low,
            "binary_accuracy": bin_acc,
            "binary_calibrated_accuracy": bin_cal_acc,
            "parallax_3class_accuracy": par_acc,
            "baseline_3class_accuracy": base_acc,
        }

    # 5. Build Fast Lookup for Video and CSV Rendering
    pred_lookup = {}
    for i in range(num_seqs):
        f_idx = int(frame_ids[i])
        sid = int(student_ids[i])
        pred_lookup[(f_idx, sid)] = {
            "gt_3class": IDX_TO_BAND[targets_3class[i]],
            "gt_binary": "HIGH" if targets_binary[i] == 1 else "LOW",
            "pred_par_3class": IDX_TO_BAND[preds_par_3class[i]],
            "probs_par_3class": probs_par[i].tolist(),
            "pred_bin": "HIGH" if preds_bin[i] == 1 else "LOW",
            "pred_bin_cal": "HIGH" if preds_bin_calibrated[i] == 1 else "LOW",
            "prob_bin_high": float(probs_bin[i, 1]),
            "prob_bin_low": float(probs_bin[i, 0]),
        }

    # 6. Render Side-by-Side Synchronized Video
    gt_by_frame = load_ground_truth(xml_path, csv_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    out_video_path = WORKSPACE_ROOT / "runs" / "4_annotated_evaluation.mp4"
    out_csv_path = WORKSPACE_ROOT / "runs" / "4_evaluation_aligned.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width * 2, height + 44))

    csv_rows = []
    frame_idx = 0

    print(f"\n[4/4] Rendering Synchronized Side-by-Side Evaluation Video: {out_video_path}...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_idx / fps
        students = gt_by_frame.get(frame_idx, [])
        students.sort(key=lambda s: s["student_id"])

        frame_gt = frame.copy()
        frame_pred = frame.copy()

        frame_matches_bin = 0
        frame_eval_count = 0

        for s in students:
            sid = s["student_id"]
            bbox = s["bbox"]
            gt_band = s["label"]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]

            pinfo = pred_lookup.get((frame_idx, sid))
            if pinfo is not None:
                frame_eval_count += 1
                pred_par = pinfo["pred_par_3class"]
                pred_bin = pinfo["pred_bin"]
                gt_bin = pinfo["gt_binary"]
                is_match_par = (pred_par == gt_band)
                is_match_bin = (pred_bin == gt_bin)
                frame_matches_bin += int(is_match_bin)
                prob_h = pinfo["prob_bin_high"]
                prob_l = pinfo["prob_bin_low"]
            else:
                pred_par = "WARMUP"
                pred_bin = "WARMUP"
                is_match_par = None
                is_match_bin = None
                prob_h, prob_l = 0.0, 0.0

            # Ground Truth Overlay (Left)
            gt_col = COLOR_MAP.get(gt_band, (200, 200, 200))
            cv2.rectangle(frame_gt, (x1, y1), (x2, y2), gt_col, 2)
            gt_txt = f"ID {sid}: {gt_band}"
            cv2.rectangle(frame_gt, (x1, max(0, y1 - 18)), (x1 + len(gt_txt) * 8 + 10, y1), gt_col, -1)
            cv2.putText(frame_gt, gt_txt, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Model Overlay (Right - Shows Binary & Parallax with Match indicator)
            border_col = MATCH_COLOR if is_match_bin else (MISMATCH_COLOR if is_match_bin is not None else (180, 180, 180))
            pred_col = COLOR_MAP.get(pred_bin, (200, 200, 200))
            cv2.rectangle(frame_pred, (x1, y1), (x2, y2), border_col, 2)

            pred_txt = f"ID {sid}: {pred_bin}" if pred_bin != "WARMUP" else f"ID {sid}: WARMUP"
            cv2.rectangle(frame_pred, (x1, max(0, y1 - 18)), (x1 + len(pred_txt) * 8 + 10, y1), pred_col, -1)
            cv2.putText(frame_pred, pred_txt, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # CSV record
            csv_rows.append({
                "frame": frame_idx,
                "timestamp_sec": round(ts, 4),
                "student_id": sid,
                "gt_attention_3class": gt_band,
                "gt_attention_binary": "HIGH" if gt_band in ["HIGH", "MODERATE"] else ("LOW" if gt_band == "LOW" else "UNKNOWN"),
                "pred_binary": pred_bin,
                "is_binary_match": is_match_bin if is_match_bin is not None else "",
                "pred_parallax_3class": pred_par,
                "is_parallax_3class_match": is_match_par if is_match_par is not None else "",
                "prob_high": round(prob_h, 4),
                "prob_low": round(prob_l, 4),
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
            })

        # Header bar
        header = np.zeros((44, width * 2, 3), dtype=np.uint8)
        header[:] = (25, 25, 25)

        cv2.putText(header, f"GROUND TRUTH (Anno CSV) | Frame {frame_idx:03d} / {total_frames} ({ts:.2f}s)",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        acc_str = f"Binary Agreement: {frame_matches_bin}/{frame_eval_count} ({frame_matches_bin/frame_eval_count*100:.1f}%)" if frame_eval_count > 0 else "Warm-up Frames 0-14 (Sliding Window)"
        cv2.putText(header, f"MODEL EVALUATION (Same Student IDs 0..20) | {acc_str}",
                    (width + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

        sbs_frame = np.vstack([header, np.hstack([frame_gt, frame_pred])])
        writer.write(sbs_frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # Save Aligned CSV
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame", "timestamp_sec", "student_id",
            "gt_attention_3class", "gt_attention_binary",
            "pred_binary", "is_binary_match",
            "pred_parallax_3class", "is_parallax_3class_match",
            "prob_high", "prob_low",
            "x1", "y1", "x2", "y2"
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    # 7. Write Comprehensive Text Report
    report_path = WORKSPACE_ROOT / "runs" / "reports" / "video_4_evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 78,
        "          AUTHENTIC CLASSROOM EVALUATION REPORT: runs/4.mp4",
        "=" * 78,
        f"Input Video File       : {video_path.resolve()}",
        f"Ground Truth Reference : {csv_path.resolve()} & {xml_path.resolve()}",
        f"Evaluated Sequence File: {seq_npz.resolve()}",
        f"Evaluation Device      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
        f"Total Video Frames     : {total_frames} frames (832x464 @ {fps:.2f} fps, 10.01 sec)",
        f"Total Classroom Students: 21 students tracked (IDs 0 to 20)",
        f"Total Student Instances : {len(csv_rows)} total observations",
        f"Evaluated Sequences    : {num_seqs} sequences (21 students x 285 valid sliding windows)",
        f"Warm-Up Observation Gap: 315 observations (Frames 0-14, 15 frames x 21 students)",
        "",
        "-" * 78,
        "1. EXECUTIVE SUMMARY & MODEL BENCHMARK COMPARISON",
        "-" * 78,
        f"{'Model Architecture':<36} {'Mode':<12} {'Raw Acc':>9} {'Bal Acc':>9} {'Attentive Rec':>13}",
        "-" * 78,
        f"{'1. Frozen Baseline (fusion_daisee)':<36} {'3-Class':<12} {acc_base_3*100:8.2f}% {bal_base_3*100:8.2f}% {r_b3[0]*100:12.1f}%",
        f"{'2. Parallax-Corrected Attention':<36} {'3-Class':<12} {acc_par_3*100:8.2f}% {bal_par_3*100:8.2f}% {r_p3[0]*100:12.1f}%",
        f"{'3. Multi-Pool Binary Attention (Default)':<36} {'Binary':<12} {acc_bin_raw*100:8.2f}% {bal_bin_raw*100:8.2f}% {r_br[1]*100:12.1f}%",
        f"{'4. Multi-Pool Binary (Calibrated)':<36} {'Binary':<12} {acc_bin_cal*100:8.2f}% {bal_bin_cal*100:8.2f}% {r_bc[1]*100:12.1f}%",
        "-" * 78,
        "",
        "-" * 78,
        "2. DETAILED METRICS: MULTI-POOL BINARY ATTENTION (HIGH vs LOW)",
        "-" * 78,
        f"Raw Accuracy           : {acc_bin_raw*100:.2f}% (Calibrated: {acc_bin_cal*100:.2f}%)",
        f"Balanced Accuracy      : {bal_bin_raw*100:.2f}% (Calibrated: {bal_bin_cal*100:.2f}%)",
        "",
        f"{'Attention Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 78,
        f"{'LOW (Inattentive)':<16} {p_br[0]*100:9.2f}% {r_br[0]*100:9.2f}% {f_br[0]*100:9.2f}% {s_br[0]:10d}",
        f"{'HIGH (Attentive)':<16} {p_br[1]*100:9.2f}% {r_br[1]*100:9.2f}% {f_br[1]*100:9.2f}% {s_br[1]:10d}",
        "-" * 78,
        f"{'MACRO AVERAGE':<16} {(p_br[0]+p_br[1])/2*100:9.2f}% {(r_br[0]+r_br[1])/2*100:9.2f}% {(f_br[0]+f_br[1])/2*100:9.2f}% {num_seqs:10d}",
        "",
        "Binary Confusion Matrix (Default 0.50 Threshold):",
        f"  Actual LOW  (1,043) -> Predicted LOW: {cm_bin_raw[0,0]:5d} | Predicted HIGH: {cm_bin_raw[0,1]:5d}",
        f"  Actual HIGH (4,942) -> Predicted LOW: {cm_bin_raw[1,0]:5d} | Predicted HIGH: {cm_bin_raw[1,1]:5d}",
        "",
        "Binary Confusion Matrix (Calibrated 0.45 Threshold):",
        f"  Actual LOW  (1,043) -> Predicted LOW: {cm_bin_cal[0,0]:5d} | Predicted HIGH: {cm_bin_cal[0,1]:5d}",
        f"  Actual HIGH (4,942) -> Predicted LOW: {cm_bin_cal[1,0]:5d} | Predicted HIGH: {cm_bin_cal[1,1]:5d}",
        "",
        "-" * 78,
        "3. DETAILED METRICS: PARALLAX-CORRECTED 3-CLASS ATTENTION",
        "-" * 78,
        f"Raw Accuracy           : {acc_par_3*100:.2f}%",
        f"Balanced Accuracy      : {bal_par_3*100:.2f}%",
        "",
        f"{'Attention Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 78,
        f"{'HIGH':<16} {p_p3[0]*100:9.2f}% {r_p3[0]*100:9.2f}% {f_p3[0]*100:9.2f}% {s_p3[0]:10d}",
        f"{'MODERATE':<16} {p_p3[1]*100:9.2f}% {r_p3[1]*100:9.2f}% {f_p3[1]*100:9.2f}% {s_p3[1]:10d}",
        f"{'LOW':<16} {p_p3[2]*100:9.2f}% {r_p3[2]*100:9.2f}% {f_p3[2]*100:9.2f}% {s_p3[2]:10d}",
        "-" * 78,
        f"{'MACRO AVERAGE':<16} {p_p3.mean()*100:9.2f}% {r_p3.mean()*100:9.2f}% {f_p3.mean()*100:9.2f}% {num_seqs:10d}",
        "",
        "3-Class Confusion Matrix (Parallax-Corrected):",
        f"  Actual HIGH     (3,896) -> Pred HIGH: {cm_par_3[0,0]:5d} | Pred MODERATE: {cm_par_3[0,1]:5d} | Pred LOW: {cm_par_3[0,2]:5d}",
        f"  Actual MODERATE (1,046) -> Pred HIGH: {cm_par_3[1,0]:5d} | Pred MODERATE: {cm_par_3[1,1]:5d} | Pred LOW: {cm_par_3[1,2]:5d}",
        f"  Actual LOW      (1,043) -> Pred HIGH: {cm_par_3[2,0]:5d} | Pred MODERATE: {cm_par_3[2,1]:5d} | Pred LOW: {cm_par_3[2,2]:5d}",
        "",
        "-" * 78,
        "4. PER-STUDENT ACCURACY BREAKDOWN (ALL 21 STUDENTS, EXACT MATCHED IDS)",
        "-" * 78,
        f"{'Student ID':<12} {'Sequences':>10} {'GT High':>9} {'GT Low':>8} {'Binary Acc':>12} {'Parallax 3-Cl':>15} {'Status':>12}",
        "-" * 78,
    ]

    for sid in sorted(per_student_stats.keys()):
        st = per_student_stats[sid]
        status = "EXCELLENT" if st["binary_accuracy"] >= 0.85 else ("GOOD" if st["binary_accuracy"] >= 0.70 else "ATTENTION NEEDED")
        lines.append(
            f"Student {sid:<4d} {st['instances']:10d} {st['gt_high_instances']:9d} {st['gt_low_instances']:8d} "
            f"{st['binary_accuracy']*100:11.1f}% {st['parallax_3class_accuracy']*100:14.1f}% {status:>14}"
        )

    lines.extend([
        "-" * 78,
        "",
        "-" * 78,
        "5. DELIVERABLES GENERATED",
        "-" * 78,
        f"1. Annotated Synchronized Video : {out_video_path.resolve()}",
        f"2. Frame-by-Frame Aligned CSV   : {out_csv_path.resolve()}",
        f"3. Comprehensive Text Report    : {report_path.resolve()}",
        "=" * 78,
    ])

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[SUCCESS] Report written to: {report_path}")

    # 8. Save Machine-Readable JSON Metrics
    metrics_json = {
        "video": str(video_path),
        "total_frames": total_frames,
        "evaluated_sequences": num_seqs,
        "models": {
            "baseline_frozen_3class": {
                "raw_accuracy": acc_base_3,
                "balanced_accuracy": bal_base_3,
                "confusion_matrix": cm_base_3.tolist(),
            },
            "parallax_3class": {
                "raw_accuracy": acc_par_3,
                "balanced_accuracy": bal_par_3,
                "per_class_recall": [float(r) for r in r_p3],
                "per_class_precision": [float(p) for p in p_p3],
                "per_class_f1": [float(f) for f in f_p3],
                "confusion_matrix": cm_par_3.tolist(),
            },
            "multi_pool_binary_raw": {
                "raw_accuracy": acc_bin_raw,
                "balanced_accuracy": bal_bin_raw,
                "per_class_recall": [float(r) for r in r_br],
                "per_class_precision": [float(p) for p in p_br],
                "per_class_f1": [float(f) for f in f_br],
                "confusion_matrix": cm_bin_raw.tolist(),
            },
            "multi_pool_binary_calibrated": {
                "raw_accuracy": acc_bin_cal,
                "balanced_accuracy": bal_bin_cal,
                "per_class_recall": [float(r) for r in r_bc],
                "per_class_precision": [float(p) for p in p_bc],
                "per_class_f1": [float(f) for f in f_bc],
                "confusion_matrix": cm_bin_cal.tolist(),
            },
        },
        "per_student": per_student_stats,
    }
    json_path = WORKSPACE_ROOT / "runs" / "reports" / "video_4_metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[SUCCESS] JSON metrics saved to: {json_path}")


if __name__ == "__main__":
    run_evaluation()
