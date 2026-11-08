"""Standalone evaluation script for frozen 3-class classifier on clip_004.

Evaluates checkpoints/classroom_attention_head_focal_v2.pt against CVAT ground truth:
- Anno/ground_truth.csv
- Anno/annotations.xml

Outputs to: evaluations/frozen_focal_v2_clip_004/
Never overwrites existing checkpoints, annotations, or experiment results.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

import sys
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.pipeline import AttentionPipeline
from train_classroom_head import ClassroomAttentionClassifier

BANDS = ("HIGH", "MODERATE", "LOW")
GT_MAP = {
    "attentive": 0,
    "high": 0,
    "moderate": 1,
    "non-attentive": 2,
    "non_attentive": 2,
    "low": 2,
}
IDX_TO_BAND = {0: "HIGH", 1: "MODERATE", 2: "LOW"}


def _iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    den = aa + bb - inter
    return inter / den if den > 0.0 else 0.0


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"error": "empty"}
    raw_acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(np.mean(f1))
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))

    return {
        "raw_accuracy": raw_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "support": [int(s) for s in supp],
        "recalls": [float(r) for r in rec],
        "precisions": [float(p) for p in prec],
        "f1s": [float(f) for f in f1],
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }


def evaluate_sequences(model: torch.nn.Module, device: torch.device, ckpt_meta: dict, out_dir: Path) -> dict:
    print("\n" + "=" * 65)
    print(" EVALUATION PART 1: CONTROLLED CVAT-ALIGNED SEQUENCES (5,985)")
    print("=" * 65)
    data = np.load(WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz")
    feats = data["features"]
    targets = data["targets"]
    student_ids = data["student_ids"]
    frame_ids = data["frame_ids"]

    train_students = ckpt_meta["train_students"]
    val_students = ckpt_meta["val_students"]

    val_mask = np.isin(student_ids, val_students)
    train_mask = np.isin(student_ids, train_students)

    with torch.no_grad():
        x_tensor = torch.tensor(feats, dtype=torch.float32).to(device)
        logits = model(x_tensor).cpu().numpy()
        preds = np.argmax(logits, axis=-1)

    val_metrics = compute_metrics_dict(targets[val_mask], preds[val_mask])
    train_metrics = compute_metrics_dict(targets[train_mask], preds[train_mask])
    overall_metrics = compute_metrics_dict(targets, preds)

    # Save sequence predictions
    np.savez_compressed(
        out_dir / "sequence_predictions.npz",
        logits=logits,
        y_pred=preds,
        y_true=targets,
        student_ids=student_ids,
        frame_ids=frame_ids,
        val_mask=val_mask,
        train_mask=train_mask,
    )

    print("\n[A. Held-Out Validation Students (6 students)]: ", val_students)
    print(f"  Sample Count      : {val_metrics['total_samples']} sequences")
    print(f"  Raw Accuracy      : {val_metrics['raw_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy : {val_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {val_metrics['macro_f1']*100:.2f}%")
    print(f"  Per-Class Recalls : HIGH={val_metrics['recalls'][0]*100:.1f}%, MOD={val_metrics['recalls'][1]*100:.1f}%, LOW={val_metrics['recalls'][2]*100:.1f}%")
    print(f"  Confusion Matrix  :\n    {val_metrics['confusion_matrix']}")

    print("\n[B. Training Students (15 students)]: ", train_students)
    print(f"  Sample Count      : {train_metrics['total_samples']} sequences (previously seen)")
    print(f"  Raw Accuracy      : {train_metrics['raw_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy : {train_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {train_metrics['macro_f1']*100:.2f}%")

    print("\n[C. Combined (All 21 Students)]: ")
    print(f"  Sample Count      : {overall_metrics['total_samples']} sequences")
    print(f"  Raw Accuracy      : {overall_metrics['raw_accuracy']*100:.2f}% (Notice: Heavily weighted by training students)")
    print(f"  Balanced Accuracy : {overall_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {overall_metrics['macro_f1']*100:.2f}%")

    results = {
        "val_held_out": val_metrics,
        "train_seen": train_metrics,
        "overall_combined": overall_metrics,
    }
    with (out_dir / "sequence_evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_end_to_end_video(model: torch.nn.Module, device: torch.device, ckpt_meta: dict, out_dir: Path) -> dict:
    print("\n" + "=" * 65)
    print(" EVALUATION PART 2: END-TO-END VIDEO INFERENCE & SPATIAL IOU ALIGNMENT")
    print("=" * 65)

    video_path = WORKSPACE_ROOT / "runs" / "Test.mp4"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth.csv"
    xml_path = WORKSPACE_ROOT / "Anno" / "annotations.xml"

    # 1. Load human GT
    gt_labels = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row["attention"].strip().lower().replace(" ", "-")
            gt_labels[(int(row["frame"]), int(row["student_id"]))] = GT_MAP[raw]

    gt_by_frame = defaultdict(list)
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        sid = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", "0"))
            label_idx = gt_labels.get((frame, sid))
            if label_idx is None:
                continue
            gt_by_frame[frame].append({
                "student_id": sid,
                "bbox": [
                    float(box.get("xtl", 0)),
                    float(box.get("ytl", 0)),
                    float(box.get("xbr", 0)),
                    float(box.get("ybr", 0)),
                ],
                "label_idx": label_idx,
                "band": IDX_TO_BAND[label_idx],
            })

    total_gt_boxes = sum(len(boxes) for boxes in gt_by_frame.values())
    print(f"Loaded {total_gt_boxes} CVAT ground-truth boxes across {len(gt_by_frame)} frames.")

    # 2. Run inference with AttentionPipeline + Neural Head
    pipe = AttentionPipeline.from_yaml(WORKSPACE_ROOT / "configs" / "pipeline.yaml", load_weights=True)
    pipe.detector.max_people = 21

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    model_preds_by_frame = defaultdict(list)
    frame_idx = 0
    window = 16

    csv_rows = []

    print(f"Running video inference on {video_path.name} ({total_frames} frames)...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps

        # Process frame through vision pipeline
        result = pipe.process_frame(ts, frame)
        students = result["students"]

        # Run neural classification for each track with >= 16 frames
        for s in students:
            track_id = int(s["student_id"])
            track = pipe.tracks.get(track_id)
            bbox = [float(v) for v in s["bbox"]]

            pred_idx = 0
            pred_band = "HIGH"
            probs = [1.0, 0.0, 0.0]

            if track is not None and len(track.feature_window) >= window:
                seq = np.stack(list(track.feature_window)[-window:], axis=0).astype(np.float32)
                with torch.no_grad():
                    inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    out_logits = model(inp).squeeze(0)
                    prob_t = torch.softmax(out_logits, dim=-1).cpu().numpy()
                    pred_idx = int(np.argmax(prob_t))
                    pred_band = IDX_TO_BAND[pred_idx]
                    probs = prob_t.tolist()

            model_preds_by_frame[frame_idx].append({
                "model_track_id": track_id,
                "bbox": bbox,
                "pred_idx": pred_idx,
                "pred_band": pred_band,
                "probs": probs,
            })

            csv_rows.append({
                "frame": frame_idx,
                "timestamp_sec": round(ts, 4),
                "model_track_id": track_id,
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
                "pred_band": pred_band,
                "prob_high": round(probs[0], 4),
                "prob_moderate": round(probs[1], 4),
                "prob_low": round(probs[2], 4),
            })

        frame_idx += 1
    cap.release()

    total_model_detections = sum(len(p) for p in model_preds_by_frame.values())
    print(f"Inference complete: {total_model_detections} student detections across {frame_idx} frames.")

    # 3. Match predictions to GT frame-by-frame by IoU >= 0.3
    y_true_all, y_pred_all = [], []
    y_true_val, y_pred_val = [], []
    y_true_train, y_pred_train = [], []

    unmatched_gt = 0
    unmatched_pred = 0
    matched_count = 0

    val_students = set(ckpt_meta["val_students"])
    train_students = set(ckpt_meta["train_students"])

    for f in range(max(frame_idx, len(gt_by_frame))):
        gt_list = gt_by_frame.get(f, [])
        pred_list = model_preds_by_frame.get(f, [])

        pairs = []
        for gi, g in enumerate(gt_list):
            for pi, p in enumerate(pred_list):
                score = _iou(g["bbox"], p["bbox"])
                if score >= 0.30:
                    pairs.append((score, gi, pi))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matched_count += 1

            gt_sid = gt_list[gi]["student_id"]
            gt_label = gt_list[gi]["label_idx"]
            p_label = pred_list[pi]["pred_idx"]

            y_true_all.append(gt_label)
            y_pred_all.append(p_label)

            if gt_sid in val_students:
                y_true_val.append(gt_label)
                y_pred_val.append(p_label)
            elif gt_sid in train_students:
                y_true_train.append(gt_label)
                y_pred_train.append(p_label)

        unmatched_gt += (len(gt_list) - len(used_g))
        unmatched_pred += (len(pred_list) - len(used_p))

    # Compute video metrics
    all_metrics = compute_metrics_dict(np.array(y_true_all), np.array(y_pred_all))
    val_video_metrics = compute_metrics_dict(np.array(y_true_val), np.array(y_pred_val))
    train_video_metrics = compute_metrics_dict(np.array(y_true_train), np.array(y_pred_train))

    coverage_stats = {
        "total_gt_instances": total_gt_boxes,
        "total_model_detections": total_model_detections,
        "matched_observations": matched_count,
        "matched_percentage": float(matched_count / total_gt_boxes * 100.0) if total_gt_boxes else 0.0,
        "unmatched_gt_instances": unmatched_gt,
        "unmatched_model_detections": unmatched_pred,
    }

    video_results = {
        "coverage": coverage_stats,
        "val_held_out_students": val_video_metrics,
        "train_seen_students": train_video_metrics,
        "all_matched_combined": all_metrics,
    }

    with (out_dir / "video_evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(video_results, f, indent=2)

    # Write frame-by-frame CSV
    with (out_dir / "predictions_frame_by_frame.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "timestamp_sec", "model_track_id", "x1", "y1", "x2", "y2",
            "pred_band", "prob_high", "prob_moderate", "prob_low",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n--- End-to-End Video Inference Coverage ---")
    print(f"  Total GT Instances       : {total_gt_boxes}")
    print(f"  Total Model Detections   : {total_model_detections}")
    print(f"  Matched Observations     : {matched_count} ({matched_count/total_gt_boxes*100:.2f}% of GT)")
    print(f"  Unmatched GT (Missed)    : {unmatched_gt}")
    print(f"  Unmatched Pred (Spurious): {unmatched_pred}")

    print("\n--- Attention Classification on Matched Observations ---")
    print(f"[A. Held-Out Validation Students ({val_video_metrics['total_samples']} matched)]: ")
    print(f"  Raw Accuracy      : {val_video_metrics['raw_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy : {val_video_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {val_video_metrics['macro_f1']*100:.2f}%")
    print(f"  Confusion Matrix  :\n    {val_video_metrics['confusion_matrix']}")

    print(f"\n[B. Training Students ({train_video_metrics['total_samples']} matched)]: ")
    print(f"  Raw Accuracy      : {train_video_metrics['raw_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy : {train_video_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {train_video_metrics['macro_f1']*100:.2f}%")

    print(f"\n[C. Combined All Students ({all_metrics['total_samples']} matched)]: ")
    print(f"  Raw Accuracy      : {all_metrics['raw_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy : {all_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score    : {all_metrics['macro_f1']*100:.2f}%")
    print(f"  Confusion Matrix  :\n    {all_metrics['confusion_matrix']}")

    return video_results


def main():
    out_dir = WORKSPACE_ROOT / "evaluations" / "frozen_focal_v2_clip_004"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = WORKSPACE_ROOT / "checkpoints" / "classroom_attention_head_focal_v2.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading candidate checkpoint: {ckpt_path.resolve()} on {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ClassroomAttentionClassifier(pretrained_fusion=fusion_base, num_classes=3, freeze_encoder=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    ckpt_meta = {
        "train_students": sorted(list(ckpt.get("train_students", []))),
        "val_students": sorted(list(ckpt.get("val_students", []))),
        "epoch": ckpt.get("epoch"),
    }
    print("Checkpoint Metadata:")
    print("  Epoch            :", ckpt_meta["epoch"])
    print("  Train Students   :", ckpt_meta["train_students"])
    print("  Val Students     :", ckpt_meta["val_students"])

    # Run evaluations
    seq_res = evaluate_sequences(model, device, ckpt_meta, out_dir)
    video_res = evaluate_end_to_end_video(model, device, ckpt_meta, out_dir)

    print(f"\nAll evaluation outputs successfully saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
