"""Standalone Audit Script: Complete Metric Recalculation & Verification.

This script executes Phase 3, Phase 5, Phase 6, and Phase 7 calculations:
1. Clip 001 evaluation against User's CVAT annotations (`runs/Annoted file for clip 1/annotations.xml`)
2. Clip 007 evaluation against CVAT ground truth (`Anno/annotations.xml` and `Anno/ground_truth.csv`)
3. Video Project 3 evaluation (heuristic check band & ground-truth audit)
4. Test 4 evaluation (heuristic check band & ground-truth audit)
5. Held-out student-disjoint validation evaluation on checkpoint `classroom_manual_binary_unquarantined.pt`
6. IoU sensitivity analysis on Clip 001 (IoU 0.15 to 0.50)
7. Exact confusion matrices, True Macro-F1, Balanced Accuracy, Class 0/1 Precision/Recall/F1, Specificity, Sensitivity.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

WORKSPACE_ROOT = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model")
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.anno_overlay import _iou, to_band
from attention_pipeline.video_accuracy import load_jsonl, check_band, _lesson_yaw
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    n = int(len(y_true))

    # Class 0 (LOW / Disengaged)
    prec_0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_0 = float(2 * prec_0 * rec_0 / (prec_0 + rec_0)) if (prec_0 + rec_0) > 0 else 0.0

    # Class 1 (HIGH / Engaged)
    prec_1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_1 = float(2 * prec_1 * rec_1 / (prec_1 + rec_1)) if (prec_1 + rec_1) > 0 else 0.0

    specificity = rec_0  # TN / (TN + FP)
    sensitivity = rec_1  # TP / (TP + FN)

    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(r_c, p_c))
        except Exception:
            pass

    return {
        "n_samples": n,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "class_0_low": {
            "support": int(tn + fp),
            "precision": round(prec_0, 4),
            "recall": round(rec_0, 4),
            "f1": round(f1_0, 4),
        },
        "class_1_high": {
            "support": int(fn + tp),
            "precision": round(prec_1, 4),
            "recall": round(rec_1, 4),
            "f1": round(f1_1, 4),
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
    }


def audit_clip_001():
    scores_path = WORKSPACE_ROOT / "runs" / "clip_001_scores.jsonl"
    gt_xml_path = WORKSPACE_ROOT / "runs" / "Annoted file for clip 1" / "annotations.xml"

    records = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    tree = ET.parse(gt_xml_path)
    root = tree.getroot()
    boxes_by_frame = defaultdict(list)
    labels_by_frame_id = {}

    for track in root.findall("track"):
        tid = int(track.get("id", 0))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", 0))
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            att = "High"
            for a in box.findall("attribute"):
                if a.get("name") == "Attention":
                    att = a.text or "High"
                    break
            boxes_by_frame[frame].append((tid, xtl, ytl, xbr, ybr))
            labels_by_frame_id[(frame, tid)] = att

    # Evaluate at standard IoU = 0.30
    def evaluate_at_iou(min_iou):
        y_true, y_pred, y_prob = [], [], []
        for row in records:
            frame = row.get("frame", 0)
            gt_boxes = boxes_by_frame.get(frame, [])
            used = set()
            for s in row.get("students", []):
                pred_band = s.get("band")
                if not pred_band or pred_band == "WARMUP":
                    continue
                sb = s.get("bbox") or [0, 0, 0, 0]
                best_i, best_iou = -1, 0.0
                for i, gtb in enumerate(gt_boxes):
                    if i in used:
                        continue
                    iou = _iou(sb, gtb[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_iou < min_iou or best_i < 0:
                    continue
                used.add(best_i)
                gt_id = gt_boxes[best_i][0]
                gt_raw = labels_by_frame_id.get((frame, gt_id), "High")
                gt_band = to_band(gt_raw)

                gt_bin = 1 if gt_band in ("HIGH", "MODERATE") else 0
                pred_bin = 1 if pred_band == "HIGH" else 0
                prob = (s.get("score") or 0.0) / 100.0

                y_true.append(gt_bin)
                y_pred.append(pred_bin)
                y_prob.append(prob)
        return compute_full_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    standard_eval = evaluate_at_iou(0.30)
    iou_sweep = {}
    for iou_thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        iou_sweep[f"iou_{iou_thresh:.2f}"] = evaluate_at_iou(iou_thresh)

    return standard_eval, iou_sweep


def audit_clip_007():
    scores_path = WORKSPACE_ROOT / "runs" / "clip_007_scores.jsonl"
    xml_path = WORKSPACE_ROOT / "Anno" / "annotations.xml"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth.csv"

    records = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    labels = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row["attention"].strip().lower().replace(" ", "-")
            labels[(int(row["frame"]), int(row["student_id"]))] = raw

    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes_by_frame = defaultdict(list)
    for track in root.findall("track"):
        sid = int(track.get("id", 0))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", 0))
            lbl = labels.get((frame, sid))
            if lbl is None:
                continue
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            boxes_by_frame[frame].append((sid, xtl, ytl, xbr, ybr, lbl))

    y_true, y_pred, y_prob = [], [], []
    for row in records:
        frame = row.get("frame", 0)
        gt_boxes = boxes_by_frame.get(frame, [])
        used = set()
        for s in row.get("students", []):
            pred_band = s.get("band")
            if not pred_band or pred_band == "WARMUP":
                continue
            sb = s.get("bbox") or [0, 0, 0, 0]
            best_i, best_iou = -1, 0.0
            for i, gtb in enumerate(gt_boxes):
                if i in used:
                    continue
                iou = _iou(sb, gtb[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_iou < 0.30 or best_i < 0:
                continue
            used.add(best_i)
            lbl = gt_boxes[best_i][5]
            gt_band = to_band(lbl)
            gt_bin = 1 if gt_band in ("HIGH", "MODERATE") else 0
            pred_bin = 1 if pred_band == "HIGH" else 0
            prob = (s.get("score") or 0.0) / 100.0

            y_true.append(gt_bin)
            y_pred.append(pred_bin)
            y_prob.append(prob)

    return compute_full_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))


def audit_heuristic_video(scores_jsonl_name: str):
    scores_path = WORKSPACE_ROOT / "runs" / scores_jsonl_name
    rows = load_jsonl(scores_path)
    y_true_binary, y_pred_binary, y_prob = [], [], []
    for row in rows:
        students = row.get("students") or []
        if not students:
            continue
        lesson = _lesson_yaw(students)
        for s in students:
            pred_band = s.get("band")
            if pred_band not in ("HIGH", "LOW"):
                continue
            gt_band = check_band(s, lesson)
            gt_bin = 1 if gt_band in ("HIGH", "MODERATE") else 0
            pred_bin = 1 if pred_band == "HIGH" else 0
            prob = (s.get("score") or 0.0) / 100.0

            y_true_binary.append(gt_bin)
            y_pred_binary.append(pred_bin)
            y_prob.append(prob)

    return compute_full_metrics(np.array(y_true_binary), np.array(y_pred_binary), np.array(y_prob))


def audit_held_out_validation():
    ckpt_path = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_unquarantined.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    data_dir = WORKSPACE_ROOT / "data" / "processed"
    clip_files = [
        data_dir / "classroom_clip_001_v3_hardened.npz",
        data_dir / "classroom_clip_004_v3_hardened.npz",
        data_dir / "classroom_clip_008_v3_hardened.npz",
    ]

    all_feats, all_sids, all_targets = [], [], []
    for p in clip_files:
        d = np.load(p, allow_pickle=True)
        all_feats.append(d["features"].astype(np.float32))
        all_sids.append(d["student_ids"].astype(np.int32))
        all_targets.append(d["targets"].astype(np.int32))

    x_all = np.concatenate(all_feats, axis=0)
    s_all = np.concatenate(all_sids, axis=0)
    raw_targets = np.concatenate(all_targets, axis=0)
    y_all = (raw_targets != 2).astype(np.int64)

    val_sids = {2, 5, 8, 14, 16, 18}
    val_mask = np.isin(s_all, list(val_sids))
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    # Reconstruct model and run inference
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 256
    val_probs = []
    with torch.no_grad():
        for start_idx in range(0, len(x_val), batch_size):
            bx = torch.from_numpy(x_val[start_idx : start_idx + batch_size]).to(device)
            p = model.predict_proba(bx)[:, 1].cpu().numpy()
            val_probs.append(p)

    val_probs = np.concatenate(val_probs, axis=0)
    eval_tau_05 = compute_full_metrics(y_val, (val_probs >= 0.50).astype(int), val_probs)
    calibrated_tau = float(ckpt.get("calibrated_threshold", 0.40))
    eval_calibrated = compute_full_metrics(y_val, (val_probs >= calibrated_tau).astype(int), val_probs)

    return {
        "val_students": sorted(list(val_sids)),
        "n_val_sequences": len(y_val),
        "n_train_sequences": len(x_all) - len(y_val),
        "total_sequences": len(x_all),
        "eval_tau_0_50": eval_tau_05,
        "calibrated_threshold": calibrated_tau,
        "eval_calibrated": eval_calibrated,
        "ckpt_saved_calibrated": ckpt.get("calibrated_metrics"),
        "ckpt_saved_default": ckpt.get("default_metrics"),
    }


def main():
    print("=" * 80)
    print("RUNNING REPRODUCIBLE AUDIT CALCULATIONS")
    print("=" * 80)

    print("\n[1/5] Auditing Clip 001...")
    clip_001_std, clip_001_iou = audit_clip_001()

    print("[2/5] Auditing Clip 007...")
    clip_007_std = audit_clip_007()

    print("[3/5] Auditing Video Project 3...")
    vp3_heur = audit_heuristic_video("video_project_3_scores.jsonl")

    print("[4/5] Auditing Test 4...")
    test_4_heur = audit_heuristic_video("test_4_scores.jsonl")

    print("[5/5] Auditing Held-Out Validation Split...")
    val_audit = audit_held_out_validation()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clip_001": {
            "ground_truth_eval_iou_0_30": clip_001_std,
            "iou_sensitivity": clip_001_iou,
        },
        "clip_007": {
            "ground_truth_eval_iou_0_30": clip_007_std,
        },
        "video_project_3": {
            "heuristic_agreement": vp3_heur,
            "ground_truth": None,
            "ground_truth_note": "No human CVAT ground-truth exists for Video Project 3. Evaluated against rule-based heuristic rubric only.",
        },
        "test_4": {
            "heuristic_agreement": test_4_heur,
            "ground_truth": None,
            "ground_truth_note": "No human CVAT ground-truth exists for test 4. Evaluated against rule-based heuristic rubric only.",
        },
        "held_out_validation": val_audit,
    }

    out_json = WORKSPACE_ROOT / "audit" / "METRIC_RECALCULATION.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON metrics audit to: {out_json}")

    # Generate CSV Summary Table
    out_csv = WORKSPACE_ROOT / "audit" / "METRIC_RECALCULATION.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Evaluation_Set",
            "Annotation_Source",
            "N_Samples",
            "Accuracy_Pct",
            "Balanced_Acc_Pct",
            "True_Macro_F1_Pct",
            "Positive_Class_F1_Pct",
            "Negative_Class_F1_Pct",
            "Sensitivity_Pct",
            "Specificity_Pct",
            "TN", "FP", "FN", "TP",
            "ROC_AUC",
            "PR_AUC"
        ])

        def write_row(name, src, m):
            writer.writerow([
                name,
                src,
                m["n_samples"],
                f"{m['accuracy']*100:.2f}",
                f"{m['balanced_accuracy']*100:.2f}",
                f"{m['macro_f1']*100:.2f}",
                f"{m['class_1_high']['f1']*100:.2f}",
                f"{m['class_0_low']['f1']*100:.2f}",
                f"{m['sensitivity']*100:.2f}",
                f"{m['specificity']*100:.2f}",
                m["confusion_matrix"]["TN"],
                m["confusion_matrix"]["FP"],
                m["confusion_matrix"]["FN"],
                m["confusion_matrix"]["TP"],
                m.get("roc_auc") or "N/A",
                m.get("pr_auc") or "N/A",
            ])

        write_row("Clip 001 (IoU >= 0.30)", "User Human CVAT XML", clip_001_std)
        write_row("Clip 007 (IoU >= 0.30)", "Repo CVAT XML + CSV", clip_007_std)
        write_row("Held-Out Validation (tau=0.50)", "Dataset Ground Truth (6 St.)", val_audit["eval_tau_0_50"])
        write_row("Held-Out Validation (tau*=0.50)", "Dataset Ground Truth (6 St. Calibrated)", val_audit["eval_calibrated"])
        write_row("Video Project 3 (Consistency)", "Heuristic Rule Rubric (NO CVAT GT)", vp3_heur)
        write_row("Test 4 (Consistency)", "Heuristic Rule Rubric (NO CVAT GT)", test_4_heur)

    print(f"Saved CSV metrics audit to: {out_csv}")
    print("\nRecalculation finished successfully.")


if __name__ == "__main__":
    main()

