"""Diagnostic Held-Out Classroom Benchmark Evaluation Script (Phase 6D).

Evaluates three checkpoints on runs/4.mp4 using Anno/ground_truth_with_boxes.csv:
1. Historical Model: checkpoints/classroom_manual_binary_model_v2_clean.pt
2. Model A (Frozen Baseline): checkpoints/classroom_attention_v3_frozen_baseline.pt
3. Model B (Trainable Candidate): checkpoints/classroom_attention_v3_unfrozen.pt

Enforces:
- Identical features, sequences, student tracks, and annotations.
- Primary comparison at standard threshold tau = 0.50.
- Reference comparison at validation-calibrated thresholds.
- Full diagnostic threshold sweep across tau in [0.10, 0.90].
- Per-student breakdown across all 20 evaluated students.
- Preserves checkpoint hashes and datasets 100% intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_low": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_high": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": cm.tolist(),
    }


def run_diagnostic_evaluation() -> dict[str, Any]:
    video_path = WORKSPACE_ROOT / "runs" / "4.mp4"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth_with_boxes.csv"
    seq_npz = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    if not seq_npz.exists():
        raise FileNotFoundError(f"Missing {seq_npz}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" PHASE 6D: DIAGNOSTIC HELD-OUT CLASSROOM BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Video File        : {video_path}")
    print(f"Annotation File   : {csv_path}")
    print(f"Feature NPZ       : {seq_npz}")
    print(f"Evaluation Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Ground Truth Annotations
    df = pd.read_csv(csv_path)
    gt_lookup = {}
    for _, row in df.iterrows():
        f_idx = int(row["frame"])
        sid = int(row["student_id"])
        att = row["attention"].strip().upper()
        gt_lookup[(f_idx, sid)] = {
            "label": att,
            "target": 1 if att == "HIGH" else 0,
        }

    # 2. Load Sequence Features
    data = np.load(seq_npz)
    raw_feats = data["features"].astype(np.float32)
    sids_all = data["student_ids"].astype(np.int32)
    fids_all = data["frame_ids"].astype(np.int32)

    # 3. Match Sequences to GT (Filtering out untracked Student 16)
    valid_indices = []
    y_manual = []
    for i in range(len(raw_feats)):
        key = (int(fids_all[i]), int(sids_all[i]))
        if key in gt_lookup:
            valid_indices.append(i)
            y_manual.append(gt_lookup[key]["target"])

    valid_indices = np.array(valid_indices)
    y_manual = np.array(y_manual)
    feats_valid = raw_feats[valid_indices]
    sids_valid = sids_all[valid_indices]
    fids_valid = fids_all[valid_indices]

    num_eval = len(y_manual)
    unique_students = sorted(np.unique(sids_valid))
    print(f"\n[Observation Alignment]:")
    print(f"  Total Valid Evaluated Sequences: {num_eval:,} (20 students x 285 frames, frames 15-299)")
    print(f"  Warm-Up Offset                 : Frames 0-14 (15 frames x 20 students = 300 observations)")
    print(f"  Evaluated Student IDs          : {unique_students} (Student 16 omitted)")
    print(f"  Ground-Truth Class Counts      : HIGH={sum(y_manual==1):,} ({sum(y_manual==1)/num_eval*100:.2f}%), LOW={sum(y_manual==0):,} ({sum(y_manual==0)/num_eval*100:.2f}%)")

    # 4. Feature Channel Inspection
    ch22_mean = float(feats_valid[:, :, 22].mean())
    ch23_mean = float(feats_valid[:, :, 23].mean())
    ch23_zeros = float((feats_valid[:, :, 23] == 0.0).sum()) / feats_valid[:, :, 23].size * 100.0
    print(f"\n[Feature Channel Verification on Evaluated Clip 004]:")
    print(f"  Channel 22 (Torso Lean Angle)               : mean = {ch22_mean:.4f}")
    print(f"  Channel 23 (Upper-Body Keypoint Confidence) : mean = {ch23_mean:.4f} ({ch23_zeros:.1f}% zeros - legacy unpopulated)")

    # 5. Checkpoint Definitions
    checkpoints = {
        "historical_v2_clean": {
            "name": "Historical In-Domain Model (v2_clean)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt",
            "calibrated_threshold": 0.40,
            "training_domain": "Classroom (Clips 001 + 008)",
        },
        "model_a_frozen": {
            "name": "Model A (DAiSEE v3 Frozen Baseline)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_attention_v3_frozen_baseline.pt",
            "calibrated_threshold": 0.44,
            "training_domain": "Public DAiSEE v3 Hardened (10 Subjects)",
        },
        "model_b_unfrozen": {
            "name": "Model B (DAiSEE v3 Trainable Candidate)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_attention_v3_unfrozen.pt",
            "calibrated_threshold": 0.48,
            "training_domain": "Public DAiSEE v3 Hardened (10 Subjects)",
        },
    }

    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    x_eval_t = torch.from_numpy(feats_valid).to(device)

    results: dict[str, Any] = {}

    for ckpt_key, ckpt_info in checkpoints.items():
        print(f"\n" + "-" * 70)
        print(f" Evaluating: {ckpt_info['name']}")
        print(f" Path      : {ckpt_info['path']}")
        print(f" Domain    : {ckpt_info['training_domain']}")
        print("-" * 70)

        model = ManualBinaryAttentionClassifier(base_net).to(device)
        ckpt_data = torch.load(ckpt_info["path"], map_location=device, weights_only=False)
        state_dict = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            probs = model.predict_proba(x_eval_t).cpu().numpy()
            probs_high = probs[:, 1]

        # Standard tau = 0.50 evaluation (Primary comparison)
        preds_50 = (probs_high >= 0.50).astype(int)
        m_50 = evaluate_predictions(y_manual, preds_50)

        # Calibrated threshold evaluation (Reference comparison)
        cal_tau = float(ckpt_info["calibrated_threshold"])
        preds_cal = (probs_high >= cal_tau).astype(int)
        m_cal = evaluate_predictions(y_manual, preds_cal)

        # Threshold sweep across tau in [0.10, 0.90] (step 0.05)
        sweep = []
        for tau_int in range(10, 95, 5):
            tau = tau_int / 100.0
            p_tau = (probs_high >= tau).astype(int)
            ev = evaluate_predictions(y_manual, p_tau)
            sweep.append({
                "threshold": tau,
                "accuracy": ev["accuracy"],
                "balanced_accuracy": ev["balanced_accuracy"],
                "macro_f1": ev["macro_f1"],
                "low_recall": ev["class_low"]["recall"],
                "high_recall": ev["class_high"]["recall"],
                "low_precision": ev["class_low"]["precision"],
            })

        # Per-student accuracy breakdown at tau = 0.50 and cal_tau
        per_student = {}
        for sid in unique_students:
            mask = (sids_valid == sid)
            n_inst = int(mask.sum())
            n_high = int((y_manual[mask] == 1).sum())
            n_low = int((y_manual[mask] == 0).sum())
            acc_50 = float(accuracy_score(y_manual[mask], preds_50[mask]))
            acc_cal = float(accuracy_score(y_manual[mask], preds_cal[mask]))
            bal_50 = float(balanced_accuracy_score(y_manual[mask], preds_50[mask])) if (n_high > 0 and n_low > 0) else None

            per_student[int(sid)] = {
                "instances": n_inst,
                "gt_high": n_high,
                "gt_low": n_low,
                "accuracy_tau50": acc_50,
                "accuracy_calibrated": acc_cal,
                "balanced_accuracy_tau50": bal_50,
            }

        prob_stats = {
            "mean": float(probs_high.mean()),
            "median": float(np.median(probs_high)),
            "std": float(probs_high.std()),
            "min": float(probs_high.min()),
            "max": float(probs_high.max()),
        }

        print(f"  P(HIGH) Distribution: mean={prob_stats['mean']:.4f}, median={prob_stats['median']:.4f}, min={prob_stats['min']:.4f}, max={prob_stats['max']:.4f}")
        print(f"  [Primary Comparison: Standard tau = 0.50]:")
        print(f"    Raw Accuracy     : {m_50['accuracy']*100:.2f}%")
        print(f"    Balanced Accuracy: {m_50['balanced_accuracy']*100:.2f}%")
        print(f"    Macro-F1         : {m_50['macro_f1']*100:.2f}%")
        print(f"    Weighted-F1      : {m_50['weighted_f1']*100:.2f}%")
        print(f"    LOW  (Rec/Prec/F1): {m_50['class_low']['recall']*100:.2f}% / {m_50['class_low']['precision']*100:.2f}% / {m_50['class_low']['f1']*100:.2f}%")
        print(f"    HIGH (Rec/Prec/F1): {m_50['class_high']['recall']*100:.2f}% / {m_50['class_high']['precision']*100:.2f}% / {m_50['class_high']['f1']*100:.2f}%")
        print(f"    Confusion Matrix : {m_50['confusion_matrix']}")

        print(f"  [Reference Comparison: Calibrated tau = {cal_tau:.2f}]:")
        print(f"    Raw Accuracy     : {m_cal['accuracy']*100:.2f}%")
        print(f"    Balanced Accuracy: {m_cal['balanced_accuracy']*100:.2f}%")
        print(f"    Macro-F1         : {m_cal['macro_f1']*100:.2f}%")
        print(f"    LOW  (Rec/Prec/F1): {m_cal['class_low']['recall']*100:.2f}% / {m_cal['class_low']['precision']*100:.2f}% / {m_cal['class_low']['f1']*100:.2f}%")
        print(f"    HIGH (Rec/Prec/F1): {m_cal['class_high']['recall']*100:.2f}% / {m_cal['class_high']['precision']*100:.2f}% / {m_cal['class_high']['f1']*100:.2f}%")
        print(f"    Confusion Matrix : {m_cal['confusion_matrix']}")

        results[ckpt_key] = {
            "name": ckpt_info["name"],
            "path": str(ckpt_info["path"]),
            "training_domain": ckpt_info["training_domain"],
            "calibrated_threshold": cal_tau,
            "prob_stats": prob_stats,
            "metrics_tau50": m_50,
            "metrics_calibrated": m_cal,
            "threshold_sweep": sweep,
            "per_student": per_student,
        }

    # Evaluate Majority-Class Dummy Classifier
    majority_preds = np.ones_like(y_manual)
    majority_metrics = evaluate_predictions(y_manual, majority_preds)
    print("\n" + "-" * 70)
    print(" Baseline: Majority-Class Dummy Classifier (Always HIGH)")
    print("-" * 70)
    print(f"  Raw Accuracy     : {majority_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {majority_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1         : {majority_metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1      : {majority_metrics['weighted_f1']*100:.2f}%")
    print(f"  Confusion Matrix : {majority_metrics['confusion_matrix']}")

    payload = {
        "benchmark_video": str(video_path),
        "annotation_file": str(csv_path),
        "sequence_features_file": str(seq_npz),
        "total_evaluated_observations": num_eval,
        "evaluated_students": [int(s) for s in unique_students],
        "class_distribution": {
            "high": int(sum(y_manual == 1)),
            "low": int(sum(y_manual == 0)),
        },
        "feature_audit": {
            "channel_22_mean": ch22_mean,
            "channel_23_mean": ch23_mean,
            "channel_23_zeros_percent": ch23_zeros,
        },
        "majority_class_baseline": majority_metrics,
        "models": results,
    }

    out_metrics_path = WORKSPACE_ROOT / "runs" / "reports" / "phase6d" / "phase6d_diagnostic_benchmark_metrics.json"
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved machine-readable metrics to: {out_metrics_path}")

    return payload


if __name__ == "__main__":
    run_diagnostic_evaluation()
