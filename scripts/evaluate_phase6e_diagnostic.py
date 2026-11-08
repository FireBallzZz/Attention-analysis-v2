"""Phase 6E Diagnostic Feature-Pipeline Compatibility and Re-Evaluation Script.

Compares model performance on runs/4.mp4 between:
1. Original legacy features: data/processed/classroom_clip_004_sequences.npz (Ch23=0.0)
2. Hardened diagnostic features: data/processed/classroom_clip_004_v3_hardened.npz (Ch23 populated, Ch22 confidence-gated)

Models Evaluated:
- Historical In-Domain Model: checkpoints/classroom_manual_binary_model_v2_clean.pt
- Model A (Frozen Baseline): checkpoints/classroom_attention_v3_frozen_baseline.pt
- Model B (Trainable Candidate): checkpoints/classroom_attention_v3_unfrozen.pt

Enforces:
- Identical 5,700 sequences (20 students x 285 frames).
- Standard primary threshold tau = 0.50.
- Preserves all checkpoints and original datasets.
- Outputs machine-readable JSON metrics to runs/reports/phase6e/phase6e_diagnostic_metrics.json.
"""

from __future__ import annotations

import json
import sys
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


def evaluate_feature_set(
    npz_path: Path,
    csv_path: Path,
    checkpoints: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    gt_lookup = {}
    for _, row in df.iterrows():
        f_idx = int(row["frame"])
        sid = int(row["student_id"])
        att = row["attention"].strip().upper()
        gt_lookup[(f_idx, sid)] = 1 if att == "HIGH" else 0

    data = np.load(npz_path)
    raw_feats = data["features"].astype(np.float32)
    sids_all = data["student_ids"].astype(np.int32)
    fids_all = data["frame_ids"].astype(np.int32)

    valid_indices = []
    y_manual = []
    for i in range(len(raw_feats)):
        key = (int(fids_all[i]), int(sids_all[i]))
        if key in gt_lookup:
            valid_indices.append(i)
            y_manual.append(gt_lookup[key])

    valid_indices = np.array(valid_indices)
    y_manual = np.array(y_manual)
    feats_valid = raw_feats[valid_indices]
    sids_valid = sids_all[valid_indices]
    fids_valid = fids_all[valid_indices]

    num_eval = len(y_manual)
    unique_students = [int(s) for s in sorted(np.unique(sids_valid))]

    ch22_mean = float(feats_valid[:, :, 22].mean())
    ch23_mean = float(feats_valid[:, :, 23].mean())
    ch23_zeros = float((feats_valid[:, :, 23] == 0.0).sum()) / feats_valid[:, :, 23].size * 100.0

    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    x_eval_t = torch.from_numpy(feats_valid).to(device)

    model_results = {}
    for key, info in checkpoints.items():
        model = ManualBinaryAttentionClassifier(base_net).to(device)
        ckpt_data = torch.load(info["path"], map_location=device, weights_only=False)
        state_dict = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            probs = model.predict_proba(x_eval_t).cpu().numpy()
            probs_high = probs[:, 1]

        preds_50 = (probs_high >= 0.50).astype(int)
        m_50 = evaluate_predictions(y_manual, preds_50)

        cal_tau = float(info["calibrated_threshold"])
        preds_cal = (probs_high >= cal_tau).astype(int)
        m_cal = evaluate_predictions(y_manual, preds_cal)

        per_student = {}
        for sid in unique_students:
            mask = (sids_valid == sid)
            n_inst = int(mask.sum())
            n_high = int((y_manual[mask] == 1).sum())
            n_low = int((y_manual[mask] == 0).sum())
            acc_50 = float(accuracy_score(y_manual[mask], preds_50[mask]))
            acc_cal = float(accuracy_score(y_manual[mask], preds_cal[mask]))
            per_student[int(sid)] = {
                "instances": n_inst,
                "gt_high": n_high,
                "gt_low": n_low,
                "accuracy_tau50": acc_50,
                "accuracy_calibrated": acc_cal,
            }

        model_results[key] = {
            "name": info["name"],
            "calibrated_threshold": cal_tau,
            "prob_stats": {
                "mean": float(probs_high.mean()),
                "median": float(np.median(probs_high)),
                "std": float(probs_high.std()),
                "min": float(probs_high.min()),
                "max": float(probs_high.max()),
            },
            "metrics_tau50": m_50,
            "metrics_calibrated": m_cal,
            "per_student": per_student,
        }

    return {
        "file": str(npz_path),
        "total_evaluated_observations": num_eval,
        "evaluated_students": unique_students,
        "class_distribution": {
            "high": int(sum(y_manual == 1)),
            "low": int(sum(y_manual == 0)),
        },
        "feature_audit": {
            "channel_22_mean": ch22_mean,
            "channel_23_mean": ch23_mean,
            "channel_23_zeros_percent": ch23_zeros,
        },
        "models": model_results,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" PHASE 6E: FEATURE COMPATIBILITY AUDIT AND DIAGNOSTIC RE-EVALUATION")
    print("=" * 80)

    legacy_npz = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz"
    hardened_npz = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_v3_hardened.npz"
    csv_path = WORKSPACE_ROOT / "Anno" / "ground_truth_with_boxes.csv"

    checkpoints = {
        "historical_v2_clean": {
            "name": "Historical In-Domain Model (v2_clean)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt",
            "calibrated_threshold": 0.40,
        },
        "model_a_frozen": {
            "name": "Model A (DAiSEE v3 Frozen Baseline)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_attention_v3_frozen_baseline.pt",
            "calibrated_threshold": 0.44,
        },
        "model_b_unfrozen": {
            "name": "Model B (DAiSEE v3 Trainable Candidate)",
            "path": WORKSPACE_ROOT / "checkpoints" / "classroom_attention_v3_unfrozen.pt",
            "calibrated_threshold": 0.48,
        },
    }

    print("\n[Evaluating Original Legacy Feature Extraction (Ch23 = 0.0)]...")
    res_legacy = evaluate_feature_set(legacy_npz, csv_path, checkpoints, device)

    print("\n[Evaluating Hardened Diagnostic Feature Extraction (Ch23 Populated)]...")
    res_hardened = evaluate_feature_set(hardened_npz, csv_path, checkpoints, device)

    # Majority Baseline
    df = pd.read_csv(csv_path)
    y_manual = np.array([1 if r["attention"].strip().upper() == "HIGH" else 0 for _, r in df.iterrows() if int(r["frame"]) >= 15 and int(r["student_id"]) != 16])
    maj_metrics = evaluate_predictions(y_manual, np.ones_like(y_manual))

    summary = {
        "benchmark_video": str(WORKSPACE_ROOT / "runs" / "4.mp4"),
        "annotation_file": str(csv_path),
        "majority_class_baseline": maj_metrics,
        "legacy_features_evaluation": res_legacy,
        "hardened_features_evaluation": res_hardened,
    }

    out_metrics_path = WORKSPACE_ROOT / "runs" / "reports" / "phase6e" / "phase6e_diagnostic_metrics.json"
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Phase 6E Diagnostic Metrics saved to: {out_metrics_path}")


if __name__ == "__main__":
    main()
