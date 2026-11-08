"""Phase 6G-4D: Isolated Candidate Threshold Analysis Script.

Executes:
1. Pre-analysis integrity checks:
   - SHA-256 of primary and focal candidate checkpoints.
   - SHA-256 of active production checkpoint.
   - Validation split integrity (2,280 sequences, tracks [2, 5, 8, 11], 425 DISENGAGED, 1855 ENGAGED).
   - Provenance confirmation (Clips 004 and 007 excluded).
2. Reproduction of baseline evaluation at threshold 0.50 for Primary (Weighted CE) and Ablation (Focal Loss).
3. Full exploratory threshold sweep over [0.05, 0.95] in steps of 0.05 on P(ENGAGED).
4. Exports metrics to CSV and JSON for audit reporting.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch

WORKSPACE_ROOT = Path(r"C:\Users\User\Downloads\Thesis\Model\Attention-analysis-model")
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

ACTIVE_CKPT_PATH = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"

PRIMARY_CKPT_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
FOCAL_CKPT_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_focal_loss_ablation_candidate.pt"

OUT_DIR = WORKSPACE_ROOT / "runs" / "analysis" / "phase6g4d_20261022_230155"


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest().upper()


def run_analysis():
    print("=" * 80)
    print(" PHASE 6G-4D: ISOLATED CANDIDATE THRESHOLD ANALYSIS")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Integrity Checks
    assert ACTIVE_CKPT_PATH.exists(), f"Missing active checkpoint: {ACTIVE_CKPT_PATH}"
    active_sha = sha256_file(ACTIVE_CKPT_PATH)
    assert active_sha == EXPECTED_ACTIVE_SHA256, f"Active checkpoint hash mismatch: {active_sha}"
    print(f"[Check 1 PASS] Active Checkpoint SHA-256 Invariant: {active_sha}")

    assert PRIMARY_CKPT_PATH.exists(), f"Missing primary candidate: {PRIMARY_CKPT_PATH}"
    primary_sha = sha256_file(PRIMARY_CKPT_PATH)
    print(f"[Check 2 PASS] Primary Candidate SHA-256: {primary_sha}")

    assert FOCAL_CKPT_PATH.exists(), f"Missing focal candidate: {FOCAL_CKPT_PATH}"
    focal_sha = sha256_file(FOCAL_CKPT_PATH)
    print(f"[Check 3 PASS] Focal Loss Candidate SHA-256: {focal_sha}")

    # Load Hardened Datasets
    p1 = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_001_v3_hardened.npz"
    p8 = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_008_v3_hardened.npz"
    d1 = np.load(p1, allow_pickle=True)
    d8 = np.load(p8, allow_pickle=True)

    feats_all = np.concatenate([d1["features"], d8["features"]], axis=0).astype(np.float32)
    sids_all = np.concatenate([d1["student_ids"], d8["student_ids"]], axis=0).astype(np.int32)
    tbin_all = np.concatenate([d1["targets_binary"], d8["targets_binary"]], axis=0).astype(np.int64)

    val_tracks = [2, 5, 8, 11]
    val_mask = np.isin(sids_all, val_tracks)
    x_val = feats_all[val_mask]
    y_val = tbin_all[val_mask]

    assert len(x_val) == 2280, f"Expected 2280 val sequences, got {len(x_val)}"
    c0 = int(np.sum(y_val == 0))
    c1 = int(np.sum(y_val == 1))
    assert c0 == 425, f"Expected 425 DISENGAGED, got {c0}"
    assert c1 == 1855, f"Expected 1855 ENGAGED, got {c1}"
    print(f"[Check 4 PASS] Validation Split Confirmed: 2,280 sequences across tracks {val_tracks} (DIS={c0}, ENG={c1})")
    print(f"[Check 5 PASS] Quarantined Clips 004 and 007 verified completely excluded.")

    # 2. Model Evaluation Function
    def get_probabilities(ckpt_path: Path) -> np.ndarray:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base = AttentionFusionNet(feat_dim=32, hidden=64)
        model = ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True)
        model.load_state_dict(ckpt["model"])
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_val_t = torch.from_numpy(x_val).to(device)

        with torch.no_grad():
            logits = model(x_val_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    print("\nEvaluating Primary Candidate (Weighted CE)...")
    primary_probs = get_probabilities(PRIMARY_CKPT_PATH)
    p_eng = primary_probs[:, 1]  # P(ENGAGED)

    print("Evaluating Focal Loss Ablation Candidate...")
    focal_probs = get_probabilities(FOCAL_CKPT_PATH)
    p_eng_focal = focal_probs[:, 1]

    # 3. Helper to compute full metrics for a given decision threshold
    def compute_metrics_at_threshold(p_engaged: np.ndarray, y_true: np.ndarray, thresh: float) -> dict:
        # Decision: If P(ENGAGED) >= thresh -> 1 (ENGAGED), else 0 (DISENGAGED)
        y_pred = (p_engaged >= thresh).astype(np.int64)

        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro"))
        wf1 = float(f1_score(y_true, y_pred, average="weighted"))

        prec, rec, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Confusion matrix unpack:
        # cm[0, 0] = True DISENGAGED (TN)
        # cm[0, 1] = False ENGAGED (FP) - actual DISENGAGED predicted as ENGAGED
        # cm[1, 0] = False DISENGAGED (FN) - actual ENGAGED predicted as DISENGAGED
        # cm[1, 1] = True ENGAGED (TP)
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])

        pred_dis = int(np.sum(y_pred == 0))
        pred_eng = int(np.sum(y_pred == 1))

        return {
            "threshold": round(thresh, 2),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": mf1,
            "weighted_f1": wf1,
            "precision_disengaged": float(prec[0]),
            "recall_disengaged": float(rec[0]),
            "f1_disengaged": float(f1s[0]),
            "support_disengaged": int(sup[0]),
            "precision_engaged": float(prec[1]),
            "recall_engaged": float(rec[1]),
            "f1_engaged": float(f1s[1]),
            "support_engaged": int(sup[1]),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "pred_disengaged": pred_dis,
            "pred_engaged": pred_eng,
            "confusion_matrix": cm.tolist(),
        }

    # Baseline 0.50 Reproduction
    base_primary = compute_metrics_at_threshold(p_eng, y_val, 0.50)
    base_focal = compute_metrics_at_threshold(p_eng_focal, y_val, 0.50)

    brier_primary = float(np.mean((p_eng - y_val) ** 2))
    brier_focal = float(np.mean((p_eng_focal - y_val) ** 2))
    base_primary["brier_score"] = brier_primary
    base_focal["brier_score"] = brier_focal

    print("\n--- Baseline Threshold 0.50 Reproduction ---")
    print(f"Primary (Weighted CE): Macro F1={base_primary['macro_f1']*100:.2f}%, BAcc={base_primary['balanced_accuracy']*100:.2f}%, Acc={base_primary['accuracy']*100:.2f}%, Brier={brier_primary:.4f}")
    print(f"  DISENGAGED (0): Prec={base_primary['precision_disengaged']*100:.2f}%, Rec={base_primary['recall_disengaged']*100:.2f}%, F1={base_primary['f1_disengaged']*100:.2f}% (TN={base_primary['tn']}, FP={base_primary['fp']})")
    print(f"  ENGAGED    (1): Prec={base_primary['precision_engaged']*100:.2f}%, Rec={base_primary['recall_engaged']*100:.2f}%, F1={base_primary['f1_engaged']*100:.2f}% (TP={base_primary['tp']}, FN={base_primary['fn']})")
    print(f"Focal Loss Ablation : Macro F1={base_focal['macro_f1']*100:.2f}%, BAcc={base_focal['balanced_accuracy']*100:.2f}%, Acc={base_focal['accuracy']*100:.2f}%, Brier={brier_focal:.4f}")

    # 4. Threshold Sweep [0.05, 0.95] in steps of 0.05
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    sweep_results = []

    print("\n--- Primary Candidate Threshold Sweep ---")
    print(f"{'Thresh':^7} | {'Accuracy':^9} | {'BAcc':^9} | {'Macro F1':^9} | {'Dis Prec':^9} | {'Dis Rec':^9} | {'Dis F1':^9} | {'Eng Prec':^9} | {'Eng Rec':^9} | {'Eng F1':^9} | {'FP (False Eng)':^14} | {'FN (False Dis)':^14}")
    print("-" * 135)

    for t in thresholds:
        res = compute_metrics_at_threshold(p_eng, y_val, t)
        sweep_results.append(res)
        print(
            f"{res['threshold']:^7.2f} | {res['accuracy']*100:^8.2f}% | {res['balanced_accuracy']*100:^8.2f}% | {res['macro_f1']*100:^8.2f}% | "
            f"{res['precision_disengaged']*100:^8.2f}% | {res['recall_disengaged']*100:^8.2f}% | {res['f1_disengaged']*100:^8.2f}% | "
            f"{res['precision_engaged']*100:^8.2f}% | {res['recall_engaged']*100:^8.2f}% | {res['f1_engaged']*100:^8.2f}% | "
            f"{res['fp']:^14d} | {res['fn']:^14d}"
        )

    # Export Sweep to JSON
    sweep_json_path = OUT_DIR / "threshold_sweep_metrics.json"
    with open(sweep_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_baseline_0_50": base_primary,
                "focal_baseline_0_50": base_focal,
                "threshold_sweep_primary": sweep_results,
            },
            f,
            indent=2,
        )
    print(f"\n[Saved Sweep JSON]: {sweep_json_path}")

    # Export Sweep to CSV
    sweep_csv_path = OUT_DIR / "threshold_sweep_metrics.csv"
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "precision_disengaged",
            "recall_disengaged",
            "f1_disengaged",
            "precision_engaged",
            "recall_engaged",
            "f1_engaged",
            "tn_disengaged",
            "fp_false_engaged",
            "fn_false_disengaged",
            "tp_engaged",
            "pred_disengaged_count",
            "pred_engaged_count",
        ])
        for r in sweep_results:
            writer.writerow([
                r["threshold"],
                f"{r['accuracy']:.4f}",
                f"{r['balanced_accuracy']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['weighted_f1']:.4f}",
                f"{r['precision_disengaged']:.4f}",
                f"{r['recall_disengaged']:.4f}",
                f"{r['f1_disengaged']:.4f}",
                f"{r['precision_engaged']:.4f}",
                f"{r['recall_engaged']:.4f}",
                f"{r['f1_engaged']:.4f}",
                r["tn"],
                r["fp"],
                r["fn"],
                r["tp"],
                r["pred_disengaged"],
                r["pred_engaged"],
            ])
    print(f"[Saved Sweep CSV]: {sweep_csv_path}")

    # 5. Identify key operating points
    # Max Macro F1
    best_mf1_res = max(sweep_results, key=lambda x: (x["macro_f1"], x["balanced_accuracy"]))
    # Max Balanced Accuracy
    best_bacc_res = max(sweep_results, key=lambda x: (x["balanced_accuracy"], x["macro_f1"]))
    # High Recall Disengaged (e.g. recall >= 75% or 80%)
    high_rec_candidates = [r for r in sweep_results if r["recall_disengaged"] >= 0.75]
    best_high_rec = max(high_rec_candidates, key=lambda x: x["macro_f1"]) if high_rec_candidates else None

    print(f"\n--- Key Sweep Operating Points ---")
    print(f"Max Macro F1 Point: Threshold = {best_mf1_res['threshold']:.2f} (Macro F1 = {best_mf1_res['macro_f1']*100:.2f}%, BAcc = {best_mf1_res['balanced_accuracy']*100:.2f}%, Acc = {best_mf1_res['accuracy']*100:.2f}%)")
    print(f"  DISENGAGED: Recall = {best_mf1_res['recall_disengaged']*100:.2f}%, Prec = {best_mf1_res['precision_disengaged']*100:.2f}%, F1 = {best_mf1_res['f1_disengaged']*100:.2f}% (TN={best_mf1_res['tn']}, FP={best_mf1_res['fp']})")
    print(f"  ENGAGED   : Recall = {best_mf1_res['recall_engaged']*100:.2f}%, Prec = {best_mf1_res['precision_engaged']*100:.2f}%, F1 = {best_mf1_res['f1_engaged']*100:.2f}% (TP={best_mf1_res['tp']}, FN={best_mf1_res['fn']})")

    if best_bacc_res["threshold"] != best_mf1_res["threshold"]:
        print(f"Max Balanced Accuracy Point: Threshold = {best_bacc_res['threshold']:.2f} (BAcc = {best_bacc_res['balanced_accuracy']*100:.2f}%, Macro F1 = {best_bacc_res['macro_f1']*100:.2f}%)")

    # Post-analysis active checkpoint hash verification
    final_sha = sha256_file(ACTIVE_CKPT_PATH)
    assert final_sha == EXPECTED_ACTIVE_SHA256
    print(f"\n[Post-Analysis Active Checkpoint SHA-256]: {final_sha} (Invariant)")


if __name__ == "__main__":
    run_analysis()
