"""Phase 6G-7A: Clip 004 Same-Session Diagnostic Evaluation.

Evaluates:
- Primary Candidate: checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt
- Active Baseline:   checkpoints/classroom_manual_binary_model_v2_clean.pt
Dataset:
- data/processed/classroom_clip_004_phase6g7a_hardened.npz (5,985 sequences)

Thresholds: 0.50 and 0.70.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
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

CANDIDATE_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
ACTIVE_PATH = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
DATA_PATH = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_phase6g7a_hardened.npz"
OUT_DIR = WORKSPACE_ROOT / "runs" / "evaluations" / "phase6g7a_clip004_diagnostic"

EXPECTED_CANDIDATE_SHA256 = "E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs_eng: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    brier = float(brier_score_loss(y_true, probs_eng))

    return {
        "confusion_matrix": {
            "true_disengaged_pred_disengaged": tn,
            "actual_disengaged_pred_engaged_fp": fp,
            "actual_engaged_pred_disengaged_fn": fn,
            "true_engaged_pred_engaged": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "class_0_disengaged": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(supp[0]),
        },
        "class_1_engaged": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(supp[1]),
        },
    }


def main():
    print("=" * 80)
    print(" PHASE 6G-7A: CLIP 004 SAME-SESSION DIAGNOSTIC EVALUATION")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pre-execution Checkpoint Verification
    cand_sha_pre = get_sha256(CANDIDATE_PATH)
    active_sha_pre = get_sha256(ACTIVE_PATH)

    print(f"Pre-Execution Candidate SHA-256: {cand_sha_pre}")
    print(f"Pre-Execution Active SHA-256:    {active_sha_pre}")
    assert cand_sha_pre == EXPECTED_CANDIDATE_SHA256, f"Candidate hash mismatch: {cand_sha_pre}"
    assert active_sha_pre == EXPECTED_ACTIVE_SHA256, f"Active hash mismatch: {active_sha_pre}"

    # 2. Load Dataset
    print(f"\nLoading dataset: {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)
    feats = data["features"].astype(np.float32)  # (5985, 16, 32)
    sids = data["student_ids"].astype(np.int32)
    fids = data["frame_ids"].astype(np.int32)
    traw = data["targets_raw"].astype(np.int32)
    y_true = data["targets_binary"].astype(np.int64)

    n_samples = len(feats)
    print(f"Loaded {n_samples} sequence observations across {len(np.unique(sids))} students.")
    assert n_samples == 5985, f"Expected 5985 observations, got {n_samples}"
    assert int(np.sum(y_true == 0)) == 1043, f"Expected 1043 DISENGAGED"
    assert int(np.sum(y_true == 1)) == 4942, f"Expected 4942 ENGAGED"

    # Evaluation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    x_tensor = torch.from_numpy(feats).to(device)

    # Function to get model probabilities
    def get_probs(ckpt_path: Path) -> np.ndarray:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base = AttentionFusionNet(feat_dim=32, hidden=64)
        model = ManualBinaryAttentionClassifier(base).to(device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(x_tensor)
            return probs.cpu().numpy()

    # 3. Model Probabilities
    print("\nRunning inference for Candidate model...")
    cand_probs = get_probs(CANDIDATE_PATH)
    cand_p_eng = cand_probs[:, 1]

    print("Running inference for Active Production baseline...")
    active_probs = get_probs(ACTIVE_PATH)
    active_p_eng = active_probs[:, 1]

    thresholds = [0.50, 0.70]
    results = {
        "evaluation_title": "Same-session annotated diagnostic evaluation — Clip 004",
        "evaluation_scope": "Same-session diagnostic evaluation on Clip 004 (frames 900-1199 of classroom_01.mp4). Not independent generalization.",
        "video_path": str(WORKSPACE_ROOT / "runs" / "4.mp4"),
        "annotation_xml": str(WORKSPACE_ROOT / "Anno" / "annotations.xml"),
        "annotation_csv": str(WORKSPACE_ROOT / "Anno" / "ground_truth.csv"),
        "dataset_path": str(DATA_PATH),
        "total_observations": n_samples,
        "matched_observations": n_samples,
        "unmatched_observations": 0,
        "excluded_observations": 0,
        "ground_truth_counts": {
            "DISENGAGED_0": int(np.sum(y_true == 0)),
            "ENGAGED_1": int(np.sum(y_true == 1)),
        },
        "candidate": {
            "name": "Phase 6G-4C Weighted Cross-Entropy Candidate",
            "checkpoint": str(CANDIDATE_PATH),
            "sha256_pre": cand_sha_pre,
            "eval_metrics": {},
        },
        "active_baseline": {
            "name": "Active Production Baseline (v2_clean)",
            "checkpoint": str(ACTIVE_PATH),
            "sha256_pre": active_sha_pre,
            "eval_metrics": {},
        },
    }

    print("\n" + "=" * 80)
    print(" RESULTS SUMMARY AT THRESHOLDS 0.50 AND 0.70")
    print("=" * 80)

    for tau in thresholds:
        print(f"\n--- EVALUATION AT THRESHOLD tau = {tau:.2f} ---")
        # Candidate
        c_pred = (cand_p_eng >= tau).astype(int)
        c_metrics = compute_metrics(y_true, c_pred, cand_p_eng)
        results["candidate"]["eval_metrics"][f"tau_{tau:.2f}"] = c_metrics

        # Active
        a_pred = (active_p_eng >= tau).astype(int)
        a_metrics = compute_metrics(y_true, a_pred, active_p_eng)
        results["active_baseline"]["eval_metrics"][f"tau_{tau:.2f}"] = a_metrics

        print(f"Candidate Model (Weighted CE):")
        print(f"  Confusion Matrix: TN={c_metrics['confusion_matrix']['true_disengaged_pred_disengaged']}, FP={c_metrics['confusion_matrix']['actual_disengaged_pred_engaged_fp']}, FN={c_metrics['confusion_matrix']['actual_engaged_pred_disengaged_fn']}, TP={c_metrics['confusion_matrix']['true_engaged_pred_engaged']}")
        print(f"  Accuracy:         {c_metrics['accuracy']*100:.2f}%")
        print(f"  Balanced Acc:     {c_metrics['balanced_accuracy']*100:.2f}%")
        print(f"  Macro F1:         {c_metrics['macro_f1']*100:.2f}%")
        print(f"  DISENGAGED (0):   P={c_metrics['class_0_disengaged']['precision']*100:.2f}%, R={c_metrics['class_0_disengaged']['recall']*100:.2f}%, F1={c_metrics['class_0_disengaged']['f1']*100:.2f}%")
        print(f"  ENGAGED (1):      P={c_metrics['class_1_engaged']['precision']*100:.2f}%, R={c_metrics['class_1_engaged']['recall']*100:.2f}%, F1={c_metrics['class_1_engaged']['f1']*100:.2f}%")
        print(f"  Brier Score:      {c_metrics['brier_score']:.4f}")

        print(f"\nActive Production Baseline (v2_clean):")
        print(f"  Confusion Matrix: TN={a_metrics['confusion_matrix']['true_disengaged_pred_disengaged']}, FP={a_metrics['confusion_matrix']['actual_disengaged_pred_engaged_fp']}, FN={a_metrics['confusion_matrix']['actual_engaged_pred_disengaged_fn']}, TP={a_metrics['confusion_matrix']['true_engaged_pred_engaged']}")
        print(f"  Accuracy:         {a_metrics['accuracy']*100:.2f}%")
        print(f"  Balanced Acc:     {a_metrics['balanced_accuracy']*100:.2f}%")
        print(f"  Macro F1:         {a_metrics['macro_f1']*100:.2f}%")
        print(f"  DISENGAGED (0):   P={a_metrics['class_0_disengaged']['precision']*100:.2f}%, R={a_metrics['class_0_disengaged']['recall']*100:.2f}%, F1={a_metrics['class_0_disengaged']['f1']*100:.2f}%")
        print(f"  ENGAGED (1):      P={a_metrics['class_1_engaged']['precision']*100:.2f}%, R={a_metrics['class_1_engaged']['recall']*100:.2f}%, F1={a_metrics['class_1_engaged']['f1']*100:.2f}%")
        print(f"  Brier Score:      {a_metrics['brier_score']:.4f}")

    # 4. Post-execution Checkpoint Verification
    cand_sha_post = get_sha256(CANDIDATE_PATH)
    active_sha_post = get_sha256(ACTIVE_PATH)
    print(f"\nPost-Execution Candidate SHA-256: {cand_sha_post}")
    print(f"Post-Execution Active SHA-256:    {active_sha_post}")
    assert cand_sha_post == EXPECTED_CANDIDATE_SHA256
    assert active_sha_post == EXPECTED_ACTIVE_SHA256

    results["candidate"]["sha256_post"] = cand_sha_post
    results["active_baseline"]["sha256_post"] = active_sha_post
    results["checkpoints_invariant"] = True

    # 5. Export JSON metrics
    json_path = OUT_DIR / "phase6g7a_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")


if __name__ == "__main__":
    main()
