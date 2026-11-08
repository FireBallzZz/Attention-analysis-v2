"""Phase 7 Step G: Final UAP Classroom Evaluation on Frozen Checkpoint.

Evaluates:
- Frozen Public Candidate: checkpoints/candidates/phase7_public_trained_binary_candidate.pt
- Production Baseline:     checkpoints/classroom_manual_binary_model_v2_clean.pt
- Phase 6G-4C Candidate:    checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt

Dataset:
- data/processed/classroom_clip_004_phase6g7a_hardened.npz (5,985 sequences across 21 students)
Thresholds:
- tau = 0.50 (Default)
- tau* = 0.90 (Frozen from public validation)

Outputs:
- runs/evaluation/phase7_public_trained_uap_final_20261023/uap_final_evaluation_metrics.json
- runs/evaluation/phase7_public_trained_uap_final_20261023/confusion_matrices/uap_final_confusion_matrices.txt
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

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

OUT_DIR = WORKSPACE_ROOT / "runs" / "evaluation" / "phase7_public_trained_uap_final_20261023"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def load_model(ckpt_path: Path, device: torch.device) -> ManualBinaryAttentionClassifier:
    base = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs_high: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    brier = float(brier_score_loss(y_true, probs_high)) if probs_high is not None else None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "confusion_matrix": {
            "true_non_attentive": tn,
            "false_attentive_fp": fp,
            "false_non_attentive_fn": fn,
            "true_attentive": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "class_0_non_attentive": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(supp[0]),
        },
        "class_1_attentive": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(supp[1]),
        },
    }


def per_student_breakdown(y_true: np.ndarray, y_pred: np.ndarray, student_ids: np.ndarray) -> list[dict]:
    results = []
    unique_students = np.unique(student_ids)
    for sid in sorted(unique_students):
        mask = (student_ids == sid)
        s_true = y_true[mask]
        s_pred = y_pred[mask]
        acc = float(accuracy_score(s_true, s_pred))
        high_true = int(np.sum(s_true == 1))
        low_true = int(np.sum(s_true == 0))
        high_pred = int(np.sum(s_pred == 1))
        low_pred = int(np.sum(s_pred == 0))
        results.append({
            "student_id": int(sid),
            "evaluated_sequences": len(s_true),
            "ground_truth_high": high_true,
            "ground_truth_low": low_true,
            "predicted_high": high_pred,
            "predicted_low": low_pred,
            "accuracy": acc,
        })
    return results


def main():
    print("=" * 80)
    print(" PHASE 7 STEP G: FINAL UAP CLASSROOM EVALUATION (FROZEN MODELS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Pre-execution Checkpoint Hash Auditing
    cand_p7_path = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase7_public_trained_binary_candidate.pt"
    active_path = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
    cand_6g4c_path = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"

    h_p7 = get_file_sha256(cand_p7_path)
    h_act = get_file_sha256(active_path)
    h_6g = get_file_sha256(cand_6g4c_path)

    print("[Checkpoint Hashes Frozen & Audited]:")
    print(f"  Phase 7 Public Candidate: {h_p7}")
    print(f"  Active Baseline Checkpoint: {h_act}")
    print(f"  Phase 6G-4C Candidate Checkpoint: {h_6g}")

    # 2. Load UAP Evaluation Data
    uap_npz_path = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_phase6g7a_hardened.npz"
    if not uap_npz_path.exists():
        raise FileNotFoundError(f"Missing UAP evaluation dataset: {uap_npz_path}")

    d = np.load(uap_npz_path, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    sids = d["student_ids"].astype(np.int32)
    fids = d["frame_ids"].astype(np.int32)
    targets_binary = d["targets_binary"].astype(np.int64)

    total_seqs = len(feats)
    n_low = int(np.sum(targets_binary == 0))
    n_high = int(np.sum(targets_binary == 1))
    print(f"\nUAP Evaluation Dataset Loaded: {total_seqs:,} sequences across {len(np.unique(sids))} students (Frames 15-299)")
    print(f"Class Distribution: NON-ATTENTIVE (0): {n_low} ({n_low/total_seqs*100:.2f}%), ATTENTIVE (1): {n_high} ({n_high/total_seqs*100:.2f}%)")

    # 3. Model Inference
    model_p7 = load_model(cand_p7_path, device)
    model_act = load_model(active_path, device)
    model_6g = load_model(cand_6g4c_path, device)

    x_uap_t = torch.from_numpy(feats).to(device)

    with torch.no_grad():
        p7_logits = model_p7(x_uap_t)
        p7_probs = torch.softmax(p7_logits, dim=-1).cpu().numpy()
        p7_probs_high = p7_probs[:, 1]

        act_logits = model_act(x_uap_t)
        act_probs = torch.softmax(act_logits, dim=-1).cpu().numpy()
        act_probs_high = act_probs[:, 1]

        p6_logits = model_6g(x_uap_t)
        p6_probs = torch.softmax(p6_logits, dim=-1).cpu().numpy()
        p6_probs_high = p6_probs[:, 1]

    # Metrics at tau = 0.50 and tau* = 0.90
    m_p7_tau50 = compute_metrics(targets_binary, (p7_probs_high >= 0.50).astype(np.int64), p7_probs_high)
    m_p7_tau90 = compute_metrics(targets_binary, (p7_probs_high >= 0.90).astype(np.int64), p7_probs_high)
    m_act_tau50 = compute_metrics(targets_binary, (act_probs_high >= 0.50).astype(np.int64), act_probs_high)
    m_6g_tau50 = compute_metrics(targets_binary, (p6_probs_high >= 0.50).astype(np.int64), p6_probs_high)

    per_student_p7_tau50 = per_student_breakdown(targets_binary, (p7_probs_high >= 0.50).astype(np.int64), sids)

    print("\n" + "=" * 80)
    print(" UAP CLASSROOM BENCHMARK (CLIP 004) PERFORMANCE MATRIX")
    print("=" * 80)
    print(f"{'Model':<35} | {'Tau':<5} | {'Accuracy':<8} | {'Bal-Acc':<8} | {'Macro-F1':<8} | {'Brier':<7} | {'Rec-0':<6} | {'Rec-1':<6}")
    print("-" * 95)
    print(f"{'Phase 7 Public Candidate':<35} | 0.50  | {m_p7_tau50['accuracy']*100:6.2f}% | {m_p7_tau50['balanced_accuracy']*100:6.2f}% | {m_p7_tau50['macro_f1']*100:6.2f}% | {m_p7_tau50['brier_score']:6.4f} | {m_p7_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_p7_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Phase 7 Public Candidate (Calibrated)':<35} | 0.90  | {m_p7_tau90['accuracy']*100:6.2f}% | {m_p7_tau90['balanced_accuracy']*100:6.2f}% | {m_p7_tau90['macro_f1']*100:6.2f}% | {m_p7_tau90['brier_score']:6.4f} | {m_p7_tau90['class_0_non_attentive']['recall']*100:5.1f}% | {m_p7_tau90['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Phase 6G-4C Candidate (Classroom-Trained)':<35} | 0.50  | {m_6g_tau50['accuracy']*100:6.2f}% | {m_6g_tau50['balanced_accuracy']*100:6.2f}% | {m_6g_tau50['macro_f1']*100:6.2f}% | {m_6g_tau50['brier_score']:6.4f} | {m_6g_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_6g_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Active Production Baseline (v2_clean)':<35} | 0.50  | {m_act_tau50['accuracy']*100:6.2f}% | {m_act_tau50['balanced_accuracy']*100:6.2f}% | {m_act_tau50['macro_f1']*100:6.2f}% | {m_act_tau50['brier_score']:6.4f} | {m_act_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_act_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print("=" * 95)

    # 4. Save JSON metrics
    uap_out = {
        "dataset": "UAP Classroom Dataset Clip 004 (Diagnostic Same-Session Benchmark)",
        "source_master_recording": "classroom_01.mp4 (Frames 900-1199, 30fps)",
        "sequences_evaluated": total_seqs,
        "students_count": len(np.unique(sids)),
        "class_distribution": {"non_attentive_0": n_low, "attentive_1": n_high},
        "generalization_status": "SAME_SESSION_BENCHMARK_ONLY (Independent classroom unverified)",
        "hashes": {
            "phase7_public_candidate_sha256": h_p7,
            "active_production_baseline_sha256": h_act,
            "phase6g4c_candidate_sha256": h_6g,
        },
        "models": {
            "phase7_public_candidate_tau50": m_p7_tau50,
            "phase7_public_candidate_tau90": m_p7_tau90,
            "phase6g4c_candidate_tau50": m_6g_tau50,
            "active_production_baseline_tau50": m_act_tau50,
        },
        "per_student_breakdown_phase7_tau50": per_student_p7_tau50,
    }
    with open(OUT_DIR / "uap_final_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(uap_out, f, indent=2)

    # 5. Save Confusion Matrices text file
    cm_txt = f"""=== Final UAP Classroom Benchmark (Clip 004) Confusion Matrices (N = {total_seqs}) ===

1. Phase 7 Public Candidate (tau = 0.50):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_p7_tau50['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_p7_tau50['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_p7_tau50['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_p7_tau50['confusion_matrix']['true_attentive']:4d}

2. Phase 7 Public Candidate (tau* = 0.90):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_p7_tau90['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_p7_tau90['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_p7_tau90['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_p7_tau90['confusion_matrix']['true_attentive']:4d}

3. Phase 6G-4C Candidate (tau = 0.50, Classroom-Trained on Clip 001/008):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_6g_tau50['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_6g_tau50['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_6g_tau50['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_6g_tau50['confusion_matrix']['true_attentive']:4d}

4. Active Production Baseline (v2_clean, tau = 0.50):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_act_tau50['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_act_tau50['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_act_tau50['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_act_tau50['confusion_matrix']['true_attentive']:4d}
"""
    (OUT_DIR / "confusion_matrices" / "uap_final_confusion_matrices.txt").write_text(cm_txt, encoding="utf-8")
    print(f"\nSaved results to {OUT_DIR / 'uap_final_evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
