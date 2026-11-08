"""Phase 7 Step F: Evaluation on Untouched Public Held-Out Test Split.

Evaluates:
- New Public Candidate: checkpoints/candidates/phase7_public_trained_binary_candidate.pt
- Production Baseline:  checkpoints/classroom_manual_binary_model_v2_clean.pt
- Phase 6G-4C Candidate: checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt
- Majority Dummy Classifier

Dataset:
- Public DAiSEE v3 Hardened Held-Out Test Split (3,563 sequences, subjects 110001 & 110004, 148 clips)
Thresholds:
- tau = 0.50 (Default)
- tau* = 0.90 (Calibrated on Public Validation Set)
Includes cluster-robust clip-level bootstrap confidence intervals (95% CI).
"""

from __future__ import annotations

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


def compute_clip_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clips: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_clips = np.unique(clips)
    n_clips = len(unique_clips)

    accs = []
    bal_accs = []
    macro_f1s = []

    for _ in range(n_bootstrap):
        sampled_clips = rng.choice(unique_clips, size=n_clips, replace=True)
        # gather indices
        indices = np.concatenate([np.where(clips == c)[0] for c in sampled_clips])
        b_true = y_true[indices]
        b_pred = y_pred[indices]

        # only evaluate if both classes present
        if len(np.unique(b_true)) > 1:
            accs.append(accuracy_score(b_true, b_pred))
            bal_accs.append(balanced_accuracy_score(b_true, b_pred))
            macro_f1s.append(f1_score(b_true, b_pred, average="macro", zero_division=0))

    def get_ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_95_lower": float(np.percentile(arr, 2.5)),
            "ci_95_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "accuracy_ci": get_ci(accs),
        "balanced_accuracy_ci": get_ci(bal_accs),
        "macro_f1_ci": get_ci(macro_f1s),
    }


def main():
    print("=" * 80)
    print(" PHASE 7 STEP F: UNTOUCHED PUBLIC HELD-OUT TEST EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split configuration
    split_path = WORKSPACE_ROOT / "configs" / "phase7_public_daisee_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)

    npz_path = WORKSPACE_ROOT / "data" / "processed" / "fusion_daisee_v3_hardened.npz"
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    scores = d["score"].astype(np.float32)
    subs = d["subject_ids"].astype(str)
    clips = d["clip_ids"].astype(str)
    y_binary = (scores >= 50.0).astype(np.int64)

    test_subs = split_cfg["held_out_test"]["subjects"]
    test_mask = np.isin(subs, test_subs)

    x_test = feats[test_mask]
    y_test = y_binary[test_mask]
    clips_test = clips[test_mask]

    n_test = len(x_test)
    n_low = int(np.sum(y_test == 0))
    n_high = int(np.sum(y_test == 1))
    print(f"Held-Out Test Split Loaded: {n_test:,} sequences across {len(np.unique(clips_test))} clips")
    print(f"Class Distribution: NON-ATTENTIVE (0): {n_low} ({n_low/n_test*100:.2f}%), ATTENTIVE (1): {n_high} ({n_high/n_test*100:.2f}%)")

    # Checkpoint paths
    cand_phase7_path = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase7_public_trained_binary_candidate.pt"
    cand_6g4c_path = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
    active_baseline_path = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"

    # Load models
    model_phase7 = load_model(cand_phase7_path, device)
    model_6g4c = load_model(cand_6g4c_path, device)
    model_active = load_model(active_baseline_path, device)

    x_test_t = torch.from_numpy(x_test).to(device)

    # 1. Inference for Phase 7 Candidate
    with torch.no_grad():
        p7_logits = model_phase7(x_test_t)
        p7_probs = torch.softmax(p7_logits, dim=-1).cpu().numpy()
        p7_probs_high = p7_probs[:, 1]

    # Evaluate at tau = 0.50 and tau* = 0.90
    m_p7_tau50 = compute_metrics(y_test, (p7_probs_high >= 0.50).astype(np.int64), p7_probs_high)
    ci_p7_tau50 = compute_clip_bootstrap_ci(y_test, (p7_probs_high >= 0.50).astype(np.int64), clips_test)

    m_p7_tau90 = compute_metrics(y_test, (p7_probs_high >= 0.90).astype(np.int64), p7_probs_high)
    ci_p7_tau90 = compute_clip_bootstrap_ci(y_test, (p7_probs_high >= 0.90).astype(np.int64), clips_test)

    # 2. Inference for Phase 6G-4C Candidate
    with torch.no_grad():
        p6_logits = model_6g4c(x_test_t)
        p6_probs = torch.softmax(p6_logits, dim=-1).cpu().numpy()
        p6_probs_high = p6_probs[:, 1]
    m_p6_tau50 = compute_metrics(y_test, (p6_probs_high >= 0.50).astype(np.int64), p6_probs_high)

    # 3. Inference for Active Production Baseline
    with torch.no_grad():
        act_logits = model_active(x_test_t)
        act_probs = torch.softmax(act_logits, dim=-1).cpu().numpy()
        act_probs_high = act_probs[:, 1]
    m_act_tau50 = compute_metrics(y_test, (act_probs_high >= 0.50).astype(np.int64), act_probs_high)

    # 4. Majority Dummy Baseline
    maj_preds = np.ones_like(y_test)
    m_maj = compute_metrics(y_test, maj_preds, np.ones_like(y_test, dtype=np.float32))

    print("\n" + "=" * 80)
    print(" PUBLIC HELD-OUT TEST PERFORMANCE COMPARISON MATRIX")
    print("=" * 80)
    print(f"{'Model':<35} | {'Tau':<5} | {'Accuracy':<8} | {'Bal-Acc':<8} | {'Macro-F1':<8} | {'Brier':<7} | {'Rec-0':<6} | {'Rec-1':<6}")
    print("-" * 95)
    print(f"{'Phase 7 Public Candidate (Ours)':<35} | 0.50  | {m_p7_tau50['accuracy']*100:6.2f}% | {m_p7_tau50['balanced_accuracy']*100:6.2f}% | {m_p7_tau50['macro_f1']*100:6.2f}% | {m_p7_tau50['brier_score']:6.4f} | {m_p7_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_p7_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Phase 7 Public Candidate (Calibrated)':<35} | 0.90  | {m_p7_tau90['accuracy']*100:6.2f}% | {m_p7_tau90['balanced_accuracy']*100:6.2f}% | {m_p7_tau90['macro_f1']*100:6.2f}% | {m_p7_tau90['brier_score']:6.4f} | {m_p7_tau90['class_0_non_attentive']['recall']*100:5.1f}% | {m_p7_tau90['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Phase 6G-4C Candidate':<35} | 0.50  | {m_p6_tau50['accuracy']*100:6.2f}% | {m_p6_tau50['balanced_accuracy']*100:6.2f}% | {m_p6_tau50['macro_f1']*100:6.2f}% | {m_p6_tau50['brier_score']:6.4f} | {m_p6_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_p6_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Active Production Baseline (v2_clean)':<35} | 0.50  | {m_act_tau50['accuracy']*100:6.2f}% | {m_act_tau50['balanced_accuracy']*100:6.2f}% | {m_act_tau50['macro_f1']*100:6.2f}% | {m_act_tau50['brier_score']:6.4f} | {m_act_tau50['class_0_non_attentive']['recall']*100:5.1f}% | {m_act_tau50['class_1_attentive']['recall']*100:5.1f}%")
    print(f"{'Majority Class Dummy Baseline':<35} | 0.50  | {m_maj['accuracy']*100:6.2f}% | {m_maj['balanced_accuracy']*100:6.2f}% | {m_maj['macro_f1']*100:6.2f}% | N/A     | {m_maj['class_0_non_attentive']['recall']*100:5.1f}% | {m_maj['class_1_attentive']['recall']*100:5.1f}%")
    print("=" * 95)

    print("\n[Bootstrap 95% Confidence Intervals (1,000 cluster resamples by clip)]:")
    print(f"  Phase 7 (tau=0.50): Accuracy 95% CI: [{ci_p7_tau50['accuracy_ci']['ci_95_lower']*100:.2f}%, {ci_p7_tau50['accuracy_ci']['ci_95_upper']*100:.2f}%]")
    print(f"  Phase 7 (tau=0.50): Bal-Acc  95% CI: [{ci_p7_tau50['balanced_accuracy_ci']['ci_95_lower']*100:.2f}%, {ci_p7_tau50['balanced_accuracy_ci']['ci_95_upper']*100:.2f}%]")
    print(f"  Phase 7 (tau=0.50): Macro-F1 95% CI: [{ci_p7_tau50['macro_f1_ci']['ci_95_lower']*100:.2f}%, {ci_p7_tau50['macro_f1_ci']['ci_95_upper']*100:.2f}%]")

    # Export public_test_metrics.json
    test_out = {
        "dataset": "DAiSEE v3 Hardened Untouched Held-Out Test Split",
        "held_out_subjects": test_subs,
        "sequences_evaluated": n_test,
        "clips_count": len(np.unique(clips_test)),
        "class_distribution": {"non_attentive_0": n_low, "attentive_1": n_high},
        "models": {
            "phase7_public_candidate_tau50": {
                "metrics": m_p7_tau50,
                "confidence_intervals": ci_p7_tau50,
            },
            "phase7_public_candidate_tau90": {
                "metrics": m_p7_tau90,
                "confidence_intervals": ci_p7_tau90,
            },
            "phase6g4c_candidate_tau50": {
                "metrics": m_p6_tau50,
            },
            "active_production_baseline_tau50": {
                "metrics": m_act_tau50,
            },
            "majority_dummy_baseline": {
                "metrics": m_maj,
            },
        },
    }
    with open(OUT_DIR / "public_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_out, f, indent=2)

    # Save confusion matrix text artifact
    cm_txt = f"""=== Public Held-Out Test Confusion Matrices (N = {n_test}) ===

1. Phase 7 Candidate (tau = 0.50):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_p7_tau50['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_p7_tau50['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_p7_tau50['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_p7_tau50['confusion_matrix']['true_attentive']:4d}

2. Phase 7 Candidate (tau* = 0.90):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_p7_tau90['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_p7_tau90['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_p7_tau90['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_p7_tau90['confusion_matrix']['true_attentive']:4d}

3. Active Production Baseline (v2_clean, tau = 0.50):
Matrix (Rows=True, Cols=Pred):
  True NON-ATTENTIVE: {m_act_tau50['confusion_matrix']['true_non_attentive']:4d} | False ATTENTIVE (FP): {m_act_tau50['confusion_matrix']['false_attentive_fp']:4d}
  False NON-ATT (FN): {m_act_tau50['confusion_matrix']['false_non_attentive_fn']:4d} | True ATTENTIVE:       {m_act_tau50['confusion_matrix']['true_attentive']:4d}
"""
    (OUT_DIR / "confusion_matrices" / "public_test_confusion_matrices.txt").write_text(cm_txt, encoding="utf-8")
    print(f"\nSaved results to {OUT_DIR / 'public_test_metrics.json'}")


if __name__ == "__main__":
    main()
