"""Controlled Retraining Script for Phase 6C Step 3: DAiSEE v3 Hardened Models.

Trains two models on the public DAiSEE v3 hardened dataset:
- Model A: Frozen Frame Encoder Baseline (freeze_frame_encoder=True)
- Model B: Trainable Frame Encoder Candidate (freeze_frame_encoder=False, differential LRs)

Enforces:
1. Strict Stratified Subject-Disjoint Partition (configs/daisee_v3_subject_split.json).
2. Pinned random seeds (seed=42) for deterministic reproducibility.
3. Class-weighted Cross-Entropy loss computed strictly from training-only counts.
4. Early stopping (patience=5) monitored on Validation Balanced Accuracy.
5. Optimal decision threshold calibration tau* in [0.10, 0.90] on Validation Macro-F1.
6. Absolute isolation: zero access to UAP classroom recordings or annotations.
7. Preserves historical checkpoints (v1.pt, v2_clean.pt) completely intact.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


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


def calibrate_threshold(y_true: np.ndarray, probs_high: np.ndarray) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Calibrate decision threshold tau in [0.10, 0.90] to maximize Validation Macro-F1."""
    best_tau = 0.50
    best_macro_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    sweep_results: list[dict[str, Any]] = []

    for tau_int in range(10, 91, 2):
        tau = tau_int / 100.0
        preds = (probs_high >= tau).astype(np.int64)
        m = evaluate_predictions(y_true, preds)
        sweep_results.append({
            "threshold": tau,
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "accuracy": m["accuracy"],
            "low_recall": m["class_low"]["recall"],
            "low_precision": m["class_low"]["precision"],
            "high_recall": m["class_high"]["recall"],
        })
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_tau = tau
            best_metrics = m

    return best_tau, best_metrics, sweep_results


def run_experiment(
    experiment_name: str,
    freeze_frame_encoder: bool,
    init_backbone_path: Path,
    out_ckpt_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    class_weights: torch.Tensor,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    lr_encoder: float = 5e-5,
    lr_temporal: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" EXPERIMENT: {experiment_name.upper()}")
    print(f" Freeze Frame Encoder: {freeze_frame_encoder}")
    print(f" Target Checkpoint   : {out_ckpt_path}")
    print("=" * 80)

    set_seed(seed)

    # Instantiate model with fresh backbone
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    init_ckpt = torch.load(init_backbone_path, map_location="cpu", weights_only=False)
    base_net.load_state_dict(init_ckpt["model"], strict=False)

    model = ManualBinaryAttentionClassifier(
        base_net, freeze_frame_encoder=freeze_frame_encoder
    ).to(device)

    # Optimizer configuration
    if freeze_frame_encoder:
        param_groups = [
            {"params": model.temporal.parameters(), "lr": lr_temporal, "weight_decay": weight_decay},
            {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
        ]
    else:
        param_groups = [
            {"params": model.frame_encoder.parameters(), "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": model.temporal.parameters(), "lr": lr_temporal, "weight_decay": weight_decay},
            {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
        ]

    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    x_val_t = torch.from_numpy(x_val).to(device)

    best_val_bal_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation evaluation at standard tau=0.50
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            v_preds = np.argmax(v_probs, axis=-1)

        v_metrics = evaluate_predictions(y_val, v_preds)

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": v_metrics["accuracy"],
            "val_balanced_accuracy": v_metrics["balanced_accuracy"],
            "val_macro_f1": v_metrics["macro_f1"],
            "val_weighted_f1": v_metrics["weighted_f1"],
            "val_low_recall": v_metrics["class_low"]["recall"],
            "val_low_precision": v_metrics["class_low"]["precision"],
            "val_low_f1": v_metrics["class_low"]["f1"],
            "val_high_recall": v_metrics["class_high"]["recall"],
            "val_high_precision": v_metrics["class_high"]["precision"],
            "val_high_f1": v_metrics["class_high"]["f1"],
            "confusion_matrix": v_metrics["confusion_matrix"],
        }
        history.append(record)

        is_best = v_metrics["balanced_accuracy"] > best_val_bal_acc
        if is_best:
            best_val_bal_acc = v_metrics["balanced_accuracy"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        star = " [*BEST BAL-ACC]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {v_metrics['accuracy']*100:.2f}% | Val Bal-Acc: {v_metrics['balanced_accuracy']*100:.2f}% | "
            f"Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | "
            f"LowRec: {v_metrics['class_low']['recall']*100:.1f}%, HighRec: {v_metrics['class_high']['recall']*100:.1f}%{star}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # Restore best checkpoint
    print(f"\nRestoring best model weights from Epoch {best_epoch} (Val Bal-Acc: {best_val_bal_acc*100:.2f}%)...")
    model.load_state_dict(best_state)
    model.eval()

    # Full validation predictions for threshold calibration
    with torch.no_grad():
        final_v_logits = model(x_val_t)
        final_v_probs = torch.softmax(final_v_logits, dim=-1).cpu().numpy()
        probs_high = final_v_probs[:, 1]

    # Metrics at standard tau=0.50
    preds_tau50 = (probs_high >= 0.50).astype(np.int64)
    metrics_tau50 = evaluate_predictions(y_val, preds_tau50)

    # Calibrate optimal threshold on validation set
    best_tau, metrics_calibrated, sweep_results = calibrate_threshold(y_val, probs_high)

    print("\n--- Model Evaluation Summary ---")
    print(f"Standard Threshold (tau = 0.50):")
    print(f"  Accuracy     : {metrics_tau50['accuracy']*100:.2f}%")
    print(f"  Balanced Acc : {metrics_tau50['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1     : {metrics_tau50['macro_f1']*100:.2f}%")
    print(f"  LOW  (Rec/Prec/F1): {metrics_tau50['class_low']['recall']*100:.1f}% / {metrics_tau50['class_low']['precision']*100:.1f}% / {metrics_tau50['class_low']['f1']*100:.1f}%")
    print(f"  HIGH (Rec/Prec/F1): {metrics_tau50['class_high']['recall']*100:.1f}% / {metrics_tau50['class_high']['precision']*100:.1f}% / {metrics_tau50['class_high']['f1']*100:.1f}%")
    print(f"  CM           : {metrics_tau50['confusion_matrix']}")

    print(f"\nCalibrated Threshold (tau* = {best_tau:.2f}):")
    print(f"  Accuracy     : {metrics_calibrated['accuracy']*100:.2f}%")
    print(f"  Balanced Acc : {metrics_calibrated['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1     : {metrics_calibrated['macro_f1']*100:.2f}%")
    print(f"  LOW  (Rec/Prec/F1): {metrics_calibrated['class_low']['recall']*100:.1f}% / {metrics_calibrated['class_low']['precision']*100:.1f}% / {metrics_calibrated['class_low']['f1']*100:.1f}%")
    print(f"  HIGH (Rec/Prec/F1): {metrics_calibrated['class_high']['recall']*100:.1f}% / {metrics_calibrated['class_high']['precision']*100:.1f}% / {metrics_calibrated['class_high']['f1']*100:.1f}%")
    print(f"  CM           : {metrics_calibrated['confusion_matrix']}")

    # Save best checkpoint
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_payload = {
        "epoch": best_epoch,
        "model": best_state,
        "experiment_name": experiment_name,
        "freeze_frame_encoder": freeze_frame_encoder,
        "seed": seed,
        "calibrated_threshold": best_tau,
        "metrics_tau50": metrics_tau50,
        "metrics_calibrated": metrics_calibrated,
        "history": history,
        "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D])",
    }
    torch.save(ckpt_payload, out_ckpt_path)
    print(f"Saved checkpoint to: {out_ckpt_path}")

    return {
        "experiment_name": experiment_name,
        "freeze_frame_encoder": freeze_frame_encoder,
        "best_epoch": best_epoch,
        "best_tau": best_tau,
        "metrics_tau50": metrics_tau50,
        "metrics_calibrated": metrics_calibrated,
        "threshold_sweep": sweep_results,
        "history": history,
        "checkpoint_path": str(out_ckpt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6C Step 3 Controlled Retraining")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--npz", default="data/processed/fusion_daisee_v3_hardened.npz")
    parser.add_argument("--split", default="configs/daisee_v3_subject_split.json")
    parser.add_argument("--init-backbone", default="checkpoints/fusion_daisee.pt")
    parser.add_argument("--out-frozen", default="checkpoints/classroom_attention_v3_frozen_baseline.pt")
    parser.add_argument("--out-unfrozen", default="checkpoints/classroom_attention_v3_unfrozen.pt")
    parser.add_argument("--metrics-out", default="runs/reports/phase6c/phase6c_step3_training_metrics.json")
    args = parser.parse_args()

    npz_path = WORKSPACE_ROOT / args.npz
    split_path = WORKSPACE_ROOT / args.split
    init_backbone_path = WORKSPACE_ROOT / args.init_backbone
    out_frozen_path = WORKSPACE_ROOT / args.out_frozen
    out_unfrozen_path = WORKSPACE_ROOT / args.out_unfrozen
    metrics_out_path = WORKSPACE_ROOT / args.metrics_out

    # Guard against accidental overwrite of historical checkpoints
    protected_paths = [
        WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v1.pt",
        WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt",
        WORKSPACE_ROOT / "checkpoints" / "baseline" / "classroom_manual_binary_model_v1.pt",
        WORKSPACE_ROOT / "checkpoints" / "baseline" / "fusion_daisee.pt",
    ]
    for p in protected_paths:
        if out_frozen_path.resolve() == p.resolve() or out_unfrozen_path.resolve() == p.resolve():
            raise RuntimeError(f"FATAL: Attempting to overwrite protected checkpoint: {p}")

    print("=" * 80)
    print(" PHASE 6C STEP 3: CONTROLLED RETRAINING PIPELINE")
    print("=" * 80)
    print(f"Dataset      : {npz_path}")
    print(f"Split Config : {split_path}")
    print(f"Init Backbone: {init_backbone_path}")
    print(f"Model A Out  : {out_frozen_path}")
    print(f"Model B Out  : {out_unfrozen_path}")
    print(f"Random Seed  : {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Data and Split Configuration
    with open(split_path, "r") as f:
        split_cfg = json.load(f)

    d = np.load(npz_path)
    feats = d["features"].astype(np.float32)
    scores = d["score"].astype(np.float32)
    clips = d["clip_ids"].astype(str)
    subjects = d["subject_ids"].astype(str)
    y_all = (scores >= 50.0).astype(np.int64)

    train_subs = set(split_cfg["train"]["subjects"])
    val_subs = set(split_cfg["validation"]["subjects"])

    train_mask = np.isin(subjects, list(train_subs))
    val_mask = np.isin(subjects, list(val_subs))

    # Anti-leakage assertions
    train_clips = set(clips[train_mask])
    val_clips = set(clips[val_mask])
    assert len(train_subs & val_subs) == 0, "Subject leakage detected!"
    assert len(train_clips & val_clips) == 0, "Clip leakage detected!"

    x_train, y_train = feats[train_mask], y_all[train_mask]
    x_val, y_val = feats[val_mask], y_all[val_mask]

    print("\n[Dataset Partition Summary]:")
    print(f"  Training   : {len(x_train):,} sequences ({len(train_clips)} clips, {len(train_subs)} subjects)")
    print(f"    Class HIGH : {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"    Class LOW  : {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"  Validation : {len(x_val):,} sequences ({len(val_clips)} clips, {len(val_subs)} subjects)")
    print(f"    Class HIGH : {sum(y_val == 1):,} ({sum(y_val == 1)/len(y_val)*100:.2f}%)")
    print(f"    Class LOW  : {sum(y_val == 0):,} ({sum(y_val == 0)/len(y_val)*100:.2f}%)")

    # Compute training-only class weights
    n_train = len(y_train)
    n_low = sum(y_train == 0)
    n_high = sum(y_train == 1)
    w_low = n_train / (2.0 * n_low)
    w_high = n_train / (2.0 * n_high)
    class_weights = torch.tensor([w_low, w_high], dtype=torch.float32, device=device)
    print(f"\n[Training-Only Loss Weights]: LOW={w_low:.4f}, HIGH={w_high:.4f} (Ratio={w_low/w_high:.2f})")

    # Evaluate Majority-Class Baseline
    majority_preds = np.ones_like(y_val)
    majority_metrics = evaluate_predictions(y_val, majority_preds)
    print("\n[Baseline 1: Majority-Class Dummy Classifier]:")
    print(f"  Accuracy     : {majority_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Acc : {majority_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1     : {majority_metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1  : {majority_metrics['weighted_f1']*100:.2f}%")
    print(f"  Confusion Mat: {majority_metrics['confusion_matrix']}")

    # Run Model A: Frozen Frame Encoder
    results_a = run_experiment(
        experiment_name="Model A (Frozen Frame Encoder)",
        freeze_frame_encoder=True,
        init_backbone_path=init_backbone_path,
        out_ckpt_path=out_frozen_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        class_weights=class_weights,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr_temporal=1e-4,
        lr_head=1e-3,
        weight_decay=1e-4,
    )

    # Run Model B: Trainable Frame Encoder (Differential LR)
    results_b = run_experiment(
        experiment_name="Model B (Trainable Frame Encoder)",
        freeze_frame_encoder=False,
        init_backbone_path=init_backbone_path,
        out_ckpt_path=out_unfrozen_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        class_weights=class_weights,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr_encoder=5e-5,
        lr_temporal=1e-4,
        lr_head=1e-3,
        weight_decay=1e-4,
    )

    # Compile Full Experiment Record
    full_summary = {
        "dataset": str(npz_path),
        "split_config": str(split_path),
        "seed": args.seed,
        "class_weights": {"low": float(w_low), "high": float(w_high)},
        "majority_baseline": majority_metrics,
        "model_a_frozen": results_a,
        "model_b_unfrozen": results_b,
    }

    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" ALL EXPERIMENTS COMPLETE")
    print(f" Metrics saved to: {metrics_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
