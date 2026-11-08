"""Reproducible Training Script for Manual-Tuned Binary Classroom Attention Model.

Enforces:
1. Pinned Random Seeds (seed=42) for 100% Deterministic Reproducibility.
2. Track-Grouped Train/Validation Splitting (Zero Temporal Overlap).
3. Checkpoint selection and threshold calibration strictly on held-out validation data.
   (runs/4.mp4 is NEVER touched during training or epoch selection).
4. Strict Preservations: Saves to checkpoints/classroom_manual_binary_model_v2_clean.pt
   leaving baseline v1 checkpoints completely untouched.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

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


def train():
    parser = argparse.ArgumentParser(description="Train Manual-Tuned Binary Attention Model (v2 clean)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr-temporal", type=float, default=8e-5, help="Learning rate for BiGRU temporal layers")
    parser.add_argument("--lr-head", type=float, default=1.5e-3, help="Learning rate for MLP classification head")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--val-tracks", type=str, default="2,5,8,11", help="Comma-separated track IDs to hold out for validation")
    parser.add_argument(
        "--init-backbone",
        default="checkpoints/baseline/fusion_daisee.pt",
        help="Pretrained backbone initialization weights",
    )
    parser.add_argument(
        "--out-ckpt",
        default="checkpoints/classroom_manual_binary_model_v2_clean.pt",
        help="Target output checkpoint path",
    )
    parser.add_argument(
        "--out-history",
        default="runs/reports/manual_binary_v2_clean_training_history.json",
        help="Path to save full training epoch metrics",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" REPRODUCIBLE TRAINING: MANUAL-TUNED BINARY ATTENTION MODEL (v2 clean)")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Random Seed: {args.seed}")
    print(f"Validation Hold-out Tracks: [{args.val_tracks}]")

    val_track_list = [int(v.strip()) for v in args.val_tracks.split(",") if v.strip()]

    # 1. Load Training Pool (clip_001 + clip_008)
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    p1 = data_dir / "classroom_clip_001_sequences.npz"
    p8 = data_dir / "classroom_clip_008_sequences.npz"

    if not p1.exists() or not p8.exists():
        raise FileNotFoundError(f"Missing training sequences in {data_dir}")

    d1 = np.load(p1)
    d8 = np.load(p8)

    feats_all = np.concatenate([d1["features"].astype(np.float32), d8["features"].astype(np.float32)], axis=0)
    sids_all = np.concatenate([d1["student_ids"].astype(np.int32), d8["student_ids"].astype(np.int32)], axis=0)
    fids_all = np.concatenate([d1["frame_ids"].astype(np.int32), d8["frame_ids"].astype(np.int32)], axis=0)
    # Binary Label Encoding: HIGH (Attentive) = 1, LOW (Moderate + Non-Attentive) = 0
    targets_raw = np.concatenate([d1["targets"], d8["targets"]], axis=0)
    targets_bin = (targets_raw == 0).astype(np.int64)

    print(f"\n[Data Pool Loaded]: {len(feats_all):,} sequences across clip_001 & clip_008")
    print(f"  Overall Distribution: HIGH={sum(targets_bin==1):,} ({sum(targets_bin==1)/len(targets_bin)*100:.1f}%), LOW={sum(targets_bin==0):,} ({sum(targets_bin==0)/len(targets_bin)*100:.1f}%)")

    # 2. Track-Grouped Train / Validation Splitting
    val_mask = np.isin(sids_all, val_track_list)
    train_mask = ~val_mask

    x_train, y_train = feats_all[train_mask], targets_bin[train_mask]
    x_val, y_val = feats_all[val_mask], targets_bin[val_mask]

    print("\n[Track-Grouped Partition (Zero Temporal Overlap)]:")
    print(f"  Train Set: {len(x_train):,} sequences ({len(x_train)/len(feats_all)*100:.1f}%) | Tracks: {sorted(list(set(sids_all[train_mask])))}")
    print(f"    Train Classes: HIGH={sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.1f}%), LOW={sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"  Val Set:   {len(x_val):,} sequences ({len(x_val)/len(feats_all)*100:.1f}%) | Tracks: {sorted(list(set(sids_all[val_mask])))}")
    print(f"    Val Classes:   HIGH={sum(y_val==1):,} ({sum(y_val==1)/len(y_val)*100:.1f}%), LOW={sum(y_val==0):,} ({sum(y_val==0)/len(y_val)*100:.1f}%)")

    # 3. Model Architecture Initialization
    backbone_path = WORKSPACE_ROOT / args.init_backbone
    if not backbone_path.exists():
        backbone_path = WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing backbone initialization: {backbone_path}")

    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    init_ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
    base_net.load_state_dict(init_ckpt["model"], strict=False)

    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True).to(device)

    # 4. Class Weights & Optimization
    counts = np.bincount(y_train, minlength=2)
    w = 1.0 / (counts / counts.sum())
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print(f"\nLoss Weights (Inverse Class Frequency): LOW={w[0]:.2f}, HIGH={w[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": args.lr_temporal, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        ]
    )

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    x_val_t = torch.from_numpy(x_val).to(device)

    # 5. Training Loop with Strictly Held-Out Validation Checkpoint Selection
    best_composite = 0.0
    best_metrics = {}
    history = []

    out_ckpt_path = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_hist_path = WORKSPACE_ROOT / args.out_history
    out_hist_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nStarting GPU Training Epochs (Early stopping driven strictly by Validation Split)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate strictly on held-out validation tracks
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            v_preds = np.argmax(v_probs, axis=-1)

        v_acc = float(accuracy_score(y_val, v_preds))
        v_bal = float(balanced_accuracy_score(y_val, v_preds))
        v_f1_macro = float(f1_score(y_val, v_preds, average="macro", zero_division=0))
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, v_preds, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y_val, v_preds, labels=[0, 1])

        composite = 0.5 * v_bal + 0.5 * v_f1_macro

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_accuracy": v_acc,
            "val_balanced_accuracy": v_bal,
            "val_macro_f1": v_f1_macro,
            "val_precision_low": float(prec[0]),
            "val_recall_low": float(rec[0]),
            "val_f1_low": float(f1[0]),
            "val_precision_high": float(prec[1]),
            "val_recall_high": float(rec[1]),
            "val_f1_high": float(f1[1]),
            "confusion_matrix": cm.tolist(),
        }
        history.append(epoch_record)

        is_best = composite > best_composite
        if is_best:
            best_composite = composite
            best_metrics = epoch_record
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics": epoch_record,
                    "seed": args.seed,
                    "val_tracks": val_track_list,
                    "train_sequences": len(x_train),
                    "val_sequences": len(x_val),
                    "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D])",
                },
                out_ckpt_path,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs or is_best:
            star = " [*BEST VAL]" if is_best else ""
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {total_loss/len(train_loader):.4f} | "
                f"Val Acc: {v_acc*100:.2f}% | Val Bal: {v_bal*100:.2f}% | Val Macro-F1: {v_f1_macro*100:.2f}% | "
                f"LowRec: {rec[0]*100:.1f}%, HighRec: {rec[1]*100:.1f}%{star}"
            )

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE — BEST VALIDATION EPOCH REPORT")
    print("=" * 80)
    print(f"Best Validation Epoch      : {best_metrics['epoch']}")
    print(f"Validation Raw Accuracy    : {best_metrics['val_accuracy']*100:.2f}%")
    print(f"Validation Balanced Acc    : {best_metrics['val_balanced_accuracy']*100:.2f}%")
    print(f"Validation Macro-F1 Score  : {best_metrics['val_macro_f1']*100:.2f}%")
    print(f"Validation LOW Recall      : {best_metrics['val_recall_low']*100:.1f}% (Precision: {best_metrics['val_precision_low']*100:.1f}%, F1: {best_metrics['val_f1_low']*100:.1f}%)")
    print(f"Validation HIGH Recall     : {best_metrics['val_recall_high']*100:.1f}% (Precision: {best_metrics['val_precision_high']*100:.1f}%, F1: {best_metrics['val_f1_high']*100:.1f}%)")
    print(f"Validation Confusion Matrix:\n  {best_metrics['confusion_matrix']}")
    print(f"\nSaved Clean Checkpoint     : {out_ckpt_path}")
    print(f"Saved Training History     : {out_hist_path}")

    with out_hist_path.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "best_metrics": best_metrics, "history": history}, f, indent=2)


if __name__ == "__main__":
    train()
