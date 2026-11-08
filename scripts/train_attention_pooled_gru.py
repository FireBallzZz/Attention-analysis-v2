"""Step 1 Optimization: Temporal Attention-Pooled GRU with Differential Learning Rates.

Upgrades:
1. Bidirectional GRU with Multi-Step Learned Attention Pooling (replaces terminal recency bias).
2. Differential Learning Rates:
   - GRU: lr=5e-5 (gentle domain adaptation)
   - Attention Pooling & Head: lr=1e-3 (rapid convergence)
3. Small-Dataset Regularization:
   - Feature dropout (p=0.1) during training to prevent sensor overfitting
   - Temporal dropout (p=0.3)
   - Weight decay (1e-3)
4. Class-Balanced Focal Loss (gamma=2.0)

Outputs to:
  checkpoints/classroom_attention_pooled_gru_v2.pt
  experiments/attention_pooled_gru_v2/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

import sys
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.utils import resolve_device

BANDS = ("HIGH", "MODERATE", "LOW")


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class TemporalAttentionPooling(nn.Module):
    """Learned attention pooling across the 16 time steps."""
    def __init__(self, feat_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        weights = self.proj(x)  # (B, T, 1)
        alpha = torch.softmax(weights, dim=1)  # (B, T, 1)
        pooled = torch.sum(alpha * x, dim=1)  # (B, D)
        return pooled, alpha.squeeze(-1)


class AttentionPooledClassroomClassifier(nn.Module):
    def __init__(
        self,
        pretrained_fusion: AttentionFusionNet,
        num_classes: int = 3,
        feat_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_encoder = pretrained_fusion.frame_encoder
        self.temporal = pretrained_fusion.temporal
        self.feat_dropout = nn.Dropout(feat_dropout)

        # Freeze frame encoder
        for p in self.frame_encoder.parameters():
            p.requires_grad = False

        # Trainable GRU
        for p in self.temporal.parameters():
            p.requires_grad = True

        fused_dim = 128  # 64 * 2 bidirectional GRU
        self.attn_pool = TemporalAttentionPooling(fused_dim, hidden_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 32)
        b, t, d = x.shape
        if self.training:
            x = self.feat_dropout(x)

        with torch.no_grad():
            encoded = self.frame_encoder(x.reshape(b * t, -1)).reshape(b, t, -1)

        gru_out, _ = self.temporal(encoded)  # (B, T, 128)
        pooled, alpha = self.attn_pool(gru_out)  # (B, 128)
        logits = self.classifier(pooled)
        return logits, alpha


class ClassroomSequenceDataset(Dataset):
    def __init__(self, npz_path: Path, student_filter: set[int] | None = None) -> None:
        data = np.load(npz_path)
        features = data["features"].astype(np.float32)
        student_ids = data["student_ids"].astype(np.int32)
        frame_ids = data["frame_ids"].astype(np.int32)
        targets = data["targets"].astype(np.int64)

        if student_filter is not None:
            mask = np.isin(student_ids, list(student_filter))
            features = features[mask]
            student_ids = student_ids[mask]
            frame_ids = frame_ids[mask]
            targets = targets[mask]

        self.features = features
        self.student_ids = student_ids
        self.frame_ids = frame_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.features[idx], dtype=torch.float32),
            "y": torch.tensor(self.targets[idx], dtype=torch.long),
            "sid": torch.tensor(self.student_ids[idx], dtype=torch.int32),
            "fid": torch.tensor(self.frame_ids[idx], dtype=torch.int32),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": float(np.mean(f1)),
        "recalls": [float(r) for r in rec],
        "precisions": [float(p) for p in prec],
        "f1s": [float(f) for f in f1],
        "support": [int(s) for s in supp],
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-gru", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--out-dir", default="experiments/attention_pooled_gru_v2")
    parser.add_argument("--out-ckpt", default="checkpoints/classroom_attention_pooled_gru_v2.pt")
    args = parser.parse_args()

    out_dir = WORKSPACE_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    print(f"Executing on device: {device}")

    # Standard audited student partition
    train_students = {0, 1, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 19, 20}
    val_students = {2, 5, 8, 10, 16, 18}
    assert len(train_students & val_students) == 0, "Leakage detected!"

    npz_path = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz"
    train_dataset = ClassroomSequenceDataset(npz_path, student_filter=train_students)
    val_dataset = ClassroomSequenceDataset(npz_path, student_filter=val_students)

    print(f"Train Dataset: {len(train_dataset)} sequences (15 students)")
    print(f"Val Dataset  : {len(val_dataset)} sequences (6 students)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load pretrained fusion base
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    base_ckpt = WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt"
    if base_ckpt.exists():
        fusion_base.load(base_ckpt, map_location=device)
        print(f"Loaded DAiSEE pretrained base: {base_ckpt.name}")

    model = AttentionPooledClassroomClassifier(pretrained_fusion=fusion_base, num_classes=3).to(device)

    # Class weights for focal loss
    train_targets = train_dataset.targets
    counts = np.bincount(train_targets, minlength=3)
    weights = len(train_targets) / (3.0 * np.maximum(counts, 1))
    alpha_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Loss weights (HIGH/MOD/LOW): {alpha_tensor.tolist()}")

    criterion = FocalLoss(alpha=alpha_tensor, gamma=args.gamma)

    # Differential learning rate optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": args.lr_gru},
            {"params": model.attn_pool.parameters(), "lr": args.lr_head},
            {"params": model.classifier.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_bal_acc = 0.0
    best_epoch = 0
    best_metrics = None
    best_preds = None
    epoch_logs = []

    print("\n--- Training Attention-Pooled GRU ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for b in train_loader:
            x = b["x"].to(device)
            y = b["y"].to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        train_loss = total_loss / max(batches, 1)

        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        val_logits = []
        val_sids = []
        val_fids = []

        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device)
                y = b["y"].to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().numpy().tolist())
                val_trues.extend(y.cpu().numpy().tolist())
                val_logits.extend(logits.cpu().numpy().tolist())
                val_sids.extend(b["sid"].numpy().tolist())
                val_fids.extend(b["fid"].numpy().tolist())

        m = compute_metrics(np.array(val_trues), np.array(val_preds))
        macro_f1 = m["macro_f1"]
        bal_acc = m["balanced_accuracy"]

        is_best = macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = macro_f1
            best_bal_acc = bal_acc
            best_epoch = epoch
            best_metrics = m
            best_preds = {
                "logits": np.array(val_logits),
                "y_pred": np.array(val_preds),
                "y_true": np.array(val_trues),
                "student_ids": np.array(val_sids),
                "frame_ids": np.array(val_fids),
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "macro_f1": macro_f1,
                    "balanced_accuracy": bal_acc,
                    "raw_accuracy": m["accuracy"],
                    "metrics": m,
                    "train_students": sorted(list(train_students)),
                    "val_students": sorted(list(val_students)),
                },
                out_ckpt,
            )

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val Macro-F1: {macro_f1*100:.2f}% | Val Bal-Acc: {bal_acc*100:.2f}% | "
            f"Recalls: [H:{m['recalls'][0]*100:.1f}%, M:{m['recalls'][1]*100:.1f}%, L:{m['recalls'][2]*100:.1f}%]{star}"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": macro_f1,
            "val_balanced_accuracy": bal_acc,
            "val_raw_accuracy": m["accuracy"],
            "recalls": m["recalls"],
            "precisions": m["precisions"],
            "f1s": m["f1s"],
        })

    # Save experiment outputs
    with (out_dir / "epoch_history.json").open("w", encoding="utf-8") as f:
        json.dump(epoch_logs, f, indent=2)

    with (out_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "metrics": best_metrics}, f, indent=2)

    if best_preds is not None:
        np.savez_compressed(out_dir / "val_predictions.npz", **best_preds)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE! Best Model from Epoch {best_epoch}:")
    print(f"  Best Val Macro-F1    : {best_macro_f1*100:.2f}%")
    print(f"  Best Val Balanced Acc: {best_bal_acc*100:.2f}%")
    print(f"  Best Val Raw Acc     : {best_metrics['accuracy']*100:.2f}%")
    print(f"  Per-Class Recalls    : HIGH: {best_metrics['recalls'][0]*100:.1f}%, MOD: {best_metrics['recalls'][1]*100:.1f}%, LOW: {best_metrics['recalls'][2]*100:.1f}%")
    print(f"  Per-Class Precisions : HIGH: {best_metrics['precisions'][0]*100:.1f}%, MOD: {best_metrics['precisions'][1]*100:.1f}%, LOW: {best_metrics['precisions'][2]*100:.1f}%")
    print("  Confusion Matrix:")
    for row in best_metrics["confusion_matrix"]:
        print("   ", row)
    print(f"Saved Checkpoint: {out_ckpt.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
