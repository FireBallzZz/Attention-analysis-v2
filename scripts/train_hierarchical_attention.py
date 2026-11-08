"""Hierarchical Two-Stage Attention Classifier for Classroom Attention Recognition.

Architecture:
- Backbone: DAiSEE-pretrained AttentionFusionNet (frame_encoder + BiGRU temporal)
- Stage 1: Binary Engagement Gate (Attentive [HIGH + MODERATE] vs Inattentive [LOW])
- Stage 2: Focus Granularity Head (HIGH vs MODERATE conditional on engagement)
- Joint Calibrated Probability Inference:
    P(LOW)      = P_s1(LOW)
    P(MODERATE) = P_s1(ENGAGED) * P_s2(MODERATE)
    P(HIGH)     = P_s1(ENGAGED) * P_s2(HIGH)

Evaluated with strict student-disjoint partitioning on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.utils import resolve_device

BANDS = ("HIGH", "MODERATE", "LOW")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


class HierarchicalAttentionNet(nn.Module):
    def __init__(self, pretrained_base: AttentionFusionNet, freeze_frame_encoder: bool = True) -> None:
        super().__init__()
        self.frame_encoder = pretrained_base.frame_encoder
        self.temporal = pretrained_base.temporal
        self.freeze_frame_encoder = freeze_frame_encoder

        if freeze_frame_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False

        fused_dim = 128  # 64 * 2 (bidirectional GRU)

        # Stage 1: Binary Engagement Gate (0: Engaged [High/Mod], 1: Low)
        self.stage1_gate = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # single logit for binary BCE
        )

        # Stage 2: Focus Granularity Head (0: High, 1: Moderate)
        self.stage2_granularity = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # 2 classes: High vs Mod
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if self.freeze_frame_encoder:
            with torch.no_grad():
                enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        else:
            enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        out, _ = self.temporal(enc)
        return out[:, -1, :]  # (B, 128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        logit_s1 = self.stage1_gate(h).squeeze(-1)  # (B,)
        logits_s2 = self.stage2_granularity(h)     # (B, 2)
        return logit_s1, logits_s2

    def predict_joint(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibrated 3-class joint probabilities: [P(HIGH), P(MODERATE), P(LOW)]."""
        self.eval()
        with torch.no_grad():
            logit_s1, logits_s2 = self.forward(x)
            p_low = torch.sigmoid(logit_s1).cpu().numpy()  # P(LOW)
            p_engaged = 1.0 - p_low                        # P(HIGH or MODERATE)

            p_s2 = torch.softmax(logits_s2, dim=-1).cpu().numpy()  # [P(HIGH|ENG), P(MOD|ENG)]
            p_high = p_engaged * p_s2[:, 0]
            p_mod = p_engaged * p_s2[:, 1]

            joint_probs = np.stack([p_high, p_mod, p_low], axis=1)  # (B, 3)
            preds = np.argmax(joint_probs, axis=1)                  # 0: HIGH, 1: MODERATE, 2: LOW
            return preds, joint_probs


class ClassroomNPZDataset(Dataset):
    def __init__(self, npz_paths: list[Path], student_filter: set[int] | None = None) -> None:
        all_feats, all_targets, all_sids, all_fids = [], [], [], []
        for p in npz_paths:
            if not p.exists():
                continue
            data = np.load(p)
            f = data["features"].astype(np.float32)
            t = data["targets"].astype(np.int64)
            s = data["student_ids"].astype(np.int32)
            fid = data["frame_ids"].astype(np.int32)
            if student_filter is not None:
                mask = np.isin(s, list(student_filter))
                f, t, s, fid = f[mask], t[mask], s[mask], fid[mask]
            all_feats.append(f)
            all_targets.append(t)
            all_sids.append(s)
            all_fids.append(fid)

        self.features = np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 16, 32), dtype=np.float32)
        self.targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
        self.student_ids = np.concatenate(all_sids, axis=0) if all_sids else np.empty((0,), dtype=np.int32)
        self.frame_ids = np.concatenate(all_fids, axis=0) if all_fids else np.empty((0,), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.features[idx]),
            "y": torch.tensor(self.targets[idx], dtype=torch.long),
            "student_id": torch.tensor(self.student_ids[idx], dtype=torch.int32),
            "frame_id": torch.tensor(self.frame_ids[idx], dtype=torch.int32),
        }


def compute_3class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    
    # Binary engagement metric: (0: High or 1: Mod) vs 2: Low
    y_true_bin = (y_true != 2).astype(int)
    y_pred_bin = (y_pred != 2).astype(int)
    bin_acc = float(accuracy_score(y_true_bin, y_pred_bin))

    return {
        "raw_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "binary_accuracy": bin_acc,
        "macro_f1": float(np.mean(f1)),
        "recalls": [float(r) for r in rec],
        "precisions": [float(p) for p in prec],
        "f1s": [float(f) for f in f1],
        "support": [int(s) for s in supp],
        "confusion_matrix": cm.tolist(),
    }


def train_hierarchical():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-temporal", type=float, default=5e-5)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="experiments/hierarchical_v1")
    parser.add_argument("--out-ckpt", default="checkpoints/classroom_hierarchical_attention_v1.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hierarchical Attention Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = WORKSPACE_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # Dataset paths
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    seq_files = [
        data_dir / "classroom_clip_004_sequences.npz",
    ]
    clip_001 = data_dir / "classroom_clip_001_sequences.npz"
    clip_008 = data_dir / "classroom_clip_008_sequences.npz"
    if clip_001.exists():
        seq_files.append(clip_001)
    if clip_008.exists():
        seq_files.append(clip_008)

    # Split design:
    # 1. Validation benchmark: Strictly clip_004 held-out students [2, 5, 8, 10, 16, 18] (CVAT human ground truth)
    # 2. Training: clip_004 training students + ALL students from independent sessions clip_001 and clip_008
    val_students = {2, 5, 8, 10, 16, 18}
    train_students_004 = {0, 1, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 19, 20}

    # Validation: strictly clip_004 held-out students
    val_ds = ClassroomNPZDataset([data_dir / "classroom_clip_004_sequences.npz"], student_filter=val_students)

    # Training: multi-clip expansion
    train_parts = [
        ClassroomNPZDataset([data_dir / "classroom_clip_004_sequences.npz"], student_filter=train_students_004)
    ]
    if clip_001.exists():
        train_parts.append(ClassroomNPZDataset([clip_001]))  # All 21 students of clip_001
    if clip_008.exists():
        train_parts.append(ClassroomNPZDataset([clip_008]))  # All 21 students of clip_008

    # Concatenate training features
    train_features = np.concatenate([p.features for p in train_parts], axis=0)
    train_targets = np.concatenate([p.targets for p in train_parts], axis=0)
    train_student_ids = np.concatenate([p.student_ids for p in train_parts], axis=0)
    train_frame_ids = np.concatenate([p.frame_ids for p in train_parts], axis=0)

    class CombinedTrainDS(Dataset):
        def __init__(self, f, t, s, fid):
            self.features = f
            self.targets = t
            self.student_ids = s
            self.frame_ids = fid
        def __len__(self): return len(self.features)
        def __getitem__(self, idx):
            return {
                "x": torch.from_numpy(self.features[idx]),
                "y": torch.tensor(self.targets[idx], dtype=torch.long),
                "student_id": torch.tensor(self.student_ids[idx], dtype=torch.int32),
                "frame_id": torch.tensor(self.frame_ids[idx], dtype=torch.int32),
            }

    train_ds = CombinedTrainDS(train_features, train_targets, train_student_ids, train_frame_ids)

    print(f"Multi-Clip Training sequences: {len(train_ds)} (across 57 students)")
    print(f"Held-Out Validation sequences : {len(val_ds)} (strictly clip_004 CVAT validation students)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model initialization
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    daisee_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt", map_location="cpu", weights_only=False)
    fusion_base.load_state_dict(daisee_ckpt["model"], strict=False)

    model = HierarchicalAttentionNet(fusion_base, freeze_frame_encoder=True).to(device)

    # Differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": args.lr_temporal, "weight_decay": 1e-4},
            {"params": model.stage1_gate.parameters(), "lr": args.lr_heads, "weight_decay": 1e-3},
            {"params": model.stage2_granularity.parameters(), "lr": args.lr_heads, "weight_decay": 1e-3},
        ]
    )

    criterion_s1 = BinaryFocalLoss(alpha=0.65, gamma=2.0)
    criterion_s2 = MultiClassFocalLoss(alpha=torch.tensor([1.0, 2.0], device=device), gamma=2.0)

    best_val_score = 0.0
    best_metrics = None
    history = []

    x_val_tensor = torch.from_numpy(val_ds.features).to(device)
    y_val = val_ds.targets

    print("\nStarting Training Epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            bx = batch["x"].to(device)
            by = batch["y"].to(device)

            # Stage 1 target: 1 if LOW (target 2), 0 if ENGAGED (target 0 or 1)
            by_s1 = (by == 2).float()

            logit_s1, logits_s2 = model(bx)
            loss_s1 = criterion_s1(logit_s1, by_s1)

            # Stage 2 loss: only for engaged students
            mask_eng = (by != 2)
            if mask_eng.sum() > 0:
                loss_s2 = criterion_s2(logits_s2[mask_eng], by[mask_eng])
            else:
                loss_s2 = torch.tensor(0.0, device=device)

            loss = loss_s1 + 1.2 * loss_s2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation evaluation
        val_preds, val_probs = model.predict_joint(x_val_tensor)
        metrics = compute_3class_metrics(y_val, val_preds)
        composite_score = 0.5 * metrics["raw_accuracy"] + 0.5 * metrics["balanced_accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_metrics": metrics,
        })

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_metrics = metrics
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_metrics": metrics,
                "train_students_004": sorted(list(train_students_004)),
                "val_students": sorted(list(val_students)),
            }, out_ckpt)

        if epoch % 2 == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {total_loss/len(train_loader):.4f} | "
                f"Val Acc: {metrics['raw_accuracy']*100:.2f}% | Bal Acc: {metrics['balanced_accuracy']*100:.2f}% | "
                f"Bin Acc: {metrics['binary_accuracy']*100:.2f}% | "
                f"Recalls: H={metrics['recalls'][0]*100:.1f}%, M={metrics['recalls'][1]*100:.1f}%, L={metrics['recalls'][2]*100:.1f}%"
            )

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE — BEST VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Best Checkpoint Saved To: {out_ckpt}")
    print(f"3-Class Raw Accuracy    : {best_metrics['raw_accuracy']*100:.2f}%")
    print(f"3-Class Balanced Accuracy: {best_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Binary Engagement Acc   : {best_metrics['binary_accuracy']*100:.2f}%")
    print(f"Macro-F1 Score          : {best_metrics['macro_f1']*100:.2f}%")
    print(f"Confusion Matrix        :\n  {best_metrics['confusion_matrix']}")

    with (out_dir / "best_metrics.json").open("w") as f:
        json.dump(best_metrics, f, indent=2)
    with (out_dir / "epoch_history.json").open("w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train_hierarchical()
