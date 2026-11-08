"""Option 2: Fine-tune classroom 3-class attention head on pretrained AttentionFusionNet encoder.

Evaluation Protocol:
- Strict student-disjoint split (zero student overlap between train and val).
- Metric: Macro-F1 & Balanced Accuracy (never aggregate accuracy alone).
- Full 3x3 confusion matrix and per-class recall reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.utils import PROJECT_ROOT, resolve_device

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


class ClassroomAttentionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_fusion: AttentionFusionNet,
        num_classes: int = 3,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.frame_encoder = pretrained_fusion.frame_encoder
        self.temporal = pretrained_fusion.temporal
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False
            for p in self.temporal.parameters():
                p.requires_grad = False

        # 128 = 64 * 2 (bidirectional GRU)
        fused_dim = 128
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_window(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        encoded = self.frame_encoder(x.reshape(b * t, -1)).reshape(b, t, -1)
        out, _ = self.temporal(encoded)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                h = self.encode_window(x)
        else:
            h = self.encode_window(x)
        return self.classifier(h)


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
            "x": torch.from_numpy(self.features[idx]),
            "y": torch.tensor(self.targets[idx], dtype=torch.long),
            "student_id": torch.tensor(self.student_ids[idx], dtype=torch.int32),
            "frame_id": torch.tensor(self.frame_ids[idx], dtype=torch.int32),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | np.ndarray]:
    cm = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    recalls = []
    precisions = []
    f1s = []

    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)

    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classroom attention classifier")
    parser.add_argument("--npz", default="data/processed/classroom_clip_004_sequences.npz")
    parser.add_argument("--fusion-ckpt", default="checkpoints/fusion_daisee.pt")
    parser.add_argument("--out", default="checkpoints/classroom_attention_head_v2.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz_path = PROJECT_ROOT / args.npz
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing sequence NPZ: {npz_path}. Run extract_classroom_sequences.py first.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device("auto")
    print(f"Using device: {device}")

    # Verified Student-Disjoint Split (Zero overlap)
    val_students = {2, 5, 8, 10, 16, 18}
    all_data = np.load(npz_path)
    all_sids = set(np.unique(all_data["student_ids"]).tolist())
    train_students = all_sids - val_students

    # Strict anti-leakage assertions
    assert len(train_students & val_students) == 0, "FATAL: Student leakage detected between train and val!"
    print(f"Train students ({len(train_students)}): {sorted(train_students)}")
    print(f"Val students   ({len(val_students)}): {sorted(val_students)}")

    train_ds = ClassroomSequenceDataset(npz_path, student_filter=train_students)
    val_ds = ClassroomSequenceDataset(npz_path, student_filter=val_students)

    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_targets = train_ds.targets
    val_targets = val_ds.targets
    print("Train class distribution:", {BANDS[c]: int((train_targets == c).sum()) for c in range(3)})
    print("Val class distribution:  ", {BANDS[c]: int((val_targets == c).sum()) for c in range(3)})

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Load pretrained AttentionFusionNet
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    fusion_ckpt_path = PROJECT_ROOT / args.fusion_ckpt
    if fusion_ckpt_path.exists():
        fusion_base.load(fusion_ckpt_path, map_location=device)
        print(f"Loaded pretrained AttentionFusionNet weights from {fusion_ckpt_path.name}")
    else:
        print(f"Warning: {fusion_ckpt_path} not found. Initializing randomly.")

    model = ClassroomAttentionClassifier(
        pretrained_fusion=fusion_base,
        num_classes=3,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Class weights for imbalanced classes (HIGH: ~65%, MODERATE: ~17%, LOW: ~18%)
    class_counts = np.bincount(train_targets, minlength=3)
    total_samples = len(train_targets)
    weights = total_samples / (3.0 * np.maximum(class_counts, 1))
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Computed loss weights (HIGH/MOD/LOW): {weights_tensor.tolist()}")

    if args.loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_macro_f1 = 0.0
    best_bal_acc = 0.0
    best_epoch = 0
    best_metrics = None

    out_ckpt_path = PROJECT_ROOT / args.out
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_trues.extend(y.cpu().numpy().tolist())

        m = compute_metrics(np.array(val_trues), np.array(val_preds))
        macro_f1 = m["macro_f1"]
        bal_acc = m["balanced_accuracy"]

        is_best = macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = macro_f1
            best_bal_acc = bal_acc
            best_epoch = epoch
            best_metrics = m
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "macro_f1": macro_f1,
                    "balanced_accuracy": bal_acc,
                    "train_students": sorted(train_students),
                    "val_students": sorted(val_students),
                    "metrics": m,
                },
                out_ckpt_path,
            )

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} | "
            f"Val Macro-F1: {macro_f1*100:.2f}% | Val Bal-Acc: {bal_acc*100:.2f}% | "
            f"Recalls: [H:{m['recalls'][0]*100:.1f}%, M:{m['recalls'][1]*100:.1f}%, L:{m['recalls'][2]*100:.1f}%]{star}"
        )

    print(f"\nTraining Complete! Best model from Epoch {best_epoch}:")
    print(f"  Best Val Macro-F1:       {best_macro_f1*100:.2f}%")
    print(f"  Best Val Balanced Acc:   {best_bal_acc*100:.2f}%")
    print(f"  Per-class Recall:        HIGH: {best_metrics['recalls'][0]*100:.1f}%, MODERATE: {best_metrics['recalls'][1]*100:.1f}%, LOW: {best_metrics['recalls'][2]*100:.1f}%")
    print(f"  Per-class Precision:     HIGH: {best_metrics['precisions'][0]*100:.1f}%, MODERATE: {best_metrics['precisions'][1]*100:.1f}%, LOW: {best_metrics['precisions'][2]*100:.1f}%")
    print(f"  Per-class F1:            HIGH: {best_metrics['f1s'][0]*100:.1f}%, MODERATE: {best_metrics['f1s'][1]*100:.1f}%, LOW: {best_metrics['f1s'][2]*100:.1f}%")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print("                  Pred HIGH   Pred MODERATE   Pred LOW")
    for r_idx, b in enumerate(BANDS):
        row_str = f"  True {b:8s}: " + "       ".join(f"{best_metrics['confusion_matrix'][r_idx, c_idx]:4d}" for c_idx in range(3))
        print(row_str)
    print(f"\nSaved checkpoint: {out_ckpt_path}")


if __name__ == "__main__":
    main()
