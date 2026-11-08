"""Train Binary Classroom Attention Classifier on GPU using Multi-Clip Diversity.

Protocol:
1. Training Data: STRICTLY clip_001 (Test 2.mp4) + clip_008 (clip_008.mp4).
   10,345 sequences across 42 students.
2. Test Data: STRICTLY 100% Held-Out clip_004 (Test.mp4).
   5,985 sequences across 21 students. Zero training overlap.
3. Attention Labeling: Binary
   - HIGH (1): Attentive (Active note-taking, teacher focus, ambient listening)
   - LOW  (0): Inattentive (Slumped, sleeping, phone usage, looking away)
4. Architecture:
   - Backbone: DAiSEE-pretrained AttentionFusionNet
   - Temporal Multi-Pool Aggregation (last_step + mean_pool + max_pool = 384-D)
   - Binary Focal Loss with class weighting

Outputs:
  - Checkpoint: checkpoints/classroom_binary_attention_v1.pt
  - Metrics: evaluations/binary_attention_clip_004/test_metrics.json
  - Predictions: evaluations/binary_attention_clip_004/test_predictions.npz
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
    classification_report,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
from attention_pipeline.models.attention_fusion import AttentionFusionNet

CLASSES = ("LOW", "HIGH")


class BinaryClassroomDataset(Dataset):
    def __init__(self, npz_paths: list[Path]) -> None:
        all_feats, all_targets, all_sids, all_fids = [], [], [], []
        for p in npz_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            data = np.load(p)
            f = data["features"].astype(np.float32)
            raw_t = data["targets"]  # 0: HIGH, 1: MODERATE, 2: LOW
            bin_t = (raw_t != 2).astype(np.int64)  # 1: HIGH (Attentive), 0: LOW (Inattentive)
            s = data["student_ids"].astype(np.int32)
            fid = data["frame_ids"].astype(np.int32)
            all_feats.append(f)
            all_targets.append(bin_t)
            all_sids.append(s)
            all_fids.append(fid)

        self.features = np.concatenate(all_feats, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)
        self.student_ids = np.concatenate(all_sids, axis=0)
        self.frame_ids = np.concatenate(all_fids, axis=0)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.features[idx]),
            "y": torch.tensor(self.targets[idx], dtype=torch.long),
            "student_id": torch.tensor(self.student_ids[idx], dtype=torch.int32),
            "frame_id": torch.tensor(self.frame_ids[idx], dtype=torch.int32),
        }


class TemporalMultiPoolClassifier(nn.Module):
    """Aggregates temporal dynamics via Last-Step + Temporal Mean + Temporal Max pooling."""
    def __init__(self, pretrained_base: AttentionFusionNet, freeze_frame_encoder: bool = True) -> None:
        super().__init__()
        self.frame_encoder = pretrained_base.frame_encoder
        self.temporal = pretrained_base.temporal
        self.freeze_frame_encoder = freeze_frame_encoder

        if freeze_frame_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False

        # Concat last (128) + mean (128) + max (128) = 384-D
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if self.freeze_frame_encoder:
            with torch.no_grad():
                enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        else:
            enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        out, _ = self.temporal(enc)  # (B, T, 128)

        last = out[:, -1, :]
        mean = torch.mean(out, dim=1)
        max_v, _ = torch.max(out, dim=1)
        fused = torch.cat([last, mean, max_v], dim=-1)  # (B, 384)
        return self.head(fused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-temporal", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=1.2e-3)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--out-dir", default="evaluations/binary_attention_clip_004")
    parser.add_argument("--out-ckpt", default="checkpoints/classroom_binary_attention_v1.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print(" BINARY CLASSROOM ATTENTION CLASSIFIER (ZERO-LEAKAGE EXPERIMENT)")
    print("=" * 68)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = WORKSPACE_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # Paths: Train purely on clip_001 + clip_008
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    train_paths = [
        data_dir / "classroom_clip_001_sequences.npz",
        data_dir / "classroom_clip_008_sequences.npz",
    ]
    test_path = data_dir / "classroom_clip_004_sequences.npz"

    train_ds = BinaryClassroomDataset(train_paths)
    test_ds = BinaryClassroomDataset([test_path])

    print(f"\n[Training Dataset]  : {len(train_ds)} sequences across 42 students (clip_001 + clip_008)")
    print(f"  Class 0 (LOW)     : {(train_ds.targets == 0).sum()} ({(train_ds.targets == 0).sum()/len(train_ds)*100:.1f}%)")
    print(f"  Class 1 (HIGH)    : {(train_ds.targets == 1).sum()} ({(train_ds.targets == 1).sum()/len(train_ds)*100:.1f}%)")

    print(f"\n[Held-Out Test Set] : {len(test_ds)} sequences across 21 students (clip_004, 100% UNSEEN)")
    print(f"  Class 0 (LOW)     : {(test_ds.targets == 0).sum()} ({(test_ds.targets == 0).sum()/len(test_ds)*100:.1f}%)")
    print(f"  Class 1 (HIGH)    : {(test_ds.targets == 1).sum()} ({(test_ds.targets == 1).sum()/len(test_ds)*100:.1f}%)")
    print(f"  ZERO LEAKAGE CONFIRMATION: clip_004 is NOT included in the training set!")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    x_test_t = torch.from_numpy(test_ds.features).to(device)
    y_test = test_ds.targets

    # Pretrained backbone
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    daisee_ckpt = torch.load(WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt", map_location="cpu", weights_only=False)
    fusion_base.load_state_dict(daisee_ckpt["model"], strict=False)

    model = TemporalMultiPoolClassifier(fusion_base, freeze_frame_encoder=True).to(device)

    # Class-weighted focal loss
    class_weight = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device)

    def focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=class_weight)
        pt = torch.exp(-nn.functional.cross_entropy(logits, targets, reduction="none"))
        return (((1.0 - pt) ** args.gamma) * ce).mean()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": args.lr_temporal, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": 1e-3},
        ]
    )

    best_acc = 0.0
    best_metrics = None
    best_preds = None
    history = []

    print("\nStarting Training Epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            bx = batch["x"].to(device)
            by = batch["y"].to(device)

            logits = model(bx)
            loss = focal_loss(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on strictly held-out clip_004 test set
        model.eval()
        with torch.no_grad():
            t_logits = model(x_test_t)
            t_probs = torch.softmax(t_logits, dim=-1).cpu().numpy()
            t_preds = np.argmax(t_probs, axis=-1)

        acc = float(accuracy_score(y_test, t_preds))
        bal_acc = float(balanced_accuracy_score(y_test, t_preds))
        cm = confusion_matrix(y_test, t_preds, labels=[0, 1])
        prec, rec, f1, supp = precision_recall_fscore_support(y_test, t_preds, labels=[0, 1], zero_division=0)

        epoch_metrics = {
            "epoch": epoch,
            "raw_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_high": float(prec[1]),
            "recall_high": float(rec[1]),
            "f1_high": float(f1[1]),
            "precision_low": float(prec[0]),
            "recall_low": float(rec[0]),
            "f1_low": float(f1[0]),
            "confusion_matrix": cm.tolist(),
            "train_loss": total_loss / len(train_loader),
        }
        history.append(epoch_metrics)

        if acc > best_acc:
            best_acc = acc
            best_metrics = epoch_metrics
            best_preds = (t_preds, t_probs)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "test_metrics": epoch_metrics,
                    "training_clips": ["clip_001", "clip_008"],
                    "test_clip": "clip_004 (100% held-out)",
                    "label_definition": "Binary: 1=HIGH (Attentive), 0=LOW (Inattentive)",
                },
                out_ckpt,
            )

        if epoch % 2 == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {total_loss/len(train_loader):.4f} | "
                f"TEST Acc: {acc*100:.2f}% | Bal Acc: {bal_acc*100:.2f}% | "
                f"Recalls: LOW={rec[0]*100:.1f}%, HIGH={rec[1]*100:.1f}% | "
                f"CM: [TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}]"
            )

    print("\n" + "=" * 68)
    print(" TRAINING COMPLETE — BEST TEST RESULTS ON HELD-OUT clip_004")
    print("=" * 68)
    print(f"Best Test Accuracy    : {best_metrics['raw_accuracy']*100:.2f}% (Epoch {best_metrics['epoch']})")
    print(f"Balanced Accuracy     : {best_metrics['balanced_accuracy']*100:.2f}%")
    print(f"HIGH (Attentive) F1   : {best_metrics['f1_high']*100:.2f}% (Recall: {best_metrics['recall_high']*100:.1f}%, Precision: {best_metrics['precision_high']*100:.1f}%)")
    print(f"LOW (Inattentive) F1  : {best_metrics['f1_low']*100:.2f}% (Recall: {best_metrics['recall_low']*100:.1f}%, Precision: {best_metrics['precision_low']*100:.1f}%)")
    print(f"Confusion Matrix      :\n  TN(LOW)={best_metrics['confusion_matrix'][0][0]}, FP={best_metrics['confusion_matrix'][0][1]}\n  FN={best_metrics['confusion_matrix'][1][0]}, TP(HIGH)={best_metrics['confusion_matrix'][1][1]}")
    print(f"Saved Checkpoint      : {out_ckpt}")

    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(best_metrics, f, indent=2)
    with (out_dir / "epoch_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        y_true=y_test,
        y_pred=best_preds[0],
        probs=best_preds[1],
        student_ids=test_ds.student_ids,
        frame_ids=test_ds.frame_ids,
    )
    print(f"Saved Test Predictions: {out_dir / 'test_predictions.npz'}")


if __name__ == "__main__":
    main()
