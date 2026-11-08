"""Reproducible Training Script for Phase 6G-4C: Binary ENGAGED/DISENGAGED Attention Classifier.

Enforces:
1. Strict pre-flight validation (SHA-256 hash, dataset paths, schema, track disjointness).
2. Pinned random seed (default=42) for deterministic reproducibility.
3. Track-grouped train/validation split (val_tracks=[2, 5, 8, 11]).
4. Supports Primary Loss (Weighted Cross-Entropy) and Ablation (Class-Balanced Focal Loss).
5. Epoch/checkpoint selection governed strictly by validation Macro F1 (tie-breaker: lowest val_loss).
6. Non-destructive: Active production checkpoint is never modified; candidates saved to checkpoints/candidates/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

ACTIVE_CHECKPOINT_PATH = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"


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


class ClassBalancedFocalLoss(nn.Module):
    """Multi-Class Focal Loss with explicit class-weighting (Lin et al., 2017).
    
    FL(pt) = - alpha_t * (1 - pt)^gamma * log(pt)
    where pt = softmax(logits)_t, and alpha_t = weight[t].
    """
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = torch.log(torch.clamp(pt, min=1e-8, max=1.0))
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt
        if self.weight is not None:
            alpha = self.weight[targets]
            loss = alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def verify_active_checkpoint() -> str:
    if not ACTIVE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Active checkpoint not found at {ACTIVE_CHECKPOINT_PATH}")
    with open(ACTIVE_CHECKPOINT_PATH, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest().upper()
    if h != EXPECTED_ACTIVE_SHA256:
        raise ValueError(f"Active checkpoint SHA-256 violation: expected {EXPECTED_ACTIVE_SHA256}, got {h}")
    return h


def run_training():
    parser = argparse.ArgumentParser(description="Phase 6G-4C Candidate Attention Classifier Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr-temporal", type=float, default=8e-5, help="Learning rate for BiGRU temporal layers")
    parser.add_argument("--lr-head", type=float, default=1.5e-3, help="Learning rate for MLP classification head")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--val-tracks", type=str, default="2,5,8,11", help="Validation track IDs")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="weighted_ce",
        choices=["weighted_ce", "focal_loss"],
        help="Loss function: weighted_ce (primary) or focal_loss (ablation)",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument(
        "--init-backbone",
        default="checkpoints/baseline/fusion_daisee.pt",
        help="Pretrained backbone initialization weights",
    )
    parser.add_argument(
        "--out-run-dir",
        type=str,
        default="runs/training/phase6g4c_run",
        help="Directory to save training artifacts and metrics",
    )
    parser.add_argument(
        "--out-ckpt",
        type=str,
        default="checkpoints/candidates/phase6g4c_candidate.pt",
        help="Path to save candidate model checkpoint",
    )
    parser.add_argument(
        "--out-history",
        type=str,
        default="",
        help="Path to save training history JSON (defaults to out-run-dir/training_history.json)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f" PHASE 6G-4C: CANDIDATE MODEL TRAINING ({args.loss_type.upper()})")
    print("=" * 80)

    # 1. Pre-Flight Checkpoint Verification
    pre_sha256 = verify_active_checkpoint()
    print(f"[Pre-Flight Checkpoint SHA-256 Verified]: {pre_sha256} (Invariant)")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Random Seed: {args.seed}")

    # Output paths isolation
    out_run_dir = WORKSPACE_ROOT / args.out_run_dir
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt_path = WORKSPACE_ROOT / args.out_ckpt
    if out_ckpt_path.exists():
        raise FileExistsError(f"Target checkpoint already exists: {out_ckpt_path}. Refusing to overwrite.")
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    out_history_path = (
        WORKSPACE_ROOT / args.out_history
        if args.out_history
        else out_run_dir / f"{args.loss_type}_history.json"
    )

    # 2. Load Hardened Datasets (Clip 001 + Clip 008)
    p1 = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_001_v3_hardened.npz"
    p8 = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_008_v3_hardened.npz"
    if not p1.exists() or not p8.exists():
        raise FileNotFoundError(f"Missing required hardened datasets: {p1} or {p8}")

    d1 = np.load(p1, allow_pickle=True)
    d8 = np.load(p8, allow_pickle=True)

    # Validate shapes and finite check
    f1, f8 = d1["features"].astype(np.float32), d8["features"].astype(np.float32)
    assert f1.shape == (5192, 16, 32) and f8.shape == (5153, 16, 32)
    assert np.all(np.isfinite(f1)) and np.all(np.isfinite(f8))

    feats_all = np.concatenate([f1, f8], axis=0)
    sids_all = np.concatenate([d1["student_ids"].astype(np.int32), d8["student_ids"].astype(np.int32)], axis=0)
    fids_all = np.concatenate([d1["frame_ids"].astype(np.int32), d8["frame_ids"].astype(np.int32)], axis=0)
    traw_all = np.concatenate([d1["targets_raw"].astype(np.int32), d8["targets_raw"].astype(np.int32)], axis=0)
    tbin_all = np.concatenate([d1["targets_binary"].astype(np.int64), d8["targets_binary"].astype(np.int64)], axis=0)

    # Binary Label Verification: 0 = DISENGAGED (Low=2), 1 = ENGAGED (High=0, Mod=1)
    assert np.all((tbin_all == 1) == (traw_all <= 1))
    assert np.all((tbin_all == 0) == (traw_all == 2))

    val_track_list = [int(v.strip()) for v in args.val_tracks.split(",") if v.strip()]
    val_mask = np.isin(sids_all, val_track_list)
    train_mask = ~val_mask

    x_train, y_train = feats_all[train_mask], tbin_all[train_mask]
    x_val, y_val = feats_all[val_mask], tbin_all[val_mask]

    assert len(x_train) == 8065, f"Expected 8,065 training sequences, got {len(x_train)}"
    assert len(x_val) == 2280, f"Expected 2,280 validation sequences, got {len(x_val)}"
    train_tracks = sorted([int(x) for x in set(sids_all[train_mask])])
    val_tracks_actual = sorted([int(x) for x in set(sids_all[val_mask])])
    assert len(set(train_tracks).intersection(set(val_tracks_actual))) == 0, "Track overlap detected!"

    print(f"\n[Dataset Partition Summary]:")
    print(f"  Training Split   : {len(x_train):,} seqs ({len(x_train)/len(feats_all)*100:.1f}%) | Tracks ({len(train_tracks)}): {train_tracks}")
    print(f"    Train Classes  : DISENGAGED (0)={sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.2f}%), ENGAGED (1)={sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"  Validation Split : {len(x_val):,} seqs ({len(x_val)/len(feats_all)*100:.1f}%) | Tracks ({len(val_tracks_actual)}): {val_tracks_actual}")
    print(f"    Val Classes    : DISENGAGED (0)={sum(y_val==0):,} ({sum(y_val==0)/len(y_val)*100:.2f}%), ENGAGED (1)={sum(y_val==1):,} ({sum(y_val==1)/len(y_val)*100:.2f}%)")

    # 3. Class Weights Computation (Mean-Normalized Convention)
    counts = np.bincount(y_train, minlength=2)
    inv_freq = 1.0 / (counts / counts.sum())
    w_mean_norm = inv_freq / inv_freq.mean()
    loss_weights_tensor = torch.tensor(w_mean_norm, dtype=torch.float32, device=device)

    print(f"\n[Loss Weight Configuration]:")
    print(f"  Class 0 (DISENGAGED, N={counts[0]}): weight = {loss_weights_tensor[0]:.4f}")
    print(f"  Class 1 (ENGAGED,    N={counts[1]}): weight = {loss_weights_tensor[1]:.4f}")
    print(f"  Weight Ratio (w0 / w1)             : {loss_weights_tensor[0] / loss_weights_tensor[1]:.4f}")

    # 4. Architecture & Pretrained Backbone Initialization
    backbone_path = WORKSPACE_ROOT / args.init_backbone
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing backbone initialization: {backbone_path}")

    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    init_ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
    base_net.load_state_dict(init_ckpt["model"], strict=False)

    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True).to(device)

    # 5. Loss Criterion Selection
    if args.loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor)
        print(f"  Criterion: Weighted Cross-Entropy Loss (Mean-Normalized)")
    elif args.loss_type == "focal_loss":
        criterion = ClassBalancedFocalLoss(weight=loss_weights_tensor, gamma=args.focal_gamma)
        print(f"  Criterion: Class-Balanced Focal Loss (gamma={args.focal_gamma})")
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # 6. Optimizer Setup
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
    y_val_t = torch.from_numpy(y_val).to(device)

    # 7. Training Loop with Epoch-Level Metrics
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    best_metrics = {}
    history = []

    print("\n" + "-" * 115)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val BAcc':^9} | {'Macro F1':^9} | {'F1 Dis':^8} | {'F1 Eng':^8} | {'Selection Status':^16}")
    print("-" * 115)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train pass
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
        avg_train_loss = train_loss_total / max(1, train_batches)

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
            y_pred = np.argmax(val_probs, axis=-1)

        val_acc = float(accuracy_score(y_val, y_pred))
        val_bacc = float(balanced_accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_val, y_pred, average="weighted"))

        prec, rec, f1s, sup = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist()

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bacc,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "precision_disengaged": float(prec[0]),
            "recall_disengaged": float(rec[0]),
            "f1_disengaged": float(f1s[0]),
            "support_disengaged": int(sup[0]),
            "precision_engaged": float(prec[1]),
            "recall_engaged": float(rec[1]),
            "f1_engaged": float(f1s[1]),
            "support_engaged": int(sup[1]),
            "confusion_matrix": cm,
        }
        history.append(epoch_record)

        # Primary selection: Highest Validation Macro F1. Tie-breaker: Lowest Validation Loss.
        is_best = False
        if macro_f1 > best_macro_f1 + 1e-5:
            is_best = True
        elif abs(macro_f1 - best_macro_f1) <= 1e-5 and val_loss < best_val_loss:
            is_best = True

        status_str = ""
        if is_best:
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = epoch_record
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status_str = f"[*] BEST (Ep {epoch:02d})"

        print(
            f"{epoch:^7d} | {avg_train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc*100:^8.2f}% | "
            f"{val_bacc*100:^8.2f}% | {macro_f1*100:^8.2f}% | {f1s[0]*100:^7.2f}% | {f1s[1]*100:^7.2f}% | {status_str:<16}"
        )

    elapsed = time.time() - start_time
    print("-" * 115)
    print(f"Training completed in {elapsed:.2f}s ({elapsed/args.epochs:.2f}s/epoch)")
    print(f"Optimal Checkpoint Selected at Epoch {best_epoch} with Validation Macro F1 = {best_macro_f1*100:.2f}% (Val Loss: {best_val_loss:.4f})")

    # 8. Save Best Checkpoint
    checkpoint_payload = {
        "epoch": best_epoch,
        "model": best_state_dict,
        "val_metrics": best_metrics,
        "seed": args.seed,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma if args.loss_type == "focal_loss" else None,
        "class_names": ["DISENGAGED", "ENGAGED"],
        "class_order": [0, 1],
        "class_weights": loss_weights_tensor.cpu().numpy().tolist(),
        "weight_convention": "mean_normalized",
        "val_tracks": val_track_list,
        "train_sequences": len(x_train),
        "val_sequences": len(x_val),
        "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D])",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(checkpoint_payload, out_ckpt_path)
    print(f"\n[Saved Candidate Checkpoint]: {out_ckpt_path}")

    # 9. Save History JSON
    with open(out_history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "epochs": history,
            },
            f,
            indent=2,
        )
    print(f"[Saved Training History]: {out_history_path}")

    # 10. Post-Training Checkpoint Invariance Verification
    post_sha256 = verify_active_checkpoint()
    print(f"[Post-Training Checkpoint SHA-256]: {post_sha256} (Invariant / 100% Preserved)")

    return best_metrics


if __name__ == "__main__":
    run_training()
