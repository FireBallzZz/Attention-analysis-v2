"""Retrain the DAiSEE pretraining backbone on the full v4 dataset.

Why this script exists
----------------------
The deployed pretraining checkpoint (`checkpoints/baseline/fusion_daisee.pt`)
has a score head whose correlation with the DAiSEE engagement label is
r = -0.05, i.e. it predicts a near-constant and learned nothing useful. The
classroom classifier then FREEZES that backbone's frame encoder, so the
deployed classroom model inherits frozen weights from a failed pretraining
run.

Two causes were identified:

  1. It was trained on the small 750-clip extract (`fusion_daisee.npz`,
     23k windows), where only 14 of 32 channels carried any variance and
     the chest/keypoint confidence channel was entirely zero.
  2. The score head and aux head were optimised jointly with a plain
     SmoothL1 loss, which lets the easy constant targets (frustration mean
     0.35, confusion mean 0.48) dominate the gradient while the hard
     engagement signal is drowned out.

This script retrains on the full v4 dataset (306,801 windows, 8,986 clips,
subject-disjoint Train/Validation/Test splits) with:

  * a dedicated regression head for the continuous engagement score,
  * a multi-task aux head (engagement, boredom, confusion, frustration),
  * per-target normalisation so each target contributes comparable gradient,
  * MAE-based early stopping on the validation engagement score,
  * a hard cap on which channels may be used, so the dead channels cannot
    silently re-enter.

Usage:
  python scripts/train_daisee_pretrain_v5.py \
      --npz data/processed/fusion_daisee_full_v4.npz \
      --out checkpoints/experiments/phase11_daisee/fusion_daisee_v5.pt
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
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet  # noqa: E402
from attention_pipeline.types import DAISEE_LABELS  # noqa: E402

# Channels that carry no signal in the DAiSEE feature extract (see
# runs/reports/phase9_rebuild_baseline/feature_schema_and_degeneracy.csv).
# The emotion block (5-11) is excluded by policy; 19/20/24 are dead because
# DAiSEE footage never produced a usable face crop for those measurements.
DEAD_CHANNELS = tuple(range(5, 12)) + (19, 20, 24) + tuple(range(25, 32))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MultiTaskPretrainer(nn.Module):
    """Backbone with a score-regression head and a 4-way aux head.

    The score head predicts the continuous engagement score in [0, 100].
    The aux head predicts the four DAiSEE ratings in [0, 3] and is trained
    as an auxiliary task only - it never feeds the binary classroom target.
    """

    def __init__(self, feat_dim: int = 32, hidden: int = 64, num_aux: int = 4) -> None:
        super().__init__()
        self.backbone = AttentionFusionNet(feat_dim=feat_dim, hidden=hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_aux),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone.encode_window(x)
        score = self.score_head(h).squeeze(-1) * 100.0
        aux = self.aux_head(h)
        return score, aux

    def export_backbone(self) -> dict[str, torch.Tensor]:
        """Return a state dict compatible with AttentionFusionNet."""
        return {
            "frame_encoder.0.weight": self.backbone.frame_encoder[0].weight,
            "frame_encoder.0.bias": self.backbone.frame_encoder[0].bias,
            "frame_encoder.2.weight": self.backbone.frame_encoder[2].weight,
            "frame_encoder.2.bias": self.backbone.frame_encoder[2].bias,
            "frame_encoder.3.weight": self.backbone.frame_encoder[3].weight,
            "frame_encoder.3.bias": self.backbone.frame_encoder[3].bias,
            "temporal.weight_ih_l0": self.backbone.temporal.weight_ih_l0,
            "temporal.weight_hh_l0": self.backbone.temporal.weight_hh_l0,
            "temporal.weight_ih_l0_reverse": self.backbone.temporal.weight_ih_l0_reverse,
            "temporal.weight_hh_l0_reverse": self.backbone.temporal.weight_hh_l0_reverse,
            "temporal.bias_ih_l0": self.backbone.temporal.bias_ih_l0,
            "temporal.bias_hh_l0": self.backbone.temporal.bias_hh_l0,
            "temporal.bias_ih_l0_reverse": self.backbone.temporal.bias_ih_l0_reverse,
            "temporal.bias_hh_l0_reverse": self.backbone.temporal.bias_hh_l0_reverse,
            "score_head.0.weight": self.backbone.score_head[0].weight,
            "score_head.0.bias": self.backbone.score_head[0].bias,
            "score_head.2.weight": self.backbone.score_head[2].weight,
            "score_head.2.bias": self.backbone.score_head[2].bias,
            "aux_head.weight": self.backbone.aux_head.weight,
            "aux_head.bias": self.backbone.aux_head.bias,
        }


def _mask_dead_channels(x: torch.Tensor) -> torch.Tensor:
    """Zero the channels that carry no signal, so the model cannot latch on."""
    mask = torch.ones(x.shape[-1], device=x.device)
    for ch in DEAD_CHANNELS:
        mask[ch] = 0.0
    return x * mask


def _correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if target.std() < 1e-9 or pred.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the DAiSEE pretraining backbone")
    parser.add_argument("--npz", default="data/processed/fusion_daisee_full_v4.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="0 = use every training window")
    parser.add_argument("--out", default="checkpoints/experiments/phase11_daisee/fusion_daisee_v5.pt")
    parser.add_argument("--metrics-out",
                        default="runs/reports/phase11_daisee_pretrain/pretrain_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = WORKSPACE_ROOT / args.npz
    out_path = WORKSPACE_ROOT / args.out
    metrics_path = WORKSPACE_ROOT / args.metrics_out

    print("=" * 82)
    print(" DAiSEE PRETRAINING v5 — score regression + multi-task aux head")
    print("=" * 82)
    print(f" dataset : {npz_path.name}")
    print(f" device  : {device}")
    print(f" seed    : {args.seed}")

    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    score = d["score"].astype(np.float32).reshape(-1)
    aux = d["aux"].astype(np.float32)
    splits = d["splits"].astype(str)
    subjects = d["subject_ids"].astype(str)

    train_mask = splits == "Train"
    val_mask = splits == "Validation"

    # Anti-leakage: the split must be subject-disjoint.
    train_subjects = set(subjects[train_mask])
    val_subjects = set(subjects[val_mask])
    assert not (train_subjects & val_subjects), "subject leakage between Train and Validation"

    if args.max_train_rows and int(train_mask.sum()) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.flatnonzero(train_mask), size=args.max_train_rows, replace=False)
        sel = np.zeros(len(feats), dtype=bool)
        sel[idx] = True
        train_mask = sel
        print(f" subsampled training set to {args.max_train_rows:,} windows")

    x_train = feats[train_mask]
    s_train = score[train_mask]
    a_train = aux[train_mask]
    x_val = feats[val_mask]
    s_val = score[val_mask]
    a_val = aux[val_mask]

    print(f" train   : {len(x_train):,} windows  ({len(train_subjects)} subjects)")
    print(f" val     : {len(x_val):,} windows  ({len(val_subjects)} subjects)")
    print(f" score   : train mean {s_train.mean():.2f}  val mean {s_val.mean():.2f}")

    # Per-target normalisation so each aux target contributes comparable
    # gradient instead of the constant ones dominating.
    aux_mean = a_train.mean(axis=0)
    aux_std = a_train.std(axis=0).clip(min=1e-3)
    print(f" aux targets: {dict(zip(DAISEE_LABELS, a_train.mean(axis=0).round(3)))}")
    print(f" aux std    : {dict(zip(DAISEE_LABELS, aux_std.round(3)))}")

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(s_train),
        torch.from_numpy((a_train - aux_mean) / aux_std),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model = MultiTaskPretrainer(feat_dim=32, hidden=64).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.frame_encoder.parameters(), "lr": args.lr_backbone},
            {"params": model.backbone.temporal.parameters(), "lr": args.lr_backbone},
            {"params": model.score_head.parameters(), "lr": args.lr_head},
            {"params": model.aux_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    score_loss = nn.SmoothL1Loss(beta=10.0)
    aux_loss = nn.SmoothL1Loss(beta=1.0)

    val_x = torch.from_numpy(x_val).to(device)
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for bx, bs, ba in loader:
            bx = _mask_dead_channels(bx.to(device))
            bs = bs.to(device)
            ba = ba.to(device)
            pred_score, pred_aux = model(bx)
            loss = score_loss(pred_score, bs) + args.aux_weight * aux_loss(pred_aux, ba)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            preds_score, preds_aux = [], []
            for i in range(0, len(val_x), 2048):
                bx = _mask_dead_channels(val_x[i:i + 2048].to(device))
                ps, pa = model(bx)
                preds_score.append(ps.cpu().numpy())
                preds_aux.append(pa.cpu().numpy())
        p_score = np.concatenate(preds_score)
        p_aux = np.concatenate(preds_aux) * aux_std + aux_mean

        mae = float(np.abs(p_score - s_val).mean())
        r_score = _correlation(p_score, s_val)
        aux_r = {
            name: _correlation(p_aux[:, i], a_val[:, i])
            for i, name in enumerate(DAISEE_LABELS)
        }
        record = {
            "epoch": epoch,
            "train_loss": running / max(len(loader), 1),
            "val_score_mae": mae,
            "val_score_correlation": r_score,
            "val_aux_correlation": aux_r,
        }
        history.append(record)
        star = ""
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " [*BEST]"
        else:
            patience_counter += 1

        aux_str = " ".join(f"{k[:4]}={v:+.3f}" for k, v in aux_r.items())
        print(f" epoch {epoch:02d}/{args.epochs}  train {record['train_loss']:.4f}  "
              f"val MAE {mae:7.4f}  r {r_score:+.4f}  |  {aux_str}{star}")

        if patience_counter >= args.patience:
            print(f" early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Final validation metrics
    with torch.no_grad():
        preds_score, preds_aux = [], []
        for i in range(0, len(val_x), 2048):
            bx = _mask_dead_channels(val_x[i:i + 2048].to(device))
            ps, pa = model(bx)
            preds_score.append(ps.cpu().numpy())
            preds_aux.append(pa.cpu().numpy())
    p_score = np.concatenate(preds_score)
    p_aux = np.concatenate(preds_aux) * aux_std + aux_mean

    final = {
        "best_epoch": best_epoch,
        "val_score_mae": float(np.abs(p_score - s_val).mean()),
        "val_score_correlation": _correlation(p_score, s_val),
        "val_score_bias": float((p_score - s_val).mean()),
        "val_aux_correlation": {
            name: _correlation(p_aux[:, i], a_val[:, i])
            for i, name in enumerate(DAISEE_LABELS)
        },
        "val_aux_mae": {
            name: float(np.abs(p_aux[:, i] - a_val[:, i]).mean())
            for i, name in enumerate(DAISEE_LABELS)
        },
    }

    print()
    print("=" * 82)
    print(" FINAL VALIDATION METRICS")
    print("=" * 82)
    print(f"   engagement score : MAE {final['val_score_mae']:.4f}  "
          f"r {final['val_score_correlation']:+.4f}  "
          f"bias {final['val_score_bias']:+.4f}")
    for name in DAISEE_LABELS:
        print(f"   {name:15s}: r {final['val_aux_correlation'][name]:+.4f}  "
              f"MAE {final['val_aux_mae'][name]:.4f}")

    # Comparison against the previously deployed baseline.
    old_path = WORKSPACE_ROOT / "checkpoints" / "baseline" / "fusion_daisee.pt"
    if old_path.exists():
        old_model = AttentionFusionNet(feat_dim=32, hidden=64).to(device).eval()
        old_ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
        old_model.load_state_dict(old_ckpt["model"])
        with torch.no_grad():
            old_preds = []
            for i in range(0, len(val_x), 2048):
                bx = _mask_dead_channels(val_x[i:i + 2048].to(device))
                old_preds.append(old_model(bx)[0].cpu().numpy())
            old_p = np.concatenate(old_preds)
        old_r = _correlation(old_p, s_val)
        old_mae = float(np.abs(old_p - s_val).mean())
        print()
        print(f"   PREVIOUS baseline : MAE {old_mae:.4f}  r {old_r:+.4f}")
        print(f"   NEW backbone      : MAE {final['val_score_mae']:.4f}  "
              f"r {final['val_score_correlation']:+.4f}")
        final["previous_baseline_val_mae"] = old_mae
        final["previous_baseline_val_correlation"] = old_r

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.export_backbone(),
            "pretrained_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": best_epoch,
            "val_metrics": final,
            "aux_mean": aux_mean,
            "aux_std": aux_std,
            "seed": args.seed,
            "dataset": str(npz_path),
            "architecture": "MultiTaskPretrainer (BiGRU + score head + aux head)",
        },
        out_path,
    )
    print(f"\n saved backbone : {out_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": str(npz_path),
                "seed": args.seed,
                "epochs": args.epochs,
                "final": final,
                "history": history,
                "dead_channels_masked": list(DEAD_CHANNELS),
                "checkpoint": str(out_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f" saved metrics  : {metrics_path}")


if __name__ == "__main__":
    main()
