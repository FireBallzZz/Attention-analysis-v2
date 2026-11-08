"""Train AttentionFusionNet on packed (T, 32) sequences.

NPZ keys:
  features: (N, T, 32) float32
  score: (N,) float32 in 0–100   (optional)
  aux: (N, 4) float32 DAiSEE 0–3  (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.timed_checkpoint import TimedCheckpoint, resume_path_for
from attention_pipeline.utils import resolve_device


class SequenceNpz(Dataset):
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.x = data["features"].astype(np.float32)
        self.score = data["score"].astype(np.float32) if "score" in data else None
        self.aux = data["aux"].astype(np.float32) if "aux" in data else None
        self.clip_ids = data["clip_ids"].astype(str) if "clip_ids" in data else None
        self.subject_ids = data["subject_ids"].astype(str) if "subject_ids" in data else None
        if self.score is None and self.aux is None:
            raise ValueError("NPZ must contain 'score' and/or 'aux'")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        item = {"x": torch.from_numpy(self.x[i])}
        if self.score is not None:
            item["score"] = torch.tensor(self.score[i])
        if self.aux is not None:
            item["aux"] = torch.from_numpy(self.aux[i])
        return item


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/processed/fusion_daisee.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="checkpoints/fusion_daisee.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint-minutes", type=float, default=30)
    p.add_argument("--fresh", action="store_true", help="Ignore a saved resume checkpoint")
    p.add_argument(
        "--split-by",
        choices=["subject", "clip"],
        default="subject",
        help="Grouped partition level: 'subject' (subject-disjoint) or 'clip' (default: subject)",
    )
    p.add_argument(
        "--allow-legacy-split",
        action="store_true",
        help="Allow random sequence-level split if clip_ids is missing from NPZ",
    )
    args = p.parse_args()
    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run: python scripts/train_all.py"
        )
    ds = SequenceNpz(path)

    # Phase 3: Subject-disjoint & Clip-grouped split with explicit anti-leakage assertions
    if ds.clip_ids is not None:
        rng = np.random.default_rng(args.seed)
        use_subject = args.split_by == "subject" and ds.subject_ids is not None
        if use_subject:
            group_keys = ds.subject_ids
            unique_groups = np.unique(group_keys)
            shuffled = rng.permutation(unique_groups)
            n_val_groups = max(1, int(round(len(shuffled) * 0.2)))
            val_group_set = set(shuffled[:n_val_groups])
            train_group_set = set(shuffled[n_val_groups:])
            train_indices = [i for i, g in enumerate(group_keys) if g in train_group_set]
            val_indices = [i for i, g in enumerate(group_keys) if g in val_group_set]

            # Strict anti-leakage assertion
            train_subjects = set(ds.subject_ids[train_indices])
            val_subjects = set(ds.subject_ids[val_indices])
            train_clips = set(ds.clip_ids[train_indices])
            val_clips = set(ds.clip_ids[val_indices])
            assert len(train_subjects & val_subjects) == 0, "Subject data leakage detected across splits!"
            assert len(train_clips & val_clips) == 0, "Clip data leakage detected across splits!"

            print(
                f"Subject-disjoint split: {len(train_subjects)} train subjects ({len(train_clips)} clips, {len(train_indices)} seqs), "
                f"{len(val_subjects)} val subjects ({len(val_clips)} clips, {len(val_indices)} seqs). Zero subject/clip overlap."
            )
        else:
            group_keys = ds.clip_ids
            unique_groups = np.unique(group_keys)
            shuffled = rng.permutation(unique_groups)
            n_val_groups = max(1, int(round(len(shuffled) * 0.2)))
            val_group_set = set(shuffled[:n_val_groups])
            train_group_set = set(shuffled[n_val_groups:])
            train_indices = [i for i, g in enumerate(group_keys) if g in train_group_set]
            val_indices = [i for i, g in enumerate(group_keys) if g in val_group_set]

            train_clips = set(ds.clip_ids[train_indices])
            val_clips = set(ds.clip_ids[val_indices])
            assert len(train_clips & val_clips) == 0, "Clip data leakage detected across splits!"

            print(
                f"Grouped clip-level split: {len(train_clips)} train clips ({len(train_indices)} seqs), "
                f"{len(val_clips)} val clips ({len(val_indices)} seqs). Zero temporal clip overlap."
            )

        train_ds = torch.utils.data.Subset(ds, train_indices)
        val_ds = torch.utils.data.Subset(ds, val_indices)
    else:
        if not args.allow_legacy_split:
            raise SystemExit(
                f"Dataset {path} does not contain 'clip_ids' or 'subject_ids' for grouped splitting.\n"
                "To prevent temporal data leakage across training and validation splits,\n"
                "re-run data generation with the updated scripts/train_all.py, or specify\n"
                "--allow-legacy-split to run with sequence-level random split for backward testing."
            )
        print("WARNING: Neither 'clip_ids' nor 'subject_ids' found in NPZ. Falling back to legacy random sequence split (risk of temporal leakage).")
        n_val = max(1, len(ds) // 5)
        split_gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val], generator=split_gen
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    score_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()
    best = 1e9
    start_epoch = 1
    out = Path(args.out)
    ckpt = TimedCheckpoint(resume_path_for(out), minutes=args.checkpoint_minutes)
    saved = None if args.fresh else ckpt.load(map_location=device)
    if saved:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        best = float(saved.get("best", best))
        start_epoch = int(saved["epoch"]) + (1 if saved.get("epoch_finished") else 0)
        print(f"Resuming fusion training at epoch {start_epoch} from {ckpt.path}")
    else:
        print(f"Fusion checkpoint every {args.checkpoint_minutes:g} min -> {ckpt.path}")

    def _payload(epoch: int, finished: bool) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "epoch_finished": finished,
            "best": best,
        }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            pred_s, pred_a = model(x)
            loss = 0.0
            if "score" in batch:
                loss = loss + score_loss(pred_s, batch["score"].to(device))
            if "aux" in batch:
                loss = loss + 0.5 * aux_loss(pred_a, batch["aux"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ckpt.maybe_save(lambda epoch=epoch: _payload(epoch, False))

        # Task 5: Rigorous validation evaluating all targets and preventing fake 0.0 loss
        model.eval()
        total_loss = 0.0
        total_score_loss = 0.0
        total_aux_loss = 0.0
        n_samples = 0
        n_score = 0
        n_aux = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                pred_s, pred_a = model(x)
                bsz = x.size(0)
                batch_loss = 0.0
                has_target = False
                if "score" in batch:
                    s_loss = score_loss(pred_s, batch["score"].to(device))
                    total_score_loss += float(s_loss.item()) * bsz
                    batch_loss = batch_loss + s_loss
                    n_score += bsz
                    has_target = True
                if "aux" in batch:
                    a_loss = aux_loss(pred_a, batch["aux"].to(device))
                    total_aux_loss += float(a_loss.item()) * bsz
                    batch_loss = batch_loss + 0.5 * a_loss
                    n_aux += bsz
                    has_target = True
                if not has_target:
                    raise RuntimeError("Validation batch has neither 'score' nor 'aux' target. Cannot evaluate loss.")
                total_loss += float(batch_loss.item()) * bsz
                n_samples += bsz

        if n_samples == 0:
            raise RuntimeError("Validation set evaluated 0 samples. Cannot compute validation loss.")

        val = total_loss / n_samples
        val_s = (total_score_loss / n_score) if n_score else float('nan')
        val_a = (total_aux_loss / n_aux) if n_aux else float('nan')
        print(f"epoch {epoch:03d}  val_total {val:.4f}  (val_score: {val_s:.4f}, val_aux: {val_a:.4f})")
        if val <= best:
            best = val
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "val": val,
                    "val_score": val_s,
                    "val_aux": val_a,
                    "epoch": epoch,
                },
                out,
            )
        ckpt.save(_payload(epoch, True), reason=f"epoch {epoch}")


if __name__ == "__main__":
    main()
