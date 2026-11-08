"""Option 2 Step 2: Fine-tune temporal GRU while keeping frame encoder frozen.

Experiment: gru_finetuned_v1
- Pretrained base: checkpoints/fusion_daisee.pt
- Initialized from: checkpoints/classroom_attention_head_focal_v2.pt
- Frozen: frame_encoder.* (0 grad updates)
- Trainable: temporal.* (AdamW lr=1e-4), classifier.* (AdamW lr=1e-3)
- Loss: Focal Loss (gamma=2.0, alpha=[0.46356538, 2.85571146, 2.02991462])
- Split: Student-disjoint within session (Train: 15 students, Val: 6 students)
- Out checkpoint: checkpoints/classroom_attention_gru_finetuned_v1.pt
- Out experiment dir: experiments/gru_finetuned_v1/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.utils import PROJECT_ROOT, resolve_device

BANDS = ("HIGH", "MODERATE", "LOW")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    ) -> None:
        super().__init__()
        self.frame_encoder = pretrained_fusion.frame_encoder
        self.temporal = pretrained_fusion.temporal

        # Freeze frame_encoder parameters explicitly
        for p in self.frame_encoder.parameters():
            p.requires_grad = False

        # Ensure temporal GRU parameters are trainable
        for p in self.temporal.parameters():
            p.requires_grad = True

        fused_dim = 128  # 64 * 2 (bidirectional GRU)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        with torch.no_grad():
            encoded = self.frame_encoder(x.reshape(b * t, -1)).reshape(b, t, -1)
        out, _ = self.temporal(encoded)
        h = out[:, -1, :]
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    recalls = []
    precisions = []
    f1s = []
    support = []

    for c in range(3):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        sup = int(cm[c, :].sum())

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        support.append(sup)

    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune temporal GRU with frozen frame encoder")
    parser.add_argument("--npz", default="data/processed/classroom_clip_004_sequences.npz")
    parser.add_argument("--init-ckpt", default="checkpoints/classroom_attention_head_focal_v2.pt")
    parser.add_argument("--fusion-ckpt", default="checkpoints/fusion_daisee.pt")
    parser.add_argument("--out-ckpt", default="checkpoints/classroom_attention_gru_finetuned_v1.pt")
    parser.add_argument("--exp-dir", default="experiments/gru_finetuned_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-gru", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz_path = PROJECT_ROOT / args.npz
    init_ckpt_path = PROJECT_ROOT / args.init_ckpt
    fusion_ckpt_path = PROJECT_ROOT / args.fusion_ckpt
    out_ckpt_path = PROJECT_ROOT / args.out_ckpt
    exp_dir_path = PROJECT_ROOT / args.exp_dir

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing classroom sequences NPZ: {npz_path}")
    if not init_ckpt_path.exists():
        raise FileNotFoundError(f"Missing baseline init checkpoint: {init_ckpt_path}")
    if not fusion_ckpt_path.exists():
        raise FileNotFoundError(f"Missing pretrained fusion checkpoint: {fusion_ckpt_path}")

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    # Hashes of input artifacts for absolute reproducibility
    input_hashes = {
        str(npz_path.relative_to(PROJECT_ROOT)): sha256_file(npz_path),
        str(init_ckpt_path.relative_to(PROJECT_ROOT)): sha256_file(init_ckpt_path),
        str(fusion_ckpt_path.relative_to(PROJECT_ROOT)): sha256_file(fusion_ckpt_path),
    }
    print("Input artifact verification:")
    for k, v in input_hashes.items():
        print(f"  {k:50s} SHA256: {v}")

    # Set exact seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device("auto")
    print(f"Device: {device}")

    # Exact audited student-disjoint split
    val_students = {2, 5, 8, 10, 16, 18}
    all_data = np.load(npz_path)
    all_sids = set(np.unique(all_data["student_ids"]).tolist())
    train_students = all_sids - val_students

    # Strict anti-leakage assertions
    assert len(train_students & val_students) == 0, "FATAL: Student leakage detected between train and val!"
    assert train_students == {0, 1, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 19, 20}, "Train student IDs mismatch audited roster!"
    assert val_students == {2, 5, 8, 10, 16, 18}, "Val student IDs mismatch audited roster!"
    print(f"Train students ({len(train_students)}): {sorted(train_students)}")
    print(f"Val students   ({len(val_students)}): {sorted(val_students)}")

    train_ds = ClassroomSequenceDataset(npz_path, student_filter=train_students)
    val_ds = ClassroomSequenceDataset(npz_path, student_filter=val_students)
    assert len(train_ds) == 4275, f"Expected 4275 train sequences, got {len(train_ds)}"
    assert len(val_ds) == 1710, f"Expected 1710 val sequences, got {len(val_ds)}"
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_targets = train_ds.targets
    val_targets = val_ds.targets
    print("Train class distribution:", {BANDS[c]: int((train_targets == c).sum()) for c in range(3)})
    print("Val class distribution:  ", {BANDS[c]: int((val_targets == c).sum()) for c in range(3)})

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Initialize model
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ClassroomAttentionClassifier(pretrained_fusion=fusion_base, num_classes=3).to(device)

    # Load from audited focal-loss checkpoint
    init_ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
    init_state = init_ckpt["model"]
    model.load_state_dict(init_state)
    print(f"Loaded weights from {init_ckpt_path.name} (trained at Epoch {init_ckpt.get('epoch')})")

    # Verify that frame_encoder matches fusion_daisee.pt bit-for-bit at initialization
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)["model"]
    for k in init_state.keys():
        if k.startswith("frame_encoder."):
            diff = torch.max(torch.abs(model.state_dict()[k] - fusion_ckpt[k])).item()
            assert diff == 0.0, f"FATAL: frame_encoder weight {k} does not match fusion_daisee.pt!"
    print("Pre-training assertion PASSED: frame_encoder matches fusion_daisee.pt bit-for-bit.")

    # Snapshot initial weights for post-training assertions
    initial_fe_weights = {k: v.clone().detach().cpu() for k, v in model.frame_encoder.state_dict().items()}
    initial_gru_weights = {k: v.clone().detach().cpu() for k, v in model.temporal.state_dict().items()}

    # Parameter groups with separate learning rates
    temporal_params = [p for p in model.temporal.parameters() if p.requires_grad]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    assert len(temporal_params) == 8, f"Expected 8 temporal GRU tensors, got {len(temporal_params)}"
    assert len(classifier_params) == 6, f"Expected 6 classifier tensors, got {len(classifier_params)}"

    optimizer = torch.optim.AdamW(
        [
            {"params": temporal_params, "lr": args.lr_gru, "weight_decay": args.weight_decay},
            {"params": classifier_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        ]
    )

    # Verified class weights
    class_counts = np.bincount(train_targets, minlength=3)
    total_samples = len(train_targets)
    weights = total_samples / (3.0 * np.maximum(class_counts, 1))
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    assert np.allclose(weights, [0.46356538, 2.85571146, 2.02991462], atol=1e-5), "Class weights mismatch!"
    print(f"Verified loss weights (HIGH/MOD/LOW): {weights_tensor.tolist()}")

    criterion = FocalLoss(alpha=weights_tensor, gamma=args.focal_gamma)

    best_macro_f1 = 0.0
    best_bal_acc = 0.0
    best_raw_acc = 0.0
    best_epoch = 0
    best_metrics = None
    best_preds = None
    best_logits = None

    epoch_history = []

    print("\n--- Starting Temporal GRU Fine-Tuning ---")
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

        # Validation on held-out students
        model.eval()
        val_preds = []
        val_trues = []
        val_all_logits = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_trues.extend(y.cpu().numpy().tolist())
                val_all_logits.extend(logits.cpu().numpy().tolist())

        val_trues_np = np.array(val_trues, dtype=np.int32)
        val_preds_np = np.array(val_preds, dtype=np.int32)
        val_logits_np = np.array(val_all_logits, dtype=np.float32)

        m = compute_metrics(val_trues_np, val_preds_np)
        macro_f1 = m["macro_f1"]
        bal_acc = m["balanced_accuracy"]
        raw_acc = m["accuracy"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_macro_f1": macro_f1,
            "val_balanced_accuracy": bal_acc,
            "val_raw_accuracy": raw_acc,
            "recalls": m["recalls"],
            "precisions": m["precisions"],
            "f1s": m["f1s"],
        }
        epoch_history.append(epoch_record)

        is_best = macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = macro_f1
            best_bal_acc = bal_acc
            best_raw_acc = raw_acc
            best_epoch = epoch
            best_metrics = m
            best_preds = val_preds_np
            best_logits = val_logits_np

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "macro_f1": macro_f1,
                    "balanced_accuracy": bal_acc,
                    "raw_accuracy": raw_acc,
                    "train_students": sorted(train_students),
                    "val_students": sorted(val_students),
                    "metrics": m,
                    "config": vars(args),
                    "input_hashes": input_hashes,
                },
                out_ckpt_path,
            )

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} | "
            f"Val Macro-F1: {macro_f1*100:.2f}% | Val Bal-Acc: {bal_acc*100:.2f}% | Val Acc: {raw_acc*100:.2f}% | "
            f"Recalls: [H:{m['recalls'][0]*100:.1f}%, M:{m['recalls'][1]*100:.1f}%, L:{m['recalls'][2]*100:.1f}%]{star}"
        )

    # Post-training integrity assertions
    print("\n--- Running Post-Training Integrity Assertions ---")
    final_fe_weights = {k: v.clone().detach().cpu() for k, v in model.frame_encoder.state_dict().items()}
    final_gru_weights = {k: v.clone().detach().cpu() for k, v in model.temporal.state_dict().items()}

    for k in initial_fe_weights.keys():
        diff = torch.max(torch.abs(initial_fe_weights[k] - final_fe_weights[k])).item()
        assert diff == 0.0, f"FATAL LEAKAGE: frame_encoder weight {k} changed during training (diff={diff})!"
    print("Assertion PASSED: All frame_encoder parameters remained 100% frozen bit-for-bit.")

    gru_changes = []
    for k in initial_gru_weights.keys():
        diff = torch.max(torch.abs(initial_gru_weights[k] - final_gru_weights[k])).item()
        gru_changes.append(diff)
    max_gru_change = max(gru_changes)
    assert max_gru_change > 1e-4, f"FATAL: No temporal GRU parameters updated! Max change = {max_gru_change}"
    print(f"Assertion PASSED: Temporal GRU parameters successfully updated (max weight delta = {max_gru_change:.6f}).")

    # Save experiment artifacts in exp_dir
    config_record = {
        "experiment_name": "gru_finetuned_v1",
        "description": "Fine-tuning temporal GRU on classroom ground truth with frozen frame encoder",
        "methodological_scope": "Student-disjoint within-session validation (same classroom recording, disjoint students)",
        "args": vars(args),
        "input_hashes": input_hashes,
        "train_students": sorted(train_students),
        "val_students": sorted(val_students),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "class_weights": weights.tolist(),
        "best_epoch": best_epoch,
    }
    with open(exp_dir_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    with open(exp_dir_path / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)

    with open(exp_dir_path / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "macro_f1": best_macro_f1,
                "balanced_accuracy": best_bal_acc,
                "raw_accuracy": best_raw_acc,
                "metrics": best_metrics,
            },
            f,
            indent=2,
        )

    np.savez_compressed(
        exp_dir_path / "val_predictions.npz",
        y_true=val_trues_np,
        y_pred=best_preds,
        logits=best_logits,
        student_ids=val_ds.student_ids,
        frame_ids=val_ds.frame_ids,
    )
    print(f"Saved all experiment artifacts to {exp_dir_path}/")

    print(f"\n================ EXPERIMENTAL RESULTS: Best Model (Epoch {best_epoch}) ================")
    print(f"  Validation Macro-F1:       {best_macro_f1*100:.2f}%")
    print(f"  Validation Balanced Acc:   {best_bal_acc*100:.2f}%")
    print(f"  Validation Raw Accuracy:   {best_raw_acc*100:.2f}%")
    print("\nPer-Class Breakdown:")
    for c, b in enumerate(BANDS):
        print(
            f"  {b:8s}: Support = {best_metrics['support'][c]:4d} | "
            f"Recall = {best_metrics['recalls'][c]*100:.2f}% | "
            f"Precision = {best_metrics['precisions'][c]*100:.2f}% | "
            f"F1 = {best_metrics['f1s'][c]*100:.2f}%"
        )
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print("                  Pred HIGH   Pred MODERATE   Pred LOW")
    for r_idx, b in enumerate(BANDS):
        row_str = f"  True {b:8s}: " + "       ".join(f"{best_metrics['confusion_matrix'][r_idx][c_idx]:4d}" for c_idx in range(3))
        print(row_str)
    print(f"\nSaved checkpoint: {out_ckpt_path}")


if __name__ == "__main__":
    main()
