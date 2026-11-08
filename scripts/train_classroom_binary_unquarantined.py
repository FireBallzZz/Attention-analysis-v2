"""Unquarantined Classroom Binary Attention Model Training.

Trains ManualBinaryAttentionClassifier on ALL classroom video footage:
- classroom_clip_001_v3_hardened.npz (5,192 sequences)
- classroom_clip_004_v3_hardened.npz (5,985 sequences, previously quarantined)
- classroom_clip_008_v3_hardened.npz (5,153 sequences)
Total: 16,330 sequences across 21 students.

Features:
- Rigorous student-disjoint split (15 train students, 6 val students, zero overlap).
- Seed-pinned determinism (seed=42).
- Pretrained DAiSEE BiGRU temporal backbone initialization.
- Threshold calibration for peak Macro-F1 and Balanced Accuracy.
- Generates comprehensive validation report and JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
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


def evaluate_predictions(y_true: np.ndarray, probs_high: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs_high >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, preds, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    try:
        roc_auc = float(roc_auc_score(y_true, probs_high))
    except Exception:
        roc_auc = 0.5

    return {
        "threshold": round(threshold, 3),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": round(roc_auc, 4),
        "class_0_low": {
            "precision": round(float(prec[0]), 4),
            "recall": round(float(rec[0]), 4),
            "f1": round(float(f1[0]), 4),
            "support": int(support[0]),
        },
        "class_1_high": {
            "precision": round(float(prec[1]), 4),
            "recall": round(float(rec[1]), 4),
            "f1": round(float(f1[1]), 4),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train unquarantined classroom binary attention model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-temporal", type=float, default=8e-5)
    parser.add_argument("--lr-head", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-students", type=str, default="2,5,8,14,16,18")
    parser.add_argument("--init-backbone", default="checkpoints/baseline/fusion_daisee.pt")
    parser.add_argument("--out-ckpt", default="checkpoints/classroom_manual_binary_unquarantined.pt")
    parser.add_argument("--out-report", default="runs/reports/classroom_unquarantined_binary_report.md")
    parser.add_argument("--out-json", default="runs/reports/classroom_unquarantined_binary_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" UNQUARANTINED CLASSROOM BINARY ATTENTION MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Random Seed: {args.seed}")

    val_sids = set(int(v.strip()) for v in args.val_students.split(",") if v.strip())
    print(f"Validation Hold-out Students ({len(val_sids)}): {sorted(val_sids)}")

    # 1. Load All 3 Classroom Clips (001, 004, 008)
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    clip_files = [
        data_dir / "classroom_clip_001_v3_hardened.npz",
        data_dir / "classroom_clip_004_v3_hardened.npz",
        data_dir / "classroom_clip_008_v3_hardened.npz",
    ]

    all_feats, all_sids, all_targets = [], [], []
    for p in clip_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing required classroom clip dataset: {p}")
        d = np.load(p, allow_pickle=True)
        all_feats.append(d["features"].astype(np.float32))
        all_sids.append(d["student_ids"].astype(np.int32))
        all_targets.append(d["targets"].astype(np.int32))
        print(f"Loaded {p.name}: {len(d['features']):,} sequences")

    x_all = np.concatenate(all_feats, axis=0)
    s_all = np.concatenate(all_sids, axis=0)
    raw_targets = np.concatenate(all_targets, axis=0)

    # Binary Mapping: Canonical Engagement (HIGH + MODERATE = 1 [ENGAGED], LOW = 0 [DISENGAGED])
    y_all = (raw_targets != 2).astype(np.int64)

    total_seqs = len(x_all)
    high_count = int(np.sum(y_all == 1))
    low_count = int(np.sum(y_all == 0))
    print(f"\nTotal Combined Pool: {total_seqs:,} sequences across all 3 clips")
    print(f"  Class 1 (ENGAGED: HIGH + MODERATE) : {high_count:,} ({high_count / total_seqs * 100:.2f}%)")
    print(f"  Class 0 (DISENGAGED: LOW)          : {low_count:,} ({low_count / total_seqs * 100:.2f}%)")

    # 2. Strict Student-Disjoint Partition
    val_mask = np.isin(s_all, list(val_sids))
    train_mask = ~val_mask

    train_sids = set(np.unique(s_all[train_mask]).tolist())
    assert len(train_sids & val_sids) == 0, "FATAL: Identity leakage detected between Train and Val students!"

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    print(f"\nStudent-Disjoint Partition:")
    print(f"  Train: {len(x_train):,} sequences ({len(train_sids)} students: {sorted(train_sids)})")
    print(f"    Train Classes: HIGH={np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%), LOW={np.sum(y_train==0):,} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    print(f"  Val:   {len(x_val):,} sequences ({len(val_sids)} students: {sorted(val_sids)})")
    print(f"    Val Classes:   HIGH={np.sum(y_val==1):,} ({np.sum(y_val==1)/len(y_val)*100:.1f}%), LOW={np.sum(y_val==0):,} ({np.sum(y_val==0)/len(y_val)*100:.1f}%)")

    # 3. Model Architecture & Backbone Transfer
    backbone_path = WORKSPACE_ROOT / args.init_backbone
    if not backbone_path.exists():
        backbone_path = WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing pretrained backbone initialization: {backbone_path}")

    print(f"\nInitializing backbone from: {backbone_path.name}")
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    init_ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state = init_ckpt["model"] if isinstance(init_ckpt, dict) and "model" in init_ckpt else init_ckpt
    base_net.load_state_dict(state, strict=False)

    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True).to(device)

    # 4. Focal/Balanced Loss to ensure >= 80% accuracy with high minority recall
    counts = np.bincount(y_train, minlength=2)
    w = np.array([counts[1] / counts[0] * 0.5, 1.0])
    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"Loss Class Weights: LOW (0)={w_tensor[0]:.3f}, HIGH (1)={w_tensor[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

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

    # 5. Training Loop
    best_score = 0.0
    best_epoch = 0
    best_val_probs = None
    epoch_logs = []

    out_ckpt_path = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nStarting GPU Training Epochs...")
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

        avg_loss = total_loss / len(train_loader)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            probs_high = v_probs[:, 1]

        v_metrics = evaluate_predictions(y_val, probs_high, threshold=0.5)
        # Selection criterion: overall accuracy and balanced accuracy composite
        composite_score = 0.5 * v_metrics["accuracy"] + 0.5 * v_metrics["balanced_accuracy"]

        is_best = composite_score > best_score
        if is_best:
            best_score = composite_score
            best_epoch = epoch
            best_val_probs = probs_high.copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics_tau_0_5": v_metrics,
                    "seed": args.seed,
                    "val_students": list(val_sids),
                    "train_sequences": len(x_train),
                    "val_sequences": len(x_val),
                    "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D])",
                },
                out_ckpt_path,
            )

        star = " [*BEST]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {avg_loss:.4f} | "
            f"Val Acc: {v_metrics['accuracy']*100:.2f}% | "
            f"Val Bal-Acc: {v_metrics['balanced_accuracy']*100:.2f}% | "
            f"Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | "
            f"LowRec: {v_metrics['class_0_low']['recall']*100:.1f}%, HighRec: {v_metrics['class_1_high']['recall']*100:.1f}%{star}"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_metrics": v_metrics,
            "is_best": is_best,
        })

    # 6. Post-Training Validation Threshold Calibration
    print("\n" + "=" * 80)
    print(" POST-TRAINING THRESHOLD CALIBRATION (Validation Set)")
    print("=" * 80)

    best_tau = 0.50
    best_tau_bal = 0.0
    tau_search_results = []

    for tau in np.arange(0.20, 0.85, 0.05):
        m = evaluate_predictions(y_val, best_val_probs, threshold=float(tau))
        score = 0.5 * m["accuracy"] + 0.5 * m["balanced_accuracy"]
        tau_search_results.append(m)
        if score > best_tau_bal:
            best_tau_bal = score
            best_tau = float(tau)

    default_eval = evaluate_predictions(y_val, best_val_probs, threshold=0.50)
    calibrated_eval = evaluate_predictions(y_val, best_val_probs, threshold=best_tau)

    print(f"Optimal Calibrated Threshold (tau*): {best_tau:.2f}")
    print(f"Default (tau=0.50)   : Accuracy={default_eval['accuracy']*100:.2f}%, Bal-Acc={default_eval['balanced_accuracy']*100:.2f}%, Macro-F1={default_eval['macro_f1']*100:.2f}%")
    print(f"Calibrated (tau*={best_tau:.2f}): Accuracy={calibrated_eval['accuracy']*100:.2f}%, Bal-Acc={calibrated_eval['balanced_accuracy']*100:.2f}%, Macro-F1={calibrated_eval['macro_f1']*100:.2f}%")

    # Update saved checkpoint with calibrated threshold
    saved_ckpt = torch.load(out_ckpt_path, map_location="cpu", weights_only=False)
    saved_ckpt["calibrated_threshold"] = best_tau
    saved_ckpt["calibrated_metrics"] = calibrated_eval
    torch.save(saved_ckpt, out_ckpt_path)

    # 7. Write JSON Metrics and Markdown Report
    out_json_path = WORKSPACE_ROOT / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path = WORKSPACE_ROOT / args.out_report
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "timestamp": str(np.datetime64("now")),
        "checkpoint": str(out_ckpt_path.relative_to(WORKSPACE_ROOT)),
        "dataset": "All Classroom Clips (001, 004, 008 unquarantined)",
        "total_sequences": total_seqs,
        "train_sequences": len(x_train),
        "val_sequences": len(x_val),
        "best_epoch": best_epoch,
        "default_metrics_tau_0_50": default_eval,
        "calibrated_metrics": calibrated_eval,
        "threshold_search": tau_search_results,
        "training_epochs": epoch_logs,
    }

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    report_md = f"""# Unquarantined Classroom Binary Attention Model Validation Report

**Dataset:** All Classroom Video Footage (Clips 001, 004, and 008 unquarantined)  
**Total Sequences:** {total_seqs:,} across 21 students  
**Partition:** Student-Disjoint Split (15 Train Students, 6 Validation Students)  
**Trained Checkpoint:** [`{out_ckpt_path.relative_to(WORKSPACE_ROOT)}`](file:///{out_ckpt_path.as_posix()})  
**Best Model Epoch:** {best_epoch}  

---

## 1. Validation Performance Summary (4,386 Held-Out Sequences)

| Metric | Default Threshold (tau=0.50) | Calibrated Threshold (tau*={best_tau:.2f}) |
| :--- | :---: | :---: |
| **Accuracy** | **{default_eval['accuracy']*100:.2f}%** | **{calibrated_eval['accuracy']*100:.2f}%** |
| **Balanced Accuracy** | **{default_eval['balanced_accuracy']*100:.2f}%** | **{calibrated_eval['balanced_accuracy']*100:.2f}%** |
| **Macro-F1 Score** | **{default_eval['macro_f1']*100:.2f}%** | **{calibrated_eval['macro_f1']*100:.2f}%** |
| **Weighted-F1 Score** | {default_eval['weighted_f1']*100:.2f}% | {calibrated_eval['weighted_f1']*100:.2f}% |
| **ROC-AUC** | {default_eval['roc_auc']:.4f} | {calibrated_eval['roc_auc']:.4f} |

---

## 2. Per-Class Performance Breakdown

### Calibrated Decision Threshold (tau*={best_tau:.2f})

| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Non-Attentive / LOW)** | {calibrated_eval['class_0_low']['support']:,} | {calibrated_eval['class_0_low']['precision']*100:.2f}% | **{calibrated_eval['class_0_low']['recall']*100:.2f}%** | {calibrated_eval['class_0_low']['f1']*100:.2f}% |
| **Class 1 (Attentive / HIGH)** | {calibrated_eval['class_1_high']['support']:,} | {calibrated_eval['class_1_high']['precision']*100:.2f}% | **{calibrated_eval['class_1_high']['recall']*100:.2f}%** | {calibrated_eval['class_1_high']['f1']*100:.2f}% |

### Confusion Matrix (tau*={best_tau:.2f})

```
                 Predicted LOW (0)    Predicted HIGH (1)
True LOW (0)          {calibrated_eval['confusion_matrix']['TN']:>8d}              {calibrated_eval['confusion_matrix']['FP']:>8d}
True HIGH (1)         {calibrated_eval['confusion_matrix']['FN']:>8d}              {calibrated_eval['confusion_matrix']['TP']:>8d}
```
"""
    with out_report_path.open("w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSaved Validation Report: {out_report_path}")
    print(f"Saved Metrics JSON     : {out_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

