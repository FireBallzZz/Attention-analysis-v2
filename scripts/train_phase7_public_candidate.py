"""Phase 7: Public-Dataset Training, Validation, and Calibration.

Trains a defensible binary student attention classifier strictly on the public DAiSEE dataset.
Enforces:
1. Strict 3-way subject-disjoint partition (configs/phase7_public_daisee_split.json).
2. Held-out test set (15.06%) remains completely untouched during training and tuning.
3. Training-only class weights (w0 = 26.1782, w1 = 0.5097).
4. Early stopping monitored on Validation Balanced Accuracy.
5. Decision threshold calibration tau* on Validation Macro F1.
6. Zero access to UAP classroom data.
7. Verification of protected checkpoint SHA-256 before and after.
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

ACTIVE_CHECKPOINT_PATH = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"
CANDIDATE_6G4C_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
EXPECTED_6G4C_SHA256 = "E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_protected_checkpoints() -> dict[str, str]:
    if not ACTIVE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Active checkpoint not found: {ACTIVE_CHECKPOINT_PATH}")
    if not CANDIDATE_6G4C_PATH.exists():
        raise FileNotFoundError(f"Existing candidate not found: {CANDIDATE_6G4C_PATH}")

    h_active = get_file_sha256(ACTIVE_CHECKPOINT_PATH)
    h_cand = get_file_sha256(CANDIDATE_6G4C_PATH)

    if h_active != EXPECTED_ACTIVE_SHA256:
        raise ValueError(f"Active checkpoint SHA violation! Expected {EXPECTED_ACTIVE_SHA256}, got {h_active}")
    if h_cand != EXPECTED_6G4C_SHA256:
        raise ValueError(f"Candidate 6G-4C SHA violation! Expected {EXPECTED_6G4C_SHA256}, got {h_cand}")

    return {"active_production": h_active, "candidate_phase6g4c": h_cand}


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs_high: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    brier = float(brier_score_loss(y_true, probs_high)) if probs_high is not None else None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "confusion_matrix": {
            "true_non_attentive": tn,
            "false_attentive_fp": fp,
            "false_non_attentive_fn": fn,
            "true_attentive": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "class_0_non_attentive": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(supp[0]),
        },
        "class_1_attentive": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(supp[1]),
        },
    }


def calibrate_threshold(y_true: np.ndarray, probs_high: np.ndarray) -> tuple[float, dict, list[dict]]:
    best_tau = 0.50
    best_macro_f1 = -1.0
    best_metrics = {}
    sweep = []

    for tau_int in range(10, 91, 2):
        tau = tau_int / 100.0
        preds = (probs_high >= tau).astype(np.int64)
        m = compute_metrics(y_true, preds, probs_high)
        sweep.append({
            "threshold": tau,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "class_0_recall": m["class_0_non_attentive"]["recall"],
            "class_0_precision": m["class_0_non_attentive"]["precision"],
            "class_1_recall": m["class_1_attentive"]["recall"],
        })
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_tau = tau
            best_metrics = m

    return best_tau, best_metrics, sweep


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Public-Dataset Candidate Training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr-temporal", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--init-backbone", default="checkpoints/fusion_daisee.pt")
    parser.add_argument("--out-ckpt", default="checkpoints/candidates/phase7_public_trained_binary_candidate.pt")
    parser.add_argument("--out-dir", default="runs/evaluation/phase7_public_trained_uap_final_20261023")
    args = parser.parse_args()

    print("=" * 80)
    print(" PHASE 7: PUBLIC-DATASET ATTENTION CLASSIFIER TRAINING")
    print("=" * 80)

    # 1. Verify protected checkpoints
    hashes_pre = verify_protected_checkpoints()
    print(f"[Protected Checkpoints Verified Invariant]:")
    for k, v in hashes_pre.items():
        print(f"  {k}: {v}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = WORKSPACE_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = WORKSPACE_ROOT / args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset and split config
    split_path = WORKSPACE_ROOT / "configs" / "phase7_public_daisee_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split config: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)

    npz_path = WORKSPACE_ROOT / "data" / "processed" / "fusion_daisee_v3_hardened.npz"
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    scores = d["score"].astype(np.float32)
    subs = d["subject_ids"].astype(str)
    clips = d["clip_ids"].astype(str)

    y_binary = (scores >= 50.0).astype(np.int64)

    train_subs = split_cfg["train"]["subjects"]
    val_subs = split_cfg["validation"]["subjects"]
    test_subs = split_cfg["held_out_test"]["subjects"]

    train_mask = np.isin(subs, train_subs)
    val_mask = np.isin(subs, val_subs)
    test_mask = np.isin(subs, test_subs)

    x_train, y_train = feats[train_mask], y_binary[train_mask]
    x_val, y_val = feats[val_mask], y_binary[val_mask]
    x_test, y_test = feats[test_mask], y_binary[test_mask]

    n_train = len(x_train)
    n_low_train = int(np.sum(y_train == 0))
    n_high_train = int(np.sum(y_train == 1))

    # Compute training-only class weights
    w0 = n_train / (2.0 * n_low_train)
    w1 = n_train / (2.0 * n_high_train)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    print(f"\n[Dataset Partitions Loaded]:")
    print(f"  Train Split      : {len(x_train):,} seqs | Non-Attentive (0): {n_low_train} ({n_low_train/len(x_train)*100:.2f}%), Attentive (1): {n_high_train} ({n_high_train/len(x_train)*100:.2f}%)")
    print(f"  Validation Split : {len(x_val):,} seqs | Non-Attentive (0): {int(np.sum(y_val==0))} ({np.mean(y_val==0)*100:.2f}%), Attentive (1): {int(np.sum(y_val==1))} ({np.mean(y_val==1)*100:.2f}%)")
    print(f"  Held-Out Test    : {len(x_test):,} seqs | Non-Attentive (0): {int(np.sum(y_test==0))} ({np.mean(y_test==0)*100:.2f}%), Attentive (1): {int(np.sum(y_test==1))} ({np.mean(y_test==1)*100:.2f}%)")
    print(f"  Class Weights (Train-Only): w0={w0:.4f}, w1={w1:.4f} (Ratio={w0/w1:.2f})")

    # Save training_config.yaml
    config_dict = {
        "pipeline_version": "Phase 7 Public-Data Training",
        "dataset_path": str(npz_path),
        "split_config": str(split_path),
        "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D] + MLP)",
        "backbone_initialization": args.init_backbone,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rates": {
            "temporal_bigru": args.lr_temporal,
            "classification_head": args.lr_head,
            "weight_decay": args.weight_decay,
        },
        "class_counts_train": {"non_attentive": n_low_train, "attentive": n_high_train},
        "class_weights": {"non_attentive_w0": float(w0), "attentive_w1": float(w1)},
        "loss_function": "Class-Weighted Cross-Entropy (weights fitted strictly on train split)",
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "pytorch_version": torch.__version__,
        },
    }
    with open(out_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config_dict, f, sort_keys=False)

    # 3. Model instantiation
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    init_backbone_path = WORKSPACE_ROOT / args.init_backbone
    if init_backbone_path.exists():
        init_ckpt = torch.load(init_backbone_path, map_location="cpu", weights_only=False)
        base_net.load_state_dict(init_ckpt["model"], strict=False)
        print(f"Loaded pretrained backbone weights from: {init_backbone_path}")
    else:
        print(f"Warning: Backbone weights not found at {init_backbone_path}, initializing randomly.")

    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True).to(device)

    param_groups = [
        {"params": model.temporal.parameters(), "lr": args.lr_temporal, "weight_decay": args.weight_decay},
        {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(device)

    # 4. Training loop
    best_val_bal_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    best_state = {}

    print("\n--- Starting Training Epochs ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation at tau = 0.50
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            probs_high_val = v_probs[:, 1]
            v_preds = (probs_high_val >= 0.50).astype(np.int64)

        v_metrics = compute_metrics(y_val, v_preds, probs_high_val)

        rec = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": v_metrics["accuracy"],
            "val_balanced_accuracy": v_metrics["balanced_accuracy"],
            "val_macro_f1": v_metrics["macro_f1"],
            "val_weighted_f1": v_metrics["weighted_f1"],
            "val_brier_score": v_metrics["brier_score"],
            "val_non_attentive_recall": v_metrics["class_0_non_attentive"]["recall"],
            "val_non_attentive_precision": v_metrics["class_0_non_attentive"]["precision"],
            "val_attentive_recall": v_metrics["class_1_attentive"]["recall"],
            "val_attentive_precision": v_metrics["class_1_attentive"]["precision"],
        }
        history.append(rec)

        is_best = v_metrics["balanced_accuracy"] > best_val_bal_acc
        if is_best:
            best_val_bal_acc = v_metrics["balanced_accuracy"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        marker = " [*BEST]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {v_metrics['accuracy']*100:.2f}% | Bal-Acc: {v_metrics['balanced_accuracy']*100:.2f}% | "
            f"Macro-F1: {v_metrics['macro_f1']*100:.2f}% | "
            f"Rec0: {v_metrics['class_0_non_attentive']['recall']*100:.1f}%, Rec1: {v_metrics['class_1_attentive']['recall']*100:.1f}%{marker}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # Save training_history.csv
    import csv
    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # 5. Restore best weights & Calibrate threshold on validation set
    print(f"\nRestoring best model from Epoch {best_epoch} (Val Bal-Acc: {best_val_bal_acc*100:.2f}%)...")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        best_v_logits = model(x_val_t)
        best_v_probs = torch.softmax(best_v_logits, dim=-1).cpu().numpy()
        probs_high_val = best_v_probs[:, 1]

    val_metrics_tau50 = compute_metrics(y_val, (probs_high_val >= 0.50).astype(np.int64), probs_high_val)
    best_tau, val_metrics_calibrated, threshold_sweep = calibrate_threshold(y_val, probs_high_val)

    print("\n--- Validation Performance Summary ---")
    print(f"Standard Threshold (tau = 0.50):")
    print(f"  Accuracy     : {val_metrics_tau50['accuracy']*100:.2f}%")
    print(f"  Balanced Acc : {val_metrics_tau50['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1     : {val_metrics_tau50['macro_f1']*100:.2f}%")
    print(f"  Brier Score  : {val_metrics_tau50['brier_score']:.4f}")
    print(f"  Confusion Mat: {val_metrics_tau50['confusion_matrix']['matrix_2x2']}")

    print(f"\nCalibrated Threshold (tau* = {best_tau:.2f}):")
    print(f"  Accuracy     : {val_metrics_calibrated['accuracy']*100:.2f}%")
    print(f"  Balanced Acc : {val_metrics_calibrated['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-F1     : {val_metrics_calibrated['macro_f1']*100:.2f}%")
    print(f"  Brier Score  : {val_metrics_calibrated['brier_score']:.4f}")
    print(f"  Confusion Mat: {val_metrics_calibrated['confusion_matrix']['matrix_2x2']}")

    # Save public_validation_metrics.json
    val_out_payload = {
        "dataset": "DAiSEE v3 Hardened Validation Split",
        "sequences_evaluated": len(y_val),
        "best_epoch": best_epoch,
        "calibrated_threshold": best_tau,
        "metrics_tau50": val_metrics_tau50,
        "metrics_calibrated": val_metrics_calibrated,
        "threshold_sweep": threshold_sweep,
    }
    with open(out_dir / "public_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_out_payload, f, indent=2)

    # 6. Save frozen candidate checkpoint
    ckpt_payload = {
        "epoch": best_epoch,
        "model": best_state,
        "experiment_name": "Phase 7 Public-Trained Attention Classifier",
        "freeze_frame_encoder": True,
        "seed": args.seed,
        "calibrated_threshold": best_tau,
        "val_metrics_tau50": val_metrics_tau50,
        "val_metrics_calibrated": val_metrics_calibrated,
        "class_weights": {"w0": float(w0), "w1": float(w1)},
        "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D] + MLP)",
        "trained_date": "2026-10-23",
        "training_data_provenance": "DAiSEE v3 Hardened Train Split (16,597 sequences, Subjects 110002, 110006, 110007, 110008, 110010, 110011, 110012, 110014)",
    }
    torch.save(ckpt_payload, out_ckpt)
    cand_sha = get_file_sha256(out_ckpt)
    print(f"\nSaved Candidate Checkpoint to: {out_ckpt}")
    print(f"Candidate Checkpoint SHA-256 : {cand_sha}")

    # 7. Post-training hash invariance check
    hashes_post = verify_protected_checkpoints()
    print("\n[Post-Training Checkpoint Invariance Check]:")
    for k, v in hashes_post.items():
        assert v == hashes_pre[k], f"Violation: {k} was altered during training!"
        print(f"  {k}: {v} (MATCH)")


if __name__ == "__main__":
    main()
