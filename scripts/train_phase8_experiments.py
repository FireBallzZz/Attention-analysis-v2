"""Phase 8: Controlled Public-Dataset Training, Validation, and Frozen Test Evaluation.

Enforces:
1. Strict 3-way subject-disjoint partition (configs/phase7_public_daisee_split.json).
2. Held-out test set remains completely untouched until model freezing.
3. Training-only class weights for weighted CE.
4. Early stopping monitored on Validation Balanced Accuracy.
5. Decision threshold calibration tau* strictly on Validation Macro-F1.
6. Zero access to UAP classroom data or forbidden paths.
7. Verification of all protected checkpoint and dataset SHA-256 hashes before and after.
8. Writes candidate checkpoints exclusively to checkpoints/experiments/phase8/.
9. Computes cluster-robust clip bootstrap confidence intervals (95% CI).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

PROTECTED_ASSETS = {
    "checkpoints/classroom_manual_binary_model_v2_clean.pt": "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E",
    "checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt": "E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E",
    "checkpoints/candidates/phase7_public_trained_binary_candidate.pt": "1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5",
    "data/processed/fusion_daisee.npz": "15B3FB28EF15AAC1564423251F36825A5BC9FBB5E2F0353D27D8978AACB5196B",
    "data/processed/fusion_daisee_v2.npz": "69E5C9984A27980AAC36BA14F190534D84AE5260145A114502D278627EE76733",
    "data/processed/fusion_daisee_v3_hardened.npz": "4505FB4428944366459B1E3AEE37678A8AD7381AA19384A64514C87F75FD2DA6",
}


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_protected_assets(stage_name: str) -> dict[str, str]:
    print(f"\n[Protected Assets Verification ({stage_name})]:")
    observed = {}
    for rel_path, exp_hash in PROTECTED_ASSETS.items():
        p = WORKSPACE_ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(f"CRITICAL: Protected asset missing: {p}")
        obs_hash = get_file_sha256(p)
        observed[rel_path] = obs_hash
        if obs_hash != exp_hash:
            raise ValueError(f"CRITICAL: Protected asset hash mismatch on {rel_path}!\nExpected: {exp_hash}\nObserved: {obs_hash}")
        print(f"  PASS: {rel_path} ({obs_hash[:16]}...)")
    return observed


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs_high: np.ndarray | None = None) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    res: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": {
            "row_order": ["0: Non-Attentive", "1: Attentive"],
            "col_order": ["0: Non-Attentive", "1: Attentive"],
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

    if probs_high is not None:
        res["brier_score"] = float(brier_score_loss(y_true, probs_high))
        try:
            res["roc_auc"] = float(roc_auc_score(y_true, probs_high))
        except Exception:
            res["roc_auc"] = None
        try:
            res["pr_auc"] = float(average_precision_score(y_true, probs_high))
        except Exception:
            res["pr_auc"] = None

    return res


def calibrate_threshold(y_true: np.ndarray, probs_high: np.ndarray) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
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


def compute_clip_bootstrap_ci(
    y_true: np.ndarray,
    probs_high: np.ndarray,
    clips: np.ndarray,
    tau: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_clips = np.unique(clips)
    n_clips = len(unique_clips)

    clip_indices = {c: np.where(clips == c)[0] for c in unique_clips}

    metric_samples = {
        "balanced_accuracy": [],
        "macro_f1": [],
        "class_0_recall": [],
        "class_0_f1": [],
        "class_1_recall": [],
        "class_1_f1": [],
    }

    for _ in range(n_bootstrap):
        sampled_clips = rng.choice(unique_clips, size=n_clips, replace=True)
        idx_list = [clip_indices[c] for c in sampled_clips]
        indices = np.concatenate(idx_list)

        y_t = y_true[indices]
        p_h = probs_high[indices]
        y_p = (p_h >= tau).astype(np.int64)

        if len(np.unique(y_t)) < 2:
            continue

        m = compute_metrics(y_t, y_p, p_h)
        metric_samples["balanced_accuracy"].append(m["balanced_accuracy"])
        metric_samples["macro_f1"].append(m["macro_f1"])
        metric_samples["class_0_recall"].append(m["class_0_non_attentive"]["recall"])
        metric_samples["class_0_f1"].append(m["class_0_non_attentive"]["f1"])
        metric_samples["class_1_recall"].append(m["class_1_attentive"]["recall"])
        metric_samples["class_1_f1"].append(m["class_1_attentive"]["f1"])

    ci_results = {}
    for metric_name, values in metric_samples.items():
        if values:
            arr = np.array(values)
            ci_results[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_lower_2.5": float(np.percentile(arr, 2.5)),
                "ci_upper_97.5": float(np.percentile(arr, 97.5)),
            }
        else:
            ci_results[metric_name] = {"mean": 0.0, "std": 0.0, "ci_lower_2.5": 0.0, "ci_upper_97.5": 0.0}

    return ci_results


def train_temporal_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor | None,
    device: torch.device,
    epochs: int = 25,
    patience: int = 6,
    batch_size: int = 128,
    lr_temporal: float = 1e-4,
    lr_head: float = 1.5e-3,
    weight_decay: float = 1e-3,
    out_ckpt_path: Path | None = None,
) -> tuple[ManualBinaryAttentionClassifier, dict[str, Any]]:
    base = AttentionFusionNet(feat_dim=32, hidden=64)
    init_backbone_path = WORKSPACE_ROOT / "checkpoints" / "fusion_daisee.pt"
    if init_backbone_path.exists():
        base.load(init_backbone_path)

    model = ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True)
    model.to(device)

    train_ds = SequenceDataset(x_train, y_train)
    val_ds = SequenceDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {"params": model.temporal.parameters(), "lr": lr_temporal, "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
    ])

    best_val_bal_acc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\nTraining {model_name} (Weighted: {class_weights is not None})...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs_high = [], [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
                probs = torch.softmax(logits, dim=-1)
                val_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                val_targets.append(by.cpu().numpy())
                val_probs_high.append(probs[:, 1].cpu().numpy())

        val_loss /= len(val_ds)
        y_val_true = np.concatenate(val_targets)
        y_val_pred = np.concatenate(val_preds)
        y_val_probs = np.concatenate(val_probs_high)

        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_probs)
        bal_acc = val_metrics["balanced_accuracy"]
        macro_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_balanced_accuracy": bal_acc,
            "val_macro_f1": macro_f1,
            "val_class_0_recall": val_metrics["class_0_non_attentive"]["recall"],
        })

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Bal-Acc: {bal_acc*100:.2f}% | Val Macro-F1: {macro_f1*100:.2f}% | Cls 0 Rec: {val_metrics['class_0_non_attentive']['recall']*100:.1f}%")

        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (Val Bal-Acc: {best_val_bal_acc*100:.2f}%).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if out_ckpt_path is not None:
        out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "best_epoch": best_epoch, "val_bal_acc": best_val_bal_acc}, out_ckpt_path)
        print(f"  Saved best checkpoint to {out_ckpt_path}")

    return model, {"history": history, "best_epoch": best_epoch, "best_val_bal_acc": best_val_bal_acc}


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Controlled Public Training & Evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    print("=" * 80)
    print(" PHASE 8: CONTROLLED PUBLIC TRAINING, VALIDATION & FROZEN TEST EVALUATION")
    print("=" * 80)

    # 1. Pre-execution hash check
    verify_protected_assets("PRE-EXECUTION")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExecution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Load dataset and split config
    npz_path = WORKSPACE_ROOT / "data" / "processed" / "fusion_daisee_v3_hardened.npz"
    split_path = WORKSPACE_ROOT / "configs" / "phase7_public_daisee_split.json"

    with open(split_path, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)

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
    test_clips = clips[test_mask]

    n_train = len(x_train)
    n_low_train = int(np.sum(y_train == 0))
    n_high_train = int(np.sum(y_train == 1))

    w0 = n_train / (2.0 * n_low_train)
    w1 = n_train / (2.0 * n_high_train)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    print(f"\nDataset Partitions Loaded from {npz_path.name}:")
    print(f"  Train Split      : {len(x_train):,} seqs | Non-Attentive (0): {n_low_train} ({n_low_train/len(x_train)*100:.2f}%), Attentive (1): {n_high_train} ({n_high_train/len(x_train)*100:.2f}%)")
    print(f"  Validation Split : {len(x_val):,} seqs | Non-Attentive (0): {int(np.sum(y_val==0))} ({np.mean(y_val==0)*100:.2f}%), Attentive (1): {int(np.sum(y_val==1))} ({np.mean(y_val==1)*100:.2f}%)")
    print(f"  Held-Out Test    : {len(x_test):,} seqs | Non-Attentive (0): {int(np.sum(y_test==0))} ({np.mean(y_test==0)*100:.2f}%), Attentive (1): {int(np.sum(y_test==1))} ({np.mean(y_test==1)*100:.2f}%)")
    print(f"  Class Weights (Train-Only): w0 = {w0:.4f}, w1 = {w1:.4f} (Ratio = {w0/w1:.2f})")

    # -------------------------------------------------------------
    # MODEL 1: Majority Class Baseline
    # -------------------------------------------------------------
    print("\n--- BASELINE 1: Majority Class Baseline ---")
    val_probs_maj = np.ones(len(x_val), dtype=np.float32)
    test_probs_maj = np.ones(len(x_test), dtype=np.float32)

    # -------------------------------------------------------------
    # MODEL 2: Class-Weighted Logistic Regression (Aggregated 32-D Features)
    # -------------------------------------------------------------
    print("\n--- BASELINE 2: Class-Weighted Logistic Regression ---")
    x_train_flat = x_train.mean(axis=1)
    x_val_flat = x_val.mean(axis=1)
    x_test_flat = x_test.mean(axis=1)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)
    log_reg.fit(x_train_flat, y_train)
    val_probs_logreg = log_reg.predict_proba(x_val_flat)[:, 1]
    test_probs_logreg = log_reg.predict_proba(x_test_flat)[:, 1]

    # -------------------------------------------------------------
    # MODEL 3: Active Production Baseline (ManualBinaryAttentionClassifier)
    # -------------------------------------------------------------
    print("\n--- BASELINE 3: Active Production Model (classroom_manual_binary_model_v2_clean.pt) ---")
    base_prod = AttentionFusionNet(feat_dim=32, hidden=64)
    prod_model = ManualBinaryAttentionClassifier(base_prod, freeze_frame_encoder=True)
    prod_model.load(WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt")
    prod_model.to(device)
    prod_model.eval()

    def get_model_probs(m: ManualBinaryAttentionClassifier, x_arr: np.ndarray) -> np.ndarray:
        loader = DataLoader(SequenceDataset(x_arr, np.zeros(len(x_arr))), batch_size=args.batch_size, shuffle=False)
        p_list = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device)
                probs = m.predict_proba(bx)
                p_list.append(probs[:, 1].cpu().numpy())
        return np.concatenate(p_list)

    val_probs_prod = get_model_probs(prod_model, x_val)
    test_probs_prod = get_model_probs(prod_model, x_test)

    # -------------------------------------------------------------
    # MODEL 4: Candidate Temporal Classifier (Unweighted CE)
    # -------------------------------------------------------------
    ckpt_unweighted = WORKSPACE_ROOT / "checkpoints" / "experiments" / "phase8" / "phase8_unweighted_candidate.pt"
    model_unweighted, info_unweighted = train_temporal_model(
        model_name="Phase 8 Unweighted Candidate",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        class_weights=None,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        out_ckpt_path=ckpt_unweighted,
    )
    val_probs_unw = get_model_probs(model_unweighted, x_val)
    test_probs_unw = get_model_probs(model_unweighted, x_test)

    # -------------------------------------------------------------
    # MODEL 5: Candidate Temporal Classifier (Class-Weighted CE)
    # -------------------------------------------------------------
    ckpt_weighted = WORKSPACE_ROOT / "checkpoints" / "experiments" / "phase8" / "phase8_class_weighted_candidate.pt"
    model_weighted, info_weighted = train_temporal_model(
        model_name="Phase 8 Class-Weighted Candidate",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        out_ckpt_path=ckpt_weighted,
    )
    val_probs_wt = get_model_probs(model_weighted, x_val)
    test_probs_wt = get_model_probs(model_weighted, x_test)

    # -------------------------------------------------------------
    # VALIDATION THRESHOLD CALIBRATION (tau* on Validation Macro-F1)
    # -------------------------------------------------------------
    models_dict = {
        "Majority Baseline": (val_probs_maj, test_probs_maj),
        "Class-Weighted Logistic Regression": (val_probs_logreg, test_probs_logreg),
        "Active Production Baseline": (val_probs_prod, test_probs_prod),
        "Phase 8 Candidate (Unweighted CE)": (val_probs_unw, test_probs_unw),
        "Phase 8 Candidate (Class-Weighted CE)": (val_probs_wt, test_probs_wt),
    }

    print("\n" + "=" * 80)
    print(" VALIDATION SET THRESHOLD CALIBRATION (tau* maximizing Validation Macro-F1)")
    print("=" * 80)

    calibration_results = {}
    for name, (val_p, _) in models_dict.items():
        tau_star, best_m, sweep = calibrate_threshold(y_val, val_p)
        default_m = compute_metrics(y_val, (val_p >= 0.50).astype(np.int64), val_p)
        calibration_results[name] = {
            "tau_star": tau_star,
            "val_metrics_at_default": default_m,
            "val_metrics_at_calibrated": best_m,
        }
        print(f"  {name:38s} | Default (0.50) Macro-F1: {default_m['macro_f1']*100:.2f}% | Calibrated tau*={tau_star:.2f} Macro-F1: {best_m['macro_f1']*100:.2f}% (Bal-Acc: {best_m['balanced_accuracy']*100:.2f}%)")

    # -------------------------------------------------------------
    # FROZEN HELD-OUT TEST EVALUATION (Executed ONCE on untouched Test Split)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FROZEN HELD-OUT TEST EVALUATION (Subjects 110001 & 110004, 3,563 seqs, 148 clips)")
    print("=" * 80)

    test_results = {}
    for name, (_, test_p) in models_dict.items():
        tau_star = calibration_results[name]["tau_star"]

        # Default tau = 0.50
        test_m_default = compute_metrics(y_test, (test_p >= 0.50).astype(np.int64), test_p)
        ci_default = compute_clip_bootstrap_ci(y_test, test_p, test_clips, tau=0.50, seed=args.seed)

        # Calibrated tau*
        test_m_calibrated = compute_metrics(y_test, (test_p >= tau_star).astype(np.int64), test_p)
        ci_calibrated = compute_clip_bootstrap_ci(y_test, test_p, test_clips, tau=tau_star, seed=args.seed)

        test_results[name] = {
            "calibrated_threshold": tau_star,
            "test_metrics_default_tau_0_50": test_m_default,
            "test_ci_default_tau_0_50": ci_default,
            "test_metrics_calibrated_tau_star": test_m_calibrated,
            "test_ci_calibrated_tau_star": ci_calibrated,
        }

        print(f"\n[{name}]:")
        print(f"  @ Default tau = 0.50:")
        print(f"    Bal-Acc: {test_m_default['balanced_accuracy']*100:.2f}% (95% CI: [{ci_default['balanced_accuracy']['ci_lower_2.5']*100:.2f}%, {ci_default['balanced_accuracy']['ci_upper_97.5']*100:.2f}%])")
        print(f"    Macro-F1: {test_m_default['macro_f1']*100:.2f}% (95% CI: [{ci_default['macro_f1']['ci_lower_2.5']*100:.2f}%, {ci_default['macro_f1']['ci_upper_97.5']*100:.2f}%])")
        print(f"    Class 0 (Non-Attentive): Rec = {test_m_default['class_0_non_attentive']['recall']*100:.2f}%, Prec = {test_m_default['class_0_non_attentive']['precision']*100:.2f}%, F1 = {test_m_default['class_0_non_attentive']['f1']*100:.2f}% (Supp = {test_m_default['class_0_non_attentive']['support']})")
        print(f"    Class 1 (Attentive)    : Rec = {test_m_default['class_1_attentive']['recall']*100:.2f}%, Prec = {test_m_default['class_1_attentive']['precision']*100:.2f}%, F1 = {test_m_default['class_1_attentive']['f1']*100:.2f}% (Supp = {test_m_default['class_1_attentive']['support']})")
        print(f"    Confusion Matrix [[TN, FP], [FN, TP]]: {test_m_default['confusion_matrix']['matrix_2x2']}")
        print(f"  @ Calibrated tau* = {tau_star:.2f}:")
        print(f"    Bal-Acc: {test_m_calibrated['balanced_accuracy']*100:.2f}% (95% CI: [{ci_calibrated['balanced_accuracy']['ci_lower_2.5']*100:.2f}%, {ci_calibrated['balanced_accuracy']['ci_upper_97.5']*100:.2f}%])")
        print(f"    Macro-F1: {test_m_calibrated['macro_f1']*100:.2f}% (95% CI: [{ci_calibrated['macro_f1']['ci_lower_2.5']*100:.2f}%, {ci_calibrated['macro_f1']['ci_upper_97.5']*100:.2f}%])")
        print(f"    Class 0 (Non-Attentive): Rec = {test_m_calibrated['class_0_non_attentive']['recall']*100:.2f}%, Prec = {test_m_calibrated['class_0_non_attentive']['precision']*100:.2f}%, F1 = {test_m_calibrated['class_0_non_attentive']['f1']*100:.2f}%")
        print(f"    Class 1 (Attentive)    : Rec = {test_m_calibrated['class_1_attentive']['recall']*100:.2f}%, Prec = {test_m_calibrated['class_1_attentive']['precision']*100:.2f}%, F1 = {test_m_calibrated['class_1_attentive']['f1']*100:.2f}%")
        print(f"    Confusion Matrix [[TN, FP], [FN, TP]]: {test_m_calibrated['confusion_matrix']['matrix_2x2']}")

    # 3. Post-execution hash check
    verify_protected_assets("POST-EXECUTION")

    # 4. Save reports
    out_dir = WORKSPACE_ROOT / "runs" / "experiments" / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "protocol": "Phase 8 Controlled Public Training and Frozen Test Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(npz_path),
        "split_config": str(split_path),
        "dataset_partitions": {
            "train": {"total": len(x_train), "non_attentive": n_low_train, "attentive": n_high_train},
            "validation": {"total": len(x_val), "non_attentive": int(np.sum(y_val == 0)), "attentive": int(np.sum(y_val == 1))},
            "held_out_test": {"total": len(x_test), "non_attentive": int(np.sum(y_test == 0)), "attentive": int(np.sum(y_test == 1))},
        },
        "class_weights": {"w0": float(w0), "w1": float(w1)},
        "calibration": calibration_results,
        "frozen_test_evaluation": test_results,
    }

    json_report_path = out_dir / "phase8_evaluation_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nFull evaluation report saved to {json_report_path}")

    # Build Markdown Summary
    md_lines = [
        "# Phase 8 Controlled Public-Dataset Training and Frozen Test Evaluation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        "**Protocol:** Subject-Disjoint Public DAiSEE Benchmark (Train: 501 clips, Val: 101 clips, Test: 148 clips)  ",
        f"**Dataset Archive:** `{npz_path.name}`  ",
        "",
        "---",
        "",
        "## 1. Frozen Held-Out Test Split Performance Comparison",
        "",
        "| Model Architecture | Threshold $\\tau$ | Accuracy | Balanced Acc (95% CI) | Macro-F1 (95% CI) | Non-Attentive Rec / Prec | Attentive Rec / Prec | Brier | ROC-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, res in test_results.items():
        m_def = res["test_metrics_default_tau_0_50"]
        ci_def = res["test_ci_default_tau_0_50"]
        m_cal = res["test_metrics_calibrated_tau_star"]
        ci_cal = res["test_ci_calibrated_tau_star"]
        tau_s = res["calibrated_threshold"]

        md_lines.append(
            f"| **{name}** | 0.50 (Default) | {m_def['accuracy']*100:.2f}% | {m_def['balanced_accuracy']*100:.2f}% [{ci_def['balanced_accuracy']['ci_lower_2.5']*100:.1f}%, {ci_def['balanced_accuracy']['ci_upper_97.5']*100:.1f}%] | {m_def['macro_f1']*100:.2f}% [{ci_def['macro_f1']['ci_lower_2.5']*100:.1f}%, {ci_def['macro_f1']['ci_upper_97.5']*100:.1f}%] | {m_def['class_0_non_attentive']['recall']*100:.1f}% / {m_def['class_0_non_attentive']['precision']*100:.1f}% | {m_def['class_1_attentive']['recall']*100:.1f}% / {m_def['class_1_attentive']['precision']*100:.1f}% | {m_def.get('brier_score', 0):.4f} | {m_def.get('roc_auc', 0):.4f} |"
        )
        if abs(tau_s - 0.50) > 1e-3:
            md_lines.append(
                f"| | {tau_s:.2f} (Calibrated $\\tau^*$) | {m_cal['accuracy']*100:.2f}% | {m_cal['balanced_accuracy']*100:.2f}% [{ci_cal['balanced_accuracy']['ci_lower_2.5']*100:.1f}%, {ci_cal['balanced_accuracy']['ci_upper_97.5']*100:.1f}%] | {m_cal['macro_f1']*100:.2f}% [{ci_cal['macro_f1']['ci_lower_2.5']*100:.1f}%, {ci_cal['macro_f1']['ci_upper_97.5']*100:.1f}%] | {m_cal['class_0_non_attentive']['recall']*100:.1f}% / {m_cal['class_0_non_attentive']['precision']*100:.1f}% | {m_cal['class_1_attentive']['recall']*100:.1f}% / {m_cal['class_1_attentive']['precision']*100:.1f}% | {m_cal.get('brier_score', 0):.4f} | {m_cal.get('roc_auc', 0):.4f} |"
            )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Confusion Matrices (Row: True Class [0, 1], Col: Predicted Class [0, 1])",
        "",
    ])

    for name, res in test_results.items():
        m_def = res["test_metrics_default_tau_0_50"]
        m_cal = res["test_metrics_calibrated_tau_star"]
        tau_s = res["calibrated_threshold"]
        md_lines.append(f"### {name}")
        md_lines.append(f"- **At $\\tau = 0.50$:** `TN={m_def['confusion_matrix']['true_non_attentive']}`, `FP={m_def['confusion_matrix']['false_attentive_fp']}`, `FN={m_def['confusion_matrix']['false_non_attentive_fn']}`, `TP={m_def['confusion_matrix']['true_attentive']}`")
        if abs(tau_s - 0.50) > 1e-3:
            md_lines.append(f"- **At $\\tau^* = {tau_s:.2f}$:** `TN={m_cal['confusion_matrix']['true_non_attentive']}`, `FP={m_cal['confusion_matrix']['false_attentive_fp']}`, `FN={m_cal['confusion_matrix']['false_non_attentive_fn']}`, `TP={m_cal['confusion_matrix']['true_attentive']}`")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 3. Checkpoint Governance & Verification",
        "",
        "All 6 protected checkpoints and datasets remained 100% bit-for-bit invariant before and after this training and evaluation pipeline:",
        "",
    ])
    for rel_p, h in PROTECTED_ASSETS.items():
        md_lines.append(f"- `{rel_p}`: `SHA-256 = {h}` (**VERIFIED INTACT**)")

    md_report_path = out_dir / "phase8_evaluation_summary.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Summary markdown saved to {md_report_path}")


if __name__ == "__main__":
    main()
