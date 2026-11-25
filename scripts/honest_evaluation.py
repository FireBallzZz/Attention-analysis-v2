"""Leakage-free re-evaluation of the classroom binary attention model.

Why this exists
---------------
The earlier evaluation had three problems:
  * clip_001 / clip_008 training labels were copied from model pre-annotations
    (generate_cvat_preannotations.py), not from people;
  * "Clip 007" was scored against Anno/annotations.xml + ground_truth.csv,
    which is the same annotation that builds the clip_004 training set;
  * the split held out per-clip CVAT track IDs, but the clips come from the
    same classroom video, so the same physical person can sit in both train
    and validation under different IDs. Epoch, config and threshold were
    also picked on the reported set.

What this script does
---------------------
  1. Uses ONLY human labels:
       clip_004  <- Anno/ground_truth.csv (human CVAT job)
       clip_001  <- runs/Annoted file for clip 1/annotations.xml (human),
                    transferred onto the existing clip_001 feature windows by
                    per-frame box IoU. clip_008 is dropped (no human labels).
  2. Assigns each track a PHYSICAL PERSON id by matching seat positions
     between clips (Hungarian on mean box centres; same fixed camera).
  3. Leave-persons-out grouped K-fold CV. Inside each training fold a set of
     persons is held out as an inner validation set that alone chooses the
     best epoch and the decision threshold. The outer fold is never touched
     until the final score.
  4. Compares against baselines (always-attentive, logistic regression on
     window statistics) on the same folds.
  5. Scores the deployed checkpoint on the same human labels (flagged as
     contaminated, because it trained on these windows).
  6. Reports accuracy, balanced accuracy, macro-F1, LOW recall/precision and
     ROC-AUC with 95% CIs from a person-level cluster bootstrap.

Writes reports/honest_eval/{results.json, REPORT.md}. Changes no existing
file, checkpoint or dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet  # noqa: E402
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier  # noqa: E402

PROC = ROOT / "data" / "processed"
HUMAN_CLIP1_XML = ROOT / "runs" / "Annoted file for clip 1" / "annotations.xml"
PRE_CLIP1_XML = ROOT / "Anno" / "clip_001_cvat_preannotated.xml"
CLIP4_XML = ROOT / "Anno" / "annotations.xml"
CLIP4_CSV = ROOT / "Anno" / "ground_truth.csv"
BACKBONE = ROOT / "checkpoints" / "baseline" / "fusion_daisee.pt"
DEPLOYED = ROOT / "checkpoints" / "classroom_manual_binary_unquarantined.pt"

ENGAGED = {"high", "moderate", "attentive"}
DISENGAGED = {"low", "non-attentive", "non_attentive"}


# --------------------------------------------------------------------------- data
def read_tracks(xml_path: Path) -> dict[int, dict[int, tuple[np.ndarray, str | None]]]:
    """track_id -> frame -> (box xyxy, attention label or None)."""
    out: dict[int, dict[int, tuple[np.ndarray, str | None]]] = defaultdict(dict)
    for t in ET.parse(xml_path).getroot().findall("track"):
        tid = int(t.get("id"))
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            att = b.find("attribute")
            box = np.array([float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")], dtype=np.float32)
            out[tid][int(b.get("frame"))] = (box, att.text.strip().lower() if att is not None else None)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def to_binary(label: str) -> int:
    if label in ENGAGED:
        return 1
    if label in DISENGAGED:
        return 0
    raise ValueError(label)


def mean_centres(tracks) -> dict[int, np.ndarray]:
    return {
        tid: np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b, _ in fr.values()], axis=0)
        for tid, fr in tracks.items()
    }


def load_human_data(min_iou: float, max_seat_px: float):
    """Return X, y, person, clip, track, frame plus provenance stats."""
    stats: dict = {}

    # clip_004: labels already equal Anno/ground_truth.csv (verified below).
    d4 = np.load(PROC / "classroom_clip_004_v3_hardened.npz", allow_pickle=True)
    gt4 = {
        (int(r["frame"]), int(r["student_id"])): r["attention"].strip().lower()
        for r in csv.DictReader(CLIP4_CSV.open(encoding="utf-8"))
    }
    y4 = np.array([to_binary(gt4[(int(f), int(s))]) for f, s in zip(d4["frame_ids"], d4["student_ids"])])
    assert np.array_equal(y4, (d4["targets"] != 2).astype(int)), "clip_004 labels drifted from Anno CSV"
    stats["clip_004"] = {"windows": int(len(y4)), "label_source": str(CLIP4_CSV.relative_to(ROOT))}

    # clip_001: transfer human labels onto windows via per-frame box IoU.
    d1 = np.load(PROC / "classroom_clip_001_v3_hardened.npz", allow_pickle=True)
    pre = read_tracks(PRE_CLIP1_XML)
    hum = read_tracks(HUMAN_CLIP1_XML)
    hum_by_frame: dict[int, list[tuple[int, np.ndarray, str]]] = defaultdict(list)
    for tid, fr in hum.items():
        for f, (box, lab) in fr.items():
            hum_by_frame[f].append((tid, box, lab))

    keep, y1, hum_tid = [], [], []
    for i, (f, s) in enumerate(zip(d1["frame_ids"], d1["student_ids"])):
        f, s = int(f), int(s)
        if f not in pre.get(s, {}):
            continue
        pbox = pre[s][f][0]
        best = max(hum_by_frame.get(f, []), key=lambda h: iou(pbox, h[1]), default=None)
        if best is None or iou(pbox, best[1]) < min_iou:
            continue
        keep.append(i)
        y1.append(to_binary(best[2]))
        hum_tid.append(best[0])
    keep = np.array(keep)
    y1 = np.array(y1)
    hum_tid = np.array(hum_tid)
    old = (d1["targets"][keep] != 2).astype(int)
    stats["clip_001"] = {
        "windows_total": int(len(d1["frame_ids"])),
        "windows_with_human_label": int(len(keep)),
        "label_source": str(HUMAN_CLIP1_XML.relative_to(ROOT)),
        "agreement_old_model_labels_vs_human": round(float((old == y1).mean()), 4),
        "old_labels_disengaged_share": round(float((old == 0).mean()), 4),
        "human_labels_disengaged_share": round(float((y1 == 0).mean()), 4),
    }

    # Physical person ids: match clip_001 human tracks to clip_004 tracks by seat.
    c1 = mean_centres(hum)
    c4 = mean_centres(read_tracks(CLIP4_XML))
    ids1, ids4 = sorted(c1), sorted(c4)
    cost = np.array([[np.linalg.norm(c1[a] - c4[b]) for b in ids4] for a in ids1])
    rows, cols = linear_sum_assignment(cost)
    person_of_1, person_of_4, pairs = {}, {}, []
    pid = 0
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_seat_px:
            person_of_1[ids1[r]] = person_of_4[ids4[c]] = pid
            pairs.append({"clip_001_track": ids1[r], "clip_004_track": ids4[c], "px": round(float(cost[r, c]), 1)})
            pid += 1
    for t in ids1:
        if t not in person_of_1:
            person_of_1[t] = pid
            pid += 1
    for t in ids4:
        if t not in person_of_4:
            person_of_4[t] = pid
            pid += 1
    stats["person_matching"] = {
        "max_seat_px": max_seat_px,
        "matched_pairs": pairs,
        "persons_total": pid,
    }

    X = np.concatenate([d4["features"], d1["features"][keep]]).astype(np.float32)
    y = np.concatenate([y4, y1]).astype(np.int64)
    person = np.concatenate(
        [[person_of_4[int(s)] for s in d4["student_ids"]], [person_of_1[int(t)] for t in hum_tid]]
    ).astype(np.int64)
    clip = np.array(["clip_004"] * len(y4) + ["clip_001"] * len(y1))
    track = np.concatenate([d4["student_ids"], hum_tid]).astype(np.int64)
    frame = np.concatenate([d4["frame_ids"], d1["frame_ids"][keep]]).astype(np.int64)
    # Training clip_001 windows (all), for reproducing the deployed model's
    # val-student overlap check.
    return X, y, person, clip, track, frame, stats


# ------------------------------------------------------------------------ metrics
def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "low_recall": float(rec[0]),
        "low_precision": float(prec[0]),
        "low_f1": float(f1[0]),
        "high_f1": float(f1[1]),
        "roc_auc": auc,
        "always_attentive_accuracy": float((y == 1).mean()),
    }


def cluster_bootstrap(y, p, thr, groups, n_boot=1000, seed=0) -> dict:
    """95% CI by resampling whole persons (windows of one person are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "low_recall", "roc_auc"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx], thr)
        for k in keys:
            samples[k].append(m[k])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in samples.items()}


def best_threshold(y, p) -> float:
    grid = np.arange(0.10, 0.91, 0.02)
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ------------------------------------------------------------------------ models
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def new_model(device) -> ManualBinaryAttentionClassifier:
    base = AttentionFusionNet(feat_dim=32, hidden=64)
    ck = torch.load(BACKBONE, map_location="cpu", weights_only=False)
    base.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=False)
    return ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True).to(device)


@torch.no_grad()
def predict(model, X, device, bs=4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs]).to(device)), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_bigru(Xtr, ytr, Xva, yva, device, seed, epochs, max_train_windows=None):
    """Same recipe as train_classroom_binary_unquarantined.py; epoch chosen on inner val."""
    set_seed(seed)
    model = new_model(device)
    counts = np.bincount(ytr, minlength=2)
    w = torch.tensor([counts[1] / max(counts[0], 1) * 0.5, 1.0], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(
        [
            {"params": model.temporal.parameters(), "lr": 8e-5, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": 1.5e-3, "weight_decay": 1e-3},
        ]
    )
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm) - 127, 128):
            b = perm[i:i + 128].to(device)
            loss = crit(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(model, Xva, device)
        score = balanced_accuracy_score(yva, (pv >= 0.5).astype(int)) if len(np.unique(yva)) > 1 else 0.0
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def window_stats(X: np.ndarray) -> np.ndarray:
    keep = X.reshape(-1, X.shape[-1]).std(0) > 1e-6
    Xk = X[:, :, keep]
    return np.concatenate([Xk.mean(1), Xk.std(1), Xk[:, -1], Xk.min(1), Xk.max(1)], axis=1)


def train_logreg(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    return lambda X: lr.predict_proba(sc.transform(X))[:, 1]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--inner-val-frac", type=float, default=0.25)
    ap.add_argument("--min-iou", type=float, default=0.30)
    ap.add_argument("--max-seat-px", type=float, default=60.0)
    ap.add_argument("--out-dir", default="reports/honest_eval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, person, clip, track, frame, stats = load_human_data(args.min_iou, args.max_seat_px)
    print(f"Human-labelled windows: {len(y)}  persons: {len(np.unique(person))}  "
          f"disengaged share: {(y == 0).mean():.3f}")
    print(json.dumps(stats["clip_001"], indent=1))

    results: dict = {"config": vars(args), "device": str(device), "data": stats}
    results["data"]["total_windows"] = int(len(y))
    results["data"]["persons"] = int(len(np.unique(person)))
    results["data"]["disengaged_share"] = float((y == 0).mean())
    results["data"]["persons_with_any_disengaged"] = int(len(np.unique(person[y == 0])))

    # --- A. deployed checkpoint on human labels (contaminated reference) ----
    dep = new_model(device)
    ck = torch.load(DEPLOYED, map_location="cpu", weights_only=False)
    dep.load_state_dict(ck["model"])
    p_dep = predict(dep, X, device)
    results["deployed_checkpoint_on_human_labels"] = {
        "note": "Trained on these exact windows (clip_004) and on clip_001 windows with model labels; "
                "NOT a held-out score. Shown for reference only.",
        "thr_0.50": metrics(y, p_dep, 0.50),
        "thr_0.45": metrics(y, p_dep, 0.45),
        "per_clip_thr_0.50": {c: metrics(y[clip == c], p_dep[clip == c], 0.5) for c in np.unique(clip)},
    }

    # --- B. leave-persons-out CV --------------------------------------------
    # Stratified so every outer fold contains disengaged windows (only ~10 of
    # the persons are ever labelled disengaged).
    gkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    oof = {name: np.full((args.seeds, len(y)), np.nan) for name in ("bigru",)}
    oof_thr = {"bigru": np.full((args.seeds, len(y)), np.nan)}
    oof_lr = np.full(len(y), np.nan)
    oof_lr_thr = np.full(len(y), np.nan)
    feats_lr = window_stats(X)
    fold_log = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=person)):
        assert not set(person[tr]) & set(person[te])
        rng = np.random.default_rng(100 + k)
        tr_persons = np.unique(person[tr])
        # Inner val: persons drawn so that it contains some disengaged windows.
        for _ in range(200):
            inner = rng.choice(tr_persons, size=max(2, int(len(tr_persons) * args.inner_val_frac)), replace=False)
            m_in = np.isin(person[tr], inner)
            if (y[tr][m_in] == 0).sum() >= 30 and (y[tr][~m_in] == 0).sum() >= 30:
                break
        fit, val = tr[~m_in], tr[m_in]

        # Logistic regression baseline
        f_lr = train_logreg(feats_lr[fit], y[fit])
        t_lr = best_threshold(y[val], f_lr(feats_lr[val]))
        oof_lr[te] = f_lr(feats_lr[te])
        oof_lr_thr[te] = t_lr

        for s in range(args.seeds):
            model = train_bigru(X[fit], y[fit], X[val], y[val], device, seed=1000 * k + s, epochs=args.epochs)
            t = best_threshold(y[val], predict(model, X[val], device))
            oof["bigru"][s, te] = predict(model, X[te], device)
            oof_thr["bigru"][s, te] = t
        fold_log.append({
            "fold": k,
            "test_persons": sorted(int(p) for p in np.unique(person[te])),
            "test_windows": int(len(te)),
            "test_disengaged": int((y[te] == 0).sum()),
            "inner_val_persons": sorted(int(p) for p in inner),
            "bigru_thresholds": [round(float(oof_thr["bigru"][s, te][0]), 2) for s in range(args.seeds)],
            "logreg_threshold": round(t_lr, 2),
        })
        print(f"fold {k}: test persons {len(np.unique(person[te]))}, windows {len(te)}, "
              f"low {int((y[te] == 0).sum())}")

    def summarise(p, thr_vec, label):
        pred = (p >= thr_vec).astype(int)
        # Hard predictions (per-fold thresholds) passed as 0/1 "probabilities".
        m = metrics(y, pred.astype(float), 0.5)
        m["roc_auc"] = float(roc_auc_score(y, p))
        ci = cluster_bootstrap(y, pred.astype(float), 0.5, person)
        ci["roc_auc"] = cluster_bootstrap(y, p, 0.5, person)["roc_auc"]
        per_clip = {}
        for c in np.unique(clip):
            mc = clip == c
            pc = metrics(y[mc], pred[mc].astype(float), 0.5)
            pc["roc_auc"] = float(roc_auc_score(y[mc], p[mc])) if len(np.unique(y[mc])) > 1 else float("nan")
            per_clip[c] = pc
        return {"model": label, "pooled": m, "ci95_person_bootstrap": ci, "per_clip": per_clip}

    cv = {}
    seed_runs = []
    for s in range(args.seeds):
        seed_runs.append(summarise(oof["bigru"][s], oof_thr["bigru"][s], f"BiGRU seed {s}")["pooled"])
    p_mean = oof["bigru"].mean(0)
    thr_mean = oof_thr["bigru"].mean(0)
    cv["bigru_seed_ensemble"] = summarise(p_mean, thr_mean, "BiGRU (DAiSEE init, mean of seeds)")
    cv["bigru_per_seed_pooled"] = seed_runs
    cv["logreg_window_stats"] = summarise(oof_lr, oof_lr_thr, "Logistic regression on window statistics")
    always = np.ones(len(y))
    cv["always_attentive"] = {
        "model": "Always predict attentive",
        "pooled": metrics(y, always * 0.75, 0.5) | {"roc_auc": 0.5},
    }
    results["leave_persons_out_cv"] = cv
    results["folds"] = fold_log

    # --- C. cross-segment transfer: train on one clip, test on the other ---
    # Different time segment and independent human labels; persons overlap,
    # so this is an upper bound on cross-person performance.
    transfer = {}
    for src, dst in (("clip_004", "clip_001"), ("clip_001", "clip_004")):
        tr_idx, te_idx = np.where(clip == src)[0], np.where(clip == dst)[0]
        rng = np.random.default_rng(7)
        tp = np.unique(person[tr_idx])
        for _ in range(200):
            inner = rng.choice(tp, size=max(2, int(len(tp) * args.inner_val_frac)), replace=False)
            m_in = np.isin(person[tr_idx], inner)
            if (y[tr_idx][m_in] == 0).sum() >= 30 and (y[tr_idx][~m_in] == 0).sum() >= 30:
                break
        fit, val = tr_idx[~m_in], tr_idx[m_in]
        ps, ts = [], []
        for s in range(args.seeds):
            model = train_bigru(X[fit], y[fit], X[val], y[val], device, seed=500 + s, epochs=args.epochs)
            ts.append(best_threshold(y[val], predict(model, X[val], device)))
            ps.append(predict(model, X[te_idx], device))
        p, t = np.mean(ps, 0), float(np.mean(ts))
        m = metrics(y[te_idx], p, t)
        m["threshold"] = t
        m["ci95_person_bootstrap"] = cluster_bootstrap(y[te_idx], p, t, person[te_idx])
        transfer[f"{src}->{dst}"] = m
        print(f"transfer {src}->{dst}: balacc {m['balanced_accuracy']:.3f} auc {m['roc_auc']:.3f}")
    results["cross_segment_transfer"] = transfer

    # Effective sample size note: windows overlap by 15/16 frames.
    nonoverlap = sum(len(np.unique(frame[(person == p) & (clip == c)] // 16))
                     for p in np.unique(person) for c in np.unique(clip))
    results["data"]["approx_non_overlapping_windows"] = int(nonoverlap)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out / 'results.json'}")
    for k in ("bigru_seed_ensemble", "logreg_window_stats", "always_attentive"):
        m = cv[k]["pooled"]
        print(f"{k:24s} acc {m['accuracy']:.3f}  balacc {m['balanced_accuracy']:.3f}  "
              f"macroF1 {m['macro_f1']:.3f}  lowRec {m['low_recall']:.3f}  auc {m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
