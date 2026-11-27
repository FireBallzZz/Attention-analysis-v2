"""DAiSEE -> classroom with unsupervised domain adaptation (no classroom labels).

Source: DAiSEE windows (data/processed/fusion_daisee_full_v4.npz, 306,801 windows,
binary engagement labels). Target: the classroom windows (human labels used
ONLY to score the result). A logistic classifier on window statistics
(mean and std of every non-constant channel) is trained on DAiSEE only.
Alignment of the classroom features to DAiSEE, done per classroom video and
without labels:

  none         raw features
  standardise  z-score DAiSEE per subject and each classroom video separately
  coral        CORAL: whiten each classroom video's covariance, re-colour with
               DAiSEE's (Sun & Saenko 2016), after standardising

Feature sets: all channels / without camera-relative head & gaze (0-4, 21).
Reported per classroom: AUC (threshold-free) and balanced accuracy with the
threshold chosen on DAiSEE (so zero classroom labels are used at all).

    python scripts/daisee_uda_eval.py
Writes reports/improvement/daisee_uda.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from loro_experiments import load_features32  # noqa: E402

ROOMS = ["classroom_01", "room_B", "room_C"]
CAMERA = (0, 1, 2, 3, 4, 21)


def stats(X, chans):
    Xk = X[:, :, chans]
    return np.concatenate([Xk.mean(1), Xk.std(1)], 1)


def zscore_groups(F, groups):
    G = F.copy()
    for g in np.unique(groups):
        m = groups == g
        G[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
    return G


def coral(Ft, Cs):
    """Align target features to source covariance Cs (both already standardised)."""
    Ct = np.cov(Ft, rowvar=False) + np.eye(Ft.shape[1])
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-6, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    return Ft @ msqrt(Ct, inv=True) @ msqrt(Cs)


def main():
    rng = np.random.default_rng(0)
    dz = np.load(ROOT / "data/processed/fusion_daisee_full_v4.npz", allow_pickle=True)
    yd_all = dz["binary_targets"].astype(int)
    lo = np.where(yd_all == 0)[0]
    hi = rng.choice(np.where(yd_all == 1)[0], size=len(lo) * 3, replace=False)
    sel = np.sort(np.concatenate([lo, hi]))
    Xd, yd, subj = dz["features"][sel].astype(np.float32), yd_all[sel], dz["subject_ids"][sel]
    C = load_features32()
    Xc, yc, room, video = C["X"], C["y"], C["room"], C["video"]
    print(f"DAiSEE windows used {len(yd)} ({(yd == 0).sum()} LOW) | classroom windows {len(yc)} ({(yc == 0).sum()} LOW)")

    nonconst = [i for i in range(32) if Xd[:, :, i].std() > 1e-6 and Xc[:, :, i].std() > 1e-6]
    sets = {"all channels": nonconst, "no camera-relative": [i for i in nonconst if i not in CAMERA]}
    res = {}
    for set_name, chans in sets.items():
        Fs, Ft = stats(Xd, chans), stats(Xc, chans)
        variants = {}
        variants["none"] = (Fs, Ft)
        Fs_z, Ft_z = zscore_groups(Fs, subj), zscore_groups(Ft, video)
        variants["standardise"] = (Fs_z, Ft_z)
        Cs = np.cov(Fs_z, rowvar=False) + np.eye(Fs_z.shape[1])
        Ft_c = Ft_z.copy()
        for v in np.unique(video):
            m = video == v
            Ft_c[m] = coral(Ft_z[m], Cs)
        variants["coral"] = (Fs_z, Ft_c)
        for vname, (S, T) in variants.items():
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(S, yd)
            ps = clf.predict_proba(S)[:, 1]
            grid = np.quantile(ps, np.linspace(0.02, 0.98, 49))
            thr = float(grid[int(np.argmax([balanced_accuracy_score(yd, (ps >= g).astype(int)) for g in grid]))])
            pt = clf.predict_proba(T)[:, 1]
            per = {r: {"auc": round(float(roc_auc_score(yc[room == r], pt[room == r])), 4),
                       "balacc": round(float(balanced_accuracy_score(yc[room == r], (pt[room == r] >= thr).astype(int))), 4)}
                   for r in ROOMS}
            mean = {k: round(float(np.mean([p[k] for p in per.values()])), 4) for k in ("auc", "balacc")}
            res[f"{set_name} | {vname}"] = {"per_room": per, "mean": mean}
            print(f"{set_name:20s} {vname:12s} AUC {mean['auc']:.3f} ({' '.join(f'{p['auc']:.2f}' for p in per.values())}) "
                  f"| balacc (DAiSEE threshold) {mean['balacc']:.3f}")
    (ROOT / "reports/improvement/daisee_uda.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
