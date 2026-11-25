"""Leave-one-classroom-out experiment harness for improving unseen-room accuracy.

Every variant is scored the same way:
  * train on two classrooms, test on the third (people, camera, room unseen);
  * the decision threshold is chosen on ~25% of people held out from the
    TRAINING rooms (balanced accuracy), never on the test room;
  * optional unsupervised per-video normalisation uses only the test video's
    own features (no labels) - this is deployable: it needs the video, not labels;
  * optional segment scoring averages a student's predictions over
    `seg_seconds` and labels the segment by its majority human label.

    python scripts/loro_experiments.py           # runs the 32-D feature variants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def score(y, p, thr):
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "acc": accuracy_score(y, pred), "balacc": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "low_recall": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
           "always_high": float((y == 1).mean())}
    try:
        out["auc"] = roc_auc_score(y, p)
    except ValueError:
        out["auc"] = float("nan")
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def best_thr(y, p):
    grid = np.quantile(p, np.linspace(0.02, 0.98, 49))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def per_video_standardise(F, video):
    """Unsupervised domain normalisation: z-score each feature within each video."""
    G = F.astype(np.float64).copy()
    for v in np.unique(video):
        m = video == v
        G[m] = (G[m] - G[m].mean(0)) / (G[m].std(0) + 1e-6)
    return G


def make_model(kind):
    if kind == "logreg":
        return lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
    if kind == "hgb":
        return lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      class_weight="balanced", l2_regularization=1.0)
    raise ValueError(kind)


def segments(video, person, time_s, seg_seconds):
    return np.array([f"{v}|{p}|{int(t // seg_seconds)}" for v, p, t in zip(video, person, time_s)])


def aggregate(seg, y, p):
    keys, inv = np.unique(seg, return_inverse=True)
    ps = np.bincount(inv, weights=p) / np.bincount(inv)
    ys = (np.bincount(inv, weights=y) / np.bincount(inv) >= 0.5).astype(int)
    return ys, ps, inv


def loro(F, y, room, person, video, *, kind="logreg", norm=False, seg=None, seed=11):
    """Returns per-room and pooled metrics. `seg` = segment ids for segment-level scoring."""
    if norm:
        F = per_video_standardise(F, video)
    pooled_p, pooled_t = np.zeros(len(y)), np.zeros(len(y))
    per = {}
    for r in sorted(np.unique(room)):
        te, tr = room == r, room != r
        rng = np.random.default_rng(seed)
        tp = np.unique(person[tr])
        for _ in range(300):
            inner = rng.choice(tp, max(2, int(len(tp) * 0.25)), replace=False)
            mi = np.isin(person, inner) & tr
            if (y[mi] == 0).sum() >= 30 and (y[tr & ~mi] == 0).sum() >= 30:
                break
        fit = tr & ~mi
        sc = StandardScaler().fit(F[fit])
        clf = make_model(kind)().fit(sc.transform(F[fit]), y[fit])
        pv = clf.predict_proba(sc.transform(F[mi]))[:, 1]
        pt = clf.predict_proba(sc.transform(F[te]))[:, 1]
        if seg is not None:
            yv, pv2, _ = aggregate(seg[mi], y[mi], pv)
            thr = best_thr(yv, pv2)
            yt, pt2, _ = aggregate(seg[te], y[te], pt)
            per[r] = score(yt, pt2, thr)
        else:
            thr = best_thr(y[mi], pv)
            per[r] = score(y[te], pt, thr)
        pooled_p[te], pooled_t[te] = pt, thr
    if seg is not None:
        ys, ps, inv = aggregate(seg, y, pooled_p)
        ts = np.bincount(inv, weights=pooled_t) / np.bincount(inv)
        pred = (ps >= ts).astype(float)
        pooled = score(ys, pred, 0.5)
    else:
        pooled = score(y, (pooled_p >= pooled_t).astype(float), 0.5)
    pooled["mean_room_auc"] = round(float(np.nanmean([v["auc"] for v in per.values()])), 4)
    pooled["mean_room_balacc"] = round(float(np.mean([v["balacc"] for v in per.values()])), 4)
    return {"per_room": per, "pooled": pooled}


def load_features32():
    """32-D windows with room / person / video / time (s)."""
    from evaluate_cross_classroom import NEW
    from honest_evaluation import load_human_data

    X, y, person, clip, track, frame, _ = load_human_data(0.30, 60.0)
    rows = {"X": [X], "y": [y], "room": [np.array(["classroom_01"] * len(y))],
            "person": [np.array([f"c01:{p}" for p in person])], "video": [clip.astype(str)],
            "t": [frame / 30.0]}
    for f, (room, label) in NEW.items():
        if room == "room_D":
            continue
        d = np.load(ROOT / "data/processed" / f, allow_pickle=True)
        rows["X"].append(d["features"])
        rows["y"].append((d["targets"] != 2).astype(int))
        rows["room"].append(np.array([room] * len(d["targets"])))
        rows["person"].append(np.array([f"{label}:{s}" for s in d["student_ids"]]))
        rows["video"].append(np.array([label] * len(d["targets"])))
        fps = 24.0 if label == "test_8" else 30.0
        rows["t"].append(d["frame_ids"] / fps)
    return {k: np.concatenate(v) for k, v in rows.items()}


def window_stats(X, drop=()):
    keep = [i for i in range(X.shape[-1]) if X.reshape(-1, X.shape[-1])[:, i].std() > 1e-6 and i not in drop]
    Xk = X[:, :, keep]
    return np.concatenate([Xk.mean(1), Xk.std(1), Xk[:, -1], Xk.min(1), Xk.max(1)], axis=1)


def show(name, res):
    p = res["pooled"]
    rooms = "  ".join(f"{r}:{v['balacc']:.2f}/{v['auc']:.2f}" for r, v in res["per_room"].items())
    print(f"{name:52s} pooled balacc {p['balacc']:.3f} acc {p['acc']:.3f} (always-HIGH {p['always_high']:.3f}) "
          f"macroF1 {p['macro_f1']:.3f} | room balacc/AUC {rooms}")


if __name__ == "__main__":
    D = load_features32()
    y, room, person, video, t = D["y"], D["room"], D["person"], D["video"], D["t"]
    DIR = (0, 1, 2, 3, 4, 21)
    res = {}
    for drop_name, drop in (("all channels", ()), ("no camera-relative", DIR)):
        F = window_stats(D["X"], drop)
        for kind in ("logreg", "hgb"):
            for norm in (False, True):
                for seg_s in (None, 5.0):
                    seg = segments(video, person, t, seg_s) if seg_s else None
                    name = f"32-D {drop_name} | {kind} | {'video-norm' if norm else 'raw'} | {'5s seg' if seg_s else 'window'}"
                    r = loro(F, y, room, person, video, kind=kind, norm=norm, seg=seg)
                    res[name] = r
                    show(name, r)
    out = ROOT / "reports/improvement"
    out.mkdir(parents=True, exist_ok=True)
    (out / "features32.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
