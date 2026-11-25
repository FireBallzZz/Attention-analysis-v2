"""Evaluate and fit the motion-based attention rule (the cue that transfers).

1. Re-measures box motion on every human-labelled video exactly as the live
   pipeline does (frames stepped at 10 fps, current box on both frames).
2. Leave-one-classroom-out: a logistic rule on log(mean motion over 1.6 s)
   relative to the class median (label-free per-video normalisation)
   is fitted on two classrooms, its threshold chosen on held-out people of
   those classrooms, and it is tested on the third classroom. Reports the
   PER-ROOM balanced accuracy (the honest number) and the pooled one.
3. Fits the final rule on all three classrooms (threshold = mean of the
   leave-one-out thresholds) -> checkpoints/motion_rule_v1.json.

    python scripts/motion_rule.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from attention_pipeline.motion import box_motion  # noqa: E402
from crop_embeddings import VIDEOS, boxes_by_frame  # noqa: E402

WINDOWS = {"1.6s": 16, "3s": 30}


def measure():
    rows = defaultdict(list)
    for room, vlabel, vfile, xml in VIDEOS:
        bf = boxes_by_frame(ROOT / xml)
        cap = cv2.VideoCapture(str(ROOT / vfile))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step, emitted, idx, prev = max(native / 10.0, 1.0), 0.0, 0, None
        hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(WINDOWS.values())))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= emitted:
                emitted += step
                for tid, lab, box in bf.get(idx, []):
                    m = box_motion(prev, frame, box)
                    if m is None:
                        continue
                    hist[tid].append(m)
                    if len(hist[tid]) >= WINDOWS["1.6s"]:
                        h = list(hist[tid])
                        rows["room"].append(room)
                        rows["video"].append(vlabel)
                        rows["person"].append(f"{vlabel}:{tid}")
                        rows["label"].append(lab)
                        rows["t"].append(idx / native)
                        for name, w in WINDOWS.items():
                            rows[f"m_{name}"].append(float(np.mean(h[-w:])))
                prev = frame
            idx += 1
        cap.release()
        print(f"  {vlabel}: {sum(1 for v in rows['video'] if v == vlabel)} windows")
    return {k: np.asarray(v) for k, v in rows.items()}


def best_thr(y, p):
    grid = np.quantile(p, np.linspace(0.02, 0.98, 49))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def relative(x, video):
    """log motion minus the median log motion of the same video (the class's own level).
    Label-free; live, the pipeline uses the running median of the class so far."""
    lx = np.log(np.maximum(x, 1e-4))
    out = np.empty_like(lx)
    for v in np.unique(video):
        m = video == v
        out[m] = lx[m] - np.median(lx[m])
    return out


def fit(z, y):
    return LogisticRegression(C=1.0, class_weight="balanced").fit(z[:, None], y)


def evaluate(D, key):
    x, y, room, person = relative(D[key], D["video"]), D["label"], D["room"], D["person"]
    per, thrs, P, T = {}, [], np.zeros(len(y)), np.zeros(len(y))
    for r in sorted(np.unique(room)):
        te, tr = room == r, room != r
        rng = np.random.default_rng(11)
        tp = np.unique(person[tr])
        inner = rng.choice(tp, max(2, int(len(tp) * 0.25)), replace=False)
        mi = np.isin(person, inner) & tr
        clf = fit(x[tr & ~mi], y[tr & ~mi])
        pv = clf.predict_proba(x[mi][:, None])[:, 1]
        t = best_thr(y[mi], pv)
        pt = clf.predict_proba(x[te][:, None])[:, 1]
        pr = (pt >= t).astype(int)
        per[r] = {"balacc": balanced_accuracy_score(y[te], pr), "acc": accuracy_score(y[te], pr),
                  "macro_f1": f1_score(y[te], pr, average="macro"), "auc": roc_auc_score(y[te], pt),
                  "low_recall": float(((pr == 0) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())),
                  "always_high": float(y[te].mean()), "n": int(te.sum())}
        thrs.append(t)
        P[te], T[te] = pt, t
    per = {r: {k: round(float(v), 4) for k, v in m.items()} for r, m in per.items()}
    pooled_pred = (P >= T).astype(int)
    summary = {k: round(float(np.mean([m[k] for m in per.values()])), 4) for k in ("balacc", "acc", "macro_f1", "auc", "low_recall", "always_high")}
    summary["pooled_balacc"] = round(float(balanced_accuracy_score(y, pooled_pred)), 4)
    summary["pooled_acc"] = round(float(accuracy_score(y, pooled_pred)), 4)
    return per, summary, float(np.mean(thrs))


def main():
    print("measuring box motion at 10 fps ...")
    D = measure()
    results = {}
    for name in WINDOWS:
        per, summ, thr = evaluate(D, f"m_{name}")
        results[name] = {"per_room": per, "mean_over_rooms": summ, "threshold": thr}
        print(f"window {name}: per-room balacc {summ['balacc']:.3f} "
              f"({' '.join(f'{r}:{m['balacc']:.2f}' for r, m in per.items())}) AUC {summ['auc']:.3f} "
              f"acc {summ['acc']:.3f} | pooled balacc {summ['pooled_balacc']:.3f}")
    best = max(results, key=lambda k: results[k]["mean_over_rooms"]["balacc"])
    clf = fit(relative(D[f"m_{best}"], D["video"]), D["label"])
    rule = {"a": float(clf.intercept_[0]), "b": float(clf.coef_[0][0]), "threshold": results[best]["threshold"],
            "window": WINDOWS[best], "trained_on": "classroom_01, room_B, room_C (human labels)",
            "expected_unseen_classroom": results[best]["mean_over_rooms"], "per_unseen_room": results[best]["per_room"]}
    (ROOT / "checkpoints/motion_rule_v1.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")
    out = ROOT / "reports/improvement"
    (out / "motion_rule.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"chosen window {best}; wrote checkpoints/motion_rule_v1.json  (b={rule['b']:.3f}: "
          f"{'more motion -> LOW' if rule['b'] < 0 else 'more motion -> HIGH'})")


if __name__ == "__main__":
    main()
