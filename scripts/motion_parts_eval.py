"""Does splitting motion into HEAD and BODY/HANDS help on unseen classrooms?

Same measurement as the live pipeline (frames 0.1 s apart, current box on both
frames), but motion is measured separately in
  head  : top 35% of the student box
  body  : the remaining 65% (torso, desk, hands)
  full  : the whole box (current final model)
Each is made class-relative (log value minus the video's median), then a
small logistic model is fitted on two classrooms and tested on the third
(threshold chosen on held-out people of the training rooms).

    python scripts/motion_parts_eval.py
Writes reports/improvement/motion_parts.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from attention_pipeline.motion import box_motion  # noqa: E402
from crop_embeddings import VIDEOS, boxes_by_frame  # noqa: E402

HEAD_FRAC = 0.35
WIN = 16


def parts(box):
    x1, y1, x2, y2 = box
    yh = y1 + (y2 - y1) * HEAD_FRAC
    return {"full": box, "head": (x1, y1, x2, yh), "body": (x1, yh, x2, y2)}


def measure():
    rows = defaultdict(list)
    for room, vlabel, vfile, xml in VIDEOS:
        bf = boxes_by_frame(ROOT / xml)
        cap = cv2.VideoCapture(str(ROOT / vfile))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step, emitted, idx, prev = max(native / 10.0, 1.0), 0.0, 0, None
        hist = defaultdict(lambda: {k: deque(maxlen=WIN) for k in ("full", "head", "body")})
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= emitted:
                emitted += step
                for tid, lab, box in bf.get(idx, []):
                    ms = {k: box_motion(prev, frame, b) for k, b in parts(box).items()}
                    if any(v is None for v in ms.values()):
                        continue
                    h = hist[tid]
                    for k, v in ms.items():
                        h[k].append(v)
                    if len(h["full"]) >= WIN:
                        rows["room"].append(room)
                        rows["video"].append(vlabel)
                        rows["person"].append(f"{vlabel}:{tid}")
                        rows["label"].append(lab)
                        for k in ("full", "head", "body"):
                            rows[k].append(float(np.mean(h[k])))
                prev = frame
            idx += 1
        cap.release()
        print(f"  {vlabel}: {sum(1 for v in rows['video'] if v == vlabel)} windows")
    return {k: np.asarray(v) for k, v in rows.items()}


def rel(x, video):
    lx = np.log(np.maximum(x, 1e-4))
    out = np.empty_like(lx)
    for v in np.unique(video):
        m = video == v
        out[m] = lx[m] - np.median(lx[m])
    return out


def best_thr(y, p):
    grid = np.quantile(p, np.linspace(0.02, 0.98, 49))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def loro(F, y, room, person):
    per = {}
    for r in sorted(np.unique(room)):
        te, tr = room == r, room != r
        rng = np.random.default_rng(11)
        tp = np.unique(person[tr])
        inner = rng.choice(tp, max(2, int(len(tp) * 0.25)), replace=False)
        mi = np.isin(person, inner) & tr
        clf = LogisticRegression(C=1.0, class_weight="balanced").fit(F[tr & ~mi], y[tr & ~mi])
        t = best_thr(y[mi], clf.predict_proba(F[mi])[:, 1])
        p = clf.predict_proba(F[te])[:, 1]
        per[r] = {"balacc": round(float(balanced_accuracy_score(y[te], (p >= t).astype(int))), 4),
                  "auc": round(float(roc_auc_score(y[te], p)), 4)}
    mean = {k: round(float(np.mean([m[k] for m in per.values()])), 4) for k in ("balacc", "auc")}
    return per, mean, clf


def main():
    print("measuring head / body / full-box motion ...")
    D = measure()
    y, room, person, video = D["label"].astype(int), D["room"], D["person"], D["video"]
    Z = {k: rel(D[k], video) for k in ("full", "head", "body")}
    print("\nsingle-part AUC per classroom (<0.5: more motion goes with LOW)")
    for k, z in Z.items():
        print(f"  {k:5s}", "  ".join(f"{r}:{roc_auc_score(y[room == r], z[room == r]):.2f}" for r in np.unique(room)))
    variants = {"full box (current)": ["full"], "head only": ["head"], "body only": ["body"],
                "head + body": ["head", "body"], "full + head + body": ["full", "head", "body"]}
    res = {}
    print()
    for name, ks in variants.items():
        F = np.stack([Z[k] for k in ks], 1)
        per, mean, clf = loro(F, y, room, person)
        res[name] = {"per_room": per, "mean": mean, "coef_last_fold": clf.coef_[0].round(3).tolist()}
        print(f"{name:20s} balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
              " ".join(f"{r}:{m['balacc']:.2f}/{m['auc']:.2f}" for r, m in per.items()) +
              f" | weights {res[name]['coef_last_fold']}")
    (ROOT / "reports/improvement/motion_parts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
