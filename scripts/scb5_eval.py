"""Do the new SCB-Dataset5 detectors (turn head, bow head, discuss) improve the
attention score on unseen classrooms?

For every matched student-frame in data/processed/pipeline_components.npz the
two detectors (checkpoints/scb5_bowturn.pt, scb5_discuss.pt) are run on that
video frame; each student gets the highest confidence per class among the
detections whose centre lies inside the student's box. Tested per classroom:

  * each cue alone (AUC; < 0.5 means the cue goes with LOW, as hoped)
  * rule: final score, then if a cue's confidence >= c, score = min(score, 40)
    (hard evidence, nothing fitted on the labels; c fixed at 0.5)
  * logistic model on [motion, cues], trained on two rooms, tested on the third

    python scripts/scb5_eval.py
Writes reports/improvement/scb5_eval.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from crop_embeddings import VIDEOS  # noqa: E402

CUES = ["BowHead", "TurnHead", "discuss"]


def detect_cues(d):
    from ultralytics import YOLO

    models = {"bowturn": YOLO(str(ROOT / "checkpoints/scb5_bowturn.pt")),
              "discuss": YOLO(str(ROOT / "checkpoints/scb5_discuss.pt"))}
    cue = np.zeros((len(d["label"]), len(CUES)))
    vfile = {v: f for _, v, f, _ in VIDEOS}
    for v in np.unique(d["video"]):
        rows = np.where(d["video"] == v)[0]
        by_t = defaultdict(list)
        for r in rows:
            by_t[round(float(d["t"][r]), 3)].append(r)
        cap = cv2.VideoCapture(str(ROOT / vfile[v]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t, rs in sorted(by_t.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            dets = []
            for name, m in models.items():
                res = m.predict(frame, imgsz=1280, conf=0.10, verbose=False)[0]
                for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy().astype(int)):
                    label = m.names[k]
                    dets.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, CUES.index(label), float(c)))
            for r in rs:
                x1, y1, x2, y2 = d["bbox"][r]
                for cx, cy, j, c in dets:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        cue[r, j] = max(cue[r, j], c)
        print(f"  {v}: cues on {len(rows)} student-frames")
    return cue


def best_thr(y, p):
    grid = np.quantile(p, np.linspace(0.02, 0.98, 49))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room = d["label"].astype(int), d["room"]
    motion = d["motion"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)
    final = np.where(study >= 0.3, np.maximum(motion, 50.0 + 1e-3), motion)   # current deployed score
    cue = detect_cues(d)
    np.save(ROOT / "data/processed/scb5_cues.npy", cue)
    rooms = np.unique(room)
    res = {"cue_auc": {}, "variants": {}}
    print("\ncue AUC per classroom (<0.5: cue goes with LOW)")
    for j, c in enumerate(CUES):
        res["cue_auc"][c] = {r: round(float(roc_auc_score(y[room == r], cue[room == r, j])), 3) for r in rooms}
        res["cue_auc"][c]["present_pct"] = round(float((cue[:, j] >= 0.5).mean() * 100), 1)
        print(f"  {c:9s}", res["cue_auc"][c])

    def report(name, score):
        per = {r: {"balacc": round(float(balanced_accuracy_score(y[room == r], (score[room == r] >= 50).astype(int))), 4),
                   "auc": round(float(roc_auc_score(y[room == r], score[room == r])), 4)} for r in rooms}
        mean = {k: round(float(np.mean([v[k] for v in per.values()])), 4) for k in ("balacc", "auc")}
        res["variants"][name] = {"per_room": per, "mean": mean}
        print(f"{name:45s} balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
              " ".join(f"{r}:{v['balacc']:.2f}" for r, v in per.items()))

    print()
    report("current final (motion + study protection)", final)
    for j, c in enumerate(CUES):
        report(f"+ {c} >= 0.5 lowers score to <= 40", np.where(cue[:, j] >= 0.5, np.minimum(final, 40.0), final))
    any_cue = (cue[:, [1, 2]] >= 0.5).any(1)
    report("+ TurnHead or discuss >= 0.5 lowers to <= 40", np.where(any_cue, np.minimum(final, 40.0), final))
    # learned combination, leave-one-classroom-out
    F = np.column_stack([np.log(np.maximum(motion, 1)), cue])
    per = {}
    for r in rooms:
        te, tr = room == r, room != r
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(F[tr], y[tr])
        t = best_thr(y[tr], clf.predict_proba(F[tr])[:, 1])
        p = clf.predict_proba(F[te])[:, 1]
        per[r] = {"balacc": round(float(balanced_accuracy_score(y[te], (p >= t).astype(int))), 4),
                  "auc": round(float(roc_auc_score(y[te], p)), 4)}
    mean = {k: round(float(np.mean([v[k] for v in per.values()])), 4) for k in ("balacc", "auc")}
    res["variants"]["learned: motion + SCB5 cues"] = {"per_room": per, "mean": mean}
    print(f"{'learned: motion + SCB5 cues':45s} balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
          " ".join(f"{r}:{v['balacc']:.2f}" for r, v in per.items()))
    (ROOT / "reports/improvement/scb5_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
