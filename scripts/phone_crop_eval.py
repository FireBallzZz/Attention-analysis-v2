"""Phone detection on enlarged student crops, used as hard evidence.

Phones are a few pixels wide in the full classroom frame, so the pipeline's
full-frame detector found a phone in only 3.9% of student-frames of the phone
clip. Here each matched student box (data/processed/pipeline_components.npz)
is cropped with 25% margin, enlarged to 640 px and passed to the local COCO
detector (yolo11n.pt, class 67 "cell phone"). The highest phone confidence per
student-frame is stored.

Rule tested (fixed in advance, nothing fitted on labels): if phone confidence
>= c, score = min(score, 30). Strict leave-one-classroom-out: the HIGH/LOW
cut-off is chosen on the two training classrooms only.

    python scripts/phone_crop_eval.py
Writes data/processed/phone_crop_conf.npy and reports/improvement/phone_crop.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from crop_embeddings import VIDEOS  # noqa: E402

ROOMS = ["classroom_01", "room_B", "room_C"]


def crop_up(frame, box, margin=0.25, size=640):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    a, b = int(max(0, x1 - mx)), int(max(0, y1 - my))
    c, d = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if c - a < 8 or d - b < 8:
        return None
    crop = frame[b:d, a:c]
    s = size / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def phone_conf(d):
    from ultralytics import YOLO

    det = YOLO(str(ROOT / "yolo11n.pt"))
    conf = np.zeros(len(d["label"]))
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
            crops, idx = [], []
            for r in rs:
                c = crop_up(frame, d["bbox"][r])
                if c is not None:
                    crops.append(c)
                    idx.append(r)
            if not crops:
                continue
            for r, res in zip(idx, det.predict(crops, imgsz=640, conf=0.05, classes=[67], verbose=False)):
                if len(res.boxes):
                    conf[r] = float(res.boxes.conf.max())
        print(f"  {v}: done")
    return conf


def strict(y, room, score):
    grid = np.arange(30, 70.5, 0.5)
    per = {}
    for r in ROOMS:
        cut = max(grid, key=lambda c: np.mean([balanced_accuracy_score(y[room == o], (score[room == o] >= c).astype(int))
                                               for o in ROOMS if o != r]))
        k = room == r
        per[r] = {"cut": float(cut), "balacc": round(float(balanced_accuracy_score(y[k], (score[k] >= cut).astype(int))), 4),
                  "auc": round(float(roc_auc_score(y[k], score[k])), 4)}
    mean = {m: round(float(np.mean([v[m] for v in per.values()])), 4) for m in ("balacc", "auc")}
    return per, mean


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, video = d["label"].astype(int), d["room"], d["video"]
    out_npy = ROOT / "data/processed/phone_crop_conf.npy"
    conf = np.load(out_npy) if out_npy.exists() and len(np.load(out_npy)) == len(y) else phone_conf(d)
    np.save(out_npy, conf)
    m = d["motion"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)
    base = np.where(study >= 0.3, np.maximum(m, 50.001), m)
    res = {"phone_rate": {}, "variants": {}}
    print("\nphone detected (conf >= 0.25) per video: on LOW students vs HIGH students")
    for v in np.unique(video):
        k = video == v
        lo, hi = (conf[k & (y == 0)] >= 0.25).mean() * 100, (conf[k & (y == 1)] >= 0.25).mean() * 100
        res["phone_rate"][str(v)] = {"low_pct": round(float(lo), 1), "high_pct": round(float(hi), 1)}
        print(f"  {v:16s} LOW {lo:5.1f}%   HIGH {hi:5.1f}%")
    for name, sc in [("current final", base)] + [
            (f"phone >= {c} -> score <= 30", np.where(conf >= c, np.minimum(base, 30.0), base)) for c in (0.25, 0.35, 0.5)]:
        per, mean = strict(y, room, sc)
        res["variants"][name] = {"per_room": per, "mean": mean}
        print(f"{name:28s} strict balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
              " ".join(f"{r}:{v['balacc']:.2f}" for r, v in per.items()))
    (ROOT / "reports/improvement/phone_crop.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
