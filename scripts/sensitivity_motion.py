"""Sensitivity of accuracy to how motion is measured (unseen classrooms).

Same protocol as scripts/motion_rule.py (human boxes, 10 fps steps, class-
relative log motion, logistic model trained on two classrooms, threshold chosen
on held-out people of those rooms, tested on the third). One parameter changed
at a time, the others at the deployed value:

  window : frames averaged per decision      (deployed 16 = 1.6 s)
  gap    : time between compared frames       (deployed 0.1 s)
  size   : crop resolution for the difference (deployed 64 px)

    python scripts/sensitivity_motion.py
Writes reports/improvement/sensitivity_motion.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from crop_embeddings import VIDEOS, boxes_by_frame  # noqa: E402
from motion_parts_eval import loro, rel  # noqa: E402

GAPS = {0.1: 1, 0.2: 2, 0.3: 3}          # seconds -> number of 10 fps steps
SIZES = (32, 64, 128)
WINDOWS = (8, 16, 24, 32, 48)
DEPLOYED = {"window": 16, "gap": 0.1, "size": 64}


def crop(frame, box, size):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def main():
    combos = [(g, s) for g in GAPS for s in SIZES]
    rows = defaultdict(list)
    for room, vlabel, vfile, xml in VIDEOS:
        bf = boxes_by_frame(ROOT / xml)
        cap = cv2.VideoCapture(str(ROOT / vfile))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step, emitted, idx = max(native / 10.0, 1.0), 0.0, 0
        past = deque(maxlen=max(GAPS.values()))       # previous analysed frames
        hist = defaultdict(lambda: {c: deque(maxlen=max(WINDOWS)) for c in combos})
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= emitted:
                emitted += step
                if len(past) == past.maxlen:
                    for tid, lab, box in bf.get(idx, []):
                        vals = {}
                        for g, s in combos:
                            a, b = crop(past[-GAPS[g]], box, s), crop(frame, box, s)
                            if a is None or b is None:
                                break
                            vals[(g, s)] = float(np.mean(cv2.absdiff(a, b)) / 255.0)
                        if len(vals) != len(combos):
                            continue
                        h = hist[tid]
                        for c, v in vals.items():
                            h[c].append(v)
                        if len(h[combos[0]]) >= max(WINDOWS):
                            rows["room"].append(room)
                            rows["video"].append(vlabel)
                            rows["person"].append(f"{vlabel}:{tid}")
                            rows["label"].append(lab)
                            for c in combos:
                                hv = list(h[c])
                                for w in WINDOWS:
                                    rows[f"{c[0]}|{c[1]}|{w}"].append(float(np.mean(hv[-w:])))
                past.append(frame)
            idx += 1
        print(f"  {vlabel}: {sum(1 for v in rows['video'] if v == vlabel)} windows")
    D = {k: np.asarray(v) for k, v in rows.items()}
    y, room, person, video = D["label"].astype(int), D["room"], D["person"], D["video"]

    def run(g, s, w):
        per, mean, _ = loro(rel(D[f"{g}|{s}|{w}"], video)[:, None], y, room, person)
        return {"mean": mean, "per_room": {r: v["balacc"] for r, v in per.items()}}

    res = {"window": {}, "gap": {}, "size": {}}
    dg, ds, dw = DEPLOYED["gap"], DEPLOYED["size"], DEPLOYED["window"]
    for w in WINDOWS:
        res["window"][w] = run(dg, ds, w)
    for g in GAPS:
        res["gap"][g] = run(g, ds, dw)
    for s in SIZES:
        res["size"][s] = run(dg, s, dw)
    for name, unit in (("window", "frames"), ("gap", "s"), ("size", "px")):
        print(f"\n{name} (deployed {DEPLOYED[name]} {unit})")
        for k, v in res[name].items():
            print(f"   {k:>5} {unit}: balacc {v['mean']['balacc']:.3f}  AUC {v['mean']['auc']:.3f}  rooms {v['per_room']}")
    # nested: choose each parameter on the two training rooms, test on the third
    nested = {}
    for name in res:
        out = {}
        for r in np.unique(room):
            others = [o for o in np.unique(room) if o != r]
            best = max(res[name], key=lambda k: np.mean([res[name][k]["per_room"][o] for o in others]))
            out[str(r)] = {"chosen": best, "balacc": res[name][best]["per_room"][r]}
        nested[name] = {"per_room": out, "mean_balacc": round(float(np.mean([v["balacc"] for v in out.values()])), 4)}
        print(f"nested choice of {name}: {out}  -> mean {nested[name]['mean_balacc']:.3f}")
    res["nested"] = nested
    (ROOT / "reports/improvement/sensitivity_motion.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
