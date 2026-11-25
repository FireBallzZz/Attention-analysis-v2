"""Optical flow vs pixel difference as the motion cue (unseen classrooms).

Same protocol as scripts/motion_parts_eval.py: frames 0.1 s apart, current
human box on both frames, 1.6 s windows, class-relative (log minus video
median), logistic model trained on two classrooms, tested on the third.
Flow: Farneback on 64x64 grey crops, mean magnitude.

    python scripts/optical_flow_eval.py
Writes reports/improvement/optical_flow.json.
"""
import json, sys
from collections import defaultdict, deque
from pathlib import Path
import cv2, numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from attention_pipeline.motion import box_motion
from crop_embeddings import VIDEOS, boxes_by_frame
from motion_parts_eval import rel, loro

def crop64(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

def flow_mag(prev, frame, box):
    a, b = crop64(prev, box), crop64(frame, box)
    if a is None or b is None:
        return None
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 3, 5, 1.1, 0)
    return float(np.mean(np.linalg.norm(f, axis=2)))

def main():
    rows = defaultdict(list)
    for room, vlabel, vfile, xml in VIDEOS:
        bf = boxes_by_frame(ROOT / xml); cap = cv2.VideoCapture(str(ROOT / vfile))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step, emitted, idx, prev = max(native / 10.0, 1.0), 0.0, 0, None
        hist = defaultdict(lambda: {"diff": deque(maxlen=16), "flow": deque(maxlen=16)})
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx >= emitted:
                emitted += step
                if prev is not None:
                    for tid, lab, box in bf.get(idx, []):
                        d, f = box_motion(prev, frame, box), flow_mag(prev, frame, box)
                        if d is None or f is None: continue
                        h = hist[tid]; h["diff"].append(d); h["flow"].append(f)
                        if len(h["diff"]) >= 16:
                            for k, v in (("room", room), ("video", vlabel), ("person", f"{vlabel}:{tid}"), ("label", lab),
                                         ("diff", float(np.mean(h["diff"]))), ("flow", float(np.mean(h["flow"])))):
                                rows[k].append(v)
                prev = frame
            idx += 1
        print(f"  {vlabel} done")
    D = {k: np.asarray(v) for k, v in rows.items()}
    y, room, person, video = D["label"].astype(int), D["room"], D["person"], D["video"]
    res = {}
    for name, F in {"pixel difference (current)": rel(D["diff"], video)[:, None],
                    "optical flow": rel(D["flow"], video)[:, None],
                    "difference + flow": np.stack([rel(D["diff"], video), rel(D["flow"], video)], 1)}.items():
        per, mean, _ = loro(F, y, room, person)
        res[name] = {"per_room": per, "mean": mean}
        print(f"{name:28s} balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " + " ".join(f"{r}:{m['balacc']:.2f}/{m['auc']:.2f}" for r, m in per.items()))
    (ROOT / "reports/improvement/optical_flow.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
