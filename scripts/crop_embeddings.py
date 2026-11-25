"""Image-crop embeddings for every human-labelled student box.

For each labelled video, every `--step`-th frame, each student box (from the
human CVAT file) is cropped and embedded by two local pretrained YOLO models:
  * scb  : checkpoints/scb_yolo_4class.pt (trained on classroom behaviour images)
  * pose : yolo11n-pose.pt               (COCO people)
Output: data/processed/crop_embeddings.npz with per-sample room, video, track,
frame, label (1 = HIGH/Moderate, 0 = LOW), box and both embeddings.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VIDEOS = [  # room, video label, video file, human CVAT xml
    ("classroom_01", "clip_004", "runs/clip_007.mp4", "Anno/annotations.xml"),
    ("classroom_01", "clip_001", "runs/clip_001.mp4", "runs/Annoted file for clip 1/annotations.xml"),
    ("room_B", "11", "runs/11.mp4", "Anno/cvat_new/11/annotations.xml"),
    ("room_B", "non_attentive_1", "runs/non attentive 1.mov", "Anno/cvat_new/NON attentive 1/annotations.xml"),
    ("room_C", "test_8", "runs/Test 8.mp4", "Anno/cvat_new/Test 8/annotations.xml"),
]
LAB = {"high": 1, "moderate": 1, "attentive": 1, "low": 0, "non-attentive": 0}


def boxes_by_frame(xml: Path, min_len: int = 16):
    out = defaultdict(list)
    for t in ET.parse(xml).getroot().findall("track"):
        bs = [b for b in t.findall("box") if b.get("outside") != "1"]
        if len(bs) < min_len:  # stray one-frame duplicate tracks
            continue
        for b in bs:
            a = b.find("attribute")
            if a is None or a.text is None:
                continue
            out[int(b.get("frame"))].append(
                (int(t.get("id")), LAB[a.text.strip().lower()],
                 [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]))
    return out


def crop(frame, box, pad=0.10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
    c, d = int(min(w, x2 + px)), int(min(h, y2 + py))
    return frame[b:d, a:c] if d > b + 4 and c > a + 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--out", default="data/processed/crop_embeddings.npz")
    args = ap.parse_args()

    from ultralytics import YOLO

    models = {"scb": YOLO(str(ROOT / "checkpoints/scb_yolo_4class.pt")), "pose": YOLO(str(ROOT / "yolo11n-pose.pt"))}
    rec = defaultdict(list)
    for room, vlabel, vfile, xml in VIDEOS:
        bf = boxes_by_frame(ROOT / xml)
        cap = cv2.VideoCapture(str(ROOT / vfile))
        f, n0 = 0, len(rec["label"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % args.step == 0 and f in bf:
                items = [(tid, lab, box, crop(frame, box)) for tid, lab, box in bf[f]]
                items = [it for it in items if it[3] is not None]
                if items:
                    crops = [it[3] for it in items]
                    for name, m in models.items():
                        emb = m.embed(crops, imgsz=args.imgsz, verbose=False)
                        rec[name].extend(e.cpu().numpy().astype(np.float32) for e in emb)
                    for tid, lab, box, _ in items:
                        rec["room"].append(room)
                        rec["video"].append(vlabel)
                        rec["track"].append(tid)
                        rec["frame"].append(f)
                        rec["label"].append(lab)
                        rec["box"].append(box)
            f += 1
        cap.release()
        print(f"{vlabel}: {len(rec['label']) - n0} crops")
    out = ROOT / args.out
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()})
    print(f"saved {out} ({len(rec['label'])} crops)")


if __name__ == "__main__":
    main()
