"""Gaze target estimation (Gaze-LLE, Ryan et al. 2025, github.com/fkryan/gazelle).

Instead of head angle relative to the camera (which flips between rooms), Gaze-LLE
predicts WHERE in the image each person is looking, plus whether the target is
inside the frame at all. Model: gazelle_dinov2_vitb14_inout (frozen DINOv2
ViT-B + GazeFollow/VideoAttentionTarget decoder). No labels are used to compute
the cues; for every matched student-frame (data/processed/pipeline_components.npz):

  inout     probability the gaze target is inside the frame (the board/teacher
            is usually behind or near the camera, i.e. out of frame)
  dy, dx    gaze target offset from the head, in head-box units (dy > 0: looking down)
  on_peer   gaze target lands inside another student's box (looking at a neighbour)

Tested per classroom (AUC) and strictly leave-one-classroom-out (cue weights
and cut-off fitted on the two training classrooms only), alone and combined
with the current final score.

    python scripts/gaze_target_eval.py
Writes data/processed/gaze_target.npz and reports/improvement/gaze_target.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "gazelle"))
from crop_embeddings import VIDEOS  # noqa: E402

ROOMS = ["classroom_01", "room_B", "room_C"]
CKPT = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt"


def head_box(b, W, H):
    x1, y1, x2, y2 = b
    w = x2 - x1
    s = 0.55 * w
    cx = (x1 + x2) / 2
    hb = (max(0, cx - s / 2) / W, max(0, y1) / H, min(W, cx + s / 2) / W, min(H, y1 + s) / H)
    return hb, (cx, y1 + s / 2), w


def compute(d):
    from PIL import Image
    from gazelle.model import get_gazelle_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_gazelle_model("gazelle_dinov2_vitb14_inout")
    model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(CKPT, map_location="cpu"))
    model.eval().to(dev)
    n = len(d["label"])
    out = {k: np.full(n, np.nan) for k in ("inout", "dx", "dy", "on_peer")}
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
            H, W = frame.shape[:2]
            img = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
            boxes = [d["bbox"][r] for r in rs]
            heads = [head_box(b, W, H) for b in boxes]
            with torch.no_grad():
                o = model({"images": img, "bboxes": [[h[0] for h in heads]]})
            for j, r in enumerate(rs):
                hm = o["heatmap"][0][j].float().cpu().numpy()
                iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
                gx, gy = (ix + 0.5) / hm.shape[1] * W, (iy + 0.5) / hm.shape[0] * H
                (hcx, hcy), w = heads[j][1], heads[j][2]
                out["inout"][r] = float(o["inout"][0][j])
                out["dx"][r] = (gx - hcx) / max(w, 1)
                out["dy"][r] = (gy - hcy) / max(w, 1)
                out["on_peer"][r] = float(any(k != j and b[0] <= gx <= b[2] and b[1] <= gy <= b[3]
                                              for k, b in enumerate(boxes)))
        print(f"  {v}: gaze targets for {len(rows)} student-frames")
    return out


def strict(y, room, score):
    grid = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    per = {}
    for r in ROOMS:
        cut = max(grid, key=lambda c: np.mean([balanced_accuracy_score(y[room == o], (score[room == o] >= c).astype(int))
                                               for o in ROOMS if o != r]))
        k = room == r
        per[r] = {"balacc": round(float(balanced_accuracy_score(y[k], (score[k] >= cut).astype(int))), 4),
                  "auc": round(float(roc_auc_score(y[k], score[k])), 4)}
    return per, {m: round(float(np.mean([v[m] for v in per.values()])), 4) for m in ("balacc", "auc")}


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, video = d["label"].astype(int), d["room"], d["video"]
    cache = ROOT / "data/processed/gaze_target.npz"
    if cache.exists() and len(np.load(cache)["inout"]) == len(y):
        g = dict(np.load(cache))
    else:
        g = compute(d)
        np.savez_compressed(cache, **g)
    res = {"cue_auc": {}, "variants": {}}
    print("\ncue AUC per classroom (> 0.5: higher value goes with HIGH attention)")
    for k in ("inout", "dx", "dy", "on_peer"):
        v = np.nan_to_num(g[k])
        if k == "dx":
            v = np.abs(v)
        res["cue_auc"][k] = {r: round(float(roc_auc_score(y[room == r], v[room == r])), 3) for r in ROOMS}
        print(f"  {k:8s} {res['cue_auc'][k]}")

    m = d["motion"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)
    phone = np.load(ROOT / "data/processed/phone_crop_conf.npy")

    def rules(s):
        s = np.where(study >= 0.3, np.maximum(s, 50.001), s)
        return np.where(phone >= 0.25, np.minimum(s, 30.0), s)

    G = np.column_stack([np.nan_to_num(g["inout"], nan=0.5), np.abs(np.nan_to_num(g["dx"])),
                         np.nan_to_num(g["dy"]), np.nan_to_num(g["on_peer"])])
    # per-video standardisation (label-free)
    for v in np.unique(video):
        k = video == v
        G[k] = (G[k] - G[k].mean(0)) / (G[k].std(0) + 1e-6)
    z = np.log(np.maximum(m, 1.0))[:, None]

    def per_fold(F):
        out = np.zeros(len(y))
        for r in ROOMS:
            tr, te = room != r, room == r
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(F[tr], y[tr])
            out[te] = clf.predict_proba(F[te])[:, 1] * 100
        return out

    variants = {
        "current final (motion + rules)": rules(m),
        "gaze cues only (per fold)": per_fold(G),
        "motion + gaze cues (per fold) + rules": rules(per_fold(np.hstack([z, G]))),
        "motion only, same learner (per fold) + rules": rules(per_fold(z)),
    }
    print()
    for name, s in variants.items():
        per, mean = strict(y, room, s)
        res["variants"][name] = {"per_room": per, "mean": mean}
        print(f"{name:45s} strict balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
              " ".join(f"{r}:{p['balacc']:.2f}/{p['auc']:.2f}" for r, p in per.items()))
    (ROOT / "reports/improvement/gaze_target.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
