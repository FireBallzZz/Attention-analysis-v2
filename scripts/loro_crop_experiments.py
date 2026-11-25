"""Leave-one-classroom-out with image-crop embeddings (see crop_embeddings.py)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from loro_experiments import loro, segments, show

FPS = {"test_8": 24.0}

def temporal_mean(E, video, track, frame, win=16):
    """Average each sample's embedding with the same student's samples in the previous `win` frames."""
    out = np.empty_like(E)
    order = np.lexsort((frame, track, video))
    keys = np.array([f"{v}|{t}" for v, t in zip(video, track)])
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        idx = idx[np.argsort(frame[idx])]
        fr = frame[idx]
        cs = np.cumsum(E[idx], axis=0)
        lo = np.searchsorted(fr, fr - win + 1)
        n = np.arange(len(idx)) - lo + 1
        prev = np.where(lo[:, None] > 0, cs[np.maximum(lo - 1, 0)], 0)
        out[idx] = (cs - prev) / n[:, None]
    return out

if __name__ == "__main__":
    d = np.load(ROOT / "data/processed/crop_embeddings.npz", allow_pickle=True)
    y, room, video, track, frame = d["label"], d["room"], d["video"], d["track"], d["frame"]
    person = np.array([f"{v}:{t}" for v, t in zip(video, track)])
    t = frame / np.array([FPS.get(v, 30.0) for v in video])
    print("samples", len(y), "LOW share", round(float((y == 0).mean()), 3))
    res = {}
    feats = {"scb": d["scb"], "pose": d["pose"], "scb+pose": np.concatenate([d["scb"], d["pose"]], 1)}
    for fname, E in feats.items():
        Em = temporal_mean(E, video, track, frame)
        for kind in ("logreg",):
            for norm in (False, True):
                for seg_s in (None, 5.0):
                    seg = segments(video, person, t, seg_s) if seg_s else None
                    name = f"crop {fname} | {kind} | {'video-norm' if norm else 'raw'} | {'5s seg' if seg_s else '1.6s window'}"
                    r = loro(Em, y, room, person, video, kind=kind, norm=norm, seg=seg)
                    res[name] = r; show(name, r)
    out = ROOT / "reports/improvement"; out.mkdir(parents=True, exist_ok=True)
    (out / "crops.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
