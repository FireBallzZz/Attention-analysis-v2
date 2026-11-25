"""Turn one pipeline run (scores JSONL + annotated video) into a dashboard view.

Shared by scripts/build_defence_dashboard.py (demo videos) and
launch_defence_dashboard.py (videos uploaded in the dashboard), so both show
the same timeline, alerts, student table and seat heatmap.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Stage-5 rules, same values as configs/pipeline.yaml `drop`
DROP = {"ema": 0.3, "low": 40.0, "sustain": 3.0, "sudden_delta": 20.0, "sudden_s": 2.0, "cooldown": 8.0}
SERIES_HZ = 2.0  # heatmap time resolution
SCB_LABELS = ("hand-raising", "reading", "writing", "using phone", "bowing head", "leaning on table")


def transcode(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so every browser can play it."""
    import imageio_ffmpeg

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-vf", "scale='min(1280,iw)':-2", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0


def clean_frame(video: Path, dst: Path, at: float = 0.33) -> tuple[int, int]:
    """Save one frame of the ORIGINAL video (no boxes) as the heatmap background.
    Returns the source frame size."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * at)))
    ok, frame = cap.read()
    cap.release()
    if ok:
        s = min(1.0, 1280 / max(w, 1))
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.resize(frame, (int(w * s), int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return w, h


def drop_events(series: list[tuple[float, float]]):
    """EMA-smoothed score; sustained < 40 for 3 s, or a >= 20-point fall within 2 s; 8 s cooldown."""
    events, hist, ema, last = [], [], None, -1e9
    for t, s in series:
        ema = s if ema is None else DROP["ema"] * s + (1 - DROP["ema"]) * ema
        hist.append((t, ema))
        if t - last < DROP["cooldown"]:
            continue
        recent_s = [(tt, v) for tt, v in hist if tt >= t - DROP["sustain"]]
        span = t - recent_s[0][0]
        kind = None
        if span >= DROP["sustain"] * 0.8 and all(v < DROP["low"] for _, v in recent_s):
            kind = "sustained"
        else:
            recent = [v for tt, v in hist if tt >= t - DROP["sudden_s"]]
            if recent and max(recent) - ema >= DROP["sudden_delta"]:
                kind = "sudden"
        if kind:
            events.append((t, kind, ema))
            last = t
    return events


def build_payload(*, key: str, title: str, room: str, labelled: bool, jsonl: Path,
                  video_url: str, poster_url: str, frame_size: tuple[int, int]) -> dict:
    """Everything the dashboard needs for one analysed video."""
    from attention_pipeline.stages.intervention import recommend
    from attention_pipeline.types import AttentionEvent

    W, H = frame_size
    rows = [json.loads(line) for line in jsonl.open(encoding="utf-8") if line.strip()]
    timeline = []
    per = defaultdict(lambda: {"scores": [], "cues": Counter(), "series": [], "boxes": [],
                               "daisee": [], "aux": [], "scb": []})
    for r in rows:
        sc = [s for s in r["students"] if s.get("score") is not None]
        if sc:
            vals = [float(s["score"]) for s in sc]
            timeline.append([round(r["timestamp"], 2), round(float(np.mean(vals)), 1),
                             round(100.0 * sum(float(s["score"]) < 50 for s in sc) / len(sc), 1), len(sc)])
        for s in sc:
            d = per[s["student_id"]]
            d["scores"].append(float(s["score"]))
            d["cues"][s.get("cue", "")] += 1
            d["series"].append((float(r["timestamp"]), float(s["score"])))
            if s.get("bbox"):
                d["boxes"].append(s["bbox"])
            if s.get("daisee_score") is not None:
                d["daisee"].append(float(s["daisee_score"]))
                if s.get("daisee_aux"):
                    d["aux"].append(s["daisee_aux"])
            if s.get("posture"):
                d["scb"].append(s["posture"])

    class_mean = float(np.mean([t[1] for t in timeline])) if timeline else None
    alerts, students, seats = [], [], []
    for sid, d in per.items():
        cue = d["cues"].most_common(1)[0][0]
        evs = drop_events(d["series"])
        for t, kind, sc in evs:
            ev = AttentionEvent(student_id=int(sid), timestamp=t, score=sc, kind=kind,
                                dominant_modality="behavior", explanation_features={})
            rec = recommend(ev, class_mean, cue)
            alerts.append({"t": round(t, 1), "student": int(sid), "kind": kind, "score": round(sc, 1),
                           "cue": cue, "recommendation": rec["teaching_recommendation"],
                           "suggestion": rec["intervention_suggestion"]})
        arr = np.array(d["scores"])
        st = {"id": int(sid), "n": int(len(arr)), "mean": round(float(arr.mean()), 1),
              "low_pct": round(float((arr < 50).mean() * 100), 1), "cue": cue, "alerts": len(evs)}
        if d["daisee"]:  # DAiSEE individual model (shown, not used in the score)
            st["daisee"] = round(float(np.mean(d["daisee"])), 1)
            if d["aux"]:
                st["daisee_aux"] = [round(float(v), 2) for v in np.mean(np.array(d["aux"], dtype=float), 0)]
        if d["scb"]:     # SCB behaviour (shown, not used in the score)
            sc = np.mean(np.array(d["scb"], dtype=float), 0)
            st["scb"] = SCB_LABELS[int(np.argmax(sc))] if len(sc) == len(SCB_LABELS) else None
            st["scb_p"] = round(float(sc.max()), 2)
        students.append(st)
        if d["boxes"] and W and H:
            b = np.median(np.array(d["boxes"], dtype=float), axis=0)
            # score series resampled to SERIES_HZ for the time-following heatmap
            bins = defaultdict(list)
            for t, v in d["series"]:
                bins[round(t * SERIES_HZ) / SERIES_HZ].append(v)
            seats.append({
                "id": int(sid),
                "cx": round(float((b[0] + b[2]) / 2 / W), 4), "cy": round(float((b[1] + b[3]) / 2 / H), 4),
                "w": round(float((b[2] - b[0]) / W), 4), "h": round(float((b[3] - b[1]) / H), 4),
                "mean": round(float(arr.mean()), 1), "low_pct": round(float((arr < 50).mean() * 100), 1),
                "cue": cue, "n": int(len(arr)),
                "series": [[t, round(float(np.mean(v)), 1)] for t, v in sorted(bins.items())],
            })
    students.sort(key=lambda s: s["mean"])
    alerts.sort(key=lambda a: a["t"])
    return {
        "key": key, "title": title, "room": room, "labelled": labelled,
        "video": video_url, "poster": poster_url, "frame": [W, H],
        "duration": rows[-1]["timestamp"] if rows else 0,
        "students_tracked": len(per), "timeline": timeline,
        "students": students, "alerts": alerts, "seats": seats,
        "class_mean": None if class_mean is None else round(class_mean, 2),
        "low_share": round(float(np.mean([t[2] for t in timeline])), 2) if timeline else None,
    }
