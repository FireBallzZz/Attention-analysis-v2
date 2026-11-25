"""Run the live pipeline on every human-labelled video and save, per matched
student-frame, every component score next to the human label:

  motion_score   class-relative motion model (0-100)
  daisee_score   DAiSEE-only network applied to the individual student (0-100)
  daisee_aux     DAiSEE ratings: engagement, boredom, confusion, frustration
  neural_score   classroom-fine-tuned BiGRU (0-100)
  scb            SCB behaviour probabilities: hand-raising, reading, writing,
                 using-phone, bowing-head, leaning-over-table
  rubric         rule-based rubric score (0-100)

Output: data/processed/pipeline_components.npz (input to scripts/ensemble_eval.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_embeddings import VIDEOS, boxes_by_frame  # noqa: E402
from evaluate_pipeline_vs_human import match  # noqa: E402


def main():
    os.chdir(ROOT)
    from attention_pipeline.pipeline import AttentionPipeline

    pipe = AttentionPipeline.from_yaml(ROOT / "configs" / "pipeline.yaml", load_weights=True)
    pipe.ensemble = None  # collect raw components, not the ensemble output
    print("DAiSEE model:", pipe.daisee_net is not None, "| motion model:", pipe.motion_scorer is not None)
    rec = {k: [] for k in ("room", "video", "track", "t", "label", "motion", "daisee", "daisee_aux",
                           "neural", "scb", "rubric", "bbox")}
    for room, vlabel, vfile, xml in VIDEOS:
        pipe.reset()
        bf = boxes_by_frame(ROOT / xml)
        native = cv2.VideoCapture(str(ROOT / vfile)).get(cv2.CAP_PROP_FPS) or 30.0
        n0 = len(rec["label"])
        for res in pipe.run_video(str(ROOT / vfile)):
            f = int(round(res["timestamp"] * native))
            gts = bf.get(f) or bf.get(f - 1) or bf.get(f + 1) or []
            ready = [st for st in res["students"]
                     if st.get("motion_score") is not None and st.get("daisee_score") is not None]
            for st, bi in match(ready, gts):
                rec["room"].append(room)
                rec["video"].append(vlabel)
                rec["track"].append(int(st["student_id"]))
                rec["t"].append(float(res["timestamp"]))
                rec["label"].append(gts[bi][1])
                rec["motion"].append(float(st["motion_score"]))
                rec["daisee"].append(float(st["daisee_score"]))
                rec["daisee_aux"].append(st["daisee_aux"])
                rec["neural"].append(float(st["neural_score"]) if st.get("neural_score") is not None else np.nan)
                rec["scb"].append(st["posture"])
                rec["rubric"].append(float(st["heuristic_score"]))
                rec["bbox"].append([float(v) for v in st["bbox"]])
        print(f"  {vlabel}: {len(rec['label']) - n0} matched student-frames")
    out = ROOT / "data/processed/pipeline_components.npz"
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
