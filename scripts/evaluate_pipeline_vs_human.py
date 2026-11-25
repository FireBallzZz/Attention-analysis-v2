"""End-to-end check: run the LIVE pipeline (configs/pipeline.yaml) on every
human-labelled video and compare its per-student scores with the human labels.

Pipeline boxes are matched to human CVAT boxes per frame (greedy, one-to-one, pipeline box at least 60% inside the human box).
Note: the motion rule's two coefficients were fitted on these same classrooms,
so this is an integration check of the live implementation, not an
unseen-classroom estimate (that is reports/improvement/motion_rule.json).

    python scripts/evaluate_pipeline_vs_human.py
Writes reports/final_model/pipeline_vs_human.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_embeddings import VIDEOS, boxes_by_frame  # noqa: E402


def contained(pipe_box, human_box):
    """Share of the pipeline box that lies inside the human box. Pipeline boxes
    are 1/3-1/2 the size of the hand-drawn ones (which include the desk), so
    plain IoU >= 0.3 discarded ~40% of correctly detected students."""
    x1, y1 = max(pipe_box[0], human_box[0]), max(pipe_box[1], human_box[1])
    x2, y2 = min(pipe_box[2], human_box[2]), min(pipe_box[3], human_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1e-6, (pipe_box[2] - pipe_box[0]) * (pipe_box[3] - pipe_box[1]))
    return inter / area


def match(students, gts, min_contained=0.6):
    """Greedy one-to-one matching of pipeline students to human boxes.
    Score = containment, ties broken by IoU. Returns [(student, human_index)]."""
    pairs = []
    for i, st in enumerate(students):
        for k, (_tid, _lab, box) in enumerate(gts):
            c = contained(st["bbox"], box)
            if c >= min_contained:
                pairs.append((c + 0.01 * iou(st["bbox"], box), i, k))
    used_s, used_g, out = set(), set(), []
    for _sc, i, k in sorted(pairs, reverse=True):
        if i in used_s or k in used_g:
            continue
        used_s.add(i)
        used_g.add(k)
        out.append((students[i], k))
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    import os

    os.chdir(ROOT)
    from attention_pipeline.pipeline import AttentionPipeline

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", default="", help="override fusion.combine_mode (model | model_floor | model_hard)")
    ap.add_argument("--out", default="reports/final_model/pipeline_vs_human.json")
    args = ap.parse_args()
    pipe = AttentionPipeline.from_yaml(ROOT / "configs" / "pipeline.yaml", load_weights=True)
    if args.combine:
        pipe.combine_mode = args.combine
    print("combine mode:", pipe.combine_mode)
    print("motion model active:", pipe.motion_scorer is not None)
    out = {}
    for room, vlabel, vfile, xml in VIDEOS:
        pipe.reset()
        bf = boxes_by_frame(ROOT / xml)
        import cv2

        native = cv2.VideoCapture(str(ROOT / vfile)).get(cv2.CAP_PROP_FPS) or 30.0
        ys, ss, fs = [], [], []
        for res in pipe.run_video(str(ROOT / vfile)):
            f = int(round(res["timestamp"] * native))
            gts = bf.get(f) or bf.get(f - 1) or bf.get(f + 1) or []
            scored = [st for st in res["students"] if st["score"] is not None]
            for st, k in match(scored, gts):
                ys.append(gts[k][1])
                ss.append(st["score"])
                fs.append(st["fusion_score"] if st["fusion_score"] is not None else st["score"])
        y, s, fz = np.array(ys), np.array(ss), np.array(fs)
        pred = (s >= 50).astype(int)
        m = {"matched": int(len(y)), "low_share": round(float((y == 0).mean()), 3),
             "auc": round(float(roc_auc_score(y, s)), 3) if len(set(y)) > 1 else None,
             "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
             "accuracy": round(float(accuracy_score(y, pred)), 3),
             "always_high": round(float(y.mean()), 3),
             "auc_model_only": round(float(roc_auc_score(y, fz)), 3) if len(set(y)) > 1 else None,
             "balanced_accuracy_model_only": round(float(balanced_accuracy_score(y, (fz >= 50).astype(int))), 3)}
        out[vlabel] = {"room": room, **m}
        print(vlabel, m)
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
