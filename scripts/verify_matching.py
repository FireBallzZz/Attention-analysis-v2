"""Are the CVAT-annotated students and the model's students really the same people?

Uses the final pipeline outputs (runs/final_outputs/*_scores.jsonl) and the
human CVAT files. Per analysed frame, model boxes are paired with human boxes
(one-to-one, model box >= 60% inside the human box). Checks:

  consistency  - each CVAT student should be paired with ONE model ID across the
                 clip; pairs that disagree with that dominant ID are suspicious
  ambiguity    - a model box that also lies >= 60% inside a SECOND human box
                 (overlapping students) could be paired with the wrong person
  verified     - pairs that are consistent AND unambiguous

Accuracy is reported on all pairs and on verified pairs only.
Also saves, per video, an image with both boxes drawn for a visual check:
runs/final_outputs/matching_check/<video>.jpg

    python scripts/verify_matching.py
Writes reports/final_model/matching_verification.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from crop_embeddings import boxes_by_frame  # noqa: E402
from evaluate_pipeline_vs_human import contained, match  # noqa: E402

VIDEOS = [("clip 007", "clip_007", "runs/clip_007.mp4", "Anno/annotations.xml"),
          ("clip 001", "clip_001", "runs/clip_001.mp4", "runs/Annoted file for clip 1/annotations.xml"),
          ("video 11", "11", "runs/11.mp4", "Anno/cvat_new/11/annotations.xml"),
          ("non attentive 1", "non_attentive_1", "runs/non attentive 1.mov", "Anno/cvat_new/NON attentive 1/annotations.xml"),
          ("Test 8", "Test_8", "runs/Test 8.mp4", "Anno/cvat_new/Test 8/annotations.xml"),
          ("Test 6 (labels all High - not counted)", "Test_6", "runs/Test 6.mp4", "Anno/cvat_new/Test 6/annotations.xml")]


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(y) == 0 or len(set(y)) < 2:
        return {"n": int(len(y))}
    p = (s >= 50).astype(int)
    return {"n": int(len(y)), "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 4),
            "accuracy": round(float(accuracy_score(y, p)), 4), "auc": round(float(roc_auc_score(y, s)), 4),
            "always_high": round(float(np.mean(y)), 4)}


def main():
    out_dir = ROOT / "runs" / "final_outputs" / "matching_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for title, stem, vfile, xml in VIDEOS:
        rows = [json.loads(line) for line in (ROOT / "runs/final_outputs" / f"{stem}_scores.jsonl").open(encoding="utf-8")]
        bf = boxes_by_frame(ROOT / xml)
        fps = cv2.VideoCapture(str(ROOT / vfile)).get(cv2.CAP_PROP_FPS) or 30.0
        pairs = []  # (frame, human_tid, model_id, label, score, ambiguous)
        for r in rows:
            f = int(round(r["timestamp"] * fps))  # exact video frame the model analysed
            gts = bf.get(f, [])
            scored = [s for s in r["students"] if s.get("score") is not None]
            for st, k in match(scored, gts):
                others = [contained(st["bbox"], g[2]) for j, g in enumerate(gts) if j != k]
                pairs.append((f, gts[k][0], int(st["student_id"]), gts[k][1], float(st["score"]),
                              bool(others and max(others) >= 0.6), st["bbox"], gts[k][2]))
        dom = {}
        by_h = defaultdict(Counter)
        for p in pairs:
            by_h[p[1]][p[2]] += 1
        for h, c in by_h.items():
            dom[h] = c.most_common(1)[0][0]
        consistent = [p[2] == dom[p[1]] for p in pairs]
        ambiguous = [p[5] for p in pairs]
        verified = [c and not a for c, a in zip(consistent, ambiguous)]
        purity = float(np.mean(consistent)) if pairs else 0.0
        y = [p[3] for p in pairs]
        s = [p[4] for p in pairs]
        yv = [p[3] for p, v in zip(pairs, verified) if v]
        sv = [p[4] for p, v in zip(pairs, verified) if v]
        report[title] = {
            "cvat_students": len({h for fr in bf.values() for h, _, _ in fr}),
            "cvat_students_paired": len(by_h),
            "pairs": len(pairs),
            "same_model_id_all_clip_pct": round(purity * 100, 1),
            "ambiguous_pct": round(float(np.mean(ambiguous)) * 100, 1) if pairs else 0.0,
            "verified_pct": round(float(np.mean(verified)) * 100, 1) if pairs else 0.0,
            "all_pairs": metrics(y, s),
            "verified_pairs": metrics(yv, sv),
        }
        # visual check: middle analysed frame with pairs
        frames = sorted({p[0] for p in pairs})
        if frames:
            fsel = frames[len(frames) // 2]
            cap = cv2.VideoCapture(str(ROOT / vfile))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fsel)
            ok, img = cap.read()
            if ok:
                sc = max(img.shape[1] / 1600, 1.0)
                th = max(2, int(2 * sc))
                for p, v in zip(pairs, verified):
                    if p[0] != fsel:
                        continue
                    hb, mb = [int(x) for x in p[7]], [int(x) for x in p[6]]
                    lab_col = (0, 200, 0) if p[3] == 1 else (0, 0, 230)
                    cv2.rectangle(img, hb[:2], hb[2:], lab_col, th)                       # human box, colour = human label
                    cv2.rectangle(img, mb[:2], mb[2:], (255, 200, 0) if v else (255, 0, 255), th)  # model box
                    cv2.line(img, ((hb[0] + hb[2]) // 2, (hb[1] + hb[3]) // 2),
                             ((mb[0] + mb[2]) // 2, (mb[1] + mb[3]) // 2), (255, 255, 255), max(1, th // 2))
                    cv2.putText(img, f"H{p[1]}={'HIGH' if p[3] else 'LOW'} / M{p[2]}:{p[4]:.0f}", (hb[0], max(15, hb[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
                cv2.putText(img, f"{title} frame {fsel}: green/red = CVAT box (human label), cyan = model box paired (verified), magenta = paired but suspicious",
                            (10, int(30 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
                cv2.imwrite(str(out_dir / f"{stem}.jpg"), img)
        r = report[title]
        print(f"{title:16s} CVAT students {r['cvat_students']:2d} paired {r['cvat_students_paired']:2d} | "
              f"same model ID whole clip {r['same_model_id_all_clip_pct']:5.1f}% | ambiguous {r['ambiguous_pct']:4.1f}% | "
              f"verified {r['verified_pct']:5.1f}% | all: {r['all_pairs']} | verified: {r['verified_pairs']}")
    p = ROOT / "reports/final_model/matching_verification.json"
    p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", p, "and images in", out_dir)


if __name__ == "__main__":
    main()
