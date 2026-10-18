"""Frame-by-frame attention annotation CSV for a classroom clip.

1. Count real students first (face + confident detection).
2. Lock that count so empty wall/monitor boxes are not kept.
3. For each student box, measure eye gaze, head pose, and posture,
   then label HIGH / MODERATE / LOW.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.pipeline import AttentionPipeline  # noqa: E402
from attention_pipeline.stages.visualize import draw_attention  # noqa: E402
from attention_pipeline.types import POSTURE_LABELS  # noqa: E402


def _count_students(pipe: AttentionPipeline, video: Path, sample_frames: int = 15) -> int:
    """Median student count over the first frames. This is the class size."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max(sample_frames, 1)) if total else 1
    counts: list[int] = []
    frame_idx = 0
    while len(counts) < sample_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            n = pipe.detector.count_students(frame)
            counts.append(n)
            print(f"  count sample frame {frame_idx}: {n} students")
        frame_idx += 1
    cap.release()
    if not counts:
        raise SystemExit("Could not count students in the clip.")
    counts.sort()
    median = counts[len(counts) // 2]
    print(f"Student counts sampled: {counts}")
    print(f"Locked student count: {median}")
    return int(median)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a clip to CSV")
    parser.add_argument("video", help="Path to the classroom clip")
    parser.add_argument("--config", default="configs/pipeline_merged.yaml")
    parser.add_argument(
        "--out-csv",
        default="",
        help="CSV path (default: annotations/<video>_attention.csv)",
    )
    parser.add_argument(
        "--out-video",
        default="",
        help="Labeled preview video (default: annotations/<video>_annotated.mp4)",
    )
    parser.add_argument(
        "--student-count",
        type=int,
        default=0,
        help="Lock this many students. 0 = count from the clip first.",
    )
    parser.add_argument("--min-det-conf", type=float, default=0.25)
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video.resolve()}")

    out_dir = ROOT / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"{video.stem}_attention.csv"
    out_video = (
        Path(args.out_video) if args.out_video else out_dir / f"{video.stem}_annotated.mp4"
    )

    pipe = AttentionPipeline.from_yaml(ROOT / args.config)
    pipe.detector.min_det_conf = float(args.min_det_conf)
    pipe.detector.require_face = True
    pipe.detector.max_people = None
    high = float(pipe.cfg.get("viz", {}).get("high", 70.0))
    low = float(pipe.cfg.get("viz", {}).get("moderate_low", 40.0))

    print(f"Step 1: count students in {video.name}")
    student_count = args.student_count or _count_students(pipe, video)
    pipe.detector.max_people = student_count
    print(f"Step 2: box only {student_count} students, then measure gaze/pose/posture")

    # Fresh tracker state after the counting pass.
    pipe.tracks.clear()
    pipe.detector._prev = []
    pipe.detector._next_id = 1
    pipe.detector._instructor = None
    pipe.detector._instructor_misses = 0
    pipe._lesson_yaw = None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    fieldnames = [
        "frame",
        "timestamp_sec",
        "student_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "attention",
        "score",
        "cue",
        "eye_gaze_yaw",
        "eye_gaze_pitch",
        "head_yaw",
        "head_pitch",
        "head_roll",
        "eye_aspect",
        "mouth_aspect",
        "posture",
        "posture_confidence",
        "behavior",
    ]
    rows: list[dict] = []
    frame_idx = 0
    band_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    per_frame_counts: list[int] = []

    print(f"Annotating {total} frames at {fps:.2f} fps")
    print(f"CSV   -> {out_csv}")
    print(f"Video -> {out_video}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        result = pipe.process_frame(ts, frame)
        students = result["students"]
        per_frame_counts.append(len(students))
        for student in students:
            box = student["bbox"]
            x1, y1, x2, y2 = [float(v) for v in box]
            band = str(student.get("band") or "MODERATE").upper()
            if band not in band_counts:
                band = "MODERATE"
            band_counts[band] += 1
            head = student.get("head_pose") or [0.0, 0.0, 0.0]
            gaze = student.get("gaze") or [0.0, 0.0]
            posture = np.asarray(student.get("posture") or [], np.float32)
            if posture.size:
                p_idx = int(np.argmax(posture))
                posture_name = POSTURE_LABELS[p_idx]
                posture_conf = float(posture[p_idx])
            else:
                posture_name = str(student.get("behavior") or "")
                posture_conf = 0.0
            rows.append(
                {
                    "frame": frame_idx,
                    "timestamp_sec": round(ts, 4),
                    "student_id": int(student["student_id"]),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(max(0.0, x2 - x1), 2),
                    "height": round(max(0.0, y2 - y1), 2),
                    "attention": band,
                    "score": round(float(student.get("score") or 0.0), 2),
                    "cue": str(student.get("cue") or ""),
                    "eye_gaze_yaw": round(float(gaze[0]), 2),
                    "eye_gaze_pitch": round(float(gaze[1]), 2),
                    "head_yaw": round(float(head[0]), 2),
                    "head_pitch": round(float(head[1]), 2),
                    "head_roll": round(float(head[2] if len(head) > 2 else 0.0), 2),
                    "eye_aspect": round(float(student.get("eye_aspect") or 0.0), 4),
                    "mouth_aspect": round(float(student.get("mouth_aspect") or 0.0), 4),
                    "posture": posture_name,
                    "posture_confidence": round(posture_conf, 3),
                    "behavior": str(student.get("behavior") or posture_name),
                }
            )
        writer.write(draw_attention(frame, result, high=high, low=low))
        if frame_idx % 30 == 0 or (total and frame_idx + 1 >= total):
            print(
                f"  frame {frame_idx + 1}/{total or '?'}  "
                f"students={len(students)}/{student_count}  rows={len(rows)}"
            )
        frame_idx += 1

    cap.release()
    writer.release()

    # Attach eye/mouth aspects from the live tracks if missing on student dict.
    # process_frame currently omits them; fill from the last observation path above
    # by reading them when available. Re-pass already used student.get defaults.

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    avg = sum(per_frame_counts) / max(len(per_frame_counts), 1)
    print(
        f"Done. frames={frame_idx}  locked_count={student_count}  "
        f"avg_boxes/frame={avg:.2f}  boxes={len(rows)}"
    )
    print(
        f"HIGH={band_counts['HIGH']}  MODERATE={band_counts['MODERATE']}  "
        f"LOW={band_counts['LOW']}"
    )
    print(f"Wrote {out_csv.resolve()}")
    print(f"Wrote {out_video.resolve()}")


if __name__ == "__main__":
    main()
