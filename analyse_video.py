"""Analyse a classroom video with the final model.

    python analyse_video.py "path/to/classroom.mp4"
    python analyse_video.py "path/to/classroom.mp4" --out results/

Writes <out>/<name>_annotated.mp4 (H.264 overlay: box, score and cue per student)
and <out>/<name>_scores.jsonl (per frame, per student: score, cue, components),
and prints a short summary. Uses configs/pipeline.yaml (class-relative motion +
hard-evidence rules + study protection).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("video")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seconds", type=float, default=None, help="only analyse the first N seconds")
    args = ap.parse_args()

    from video_processor import JobResult, VideoProcessor

    src = Path(args.video).resolve()
    if not src.exists():
        sys.exit(f"Video not found: {src}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = src.stem.replace(" ", "_")
    job = JobResult(job_id=name, input_name=src.name, input_path=str(src), queued_at=time.time())
    VideoProcessor().run(job, src, out / f"{name}_annotated.mp4", out / f"{name}_scores.jsonl", max_seconds=args.seconds)
    if job.status != "done":
        sys.exit(f"Failed: {job.error}")
    r = job.to_dict()["results"]
    print(f"\n{src.name}: {r['students_tracked']} students tracked | mean score {r['mean_score']:.1f} | "
          f"{r['attentive_pct']:.1f}% of observations HIGH | {len(r['drop_events'])} teacher alerts")
    print(f"Output video : {out / (name + '_annotated.mp4')}")
    print(f"Scores (JSONL): {out / (name + '_scores.jsonl')}")


if __name__ == "__main__":
    main()
