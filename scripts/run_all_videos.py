"""Run the final pipeline (configs/pipeline.yaml) on every video in runs/.

Writes, for each input video, into runs/final_outputs/:
  <name>_annotated.mp4   H.264 overlay video (plays in any browser / player)
  <name>_scores.jsonl    per-frame, per-student scores and components
and a summary table runs/final_outputs/SUMMARY.md.

    python scripts/run_all_videos.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from video_processor import JobResult, VideoProcessor  # noqa: E402

EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def main() -> None:
    src_dir = ROOT / "runs"
    out_dir = ROOT / "runs" / "final_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in src_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in EXT
                    and not any(t in p.stem.lower() for t in ("overlay", "annotated")))
    print(f"{len(videos)} videos:", ", ".join(v.name for v in videos))
    proc = VideoProcessor()
    rows = []
    for v in videos:
        name = v.stem.replace(" ", "_")
        job = JobResult(job_id=name, input_name=v.name, input_path=str(v), queued_at=time.time())
        t0 = time.time()
        proc.run(job, v, out_dir / f"{name}_annotated.mp4", out_dir / f"{name}_scores.jsonl")
        d = job.to_dict()
        r = d["results"]
        rows.append((v.name, job.status, d["video"]["duration_s"], r["students_tracked"], r["mean_score"],
                     r["attentive_pct"], len(r["drop_events"]), time.time() - t0, job.error))
        print(f"  {v.name}: {job.status} | {r['students_tracked']} students | mean score {r['mean_score']} | "
              f"{len(r['drop_events'])} alerts | {time.time() - t0:.0f}s {job.error or ''}")
    lines = ["# Final pipeline outputs", "",
             "Model: class-relative motion + hard-evidence floor (`configs/pipeline.yaml`). "
             "Score >= 50 = HIGH (attentive).", "",
             "| Input video | Status | Length (s) | Students tracked | Mean score | % HIGH | Alerts | Output |",
             "|---|---|---|---|---|---|---|---|"]
    for name, st, dur, n, mean, hi, al, sec, err in rows:
        out = f"`{Path(name).stem.replace(' ', '_')}_annotated.mp4`" if st == "done" else (err or "")
        lines.append(f"| {name} | {st} | {dur:.1f} | {n} | {mean:.1f} | {hi:.1f}% | {al} | {out} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
