"""Batch Runner to generate annotated overlay videos and score files for all 7 benchmark test videos."""

from __future__ import annotations

import json
import time
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "configs" / "pipeline.yaml"
RUNS_DIR = WORKSPACE_ROOT / "runs"

TEST_VIDEOS = [
    ("Test 6.mp4", "test_6_overlay.mp4", "test_6_scores.jsonl"),
    ("Test 5.mp4", "test_5_overlay.mp4", "test_5_scores.jsonl"),
    ("clip_001.mp4", "clip_001_overlay.mp4", "clip_001_scores.jsonl"),
    ("clip_007.mp4", "clip_007_overlay.mp4", "clip_007_scores.jsonl"),
    ("test 4.mp4", "test_4_overlay.mp4", "test_4_scores.jsonl"),
    ("Video Project 3.mp4", "video_project_3_overlay.mp4", "video_project_3_scores.jsonl"),
    ("Test 7.mp4", "test_7_overlay.mp4", "test_7_scores.jsonl"),
]

def run_video_pipeline(pipe, in_video: Path, out_video: Path, out_jsonl: Path):
    pipe.reset()
    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input video: {in_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    jsonl_handle = out_jsonl.open("w", encoding="utf-8")

    infer_fps = float(pipe.cfg.get("fps", 10))
    infer_dt = 1.0 / max(infer_fps, 1.0)
    next_infer = 0.0

    frame_count = 0
    t_start = time.time()
    from attention_pipeline.stages.visualize import draw_attention, attention_band
    high = float(pipe.cfg.get("viz", {}).get("high", 50.0))
    low = float(pipe.cfg.get("viz", {}).get("moderate_low", 40.0))
    last_result = {"timestamp": 0.0, "students": [], "events": []}

    print(f"\n---> Processing {in_video.name} ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        if timestamp >= next_infer:
            last_result = pipe.process_frame(timestamp, frame)
            next_infer = timestamp + infer_dt

            jsonl_handle.write(
                json.dumps({
                    "frame": frame_count,
                    "timestamp": round(timestamp, 3),
                    "students": [
                        {
                            **s,
                            "band": s.get("band") or (attention_band(float(s["score"]), high, low) if s.get("score") is not None else "WARMUP"),
                        }
                        for s in last_result["students"]
                    ],
                    "drops": len(last_result["events"]),
                }) + "\n"
            )

        drawn = draw_attention(frame, last_result, high=high, low=low)
        writer.write(drawn)
        frame_count += 1

        if frame_count % (int(fps) * 5) == 0:
            elapsed = time.time() - t_start
            print(f"     [{in_video.name}] {frame_count}/{total_frames} frames ({frame_count/fps:.1f}s) | Speed: {frame_count/max(elapsed, 1e-4):.1f} FPS")

    cap.release()
    writer.release()
    jsonl_handle.close()
    elapsed = time.time() - t_start
    print(f"[DONE] {in_video.name} finished in {elapsed:.1f}s -> {out_video.name} ({out_video.stat().st_size / 1024 / 1024:.1f} MB)")

def main():
    import sys
    sys.path.insert(0, str(WORKSPACE_ROOT))
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
    from attention_pipeline.pipeline import AttentionPipeline

    print("=" * 80)
    print(" BATCH RUNNER: GENERATING ANNOTATED OVERLAYS FOR ALL 7 TEST VIDEOS")
    print("=" * 80)
    print(f"Loading pipeline from: {CONFIG_PATH.resolve()}...")
    pipe = AttentionPipeline.from_yaml(CONFIG_PATH, load_weights=True)

    t_batch_start = time.time()
    for in_name, out_name, jsonl_name in TEST_VIDEOS:
        in_path = RUNS_DIR / in_name
        out_path = RUNS_DIR / out_name
        jsonl_path = RUNS_DIR / jsonl_name
        if in_path.exists():
            run_video_pipeline(pipe, in_path, out_path, jsonl_path)
        else:
            print(f"[SKIP] Video not found: {in_path}")

    total_elapsed = time.time() - t_batch_start
    print("\n" + "=" * 80)
    print(f" ALL 7 BENCHMARK VIDEOS COMPLETED IN {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()

