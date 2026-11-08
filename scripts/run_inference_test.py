"""Inference-Only Test Runner for Classroom Student Attention Model.

Strict Compliance:
- Zero modifications to checkpoints, model weights, or decision thresholds.
- Reuses established AttentionPipeline (YOLO11n-pose + ByteTrack + L2CS + SCB + Phase 6B 32-D features + ManualBinaryAttentionClassifier).
- Respects 16-frame sequence requirement & 15-frame warm-up (no artificial frame padding).
- Exports annotated MP4 with bounding boxes, stable track IDs, HIGH/ATTENTIVE and LOW/INATTENTIVE labels, and P(HIGH) probabilities.
- Exports complete observation CSV.
- Zero accuracy claims (inference/output inspection only).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.utils import load_yaml, project_path

# BGR color palette
COLOR_HIGH = (40, 190, 40)       # Vibrant Green: HIGH / ATTENTIVE
COLOR_LOW = (30, 30, 220)        # Vibrant Red: LOW / INATTENTIVE
COLOR_WARMUP = (140, 140, 140)   # Neutral Gray: WARMUP (< 16 frames)
COLOR_TEXT = (255, 255, 255)     # Crisp White text
COLOR_BG_PANEL = (20, 20, 20)    # Dark charcoal for HUD


def draw_label_with_bg(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = COLOR_TEXT,
    scale: float = 0.45,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw text with a solid contrasting background rectangle for guaranteed readability."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    h_img, w_img = img.shape[:2]
    
    x1 = max(0, x)
    y1 = max(0, y - th - padding * 2)
    x2 = min(w_img - 1, x1 + tw + padding * 2)
    y2 = min(h_img - 1, y1 + th + padding * 2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        img,
        text,
        (x1 + padding, y2 - padding - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame: np.ndarray,
    result: dict,
    frame_idx: int,
    total_frames: int,
    fps: float,
    threshold: float,
) -> np.ndarray:
    """Annotate frame with bounding boxes, labels, probabilities, and HUD."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    scale = max(0.5, min(w / 1280.0, 1.0))
    thick = 2 if w >= 800 else 1

    students = result.get("students", [])
    counts = {"HIGH": 0, "LOW": 0, "WARMUP": 0}

    for s in students:
        sid = s["student_id"]
        bbox = [int(v) for v in s["bbox"]]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w - 1, bbox[2]), min(h - 1, bbox[3])
        
        band = s.get("band", "WARMUP")
        fusion_score = s.get("fusion_score")
        
        if fusion_score is not None:
            p_high = fusion_score / 100.0
            p_low = 1.0 - p_high
            if p_high >= threshold:
                pred_label = "HIGH / ATTENTIVE"
                box_color = COLOR_HIGH
                counts["HIGH"] += 1
            else:
                pred_label = "LOW / INATTENTIVE"
                box_color = COLOR_LOW
                counts["LOW"] += 1
            text = f"ID {sid} | {pred_label} | P(H): {p_high:.2f}"
        else:
            pred_label = "WARMUP"
            box_color = COLOR_WARMUP
            counts["WARMUP"] += 1
            text = f"ID {sid} | WARMUP (seq < 16)"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thick + 1)

        # Label position: above box if possible, else below
        label_y = y1 - 4 if y1 > 28 else y2 + 22
        draw_label_with_bg(annotated, text, x1, label_y, box_color, scale=0.48 * scale, thickness=thick)

    # Top status HUD Panel
    ts = result.get("timestamp", frame_idx / max(fps, 1.0))
    hud_h = int(36 * scale)
    hud_w = int(580 * scale)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h * 2 + 10), COLOR_BG_PANEL, -1)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h * 2 + 10), (80, 80, 80), 1)

    line1 = f"Frame: {frame_idx}/{total_frames} ({ts:.2f}s) | Active Students: {len(students)}"
    line2 = f"HIGH: {counts['HIGH']}  |  LOW: {counts['LOW']}  |  WARMUP: {counts['WARMUP']}  |  Threshold: {threshold:.2f}"
    
    cv2.putText(annotated, line1, (18, 10 + int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(annotated, line2, (18, 10 + int(46 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def run_inference(
    video_path: Path,
    config_path: Path,
    out_dir: Path,
    max_frames: int | None = None,
    is_smoke: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" CLASSROOM STUDENT ATTENTION: INFERENCE-ONLY TEST")
    print("=" * 80)
    print(f"Video Path       : {video_path.resolve()}")
    print(f"Config Path      : {config_path.resolve()}")
    print(f"Output Directory : {out_dir.resolve()}")
    print(f"Mode             : {'SMOKE TEST' if is_smoke else 'FULL RUN'}")
    
    # 1. Verify Video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path.resolve()}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video with OpenCV: {video_path.resolve()}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(native_fps, 1.0)

    print(f"Video Specs      : {width}x{height} @ {native_fps:.2f} fps | {total_frames} frames ({duration_sec:.2f}s)")

    # 2. Load Pipeline & Checkpoint
    cfg = load_yaml(str(config_path))
    pipe = AttentionPipeline(cfg, load_weights=True)
    
    ckpt_path = project_path(cfg.get("models", {}).get("manual_binary_weights"))
    threshold = float(pipe.binary_threshold)
    
    print(f"Active Checkpoint: {ckpt_path.resolve() if ckpt_path else 'None'}")
    print(f"Decision Thresh  : {threshold:.2f}")
    print(f"Sequence Length  : {pipe.scorer.window} frames")
    print(f"Hardware Device  : {pipe.device}")
    print(f"Dataset Modalities: {pipe.dataset_status}")

    # 3. Setup Outputs
    target_frames = min(total_frames, max_frames) if max_frames is not None else total_frames
    video_out_path = out_dir / ("annotated_smoke.mp4" if is_smoke else "annotated_classroom_inference.mp4")
    csv_out_path = out_dir / ("observations_smoke.csv" if is_smoke else "student_attention_observations.csv")
    summary_json_path = out_dir / ("summary_smoke.json" if is_smoke else "inference_summary.json")
    report_txt_path = out_dir / ("report_smoke.txt" if is_smoke else "inference_report.txt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, native_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter at {video_out_path}")

    csv_file = csv_out_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_number",
        "timestamp_seconds",
        "track_id",
        "prediction_label",
        "prob_high_attentive",
        "prob_low_inattentive",
        "is_warmup",
        "decision_threshold",
        "bbox_xtl",
        "bbox_ytl",
        "bbox_xbr",
        "bbox_ybr",
    ])

    # 4. Run Frame Processing Loop
    tracked_students: set[int] = set()
    total_observations = 0
    scored_observations = 0
    warmup_observations = 0
    high_count = 0
    low_count = 0
    errors_warnings: list[str] = []

    t_start = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and frame_idx >= max_frames):
                break

            ts = frame_idx / native_fps
            t_frame_start = time.time()
            
            try:
                res = pipe.process_frame(ts, frame)
            except Exception as e:
                err_msg = f"Frame {frame_idx} processing error: {e}"
                errors_warnings.append(err_msg)
                print(f"[WARNING] {err_msg}")
                res = {"timestamp": ts, "students": [], "events": []}

            # Process observations
            students = res.get("students", [])
            for s in students:
                sid = int(s["student_id"])
                tracked_students.add(sid)
                total_observations += 1
                
                fusion_score = s.get("fusion_score")
                bbox = s.get("bbox", [0, 0, 0, 0])
                
                if fusion_score is not None:
                    scored_observations += 1
                    p_high = float(fusion_score / 100.0)
                    p_low = 1.0 - p_high
                    is_warmup = False
                    if p_high >= threshold:
                        pred_label = "HIGH / ATTENTIVE"
                        high_count += 1
                    else:
                        pred_label = "LOW / INATTENTIVE"
                        low_count += 1
                    p_high_str = f"{p_high:.4f}"
                    p_low_str = f"{p_low:.4f}"
                else:
                    warmup_observations += 1
                    is_warmup = True
                    pred_label = "WARMUP"
                    p_high_str = ""
                    p_low_str = ""

                csv_writer.writerow([
                    frame_idx,
                    f"{ts:.4f}",
                    sid,
                    pred_label,
                    p_high_str,
                    p_low_str,
                    is_warmup,
                    f"{threshold:.2f}",
                    f"{bbox[0]:.1f}",
                    f"{bbox[1]:.1f}",
                    f"{bbox[2]:.1f}",
                    f"{bbox[3]:.1f}",
                ])

            # Annotate and write frame
            annotated = annotate_frame(frame, res, frame_idx, target_frames, native_fps, threshold)
            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 25 == 0 or frame_idx == target_frames:
                elapsed_so_far = time.time() - t_start
                cur_fps = frame_idx / max(elapsed_so_far, 1e-4)
                print(f"Processed frame {frame_idx}/{target_frames} ({frame_idx/target_frames*100:.1f}%) | Speed: {cur_fps:.1f} FPS | Active Tracks: {len(tracked_students)}")

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    total_time = time.time() - t_start
    processing_fps = frame_idx / max(total_time, 1e-4)

    # 5. Output Verification
    if not video_out_path.exists() or video_out_path.stat().st_size == 0:
        raise RuntimeError(f"Output video {video_out_path} is missing or empty!")
    
    # Verify readability
    verify_cap = cv2.VideoCapture(str(video_out_path))
    v_opened = verify_cap.isOpened()
    v_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    verify_cap.release()

    if not v_opened or v_frames == 0:
        raise RuntimeError(f"Output video {video_out_path} could not be reopened or has 0 frames!")

    summary = {
        "execution_timestamp": datetime.now().isoformat(),
        "mode": "smoke_test" if is_smoke else "full_inference",
        "input_video": str(video_path.resolve()),
        "checkpoint_path": str(ckpt_path.resolve()) if ckpt_path else "None",
        "decision_threshold": threshold,
        "sequence_window_length": pipe.scorer.window,
        "input_specs": {
            "width": width,
            "height": height,
            "native_fps": native_fps,
            "total_frames_in_source": total_frames,
            "duration_seconds": duration_sec,
        },
        "execution_metrics": {
            "frames_processed": frame_idx,
            "total_runtime_seconds": total_time,
            "processing_fps": processing_fps,
            "unique_students_detected": len(tracked_students),
            "student_track_ids": sorted(list(tracked_students)),
            "total_observations": total_observations,
            "scored_observations": scored_observations,
            "warmup_observations": warmup_observations,
            "high_attentive_count": high_count,
            "low_inattentive_count": low_count,
        },
        "output_files": {
            "video_path": str(video_out_path.resolve()),
            "video_size_bytes": video_out_path.stat().st_size,
            "video_verified_frames": v_frames,
            "csv_path": str(csv_out_path.resolve()),
            "csv_size_bytes": csv_out_path.stat().st_size,
            "report_path": str(report_txt_path.resolve()),
            "summary_path": str(summary_json_path.resolve()),
        },
        "errors_and_warnings": errors_warnings,
        "evaluation_safeguards": {
            "ground_truth_present": False,
            "accuracy_claimed": False,
            "statement": "Inference-only output inspection. No ground truth is available; accuracy is not computed or claimed.",
        },
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Text Report
    report_text = f"""================================================================================
CLASSROOM ATTENTION INFERENCE REPORT ({'SMOKE TEST' if is_smoke else 'FULL RUN'})
================================================================================
Generated: {summary['execution_timestamp']}

1. RUN SPECIFICATION & ASSETS
-----------------------------
Input Video          : {video_path.resolve()}
Active Checkpoint    : {ckpt_path.resolve() if ckpt_path else 'None'}
Decision Threshold   : {threshold:.2f}
Sequence Requirement: {pipe.scorer.window} frames (First {pipe.scorer.window - 1} frames strictly warmup)
Video Dimensions     : {width} x {height} @ {native_fps:.2f} FPS
Source Duration      : {duration_sec:.2f} seconds ({total_frames} frames)

2. EXECUTION PERFORMANCE
------------------------
Frames Processed     : {frame_idx} / {target_frames}
Total Wall Time      : {total_time:.2f} seconds
Average Inference FPS: {processing_fps:.2f} FPS
Errors / Warnings    : {len(errors_warnings)}

3. DETECTION & PREDICTION SUMMARY
---------------------------------
Unique Students Tracked : {len(tracked_students)} (IDs: {sorted(list(tracked_students))})
Total Observations      : {total_observations}
Scored Observations     : {scored_observations} ({scored_observations/max(total_observations,1)*100:.1f}%)
Warm-up Observations    : {warmup_observations} ({warmup_observations/max(total_observations,1)*100:.1f}%)
HIGH / ATTENTIVE Labels : {high_count} ({high_count/max(scored_observations,1)*100:.1f}% of scored)
LOW / INATTENTIVE Labels: {low_count} ({low_count/max(scored_observations,1)*100:.1f}% of scored)

4. VERIFIED ARTIFACTS
---------------------
Annotated Video : {video_out_path.resolve()} ({video_out_path.stat().st_size:,} bytes, {v_frames} frames)
Observation CSV : {csv_out_path.resolve()} ({csv_out_path.stat().st_size:,} bytes)
Summary JSON    : {summary_json_path.resolve()}

5. ACADEMIC & METHODOLOGICAL SAFEGUARDS
---------------------------------------
- Unsupervised / Inference-Only: No ground-truth annotations are available for this video.
- Zero Accuracy Claim: Accuracy and error rates are strictly omitted.
- No Artificial Padding: Incomplete temporal sequences were kept as WARMUP.
================================================================================
"""
    with report_txt_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run inference on classroom video")
    parser.add_argument("--video", default="runs/new_classroom_test.mp4", help="Input video path")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML path")
    parser.add_argument("--out-dir", default=None, help="Directory to save artifacts")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (first 30 frames)")
    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    
    if args.out_dir is None:
        if args.smoke:
            out_dir = WORKSPACE_ROOT / "runs" / "inference_tests" / "smoke_test"
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = WORKSPACE_ROOT / "runs" / "inference_tests" / f"run_{stamp}"
    else:
        out_dir = Path(args.out_dir)

    max_frames = 30 if args.smoke and args.max_frames is None else args.max_frames

    run_inference(
        video_path=video_path,
        config_path=config_path,
        out_dir=out_dir,
        max_frames=max_frames,
        is_smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
