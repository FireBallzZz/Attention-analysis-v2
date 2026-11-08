"""Production Runner for the Classroom Multimodal Attention Pipeline.

Executes the full 6-stage framework:
1. Video Ingestion & Temporal Normalization
2. YOLO11-Pose Detection + Tracking (COCO-17 Keypoints)
3. Multimodal Feature Extraction (L2CS Gaze + SCB Actions + OpenSeeFace Pose)
4. Domain-Adapted Classroom BiGRU Attention Scorer (0-100)
5. Automated Attention-Drop Detector (Sudden & Sustained drops)
6. Pedagogical Recommendation Engine (Teacher Intervention Prompts)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import cv2
import numpy as np

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.visualize import attention_band, draw_attention


def main():
    parser = argparse.ArgumentParser(description="Run Production Classroom Attention Pipeline")
    parser.add_argument(
        "--video",
        default=r"C:\Users\User\Downloads\Thesis\DEFENCE\classroom_01.mp4",
        help="Path to classroom video file",
    )
    parser.add_argument(
        "--config",
        default=str(WORKSPACE_ROOT / "configs" / "pipeline.yaml"),
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--out-video",
        default=str(WORKSPACE_ROOT / "runs" / "classroom_production_overlay.mp4"),
        help="Target annotated output video path",
    )
    parser.add_argument(
        "--out-jsonl",
        default=str(WORKSPACE_ROOT / "runs" / "classroom_production_scores.jsonl"),
        help="Target scores JSONL path",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duration in seconds to process (default: entire video)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    print("=" * 80)
    print(" CLASSROOM MULTIMODAL ATTENTION PIPELINE — PRODUCTION RUNNER")
    print("=" * 80)
    print(f"Input Video   : {video_path.name}")
    print(f"Active Config : {config_path.name}")
    print(f"Target Output : {Path(args.out_video).name}")

    print("\n[Stage 1-4] Initializing Pipeline & Loading Domain-Adapted Weights...")
    pipe = AttentionPipeline.from_yaml(config_path, load_weights=True)
    status = pipe.dataset_status
    print(
        f"  * L2CS-Net Gaze Backbone : {'[ONLINE]' if status['l2cs_gaze'] else '[OFFLINE]'}\n"
        f"  * SCB Behavior YOLO      : {'[ONLINE]' if status['scb_behavior'] else '[OFFLINE]'}\n"
        f"  * Multimodal BiGRU Scorer: {'[ONLINE]' if status['fusion'] else '[OFFLINE]'}"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(args.seconds * fps) if args.seconds else total_frames

    print(f"\n[Video Specifications]: {width}x{height} @ {fps:.1f} FPS | Total Frames: {total_frames:,}")
    if args.seconds:
        print(f"Processing window: First {args.seconds:.1f} seconds ({max_frames:,} frames)")

    out_video_path = Path(args.out_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    out_jsonl_path = Path(args.out_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_handle = out_jsonl_path.open("w", encoding="utf-8")

    infer_fps = float(pipe.cfg.get("fps", 10))
    infer_dt = 1.0 / max(infer_fps, 1.0)
    next_infer = 0.0

    frame_count = 0
    t_start = time.time()
    all_events = []
    total_student_obs = 0
    high_count = 0
    low_count = 0

    # Thresholds for visual color coding
    high = float(pipe.cfg.get("viz", {}).get("high", 50.0))
    low = float(pipe.cfg.get("viz", {}).get("moderate_low", 40.0))
    last_result: dict = {"timestamp": 0.0, "students": [], "events": []}

    print("\n[Real-Time Inference Running at 10 FPS Target]...")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        if timestamp >= next_infer:
            last_result = pipe.process_frame(timestamp, frame)
            next_infer = timestamp + infer_dt

            # Collect metrics
            thresh_score = pipe.binary_threshold * 100.0
            for s in last_result["students"]:
                total_student_obs += 1
                score = s.get("score")
                if score is not None:
                    if score >= thresh_score:
                        high_count += 1
                    else:
                        low_count += 1

            for ev in last_result["events"]:
                all_events.append(ev)
                event_obj = ev["event"]
                rec_obj = ev.get("recommendation") or {}
                print(
                    f"  [! ATTENTION DROP] t={event_obj.timestamp:5.1f}s | Student {event_obj.student_id:2d} | "
                    f"Score: {event_obj.score:4.1f}/100 | Kind: {event_obj.kind} | Trigger: {event_obj.dominant_modality}"
                )
                if "teaching_recommendation" in rec_obj:
                    print(f"      Pedagogical Move -> {rec_obj['teaching_recommendation']}")

            # Save rich jsonl row compatible with video_accuracy.py and score_ground_truth
            jsonl_handle.write(
                json.dumps(
                    {
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
                    }
                )
                + "\n"
            )

        # Draw overlays on the frame
        drawn = draw_attention(frame, last_result, high=high, low=low)
        writer.write(drawn)
        frame_count += 1

        if frame_count % (int(fps) * 5) == 0:
            elapsed = time.time() - t_start
            proc_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count:,}/{max_frames:,} frames ({frame_count/fps:.1f}s) | Processing Speed: {proc_fps:.1f} FPS")

    cap.release()
    writer.release()
    jsonl_handle.close()

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(" INFERENCE RUN COMPLETE — SUMMARY OF OUTCOMES")
    print("=" * 80)
    print(f"Total Frames Processed     : {frame_count:,} frames ({frame_count/fps:.1f} seconds)")
    print(f"Total Elapsed Execution    : {elapsed:.1f} seconds ({frame_count/elapsed:.1f} effective FPS)")
    print(f"Total Student Observations : {total_student_obs:,}")
    valid_scored = high_count + low_count
    if valid_scored > 0:
        print(f"  * Attentive (HIGH) State : {high_count:,} ({high_count/valid_scored*100:.1f}%)")
        print(f"  * Distracted (LOW) State : {low_count:,} ({low_count/valid_scored*100:.1f}%)")
    print(f"Total Drop Events Flagged  : {len(all_events):,}")
    print(f"Annotated Video Output     : {out_video_path.resolve()}")
    print(f"Detailed Scores Log        : {out_jsonl_path.resolve()}")

    try:
        from attention_pipeline.video_accuracy import write_video_accuracy, load_jsonl, check_band, _lesson_yaw

        accuracy_report_path = write_video_accuracy(video_path, out_video_path, out_jsonl_path)
        print(f"Video Accuracy Report      : {accuracy_report_path.resolve()}")

        # Compute empirical binary accuracy against independent multimodal ground-truth check band
        rows = load_jsonl(out_jsonl_path)
        y_true_binary = []
        y_pred_binary = []
        for row in rows:
            students = row.get("students") or []
            if not students:
                continue
            lesson = _lesson_yaw(students)
            for s in students:
                pred_band = s.get("band")
                if pred_band not in ("HIGH", "LOW"):
                    continue
                gt_band = check_band(s, lesson)
                # Binary mapping: Class 1 (Engaged) = HIGH or MODERATE; Class 0 (Disengaged) = LOW
                gt_bin = 1 if gt_band in ("HIGH", "MODERATE") else 0
                pred_bin = 1 if pred_band == "HIGH" else 0
                y_true_binary.append(gt_bin)
                y_pred_binary.append(pred_bin)

        if y_true_binary:
            y_true_arr = np.array(y_true_binary)
            y_pred_arr = np.array(y_pred_binary)
            tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
            tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
            fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
            fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
            bin_acc = (tp + tn) / len(y_true_binary) * 100.0
            prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            print("-" * 80)
            print(" VIDEO EMPIRICAL ACCURACY & EVALUATION")
            print("-" * 80)
            print(f"Total Student Comparisons  : {len(y_true_binary):,}")
            print(f"Binary Attention Accuracy  : {bin_acc:.2f}%")
            print(f"Precision (Attentive)      : {prec:.2f}%")
            print(f"Recall (Attentive)         : {rec:.2f}%")
            print(f"Macro F1-Score             : {f1:.2f}%")
            print(f"Confusion Matrix           : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print("-" * 80)
    except Exception as e:
        print(f"Accuracy calculation notice: {e}")

    print("=" * 80)


if __name__ == "__main__":
    main()

