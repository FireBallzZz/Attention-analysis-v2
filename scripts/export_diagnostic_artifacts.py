"""Diagnostic artifact exporter for False-LOW Investigation.

Generates:
1. Per-track distribution summary (CSV).
2. Cropped and full-frame visual images with bounding boxes for manual review.
3. Diagnostic 32-D feature tables (CSV and JSON).
4. Feature ablation & sensitivity analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.pipeline import AttentionPipeline

DIAG_DIR = WORKSPACE_ROOT / "runs" / "inference_tests" / "run_new_classroom_test_20261022_210515" / "false_low_diagnostic"
FRAMES_DIR = DIAG_DIR / "frames"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "ch0_head_yaw", "ch1_head_pitch", "ch2_head_roll", "ch3_gaze_yaw", "ch4_gaze_pitch",
    "ch5_emo_angry", "ch6_emo_disgust", "ch7_emo_fear", "ch8_emo_happy", "ch9_emo_sad", "ch10_emo_surprise", "ch11_emo_neutral",
    "ch12_scb_raise", "ch13_scb_read", "ch14_scb_write", "ch15_scb_phone", "ch16_scb_bow", "ch17_scb_lean",
    "ch18_motion", "ch19_ear", "ch20_mar", "ch21_facing_front", "ch22_torso_lean", "ch23_upper_kpt_conf",
    "ch24_pad", "ch25_pad", "ch26_pad", "ch27_pad", "ch28_pad", "ch29_pad", "ch30_pad", "ch31_pad",
]


def run_diagnostic():
    print("=" * 80)
    print(" EXPORTING DIAGNOSTIC ARTIFACTS FOR FALSE-LOW INVESTIGATION")
    print("=" * 80)

    pipe = AttentionPipeline.from_yaml("configs/pipeline.yaml")
    cap = cv2.VideoCapture("runs/new_classroom_test.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    target_tids = [1, 2, 3, 4, 5, 7, 8, 13, 14]
    target_frames = [30, 50, 75, 100]

    extracted_records = []
    saved_frames = set()
    model = pipe.fusion
    model.eval()

    ablation_results = []

    for f_idx in range(110):
        ok, frame = cap.read()
        if not ok:
            break
        ts = f_idx / fps
        res = pipe.process_frame(ts, frame)

        if f_idx in target_frames:
            # Save full annotated frame once
            if f_idx not in saved_frames:
                saved_frames.add(f_idx)
                full_annotated = frame.copy()
                for s_all in res.get("students", []):
                    b_all = [int(v) for v in s_all["bbox"]]
                    band_str = str(s_all.get("band", "WARMUP"))
                    if band_str == "HIGH":
                        b_col = (40, 190, 40)
                    elif band_str == "LOW":
                        b_col = (30, 30, 220)
                    else:
                        b_col = (140, 140, 140)
                    cv2.rectangle(full_annotated, (b_all[0], b_all[1]), (b_all[2], b_all[3]), b_col, 2)
                    score_val = s_all.get("score")
                    score_str = f"{score_val:.0f}" if score_val is not None else "W"
                    tag = f"ID {s_all['student_id']} | {band_str} | {score_str}"
                    cv2.putText(
                        full_annotated,
                        tag,
                        (b_all[0], max(16, b_all[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                full_name = f"full_frame_{f_idx:03d}.jpg"
                cv2.imwrite(str(FRAMES_DIR / full_name), full_annotated)

            for s in res.get("students", []):
                tid = s["student_id"]
                if tid in target_tids:
                    t = pipe.tracks.get(tid)
                    if t and len(t.feature_window) >= 16:
                        w = np.stack(t.feature_window[-16:], axis=0)  # (16, 32)
                        latest_feat = w[-1].copy()
                        fusion_score = s.get("fusion_score")
                        p_high = fusion_score / 100.0 if fusion_score is not None else None
                        band = str(s.get("band", "WARMUP"))
                        bbox = [int(v) for v in s["bbox"]]

                        extracted_records.append({
                            "frame_number": f_idx,
                            "timestamp_seconds": round(ts, 4),
                            "track_id": tid,
                            "prediction": band,
                            "p_high": round(p_high, 4) if p_high is not None else None,
                            "bbox": bbox,
                            "features": latest_feat.tolist(),
                        })

                        # Save cropped student image
                        h_img, w_img = frame.shape[:2]
                        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                        x2, y2 = min(w_img - 1, bbox[2]), min(h_img - 1, bbox[3])
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            safe_band = band.replace("/", "_").replace(" ", "_")
                            crop_name = f"frame_{f_idx:03d}_track_{tid}_{safe_band}.jpg"
                            # Add label to crop
                            cv2.putText(
                                crop,
                                f"ID {tid} {band} P(H)={p_high:.2f}" if p_high is not None else f"ID {tid} WARMUP",
                                (4, 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.imwrite(str(FRAMES_DIR / crop_name), crop)

                        # Test Channel Ablation on this exact window
                        with torch.no_grad():
                            t_orig = torch.from_numpy(w).unsqueeze(0).to(pipe.device)
                            p_live = float(model.predict_proba(t_orig)[0, 1].item())

                            # Ablation 1: Channel 23 = 0.0 (training condition)
                            w_c23_zero = w.copy()
                            w_c23_zero[:, 23] = 0.0
                            t_c23 = torch.from_numpy(w_c23_zero).unsqueeze(0).to(pipe.device)
                            p_c23_zero = float(model.predict_proba(t_c23)[0, 1].item())

                            # Ablation 2: Channel 2 = 0.0 (neutral roll)
                            w_c2_zero = w.copy()
                            w_c2_zero[:, 2] = 0.0
                            t_c2 = torch.from_numpy(w_c2_zero).unsqueeze(0).to(pipe.device)
                            p_c2_zero = float(model.predict_proba(t_c2)[0, 1].item())

                            # Ablation 3: Both Ch 23=0 and Ch 2=0
                            w_both = w.copy()
                            w_both[:, 23] = 0.0
                            w_both[:, 2] = 0.0
                            t_both = torch.from_numpy(w_both).unsqueeze(0).to(pipe.device)
                            p_both_zero = float(model.predict_proba(t_both)[0, 1].item())

                        ablation_results.append({
                            "frame": f_idx,
                            "track_id": tid,
                            "live_p_high": round(p_live, 4),
                            "live_prediction": "HIGH" if p_live >= 0.50 else "LOW",
                            "p_high_ch23_zero": round(p_c23_zero, 4),
                            "pred_ch23_zero": "HIGH" if p_c23_zero >= 0.50 else "LOW",
                            "p_high_ch2_zero": round(p_c2_zero, 4),
                            "pred_ch2_zero": "HIGH" if p_c2_zero >= 0.50 else "LOW",
                            "p_high_both_zero": round(p_both_zero, 4),
                            "pred_both_zero": "HIGH" if p_both_zero >= 0.50 else "LOW",
                            "ch23_val": round(float(latest_feat[23]), 4),
                            "ch2_val": round(float(latest_feat[2]), 4),
                            "ch21_facing_front": round(float(latest_feat[21]), 4),
                            "ch22_torso_lean": round(float(latest_feat[22]), 4),
                        })

    cap.release()

    # Save feature records to JSON
    with (DIAG_DIR / "diagnostic_feature_records.json").open("w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2)

    # Save 32-D feature table to CSV
    csv_rows = []
    for r in extracted_records:
        row = {
            "frame_number": r["frame_number"],
            "timestamp_seconds": r["timestamp_seconds"],
            "track_id": r["track_id"],
            "prediction": r["prediction"],
            "p_high": r["p_high"],
        }
        for idx, col_name in enumerate(FEATURE_NAMES):
            row[col_name] = round(r["features"][idx], 4)
        csv_rows.append(row)

    feat_df = pd.DataFrame(csv_rows)
    feat_df.to_csv(DIAG_DIR / "diagnostic_feature_table.csv", index=False)

    # Save ablation table to CSV
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(DIAG_DIR / "channel_ablation_analysis.csv", index=False)

    print(f"Exported {len(extracted_records)} feature observations.")
    print(f"Exported {len(ablation_results)} ablation records.")
    print(f"Files saved in {DIAG_DIR}")


if __name__ == "__main__":
    run_diagnostic()
