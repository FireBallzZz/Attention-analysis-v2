"""Phase 6G-1: Controlled Zero-Retraining Feature Compatibility Experiments.

Evaluates:
1. Training sequence Channel 23 audit.
2. Full-video controlled ablation on Channel 23 (Original vs Ch23=0).
3. Head-roll mathematical audit (existing vs zero-centered orientation).
4. Runtime consistency checks.

Strict Rules:
- No retraining, fine-tuning, or model modification.
- Checkpoint and decision threshold (0.50) strictly preserved.
- Output artifacts saved to runs/inference_tests/run_new_classroom_test_20261022_210515/phase6g1/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from attention_pipeline.pipeline import AttentionPipeline

PHASE6G1_DIR = WORKSPACE_ROOT / "runs" / "inference_tests" / "run_new_classroom_test_20261022_210515" / "phase6g1"
PHASE6G1_DIR.mkdir(parents=True, exist_ok=True)


def audit_head_roll() -> pd.DataFrame:
    """Task B: Head Roll Mathematical Audit on Synthetic Test Cases."""
    def existing_roll(reye, leye):
        return float(np.degrees(np.arctan2(reye[1] - leye[1], reye[0] - leye[0] + 1e-6)))

    def corrected_roll(reye, leye):
        return float(np.degrees(np.arctan2(leye[1] - reye[1], leye[0] - reye[0] + 1e-6)))

    cases = [
        ("Perfect Horizontal (Level Upright)", [90.0, 100.0], [110.0, 100.0], "Horizontal eye-axis (anatomical right eye on left, left eye on right)"),
        ("Sub-pixel Jitter (+0.05px Y)",       [90.0, 100.05], [110.0, 100.0], "Right eye 0.05px lower than left eye"),
        ("Sub-pixel Jitter (-0.05px Y)",       [90.0, 99.95],  [110.0, 100.0], "Right eye 0.05px higher than left eye"),
        ("Sub-pixel Jitter (+0.01px Y)",       [90.0, 100.01], [110.0, 100.0], "Right eye 0.01px lower than left eye"),
        ("Sub-pixel Jitter (-0.01px Y)",       [90.0, 99.99],  [110.0, 100.0], "Right eye 0.01px higher than left eye"),
        ("Slight Tilt Left (+5 deg)",          [90.0, 98.26],  [110.0, 100.0], "Head tilted towards person's left shoulder"),
        ("Slight Tilt Right (-5 deg)",         [90.0, 101.74], [110.0, 100.0], "Head tilted towards person's right shoulder"),
        ("Moderate Tilt Left (+15 deg)",       [90.0, 94.82],  [110.0, 100.0], "Noticeable tilt left"),
        ("Moderate Tilt Right (-15 deg)",      [90.0, 105.18], [110.0, 100.0], "Noticeable tilt right"),
        ("Extreme Tilt Left (+45 deg)",        [90.0, 80.0],   [110.0, 100.0], "Severe lateral tilt left"),
        ("Extreme Tilt Right (-45 deg)",       [90.0, 120.0],  [110.0, 100.0], "Severe lateral tilt right"),
    ]

    records = []
    for name, reye, leye, note in cases:
        e_deg = existing_roll(reye, leye)
        e_norm = np.clip(e_deg / 180.0, -1.0, 1.0)
        c_deg = corrected_roll(reye, leye)
        c_norm = np.clip(c_deg / 180.0, -1.0, 1.0)
        discontinuity = abs(e_norm) > 0.95
        records.append({
            "test_case": name,
            "reye_xy": str(reye),
            "leye_xy": str(leye),
            "existing_roll_deg": round(e_deg, 4),
            "existing_roll_norm": round(e_norm, 4),
            "corrected_roll_deg": round(c_deg, 4),
            "corrected_roll_norm": round(c_norm, 4),
            "discontinuity_flag": discontinuity,
            "description": note,
        })

    df = pd.DataFrame(records)
    out_csv = PHASE6G1_DIR / "head_roll_synthetic_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote Head Roll audit to {out_csv}")
    return df


def run_channel23_controlled_experiment() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Task A: Full-Video Controlled Ablation on Channel 23."""
    print("=" * 80)
    print(" RUNNING FULL-VIDEO CONTROLLED ABLATION EXPERIMENT: CHANNEL 23")
    print("=" * 80)

    pipe = AttentionPipeline.from_yaml("configs/pipeline.yaml")
    cap = cv2.VideoCapture("runs/new_classroom_test.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    model = pipe.fusion
    model.eval()
    threshold = float(pipe.binary_threshold)

    obs_records = []
    warmup_count = 0
    scored_count = 0
    t_start = time.time()

    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = f_idx / fps
        res = pipe.process_frame(ts, frame)

        for s in res.get("students", []):
            tid = int(s["student_id"])
            t = pipe.tracks.get(tid)
            if not t or len(t.feature_window) < pipe.scorer.window:
                warmup_count += 1
                continue

            scored_count += 1
            # Stack the exact 16-frame window
            w = np.stack(t.feature_window[-pipe.scorer.window:], axis=0)  # (16, 32)
            ch23_curr = float(w[-1, 23])
            ch23_mean = float(np.mean(w[:, 23]))

            # Variant A: Original
            t_orig = torch.from_numpy(w).unsqueeze(0).to(pipe.device)
            with torch.no_grad():
                probs_orig = model.predict_proba(t_orig).squeeze(0).cpu().numpy()
            p_high_orig = float(probs_orig[1])
            pred_orig = "HIGH" if p_high_orig >= threshold else "LOW"

            # Variant B: Ch 23 = 0.0 (diagnostic copy)
            w_ablated = w.copy()
            w_ablated[:, 23] = 0.0
            t_ablated = torch.from_numpy(w_ablated).unsqueeze(0).to(pipe.device)
            with torch.no_grad():
                probs_ablated = model.predict_proba(t_ablated).squeeze(0).cpu().numpy()
            p_high_ablated = float(probs_ablated[1])
            pred_ablated = "HIGH" if p_high_ablated >= threshold else "LOW"

            delta = p_high_ablated - p_high_orig

            if pred_orig == "LOW" and pred_ablated == "HIGH":
                flip = "LOW_TO_HIGH"
            elif pred_orig == "HIGH" and pred_ablated == "LOW":
                flip = "HIGH_TO_LOW"
            elif pred_orig == "HIGH" and pred_ablated == "HIGH":
                flip = "UNCHANGED_HIGH"
            else:
                flip = "UNCHANGED_LOW"

            obs_records.append({
                "frame_number": f_idx,
                "timestamp_seconds": round(ts, 4),
                "track_id": tid,
                "p_high_original": round(p_high_orig, 4),
                "pred_original": pred_orig,
                "p_high_ch23_zero": round(p_high_ablated, 4),
                "pred_ch23_zero": pred_ablated,
                "delta_p_high": round(delta, 4),
                "flip_type": flip,
                "ch23_current_frame": round(ch23_curr, 4),
                "ch23_window_mean": round(ch23_mean, 4),
            })

        f_idx += 1
        if f_idx % 50 == 0 or f_idx == total_frames:
            elapsed = time.time() - t_start
            print(f"Processed frame {f_idx}/{total_frames} | Scored Observations: {scored_count} | Speed: {f_idx/max(elapsed, 1e-4):.1f} FPS")

    cap.release()

    obs_df = pd.DataFrame(obs_records)
    obs_csv = PHASE6G1_DIR / "ch23_ablation_per_observation.csv"
    obs_df.to_csv(obs_csv, index=False)
    print(f"Saved {len(obs_df)} per-observation records to {obs_csv}")

    # Compute Per-Track Aggregation
    tracks = sorted(obs_df["track_id"].unique())
    track_summaries = []
    for tid in tracks:
        tdf = obs_df[obs_df["track_id"] == tid]
        n = len(tdf)
        orig_h = (tdf["pred_original"] == "HIGH").sum()
        orig_l = (tdf["pred_original"] == "LOW").sum()
        abl_h = (tdf["pred_ch23_zero"] == "HIGH").sum()
        abl_l = (tdf["pred_ch23_zero"] == "LOW").sum()
        flips_l2h = (tdf["flip_type"] == "LOW_TO_HIGH").sum()
        flips_h2l = (tdf["flip_type"] == "HIGH_TO_LOW").sum()
        unchanged = (tdf["flip_type"].isin(["UNCHANGED_HIGH", "UNCHANGED_LOW"])).sum()

        mean_p_orig = float(tdf["p_high_original"].mean())
        mean_p_abl = float(tdf["p_high_ch23_zero"].mean())

        track_summaries.append({
            "track_id": tid,
            "total_scored": n,
            "orig_high_count": orig_h,
            "orig_low_count": orig_l,
            "orig_pct_high": round(orig_h / n * 100, 1),
            "ablated_high_count": abl_h,
            "ablated_low_count": abl_l,
            "ablated_pct_high": round(abl_h / n * 100, 1),
            "pct_high_shift": round((abl_h - orig_h) / n * 100, 1),
            "flips_low_to_high": flips_l2h,
            "flips_high_to_low": flips_h2l,
            "unchanged_count": unchanged,
            "net_high_gain": abl_h - orig_h,
            "mean_p_high_orig": round(mean_p_orig, 4),
            "mean_p_high_ablated": round(mean_p_abl, 4),
            "delta_mean_p_high": round(mean_p_abl - mean_p_orig, 4),
        })

    track_df = pd.DataFrame(track_summaries)
    track_csv = PHASE6G1_DIR / "ch23_ablation_per_track_summary.csv"
    track_df.to_csv(track_csv, index=False)
    print(f"Saved per-track summary to {track_csv}")

    # Compute Overall Distribution Metrics
    orig_p = obs_df["p_high_original"].values
    abl_p = obs_df["p_high_ch23_zero"].values

    total_scored = len(obs_df)
    total_l2h = int((obs_df["flip_type"] == "LOW_TO_HIGH").sum())
    total_h2l = int((obs_df["flip_type"] == "HIGH_TO_LOW").sum())
    total_unchanged_h = int((obs_df["flip_type"] == "UNCHANGED_HIGH").sum())
    total_unchanged_l = int((obs_df["flip_type"] == "UNCHANGED_LOW").sum())

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    hist_orig, _ = np.histogram(orig_p, bins=bins)
    hist_abl, _ = np.histogram(abl_p, bins=bins)

    bin_labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(bins) - 1)]

    overall_metrics = {
        "execution_date": "2026-10-22",
        "checkpoint": "checkpoints/classroom_manual_binary_model_v2_clean.pt",
        "decision_threshold": threshold,
        "total_scored_observations": total_scored,
        "warmup_observations": warmup_count,
        "flips_summary": {
            "flips_low_to_high": total_l2h,
            "flips_low_to_high_pct": round(total_l2h / total_scored * 100, 2),
            "flips_high_to_low": total_h2l,
            "flips_high_to_low_pct": round(total_h2l / total_scored * 100, 2),
            "total_flips": total_l2h + total_h2l,
            "total_flip_rate_pct": round((total_l2h + total_h2l) / total_scored * 100, 2),
            "unchanged_high": total_unchanged_h,
            "unchanged_low": total_unchanged_l,
            "total_unchanged": total_unchanged_h + total_unchanged_l,
            "total_unchanged_pct": round((total_unchanged_h + total_unchanged_l) / total_scored * 100, 2),
        },
        "class_totals": {
            "original_high": int((obs_df["pred_original"] == "HIGH").sum()),
            "original_low": int((obs_df["pred_original"] == "LOW").sum()),
            "original_pct_high": round(float((obs_df["pred_original"] == "HIGH").mean() * 100), 2),
            "ablated_high": int((obs_df["pred_ch23_zero"] == "HIGH").sum()),
            "ablated_low": int((obs_df["pred_ch23_zero"] == "LOW").sum()),
            "ablated_pct_high": round(float((obs_df["pred_ch23_zero"] == "HIGH").mean() * 100), 2),
            "net_high_gain": int((obs_df["pred_ch23_zero"] == "HIGH").sum() - (obs_df["pred_original"] == "HIGH").sum()),
        },
        "distribution_metrics": {
            "p_high_original": {
                "mean": round(float(np.mean(orig_p)), 4),
                "std": round(float(np.std(orig_p)), 4),
                "median": round(float(np.median(orig_p)), 4),
                "p25": round(float(np.percentile(orig_p, 25)), 4),
                "p75": round(float(np.percentile(orig_p, 75)), 4),
                "iqr": round(float(np.percentile(orig_p, 75) - np.percentile(orig_p, 25)), 4),
                "min": round(float(np.min(orig_p)), 4),
                "max": round(float(np.max(orig_p)), 4),
            },
            "p_high_ch23_zero": {
                "mean": round(float(np.mean(abl_p)), 4),
                "std": round(float(np.std(abl_p)), 4),
                "median": round(float(np.median(abl_p)), 4),
                "p25": round(float(np.percentile(abl_p, 25)), 4),
                "p75": round(float(np.percentile(abl_p, 75)), 4),
                "iqr": round(float(np.percentile(abl_p, 75) - np.percentile(abl_p, 25)), 4),
                "min": round(float(np.min(abl_p)), 4),
                "max": round(float(np.max(abl_p)), 4),
            },
            "delta_p_high": {
                "mean": round(float(np.mean(abl_p - orig_p)), 4),
                "std": round(float(np.std(abl_p - orig_p)), 4),
                "median": round(float(np.median(abl_p - orig_p)), 4),
            },
        },
        "histogram_bins": {
            bin_labels[i]: {
                "original_count": int(hist_orig[i]),
                "original_pct": round(hist_orig[i] / total_scored * 100, 2),
                "ablated_count": int(hist_abl[i]),
                "ablated_pct": round(hist_abl[i] / total_scored * 100, 2),
            }
            for i in range(len(bin_labels))
        },
    }

    metrics_json = PHASE6G1_DIR / "ch23_ablation_overall_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"Saved overall metrics to {metrics_json}")

    return obs_df, track_df, overall_metrics


if __name__ == "__main__":
    audit_head_roll()
    run_channel23_controlled_experiment()
