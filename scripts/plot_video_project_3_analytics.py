"""Generate Publication-Quality Visual Analytics for Video Project 3.

Plots:
1. Class-wide temporal attention trend over 22.6 seconds.
2. Per-student engagement score ranking and attentive/distracted breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SCORES_JSONL = WORKSPACE_ROOT / "runs" / "video_project_3_scores.jsonl"
METRICS_JSON = WORKSPACE_ROOT / "runs" / "reports" / "video_project_3_metrics.json"
PLOTS_DIR = WORKSPACE_ROOT / "runs" / "reports"


def main():
    if not SCORES_JSONL.exists() or not METRICS_JSON.exists():
        print("Required files not found.")
        return

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Timeline extraction
    records = []
    with open(SCORES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    timestamps = []
    mean_scores = []
    drop_timestamps = []

    for rec in records:
        t = rec["timestamp"]
        valid_s = [s["score"] for s in rec["students"] if s.get("score") is not None]
        if valid_s:
            timestamps.append(t)
            mean_scores.append(np.mean(valid_s))
            if rec.get("drops", 0) > 0:
                drop_timestamps.append(t)

    # Plot 1: Class-Wide Attention Curve
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(timestamps, mean_scores, color="#1f77b4", linewidth=2.2, label="Class Average Attention (0-100)")
    plt.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.5, label="Attention Decision Boundary (τ = 50.0)")

    # Highlight drop events
    if drop_timestamps:
        plt.scatter(
            drop_timestamps,
            [mean_scores[timestamps.index(t)] for t in drop_timestamps],
            color="#ff7f0e",
            s=40,
            zorder=5,
            label="Attention Drop Events Flagged",
        )

    plt.fill_between(timestamps, 50.0, mean_scores, where=np.array(mean_scores) >= 50.0, color="#2ca02c", alpha=0.15, label="Attentive Zone")
    plt.fill_between(timestamps, 50.0, mean_scores, where=np.array(mean_scores) < 50.0, color="#d62728", alpha=0.15, label="Disengaged Zone")

    plt.title("Classroom Multimodal Attention Dynamics — Video Project 3", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Timeline (Seconds)", fontsize=11)
    plt.ylabel("Attention Score (0 - 100)", fontsize=11)
    plt.ylim(0, 100)
    plt.xlim(0, max(timestamps) if timestamps else 23)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_timeline = PLOTS_DIR / "video_project_3_timeline.png"
    plt.savefig(out_timeline)
    plt.close()
    print(f"Saved: {out_timeline}")

    # Plot 2: Per-Student Mean Score Bar Chart
    valid_students = [s for s in metrics["students"] if s["mean_score"] is not None and s["scored_observations"] >= 20]
    valid_students.sort(key=lambda x: x["mean_score"])

    s_labels = [f"Student {s['student_id']:02d}" for s in valid_students]
    s_means = [s["mean_score"] for s in valid_students]
    s_colors = ["#2ca02c" if m >= 50.0 else "#d62728" for m in s_means]

    plt.figure(figsize=(10, max(6, len(valid_students) * 0.32)), dpi=300)
    bars = plt.barh(s_labels, s_means, color=s_colors, height=0.65, edgecolor="none")
    plt.axvline(50.0, color="#333333", linestyle="--", linewidth=1.2, label="Threshold (50.0)")

    for bar, score in zip(bars, s_means):
        plt.text(score + 1.2, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", fontsize=8, color="#333333")

    plt.title("Per-Student Mean Attention Score Ranking — Video Project 3", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Average Attention Score (0 - 100)", fontsize=11)
    plt.ylabel("Student Identifier", fontsize=11)
    plt.xlim(0, 105)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_students = PLOTS_DIR / "video_project_3_student_ranking.png"
    plt.savefig(out_students)
    plt.close()
    print(f"Saved: {out_students}")


if __name__ == "__main__":
    main()

