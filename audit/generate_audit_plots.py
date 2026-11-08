"""Generate High-Resolution Publication Charts for Audit Report.

Generates:
1. `audit/confusion_matrices_audit.png`: 3-panel normalized confusion matrix heatmaps
2. `audit/iou_sensitivity_curve.png`: Dual-axis plot of IoU threshold vs Accuracy & Samples (Selection Bias Proof)
3. `audit/runtime_breakdown.png`: Component latency breakdown on RTX 3060
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model")
AUDIT_DIR = WORKSPACE_ROOT / "audit"

# Set publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.8


def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    # 1. Clip 001 (TN=0, FP=51, FN=1, TP=1015)
    cm1 = np.array([[0, 51], [1, 1015]])
    cm1_norm = np.array([[0.0, 1.0], [1/1016, 1015/1016]])

    # 2. Clip 007 (TN=3, FP=165, FN=2, TP=693)
    cm2 = np.array([[3, 165], [2, 693]])
    cm2_norm = np.array([[3/168, 165/168], [2/695, 693/695]])

    # 3. Held-Out Validation (TN=295, FP=327, FN=237, TP=3743)
    cm3 = np.array([[295, 327], [237, 3743]])
    cm3_norm = np.array([[295/622, 327/622], [237/3980, 3743/3980]])

    configs = [
        (axes[0], cm1, cm1_norm, "Clip 001 (IoU >= 0.30)\nN = 1,067 | Acc: 95.13% | Bal-Acc: 49.95%"),
        (axes[1], cm2, cm2_norm, "Clip 007 (IoU >= 0.30)\nN = 863 | Acc: 80.65% | Bal-Acc: 50.75%"),
        (axes[2], cm3, cm3_norm, "Held-Out Validation (tau=0.50)\nN = 4,602 | Acc: 87.74% | Bal-Acc: 70.74%"),
    ]

    for ax, raw, norm, title in configs:
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred LOW", "Pred HIGH"], fontsize=10, fontweight="bold")
        ax.set_yticklabels(["True LOW", "True HIGH"], fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#0F2942")

        for i in range(2):
            for j in range(2):
                val = raw[i, j]
                pct = norm[i, j] * 100.0
                txt_color = "white" if norm[i, j] > 0.55 else "#1A202C"
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = AUDIT_DIR / "confusion_matrices_audit.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_iou_sensitivity():
    # Load JSON data
    json_path = AUDIT_DIR / "METRIC_RECALCULATION.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)["clip_001"]["iou_sensitivity"]

    ious = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    accs = [data[f"iou_{iou:.2f}"]["accuracy"] * 100.0 for iou in ious]
    samples = [data[f"iou_{iou:.2f}"]["n_samples"] for iou in ious]
    fps = [data[f"iou_{iou:.2f}"]["confusion_matrix"]["FP"] for iou in ious]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)

    color_acc = "#1A5276"
    color_n = "#C0392B"

    ax1.set_xlabel("Spatial Matching IoU Cutoff Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Binary Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    line1 = ax1.plot(ious, accs, color=color_acc, marker="o", linewidth=2.5, markersize=7, label="Binary Accuracy (%)")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(92.0, 98.0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Matched Student Observations (N)", color=color_n, fontsize=11, fontweight="bold")
    line2 = ax2.plot(ious, samples, color=color_n, marker="s", linewidth=2.5, linestyle="--", markersize=7, label="Matched Samples (N)")
    ax2.tick_params(axis="y", labelcolor=color_n)
    ax2.set_ylim(500, 1500)

    # Title & Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True, framealpha=0.9)
    plt.title("Empirical Evidence of Selection Bias in IoU Sensitivity Analysis\n(Higher IoU prunes ambiguous/occluded students, artificially inflating accuracy)", fontsize=11, fontweight="bold", pad=12, color="#0F2942")

    plt.tight_layout()
    out_path = AUDIT_DIR / "iou_sensitivity_curve.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_runtime_breakdown():
    json_path = AUDIT_DIR / "RUNTIME_BENCHMARK.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stages = [
        "BiGRU Temporal Inference (20 st.)",
        "Video Frame Decoding (1080p)",
        "Visualization & Alpha Blending",
        "Tracking & Box EMA Smoothing",
        "L2CS Gaze & SCB-6 Behavior",
        "YOLO11n-Pose Landmark Extraction",
    ]
    latencies = [
        data["bigru_temporal_ms"],
        data["frame_decode_ms"],
        1.51,
        2.10,
        6.40,
        data["yolo11n_pose_ms"],
    ]
    colors_list = ["#27AE60", "#2980B9", "#8E44AD", "#F39C12", "#E67E22", "#C0392B"]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=300)
    bars = ax.barh(stages, latencies, color=colors_list, height=0.6)

    ax.set_xlabel("Latency per Frame (Milliseconds) — Lower is Better", fontsize=10, fontweight="bold")
    ax.set_title(f"Hardware Runtime Profiling on {data['device_name']}\n(Total: 32.66 ms / Frame = ~30.6 FPS Real-Time Processing @ 1080p)", fontsize=11, fontweight="bold", pad=10, color="#0F2942")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, latencies):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.2f} ms", va="center", ha="left", fontsize=9, fontweight="bold", color="#2C3E50")

    ax.set_xlim(0, 26)
    plt.tight_layout()
    out_path = AUDIT_DIR / "runtime_breakdown.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_confusion_matrices()
    plot_iou_sensitivity()
    plot_runtime_breakdown()

