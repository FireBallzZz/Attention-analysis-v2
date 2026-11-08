"""Generates publication-ready 2D Spatial Classroom Attention Heatmaps from scores JSONL.

Visualizes student desk seating locations, attention heat levels (Green -> Gold -> Red),
and primary behavioral cues (phone, reading, sleeping, peer-talking).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def generate_spatial_heatmap(scores_jsonl: Path, out_png: Path, title: str | None = None):
    if not scores_jsonl.exists():
        print(f"File not found: {scores_jsonl}")
        return

    student_data = defaultdict(lambda: {
        "x": [],
        "y": [],
        "w": [],
        "h": [],
        "scores": [],
        "cues": defaultdict(int)
    })

    frame_width = 1920
    frame_height = 1080

    with open(scores_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            for s in entry.get("students", []):
                sid = s.get("student_id")
                bbox = s.get("bbox")
                score = s.get("score")
                cue = s.get("cue", "unknown")

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = x2 - x1
                    bh = y2 - y1
                    student_data[sid]["x"].append(cx)
                    student_data[sid]["y"].append(cy)
                    student_data[sid]["w"].append(bw)
                    student_data[sid]["h"].append(bh)
                    if x2 > frame_width:
                        frame_width = int(x2 * 1.1)
                    if y2 > frame_height:
                        frame_height = int(y2 * 1.1)

                if score is not None:
                    student_data[sid]["scores"].append(float(score))
                student_data[sid]["cues"][cue] += 1

    if not student_data:
        print("No student observations found.")
        return

    # Aggregate student stats
    students_summary = []
    for sid, d in student_data.items():
        if not d["x"] or not d["scores"]:
            continue
        mean_x = float(np.median(d["x"]))
        mean_y = float(np.median(d["y"]))
        mean_w = float(np.median(d["w"]))
        mean_h = float(np.median(d["h"]))
        mean_score = float(np.mean(d["scores"]))
        top_cue = max(d["cues"].items(), key=lambda item: item[1])[0]

        students_summary.append({
            "id": sid,
            "x": mean_x,
            "y": mean_y,
            "w": mean_w,
            "h": mean_h,
            "score": mean_score,
            "cue": top_cue
        })

    # Sort front to back for rendering
    students_summary.sort(key=lambda s: s["y"])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor("#0F141C")
    ax.set_facecolor("#151D28")

    # Colormap: Red (Low) -> Gold/Yellow (Moderate) -> Vibrant Green (High)
    colors = ["#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#2EC4B6"]
    cmap = LinearSegmentedColormap.from_list("attention_heat", colors, N=100)

    # Plot Teacher Podium / Board Line at the top or front
    ax.axhline(0, color="#4A5568", linestyle="--", alpha=0.5)
    ax.text(frame_width / 2.0, 30, "[ FRONT OF CLASSROOM / TEACHER LECTURE PODIUM ]",
            color="#A0AEC0", fontsize=11, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A202C", edgecolor="#4A5568", alpha=0.9))

    # Plot each student's spatial interaction zone
    for s in students_summary:
        norm_score = max(0.0, min(1.0, s["score"] / 100.0))
        node_color = cmap(norm_score)
        
        # Outer glow ring
        glow = patches.Ellipse(
            (s["x"], s["y"]),
            width=s["w"] * 1.15,
            height=s["h"] * 0.95,
            edgecolor=node_color,
            facecolor=node_color,
            alpha=0.25,
            linewidth=2
        )
        ax.add_patch(glow)

        # Main desk/student marker
        rect = patches.FancyBboxPatch(
            (s["x"] - s["w"] * 0.45, s["y"] - s["h"] * 0.35),
            s["w"] * 0.9,
            s["h"] * 0.7,
            boxstyle="round,pad=8",
            edgecolor=node_color,
            facecolor="#1A202C",
            linewidth=2.5,
            alpha=0.92
        )
        ax.add_patch(rect)

        # Student Label
        id_str = f"S{s['id']:02d}"
        score_str = f"{s['score']:.1f}"
        cue_str = s['cue']
        if len(cue_str) > 16:
            cue_str = cue_str[:14] + ".."

        ax.text(s["x"], s["y"] - s["h"] * 0.12, id_str,
                color="white", fontsize=11, fontweight="heavy", ha="center", va="center")
        ax.text(s["x"], s["y"] + s["h"] * 0.06, f"Score: {score_str}",
                color=node_color, fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(s["x"], s["y"] + s["h"] * 0.22, f"[{cue_str}]",
                color="#CBD5E0", fontsize=8, fontstyle="italic", ha="center", va="center")

    # Invert Y axis so (0,0) is top-left, matching camera perspective
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)

    # Title & Metadata
    plot_title = title or f"Spatial Classroom Attention & Seating Heatmap ({scores_jsonl.stem})"
    plt.title(plot_title, color="white", fontsize=15, fontweight="bold", pad=25)
    ax.set_xlabel("Horizontal Camera Coordinate (px)", color="#A0AEC0", fontsize=10, labelpad=10)
    ax.set_ylabel("Perspective Row Depth (px)", color="#A0AEC0", fontsize=10, labelpad=10)
    ax.tick_params(colors="#718096", labelsize=9)

    # Grid lines
    ax.grid(color="#2D3748", linestyle=":", alpha=0.6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Multimodal Attention Score (0 - 100)", color="white", fontsize=10, labelpad=12)
    cbar.ax.tick_params(colors="white", labelsize=9)

    # Summary Card
    all_scores = [s["score"] for s in students_summary]
    class_mean = float(np.mean(all_scores)) if all_scores else 0.0
    high_count = sum(1 for s in students_summary if s["score"] >= 50.0)
    low_count = len(students_summary) - high_count

    summary_text = (
        f"Classroom Summary:\n"
        f"• Tracked Students : {len(students_summary)}\n"
        f"• Class Mean Score : {class_mean:.1f} / 100\n"
        f"• Attentive (HIGH) : {high_count} ({high_count/len(students_summary)*100:.0f}%)\n"
        f"• Distracted (LOW) : {low_count} ({low_count/len(students_summary)*100:.0f}%)"
    )
    plt.figtext(0.14, 0.15, summary_text, color="white", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A202C", edgecolor="#4A5568", alpha=0.85))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Spatial heatmap saved: {out_png.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Generate Classroom Spatial Attention Heatmap")
    parser.add_argument("--scores", required=True, help="Path to scores JSONL file")
    parser.add_argument("--out", required=True, help="Path to output PNG image")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    generate_spatial_heatmap(Path(args.scores), Path(args.out), args.title)

if __name__ == "__main__":
    main()

