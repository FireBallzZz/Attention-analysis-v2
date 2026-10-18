"""Compare a model run on Anno/clip_004.mp4 to Anno/ground_truth.csv.

1. Run the model and write a jsonl of predictions.
2. Match each ground-truth box (from annotations.xml) to a prediction by IoU.
3. Compare attention labels and write a new accuracy .txt report.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GT_TO_BAND = {
    "attentive": "HIGH",
    "high": "HIGH",
    "moderate": "MODERATE",
    "non-attentive": "LOW",
    "non_attentive": "LOW",
    "low": "LOW",
}
BANDS = ("HIGH", "MODERATE", "LOW")


def _band(label: str) -> str:
    return GT_TO_BAND.get(label.strip().lower().replace(" ", "-"), "MODERATE")


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def load_gt(csv_path: Path, xml_path: Path) -> dict[int, list[dict]]:
    labels: dict[tuple[int, int], str] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[(int(row["frame"]), int(row["student_id"]))] = row["attention"]

    by_frame: dict[int, list[dict]] = defaultdict(list)
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        sid = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", "0"))
            label = labels.get((frame, sid))
            if label is None:
                continue
            by_frame[frame].append(
                {
                    "student_id": sid,
                    "bbox": [
                        float(box.get("xtl", 0)),
                        float(box.get("ytl", 0)),
                        float(box.get("xbr", 0)),
                        float(box.get("ybr", 0)),
                    ],
                    "band": _band(label),
                    "label": label,
                }
            )
    return by_frame


def load_pred(path: Path) -> dict[int, list[dict]]:
    """Load model boxes from a every-frame CSV or a jsonl dump."""
    if path.suffix.lower() == ".csv":
        return _load_pred_csv(path)
    return _load_pred_jsonl(path)


def _load_pred_csv(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            band = str(row.get("attention") or "MODERATE").upper()
            if band not in BANDS:
                band = _band(band)
            by_frame[int(row["frame"])].append(
                {
                    "bbox": [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ],
                    "band": band,
                }
            )
    return by_frame


def _load_pred_jsonl(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            students = []
            for s in row.get("students") or []:
                band = str(s.get("band") or "MODERATE").upper()
                if band not in BANDS:
                    score = float(s.get("score") or 50)
                    band = "HIGH" if score >= 70 else "LOW" if score < 40 else "MODERATE"
                students.append({"bbox": [float(v) for v in s["bbox"]], "band": band})
            frame = int(row["frame"]) if "frame" in row else index
            by_frame[frame] = students
    return by_frame


def match_frame(gt_list: list[dict], pred_list: list[dict], iou_thr: float) -> list[tuple[str, str]]:
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_list):
        for pi, p in enumerate(pred_list):
            score = _iou(g["bbox"], p["bbox"])
            if score >= iou_thr:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[str, str]] = []
    for _score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gt_list[gi]["band"], pred_list[pi]["band"]))
    return matched


def prf1(y_true: list[str], y_pred: list[str]) -> dict:
    support = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / support if support else 0.0
    per = {}
    macros = []
    for band in BANDS:
        tp = sum(t == band and p == band for t, p in zip(y_true, y_pred))
        fp = sum(t != band and p == band for t, p in zip(y_true, y_pred))
        fn = sum(t == band and p != band for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[band] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(t == band for t in y_true),
        }
        macros.append((prec, rec, f1))
    macro = {
        "precision": sum(m[0] for m in macros) / 3,
        "recall": sum(m[1] for m in macros) / 3,
        "f1": sum(m[2] for m in macros) / 3,
    }
    return {"accuracy": acc, "support": support, "per_band": per, "macro": macro}


def write_report(
    path: Path,
    metrics: dict,
    video: Path,
    csv_path: Path,
    pred_path: Path,
    matched: int,
    unmatched_gt: int,
) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.video_accuracy import load_model_metrics, model_lines

    model = load_model_metrics()
    overall_model = ((model or {}).get("overall") or {}).get("accuracy")
    lines = [
        "=" * 72,
        "VIDEO ACCURACY REPORT",
        "=" * 72,
        "",
        f"Video      : {video.resolve()}",
        f"Ground truth: {csv_path.resolve()}",
        f"Predictions : {pred_path.resolve()}",
        f"Matched boxes : {matched}",
        f"Unmatched GT  : {unmatched_gt}",
        "",
        *model_lines(model),
        "",
        "=" * 72,
        "GROUND TRUTH ACCURACY",
        "=" * 72,
        "",
        f"{'Band':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}",
        "-" * 72,
    ]
    for band in BANDS:
        item = metrics["per_band"][band]
        lines.append(
            f"{band:<12} {item['precision']*100:7.2f}% {item['recall']*100:7.2f}% "
            f"{item['f1']*100:7.2f}% {item['support']:8d}"
        )
    macro = metrics["macro"]
    lines.extend(
        [
            "-" * 72,
            f"{'MACRO':<12} {macro['precision']*100:7.2f}% {macro['recall']*100:7.2f}% "
            f"{macro['f1']*100:7.2f}%",
            "",
            f"MODEL ACCURACY: {overall_model*100:.2f}%" if overall_model is not None else "MODEL ACCURACY: n/a",
            f"GROUND TRUTH ACCURACY: {metrics['accuracy']*100:.2f}%",
            f"Students compared: {metrics['support']}",
            "=" * 72,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy vs Anno ground_truth.csv")
    parser.add_argument("--video", default=str(ROOT / "Anno" / "clip_004.mp4"))
    parser.add_argument("--csv", default=str(ROOT / "Anno" / "ground_truth.csv"))
    parser.add_argument("--xml", default=str(ROOT / "Anno" / "annotations.xml"))
    parser.add_argument(
        "--pred",
        default=str(ROOT / "runs" / "anno_clip_004_pred.csv"),
        help="Model predictions: every-frame CSV from annotate_clip_csv.py, or jsonl",
    )
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument(
        "--out",
        default="",
        help="Report path (default: runs/reports/clip_004_gt_<stamp>_accuracy.txt)",
    )
    args = parser.parse_args()

    video = Path(args.video)
    csv_path = Path(args.csv)
    xml_path = Path(args.xml)
    pred_path = Path(args.pred)
    for path in (csv_path, xml_path, pred_path):
        if not path.exists():
            raise SystemExit(f"Missing {path.resolve()}")

    gt = load_gt(csv_path, xml_path)
    pred = load_pred(pred_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    unmatched = 0
    for frame in sorted(gt):
        matched = match_frame(gt[frame], pred.get(frame, []), args.iou)
        y_true.extend(t for t, _p in matched)
        y_pred.extend(p for _t, p in matched)
        unmatched += len(gt[frame]) - len(matched)

    metrics = prf1(y_true, y_pred)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "runs" / "reports" / f"clip_004_gt_{stamp}_accuracy.txt"
    write_report(out, metrics, video, csv_path, pred_path, len(y_true), unmatched)
    print(
        f"GROUND-TRUTH ACCURACY: {metrics['accuracy']*100:.2f}%  "
        f"matched={len(y_true)}  unmatched_gt={unmatched}"
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
