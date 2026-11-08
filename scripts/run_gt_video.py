"""Run attention on a video with Anno ground truth and write both accuracies.

Uses GT boxes from annotations.xml, the GT attention calibrator for labels,
compares to ground_truth.csv, and writes a report with MODEL + GROUND TRUTH.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.gt_calibrator import GtAttentionCalibrator  # noqa: E402
from attention_pipeline.pipeline import AttentionPipeline  # noqa: E402
from attention_pipeline.utils import resolve_device  # noqa: E402
from attention_pipeline.video_accuracy import load_model_metrics, model_lines  # noqa: E402

BANDS = ("HIGH", "MODERATE", "LOW")
GT_MAP = {
    "attentive": "HIGH",
    "high": "HIGH",
    "moderate": "MODERATE",
    "non-attentive": "LOW",
    "non_attentive": "LOW",
    "low": "LOW",
}
COLORS = {"HIGH": (0, 200, 0), "MODERATE": (0, 165, 255), "LOW": (0, 0, 220)}


def load_gt(xml_path: Path, csv_path: Path):
    labels = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[(int(row["frame"]), int(row["student_id"]))] = GT_MAP[
                row["attention"].strip().lower().replace(" ", "-")
            ]
    by_frame = defaultdict(list)
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        sid = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", "0"))
            band = labels.get((frame, sid))
            if band is None:
                continue
            by_frame[frame].append(
                {
                    "id": sid,
                    "bbox": np.array(
                        [
                            float(box.get("xtl", 0)),
                            float(box.get("ytl", 0)),
                            float(box.get("xbr", 0)),
                            float(box.get("ybr", 0)),
                        ],
                        np.float32,
                    ),
                    "gt": band,
                }
            )
    return by_frame


def head_pose_from_kpts(kpts):
    if kpts is None or kpts.shape[0] < 3:
        return np.zeros(3, np.float32)
    nose, leye, reye = kpts[0], kpts[1], kpts[2]
    if float(nose[2] + leye[2] + reye[2]) < 0.4:
        return np.zeros(3, np.float32)
    mid = (leye[:2] + reye[:2]) / 2.0
    eye_w = abs(float(reye[0] - leye[0])) + 1e-6
    yaw = float(np.degrees(np.arctan2(nose[0] - mid[0], eye_w)))
    pitch = float(np.degrees(np.arctan2(nose[1] - mid[1], eye_w)))
    roll = float(np.degrees(np.arctan2(reye[1] - leye[1], eye_w)))
    return np.array([yaw, pitch, roll], np.float32)


def best_kpts(dets, bbox):
    best = None
    best_iou = 0.0
    for det in dets:
        a, b = bbox, det.bbox_xyxy
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            continue
        aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        iou = inter / (aa + bb - inter + 1e-6)
        if iou > best_iou:
            best_iou = iou
            best = det.keypoints
    return best if best_iou >= 0.15 else None


def crop_face(frame, bbox, kpts):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if kpts is not None and kpts.shape[0] >= 5 and float(kpts[0, 2] + kpts[1, 2]) > 0.2:
        eyes = kpts[[0, 1, 2, 3, 4], :2]
        cx, cy = eyes.mean(axis=0)
        scale = float(np.linalg.norm(kpts[1, :2] - kpts[2, :2]) + 1.0)
        side = max(int(scale * 3.5), 32)
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = x1 + side, y1 + side
    else:
        bh = max(y2 - y1, 1)
        y2 = y1 + int(bh * 0.45)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2]


def prf1(y_true, y_pred):
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
        per[band] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(t == band for t in y_true)}
        macros.append((prec, rec, f1))
    macro = {
        "precision": sum(m[0] for m in macros) / 3,
        "recall": sum(m[1] for m in macros) / 3,
        "f1": sum(m[2] for m in macros) / 3,
    }
    return {"accuracy": acc, "support": support, "per_band": per, "macro": macro}


def write_report(path: Path, video: Path, gt_metrics: dict, out_video: Path, out_csv: Path) -> None:
    model = load_model_metrics()
    lines = [
        "=" * 72,
        "VIDEO ACCURACY REPORT",
        "=" * 72,
        "",
        "Model accuracy is the saved validation score.",
        "Ground-truth accuracy is agreement with Anno/ground_truth.csv.",
        "",
        f"Input video : {video.resolve()}",
        f"Overlay     : {out_video.resolve()}",
        f"Predictions : {out_csv.resolve()}",
        "",
        *model_lines(model),
        "",
        "=" * 72,
        "GROUND TRUTH ACCURACY",
        "=" * 72,
        "",
        "Compared to the manual Anno labels for this video.",
        "",
        f"{'Band':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}",
        "-" * 72,
    ]
    for band in BANDS:
        item = gt_metrics["per_band"][band]
        lines.append(
            f"{band:<12} {item['precision']*100:7.2f}% {item['recall']*100:7.2f}% "
            f"{item['f1']*100:7.2f}% {item['support']:8d}"
        )
    macro = gt_metrics["macro"]
    overall_model = ((model or {}).get("overall") or {}).get("accuracy")
    lines.extend(
        [
            "-" * 72,
            f"{'MACRO':<12} {macro['precision']*100:7.2f}% {macro['recall']*100:7.2f}% "
            f"{macro['f1']*100:7.2f}%",
            "",
            f"GROUND TRUTH ACCURACY: {gt_metrics['accuracy']*100:.2f}%",
            f"Students compared: {gt_metrics['support']}",
            "",
            f"MODEL ACCURACY: {overall_model*100:.2f}%" if overall_model is not None else "MODEL ACCURACY: n/a",
            f"GROUND TRUTH ACCURACY: {gt_metrics['accuracy']*100:.2f}%",
            "=" * 72,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "Anno" / "clip_004.mp4"))
    parser.add_argument("--csv", default=str(ROOT / "Anno" / "ground_truth.csv"))
    parser.add_argument("--xml", default=str(ROOT / "Anno" / "annotations.xml"))
    parser.add_argument("--config", default="configs/pipeline_merged.yaml")
    parser.add_argument("--out-csv", default=str(ROOT / "runs" / "anno_clip_004_gtpred.csv"))
    parser.add_argument("--out-video", default=str(ROOT / "runs" / "anno_clip_004_gtpred.mp4"))
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        candidates = [
            ROOT / "Anno" / "clip_004.mp4",
            ROOT.parent.parent / "classroom_01_10s_clips" / "clip_004.mp4",
            ROOT.parent / "classroom_01_10s_clips" / "clip_004.mp4",
            ROOT / "video" / "clip_004.mp4",
        ]
        for cand in candidates:
            if cand.exists():
                video = cand
                break
    csv_path = Path(args.csv)
    xml_path = Path(args.xml)
    out_csv = Path(args.out_csv)
    out_video = Path(args.out_video)
    for path in (video, csv_path, xml_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    by_frame = load_gt(xml_path, csv_path)
    pipe = AttentionPipeline.from_yaml(ROOT / args.config)
    pipe.detector.require_face = False
    pipe.detector.min_det_conf = 0.05
    pipe.detector.max_people = None
    cal = GtAttentionCalibrator(device=str(pipe.device))
    if not cal.enabled:
        raise SystemExit("Missing checkpoints/gt_attention_calibrator.pt. Run scripts/train_gt_calibrator.py")
    cal.reset()

    if "clip_004" in video.name:
        print("\n" + "!" * 72)
        print("WARNING: 'gt_attention_calibrator.pt' was trained on clip_004.mp4.")
        print("Evaluating all students on this video causes train-test data contamination.")
        print("For defensible benchmarks, evaluate on held-out students or independent clips")
        print("(e.g., clip_001.mp4 or clip_008.mp4).")
        print("!" * 72 + "\n")

    device = resolve_device("auto")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    fieldnames = [
        "frame",
        "timestamp_sec",
        "student_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "attention",
        "gt_attention",
        "score",
        "eye_gaze_yaw",
        "eye_gaze_pitch",
        "head_yaw",
        "head_pitch",
        "head_roll",
    ]
    rows = []
    y_true = []
    y_pred = []
    frame_idx = 0
    print(f"Running GT-box attention on {video.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = pipe.detector.infer(frame)
        vis = frame.copy()
        for item in by_frame.get(frame_idx, []):
            kpts = best_kpts(dets, item["bbox"])
            head = head_pose_from_kpts(kpts)
            gaze = head[:2].copy()
            crop = crop_face(frame, item["bbox"], kpts)
            if crop is not None:
                rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                tensor = TF.to_tensor(rgb).unsqueeze(0).to(device)
                tensor = (tensor - mean) / std
                with torch.no_grad():
                    gaze = pipe.gaze.predict_degrees(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            band, score = cal.predict(item["id"], gaze, head, item["bbox"], width, height)
            y_true.append(item["gt"])
            y_pred.append(band)
            x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
            rows.append(
                {
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "student_id": item["id"],
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "attention": band,
                    "gt_attention": item["gt"],
                    "score": round(score, 2),
                    "eye_gaze_yaw": round(float(gaze[0]), 2),
                    "eye_gaze_pitch": round(float(gaze[1]), 2),
                    "head_yaw": round(float(head[0]), 2),
                    "head_pitch": round(float(head[1]), 2),
                    "head_roll": round(float(head[2]), 2),
                }
            )
            color = COLORS[band]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                vis,
                f"id{item['id']}:{band}",
                (int(x1), max(14, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        writer.write(vis)
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}")
        frame_idx += 1
    cap.release()
    writer.release()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    metrics = prf1(y_true, y_pred)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = ROOT / "runs" / "reports" / f"{video.stem}_gt_{stamp}_accuracy.txt"
    write_report(report, video, metrics, out_video, out_csv)
    print(f"GROUND TRUTH ACCURACY: {metrics['accuracy']*100:.2f}%")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_video}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
