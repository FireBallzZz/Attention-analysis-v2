"""Extract 32-D multimodal feature sequences from classroom clip_004 aligned with CVAT ground truth.

Generates: data/processed/classroom_clip_004_sequences.npz
Keys:
  features:    (N, 16, 32) float32
  student_ids: (N,) int32 (0..20)
  frame_ids:   (N,) int32 (0..299)
  targets:     (N,) int32 (0: HIGH, 1: MODERATE, 2: LOW)
  bands:       (N,) str ('HIGH', 'MODERATE', 'LOW')
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.types import FEAT_DIM, StudentFrame
from attention_pipeline.features import pack_features
from attention_pipeline.utils import PROJECT_ROOT, resolve_device

GT_TO_IDX = {
    "attentive": 0,
    "high": 0,
    "moderate": 1,
    "non-attentive": 2,
    "non_attentive": 2,
    "low": 2,
}
IDX_TO_BAND = {0: "HIGH", 1: "MODERATE", 2: "LOW"}


def load_gt(xml_path: Path, csv_path: Path):
    labels = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row["attention"].strip().lower().replace(" ", "-")
            labels[(int(row["frame"]), int(row["student_id"]))] = GT_TO_IDX[raw]

    by_frame = defaultdict(list)
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        sid = int(track.get("id", "0"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame", "0"))
            label_idx = labels.get((frame, sid))
            if label_idx is None:
                continue
            bbox = np.array(
                [
                    float(box.get("xtl", 0)),
                    float(box.get("ytl", 0)),
                    float(box.get("xbr", 0)),
                    float(box.get("ybr", 0)),
                ],
                dtype=np.float32,
            )
            by_frame[frame].append({
                "sid": sid,
                "bbox": bbox,
                "label_idx": label_idx,
            })
    return by_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="runs/Test.mp4")
    parser.add_argument("--xml", default="Anno/annotations.xml")
    parser.add_argument("--csv", default="Anno/ground_truth.csv")
    parser.add_argument("--out", default="data/processed/classroom_clip_004_sequences.npz")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame in master video")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames to extract")
    parser.add_argument("--force", action="store_true", help="Allow overwriting target output")
    args = parser.parse_args()

    video_path = PROJECT_ROOT / args.video
    xml_path = PROJECT_ROOT / args.xml
    csv_path = PROJECT_ROOT / args.csv
    out_path = PROJECT_ROOT / args.out
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Target dataset already exists: {out_path}. Refusing to overwrite to protect artifacts.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth from {xml_path.name} and {csv_path.name}...")
    gt_by_frame = load_gt(xml_path, csv_path)
    total_gt_frames = len(gt_by_frame)
    print(f"Loaded ground-truth for {total_gt_frames} frames.")

    print("Initializing pipeline extractors...")
    pipe = AttentionPipeline.from_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml", load_weights=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video {video_path.name}: {total_frames} frames, {fps:.1f} fps (Extracting {args.num_frames} frames from start_frame={args.start_frame})")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # Buffers per student
    student_histories: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=args.window))
    prev_frames: dict[int, np.ndarray] = {}

    out_features = []
    out_student_ids = []
    out_frame_ids = []
    out_targets = []
    out_bands = []

    t_start = time.time()
    frame_idx = 0

    while frame_idx < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps
        gt_students = gt_by_frame.get(frame_idx, [])

        # Run student detection to get pose keypoints
        detections = pipe.detector.infer(frame)
        scb_dets = pipe.scb.infer(frame)
        scb_cfg = pipe.cfg.get("scb", {})
        min_iou = float(scb_cfg.get("match_iou", 0.4))
        blend = float(scb_cfg.get("blend", 0.65))

        for gt_s in gt_students:
            sid = gt_s["sid"]
            bbox = gt_s["bbox"]
            label_idx = gt_s["label_idx"]

            # Match GT box to pipeline detection by IoU to find keypoints
            best_det = None
            best_iou = 0.0
            for det in detections:
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
                    best_det = det

            kpts = best_det.keypoints if (best_det is not None and best_iou >= 0.20) else np.zeros((17, 3), dtype=np.float32)

            # Extract face / gaze
            # Mock a Detection object with GT bbox if detection missed or IoU low
            from attention_pipeline.stages.detect_track import Detection
            det_for_face = best_det if (best_det is not None and best_iou >= 0.30) else Detection(
                bbox_xyxy=bbox,
                conf=1.0,
                track_id=sid,
                keypoints=kpts,
            )
            face = pipe.faces.extract(frame, det_for_face, sid)
            posture = pipe.posture_x.predict(kpts, bbox)

            # Match SCB
            scb_matches = pipe.scb.match_all([bbox], scb_dets, min_iou=min_iou)
            scb_probs = scb_matches[0] if scb_matches else None

            if scb_probs is not None:
                posture = (blend * scb_probs + (1.0 - blend) * posture).astype(np.float32)
                posture = posture / max(float(posture.sum()), 1e-6)

            # Compute optical motion for student box
            x1_i, y1_i = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2_i, y2_i = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
            crop = frame[y1_i:y2_i, x1_i:x2_i]
            motion = 0.0
            if sid in prev_frames and prev_frames[sid].shape == crop.shape:
                diff = cv2.absdiff(crop, prev_frames[sid])
                motion = float(np.mean(diff) / 255.0)
            if crop.size > 0:
                prev_frames[sid] = crop.copy()

            # Pack 32-D features
            feat = pack_features(
                head_pose=face.head_pose,
                gaze=face.gaze,
                emotion_probs=face.emotion_probs,
                posture_probs=posture,
                motion=motion,
                eye_aspect=face.eye_aspect,
                mouth_aspect=face.mouth_aspect,
                keypoints=kpts,
            )

            buf = student_histories[sid]
            buf.append(feat)

            if len(buf) >= args.window:
                seq = np.stack(list(buf)[-args.window:], axis=0).astype(np.float32)
                out_features.append(seq)
                out_student_ids.append(sid)
                out_frame_ids.append(frame_idx)
                out_targets.append(label_idx)
                out_bands.append(IDX_TO_BAND[label_idx])

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            print(f"Processed frame [{frame_idx}/{total_frames}] | Seqs: {len(out_features)} | Elapsed: {elapsed:.1f}s")

    cap.release()

    features_arr = np.stack(out_features).astype(np.float32)
    student_ids_arr = np.asarray(out_student_ids, dtype=np.int32)
    frame_ids_arr = np.asarray(out_frame_ids, dtype=np.int32)
    targets_raw_arr = np.asarray(out_targets, dtype=np.int32)
    bands_raw_arr = np.asarray(out_bands, dtype=str)

    # Authorized Binary Target Mapping:
    # 1 = ENGAGED (High: 0, Moderate: 1), 0 = DISENGAGED (Low: 2)
    targets_binary_arr = (targets_raw_arr <= 1).astype(np.int64)
    bands_binary_arr = np.where(targets_binary_arr == 1, "ENGAGED", "DISENGAGED")

    metadata = np.array([{
        "schema_version": "3.0.0_hardened",
        "source_video": str(video_path),
        "source_xml": str(xml_path),
        "source_csv": str(csv_path),
        "start_frame": args.start_frame,
        "num_frames": args.num_frames,
        "window": args.window,
        "feature_dim": FEAT_DIM,
        "head_roll_version": "v3_zero_centered",
        "channel_23_version": "phase6b_upper_body_confidence",
        "target_mapping": "0=DISENGAGED (Low), 1=ENGAGED (High+Moderate)",
    }], dtype=object)

    np.savez_compressed(
        out_path,
        features=features_arr,
        student_ids=student_ids_arr,
        frame_ids=frame_ids_arr,
        targets=targets_raw_arr,           # Backward-compatible 3-class target key
        bands=bands_raw_arr,               # Backward-compatible 3-class band key
        targets_raw=targets_raw_arr,       # Explicit raw 3-class target
        targets_binary=targets_binary_arr, # Explicit binary target
        bands_raw=bands_raw_arr,
        bands_binary=bands_binary_arr,
        metadata=metadata,
    )

    print(f"\nSuccessfully extracted {len(features_arr)} sequences to {out_path}!")
    print(f"Features shape: {features_arr.shape}")
    print(f"Targets shape:  {targets_raw_arr.shape}")
    print(f"Unique students: {np.unique(student_ids_arr).tolist()} (count={len(np.unique(student_ids_arr))})")
    print("\n[Raw 3-Class Ground Truth]:")
    for idx, name in IDX_TO_BAND.items():
        cnt = int((targets_raw_arr == idx).sum())
        print(f"  Class {idx} ({name}): {cnt} ({cnt/len(targets_raw_arr)*100:.1f}%)")
    print("\n[Proposed Binary Target]:")
    c0 = int((targets_binary_arr == 0).sum())
    c1 = int((targets_binary_arr == 1).sum())
    print(f"  Class 0 (DISENGAGED): {c0} ({c0/len(targets_binary_arr)*100:.1f}%)")
    print(f"  Class 1 (ENGAGED):    {c1} ({c1/len(targets_binary_arr)*100:.1f}%)")


if __name__ == "__main__":
    main()
