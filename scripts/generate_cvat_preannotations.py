"""Generate CVAT for video 1.1 pre-annotation XML for a new classroom video clip.

Uses YOLO detector + ByteTrack + frozen neural classifier
(checkpoints/classroom_attention_head_focal_v2.pt) to produce high-quality
candidate bounding box tracks and initial attention attributes.

Output can be uploaded directly to CVAT via:
  Task Actions -> Upload Annotations -> CVAT for video 1.1 (XML).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch

import sys
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.pipeline import AttentionPipeline
from train_classroom_head import ClassroomAttentionClassifier

IDX_TO_CVAT_BAND = {0: "High", 1: "Moderate", 2: "Low"}


def main():
    parser = argparse.ArgumentParser(description="Generate CVAT 1.1 XML pre-annotations")
    parser.add_argument("video", help="Path to input video clip (e.g., runs/Test 2.mp4 or runs/clip_008.mp4)")
    parser.add_argument("--out-xml", default="", help="Output CVAT XML path (default: Anno/<video_stem>_preannotated.xml)")
    parser.add_argument("--student-count", type=int, default=21, help="Locked student count")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--ckpt", default="checkpoints/classroom_attention_head_focal_v2.pt")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = WORKSPACE_ROOT / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_xml = Path(args.out_xml) if args.out_xml else WORKSPACE_ROOT / "Anno" / f"{video_path.stem}_cvat_preannotated.xml"
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading neural classifier from: {args.ckpt} on {device}")
    ckpt = torch.load(WORKSPACE_ROOT / args.ckpt, map_location=device, weights_only=False)
    fusion_base = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ClassroomAttentionClassifier(pretrained_fusion=fusion_base, num_classes=3, freeze_encoder=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"Initializing pipeline from {args.config}...")
    pipe = AttentionPipeline.from_yaml(WORKSPACE_ROOT / args.config, load_weights=True)
    pipe.detector.max_people = args.student_count

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    print(f"Processing video: {video_path.name} ({width}x{height}, {fps:.2f} fps, {total_frames} frames)")

    # Store tracks: track_id -> dict of frame -> (bbox, band_str)
    tracks_data: dict[int, dict[int, tuple[list[float], str]]] = defaultdict(dict)
    frame_idx = 0
    window = 16

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        result = pipe.process_frame(ts, frame)
        students = result["students"]

        for s in students:
            track_id = int(s["student_id"])
            bbox = [float(v) for v in s["bbox"]]
            track = pipe.tracks.get(track_id)

            pred_band = "High"
            if track is not None and len(track.feature_window) >= window:
                seq = np.stack(list(track.feature_window)[-window:], axis=0).astype(np.float32)
                with torch.no_grad():
                    inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = model(inp).squeeze(0)
                    pred_idx = int(torch.argmax(logits).item())
                    pred_band = IDX_TO_CVAT_BAND[pred_idx]
            else:
                # During warm-up, use heuristic band
                pred_band = str(s.get("band", "High")).capitalize()
                if pred_band not in ["High", "Moderate", "Low"]:
                    pred_band = "High"

            tracks_data[track_id][frame_idx] = (bbox, pred_band)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} processed...")

    cap.release()
    print(f"Tracking complete. Found {len(tracks_data)} raw tracks across {frame_idx} frames.")

    # Keep top N tracks with most frame presence
    sorted_tracks = sorted(tracks_data.items(), key=lambda kv: len(kv[1]), reverse=True)
    kept_tracks = sorted_tracks[: args.student_count]
    print(f"Locked top {len(kept_tracks)} tracks for CVAT export.")

    # Build CVAT XML 1.1 tree
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = video_path.stem
    ET.SubElement(task, "size").text = str(total_frames)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(total_frames - 1)
    ET.SubElement(task, "created").text = datetime.now(timezone.utc).isoformat()
    ET.SubElement(task, "updated").text = datetime.now(timezone.utc).isoformat()

    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)

    labels = ET.SubElement(task, "labels")
    label_el = ET.SubElement(labels, "label")
    ET.SubElement(label_el, "name").text = "Students"
    ET.SubElement(label_el, "color").text = "#66ff66"
    ET.SubElement(label_el, "type").text = "rectangle"

    attributes = ET.SubElement(label_el, "attributes")
    attr_el = ET.SubElement(attributes, "attribute")
    ET.SubElement(attr_el, "name").text = "Attention"
    ET.SubElement(attr_el, "mutable").text = "True"
    ET.SubElement(attr_el, "input_type").text = "radio"
    ET.SubElement(attr_el, "default_value").text = "High"
    ET.SubElement(attr_el, "values").text = "High\nLow\nModerate"

    # Export tracks (re-indexed 0 to student_count - 1)
    for new_id, (_orig_id, frame_boxes) in enumerate(kept_tracks):
        track_el = ET.SubElement(root, "track", id=str(new_id), label="Students", source="semi-auto")
        
        # Sort frames
        for f in range(total_frames):
            if f in frame_boxes:
                bbox, band = frame_boxes[f]
                box_el = ET.SubElement(
                    track_el,
                    "box",
                    frame=str(f),
                    keyframe="1" if f % 15 == 0 else "0",
                    outside="0",
                    occluded="0",
                    xtl=f"{bbox[0]:.2f}",
                    ytl=f"{bbox[1]:.2f}",
                    xbr=f"{bbox[2]:.2f}",
                    ybr=f"{bbox[3]:.2f}",
                    z_order="0",
                )
                att_el = ET.SubElement(box_el, "attribute", name="Attention")
                att_el.text = band
            else:
                # Outside / out of frame
                box_el = ET.SubElement(
                    track_el,
                    "box",
                    frame=str(f),
                    keyframe="1",
                    outside="1",
                    occluded="0",
                    xtl="0.00",
                    ytl="0.00",
                    xbr="1.00",
                    ybr="1.00",
                    z_order="0",
                )
                att_el = ET.SubElement(box_el, "attribute", name="Attention")
                att_el.text = "High"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    print(f"\nSuccessfully wrote CVAT 1.1 pre-annotation XML to: {out_xml.resolve()}")
    print("Ready for upload to CVAT!")


if __name__ == "__main__":
    main()
