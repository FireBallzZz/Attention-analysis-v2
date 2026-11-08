"""Extract hardened multimodal feature sequences for the controlled 750-clip DAiSEE development set.

Generates: data/processed/fusion_daisee_v3_hardened.npz
Uses the Phase 6B feature extraction pipeline with:
- Hardened torso lean with confidence gating (c >= 0.25) and shoulder-tilt fallback
- Hardened Channel 23 upper-body keypoint confidence (mean of nose, eyes, shoulders)
- Strictly preserved channels 24-31 (constant 0.0)

Keys:
  features:       (N, 16, 32) float32
  score:          (N,) float32 in [0, 100]
  aux:            (N, 4) float32 in [0, 3] (engagement, boredom, confusion, frustration)
  clip_ids:       (N,) str
  subject_ids:    (N,) str
  frame_indices:  (N, 16) int32 (sampled frame index of each step in window)
  clip:           (750,) str (unique sorted clips)
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path
import time
import numpy as np
import torch

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.source import iter_frames
from attention_pipeline.types import FEAT_DIM
from attention_pipeline.utils import resolve_daisee_root, PROJECT_ROOT


def read_daisee_labels() -> dict[str, dict[str, float]]:
    root = resolve_daisee_root()
    labels_dir = (root / "DAiSEE" / "Labels") if (root / "DAiSEE" / "Labels").exists() else (root / "Labels")
    names = ["AllLabels.csv", "TrainLabels.csv", "ValidationLabels.csv", "TestLabels.csv"]
    labels: dict[str, dict[str, float]] = {}
    for name in names:
        csv_path = labels_dir / name
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
                clip_id = str(row.get("ClipID", ""))
                if not clip_id:
                    continue
                stem = Path(clip_id).stem
                labels[stem] = {
                    "boredom": float(row.get("Boredom", 0)),
                    "engagement": float(row.get("Engagement", 0)),
                    "confusion": float(row.get("Confusion", 0)),
                    "frustration": float(row.get("Frustration", 0)),
                }
    return labels


def find_daisee_videos(root: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in list(root.rglob("*.avi")) + list(root.rglob("*.mp4"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-npz", default="data/processed/fusion_daisee_v2.npz")
    parser.add_argument("--out-npz", default="data/processed/fusion_daisee_v3_hardened.npz")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for testing")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--save-interval", type=int, default=25)
    args = parser.parse_args()

    src_path = PROJECT_ROOT / args.src_npz
    out_path = PROJECT_ROOT / args.out_npz
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source NPZ not found: {src_path}")

    src_data = np.load(src_path)
    target_clips = sorted(list(src_data["clip"]))
    if args.limit:
        target_clips = target_clips[: args.limit]
    print(f"Targeting {len(target_clips)} clips from {src_path.name}")

    root = resolve_daisee_root()
    video_map = find_daisee_videos(root)
    labels_map = read_daisee_labels()

    missing_vids = [c for c in target_clips if c not in video_map]
    if missing_vids:
        raise RuntimeError(f"Missing {len(missing_vids)} video files on disk, e.g., {missing_vids[:3]}")

    missing_labs = [c for c in target_clips if c not in labels_map]
    if missing_labs:
        raise RuntimeError(f"Missing labels for {len(missing_labs)} clips, e.g., {missing_labs[:3]}")

    # Resume capability
    done: set[str] = set()
    features: list[np.ndarray] = []
    scores: list[float] = []
    auxs: list[np.ndarray] = []
    clip_ids: list[str] = []
    subject_ids: list[str] = []
    frame_indices: list[np.ndarray] = []

    if out_path.exists():
        prev = np.load(out_path)
        done = set(str(x) for x in prev["clip"])
        print(f"Resuming: {out_path.name} already has {len(done)} clips completed")
        features = list(prev["features"])
        scores = list(prev["score"].astype(np.float32))
        auxs = list(prev["aux"].astype(np.float32))
        clip_ids = list(prev["clip_ids"].astype(str))
        subject_ids = list(prev["subject_ids"].astype(str))
        if "frame_indices" in prev:
            frame_indices = list(prev["frame_indices"].astype(np.int32))

    clips_to_process = [c for c in target_clips if c not in done]
    print(f"Clips remaining to process: {len(clips_to_process)} / {len(target_clips)}")

    if not clips_to_process:
        print(f"All {len(target_clips)} clips already present in {out_path.name}!")
        return

    pipe = AttentionPipeline.from_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml", load_weights=True)
    pipe.detector.tiles = False
    pipe.detector.imgsz = 640
    window = int(pipe.cfg.get("fusion", {}).get("window", 16))

    t_start = time.time()
    for idx, c_stem in enumerate(clips_to_process, start=1):
        vpath = video_map[c_stem]
        lab = labels_map[c_stem]

        # Verify subject ID from directory structure and clip stem
        subj_folder = vpath.parent.parent.name
        stem_subj = c_stem[:6] if len(c_stem) >= 6 else c_stem
        if subj_folder != stem_subj:
            print(f"Warning: Subject folder {subj_folder} differs from stem {stem_subj}")
        subj_id = subj_folder

        score = float(np.clip(lab["engagement"] / 3.0 * 100.0, 0, 100))
        aux = np.array(
            [lab["engagement"], lab["boredom"], lab["confusion"], lab["frustration"]],
            dtype=np.float32,
        )

        pipe.tracks.clear()
        pipe._prev_crops.clear()
        pipe._lesson_yaw = None
        produced = 0

        track_frame_buffers: dict[int, deque] = {}
        frame_seq_idx = 0

        for ts, frame in iter_frames(str(vpath), target_fps=args.fps):
            pipe.process_frame(ts, frame)
            for track_id, track in pipe.tracks.items():
                if track_id not in track_frame_buffers:
                    track_frame_buffers[track_id] = deque(maxlen=window)
                track_frame_buffers[track_id].append(frame_seq_idx)

                if len(track.feature_window) >= window:
                    seq = np.stack(list(track.feature_window)[-window:], axis=0).astype(np.float32)
                    f_idx_window = np.array(list(track_frame_buffers[track_id]), dtype=np.int32)
                    features.append(seq)
                    scores.append(score)
                    auxs.append(aux)
                    clip_ids.append(c_stem)
                    subject_ids.append(subj_id)
                    frame_indices.append(f_idx_window)
                    produced += 1
                break
            frame_seq_idx += 1

        if produced == 0:
            blank = np.zeros((window, FEAT_DIM), dtype=np.float32)
            features.append(blank)
            scores.append(score)
            auxs.append(aux)
            clip_ids.append(c_stem)
            subject_ids.append(subj_id)
            frame_indices.append(np.arange(window, dtype=np.int32))

        done.add(c_stem)

        if idx % args.save_interval == 0 or idx == len(clips_to_process):
            elapsed = time.time() - t_start
            rate = idx / elapsed
            remaining = (len(clips_to_process) - idx) / rate if rate > 0 else 0
            np.savez_compressed(
                out_path,
                features=np.stack(features).astype(np.float32),
                score=np.asarray(scores, dtype=np.float32),
                aux=np.stack(auxs).astype(np.float32),
                clip_ids=np.asarray(clip_ids, dtype=str),
                subject_ids=np.asarray(subject_ids, dtype=str),
                frame_indices=np.stack(frame_indices).astype(np.int32),
                clip=np.asarray(sorted(done), dtype=str),
            )
            print(
                f"Progress: [{len(done)}/{len(target_clips)}] clips "
                f"({len(features)} seqs) | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m"
            )

    print(f"Extraction complete! Saved {len(features)} sequences from {len(done)} clips to {out_path}")


if __name__ == "__main__":
    main()
