"""Extract multimodal feature sequences from DAiSEE master manifest.

Version: v4 (Production Rebuild)
Features:
- Reads directly from verified manifest (e.g. data/manifests/daisee_master_manifest_v1.csv)
- Fully resumable from last saved checkpoint
- Atomic saving via .npz.tmp -> .replace() to prevent archive corruption on interruption
- Preserves Phase 6B/6C hardened features (torso lean shoulder fallback, upper body confidence)
- Preserves 32-D feature schema and records explicit splits and binary targets

Keys in output NPZ:
  features:        (N, 16, 32) float32
  score:           (N,) float32 in [0, 100]
  aux:             (N, 4) float32 in [0, 3] (engagement, boredom, confusion, frustration)
  clip_ids:        (N,) str
  subject_ids:     (N,) str
  splits:          (N,) str ('Train', 'Validation', 'Test')
  binary_targets:  (N,) int64 (0 = NON-ATTENTIVE, 1 = ATTENTIVE)
  frame_indices:   (N, 16) int32
  clip:            (M,) str (unique completed clip stems)
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.source import iter_frames
from attention_pipeline.types import FEAT_DIM
from attention_pipeline.utils import resolve_daisee_root, PROJECT_ROOT


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAiSEE feature sequences from master manifest")
    parser.add_argument("--manifest", default="data/manifests/daisee_master_manifest_v1.csv", help="Path to manifest CSV")
    parser.add_argument("--out-npz", default="data/processed/fusion_daisee_full_v4.npz", help="Destination NPZ file")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for testing/smoke test")
    parser.add_argument("--split", type=str, default=None, choices=["Train", "Validation", "Test"], help="Filter by split")
    parser.add_argument("--fps", type=float, default=5.0, help="Sampling frame rate")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval in clips")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    out_path = PROJECT_ROOT / args.out_npz
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Targeting {len(rows)} clips from {manifest_path.name}")

    daisee_root = resolve_daisee_root()

    # Resume capability
    done: set[str] = set()
    features: list[np.ndarray] = []
    scores: list[float] = []
    auxs: list[np.ndarray] = []
    clip_ids: list[str] = []
    subject_ids: list[str] = []
    splits: list[str] = []
    binary_targets: list[int] = []
    frame_indices: list[np.ndarray] = []

    if out_path.exists():
        with np.load(out_path) as prev:
            done = set(str(x) for x in prev["clip"])
            print(f"Resuming: {out_path.name} already has {len(done)} clips completed")
            features = list(prev["features"])
            scores = list(prev["score"].astype(np.float32))
            auxs = list(prev["aux"].astype(np.float32))
            clip_ids = list(prev["clip_ids"].astype(str))
            subject_ids = list(prev["subject_ids"].astype(str))
            if "splits" in prev:
                splits = list(prev["splits"].astype(str))
            else:
                # Fallback reconstruction from manifest
                row_map = {r["clip_id"]: r["split"] for r in rows}
                splits = [row_map.get(c, "Train") for c in clip_ids]
            if "binary_targets" in prev:
                binary_targets = list(prev["binary_targets"].astype(np.int64))
            else:
                binary_targets = [0 if a[0] < 2.0 else 1 for a in auxs]
            if "frame_indices" in prev:
                frame_indices = list(prev["frame_indices"].astype(np.int32))

    clips_to_process = [r for r in rows if r["clip_id"] not in done]
    print(f"Clips remaining to process: {len(clips_to_process)} / {len(rows)}")

    if not clips_to_process:
        print(f"All {len(rows)} target clips already present in {out_path.name}!")
        return

    pipe = AttentionPipeline.from_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml", load_weights=True)
    pipe.detector.tiles = False
    pipe.detector.imgsz = 640
    window = int(pipe.cfg.get("fusion", {}).get("window", 16))

    t_start = time.time()
    for idx, r in enumerate(clips_to_process, start=1):
        c_stem = r["clip_id"]
        vpath = daisee_root / r["video_path"]
        if not vpath.exists():
            # Try finding video file if relative path doesn't resolve directly
            matches = list(daisee_root.rglob(f"{c_stem}.*"))
            if matches:
                vpath = matches[0]
            else:
                raise FileNotFoundError(f"Video file missing: {r['video_path']}")

        subj_id = r["subject_id"]
        split_name = r["split"]
        eng = float(r["engagement"])
        score = float(np.clip(eng / 3.0 * 100.0, 0, 100))
        b_target = int(r["binary_target"])
        aux = np.array(
            [eng, float(r["boredom"]), float(r["confusion"]), float(r["frustration"])],
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
                    splits.append(split_name)
                    binary_targets.append(b_target)
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
            splits.append(split_name)
            binary_targets.append(b_target)
            frame_indices.append(np.arange(window, dtype=np.int32))

        done.add(c_stem)

        if idx % args.save_interval == 0 or idx == len(clips_to_process):
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(clips_to_process) - idx) / rate if rate > 0 else 0

            # Atomic save: write to temporary file, then replace
            tmp_path = out_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp_path,
                features=np.stack(features).astype(np.float32),
                score=np.asarray(scores, dtype=np.float32),
                aux=np.stack(auxs, dtype=np.float32),
                clip_ids=np.asarray(clip_ids, dtype=str),
                subject_ids=np.asarray(subject_ids, dtype=str),
                splits=np.asarray(splits, dtype=str),
                binary_targets=np.asarray(binary_targets, dtype=np.int64),
                frame_indices=np.stack(frame_indices).astype(np.int32),
                clip=np.asarray(sorted(done), dtype=str),
            )
            tmp_path.replace(out_path)

            print(
                f"Progress: [{len(done)}/{len(rows)}] clips "
                f"({len(features)} seqs) | Rate: {rate:.2f} clips/s | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m"
            )

    # Save sidecar metadata JSON
    sidecar_path = out_path.with_suffix(".json")
    meta = {
        "out_path": str(out_path),
        "manifest_path": str(manifest_path),
        "total_clips_completed": len(done),
        "total_sequences_generated": len(features),
        "feature_dim": FEAT_DIM,
        "window_size": window,
        "sampling_fps": args.fps,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete! Saved {len(features)} sequences from {len(done)} clips to {out_path}")
    print(f"Sidecar metadata saved to {sidecar_path}")


if __name__ == "__main__":
    main()
