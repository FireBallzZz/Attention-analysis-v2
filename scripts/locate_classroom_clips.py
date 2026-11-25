"""Locate classroom clips inside the master recording.

PROBLEM
-------
`extract_classroom_sequences.py` rebuilds a classroom dataset from a source
video plus a frame offset. Several source videos referenced by the dataset
metadata no longer exist on disk (`runs/4.mp4`), but the master recording
`DEFENCE/classroom_01.mp4` (3355 frames, 832x464) contains every clip as a
contiguous 300-frame segment.

METHOD
------
Optical motion (feature channel 18) is a frame-to-frame derivative. Re-
extracting it from the master at a candidate offset and comparing against the
motion stored in an existing dataset reproduces the stored values exactly at
the correct offset, and not at any other.

This script is validated before use: it must recover the offsets already
recorded in the dataset metadata for clip_001 (master@0) and clip_008
(master@2100). Only then is it applied to clip_004, whose metadata is absent.

Usage:
    python scripts/locate_classroom_clips.py
    python scripts/locate_classroom_clips.py --clip clip_004
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

MASTER_CANDIDATES = [
    WORKSPACE_ROOT.parent.parent / "DEFENCE" / "classroom_01.mp4",
    WORKSPACE_ROOT / "runs" / "classroom_01.mp4",
]
ANNO_DIR = WORKSPACE_ROOT / "Anno"
DATA_DIR = WORKSPACE_ROOT / "data" / "processed"

WINDOW = list(range(120, 180))     # 60 consecutive frames used per candidate
COARSE_STEP = 10                  # coarse scan stride
FINE_RADIUS = 12                  # refinement window around the coarse peak

CLIPS = {
    "clip_001": {
        "npz": "classroom_clip_001_v3_hardened.npz",
        "xml": "clip_001_cvat_preannotated.xml",
        "expected_offset": 0,
    },
    "clip_004": {
        "npz": "classroom_clip_004_v3_hardened.npz",
        "xml": "annotations.xml",
        "expected_offset": None,
    },
    "clip_008": {
        "npz": "classroom_clip_008_v3_hardened.npz",
        "xml": "clip_008_cvat_preannotated.xml",
        "expected_offset": 2100,
    },
}


def resolve_master() -> Path:
    for candidate in MASTER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Master recording not found. Looked in: {MASTER_CANDIDATES}")


def parse_cvat_boxes(xml_path: Path) -> dict[int, dict[int, tuple]]:
    """frame -> {track_id: (x1, y1, x2, y2)} from a CVAT 1.1 export.

    The student id lives on the <track> element; <box> only carries the
    geometry and the frame number.
    """
    root = ET.parse(xml_path).getroot()
    boxes: dict[int, dict[int, tuple]] = {}
    for track in root.iter("track"):
        try:
            student_id = int(track.get("id"))
        except (TypeError, ValueError):
            continue
        for box in track.iter("box"):
            try:
                frame = int(box.get("frame"))
            except (TypeError, ValueError):
                continue
            try:
                geometry = (
                    float(box.get("xtl")), float(box.get("ytl")),
                    float(box.get("xbr")), float(box.get("ybr")),
                )
            except (TypeError, ValueError):
                continue
            boxes.setdefault(frame, {})[student_id] = geometry
    return boxes


def load_stored_motion(npz_path: Path) -> dict[tuple[int, int], float]:
    """(student_id, frame) -> stored channel-18 motion from the last timestep."""
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    student_ids = data["student_ids"].astype(int)
    frame_ids = data["frame_ids"].astype(int)
    return {
        (int(student), int(frame)): float(window[-1][18])
        for student, frame, window in zip(student_ids, frame_ids, features)
    }


class MotionMatcher:
    """Compares stored optical motion against motion re-extracted from video."""

    def __init__(self, video: Path, gt_boxes: dict[int, dict[int, tuple]],
                 stored: dict[tuple[int, int], float]) -> None:
        self.cap = cv2.VideoCapture(str(video))
        self.gt_boxes = gt_boxes
        self.stored = stored

    def close(self) -> None:
        self.cap.release()

    def score(self, offset: int) -> tuple[float, float]:
        """Return (mean absolute error, correlation) at this offset."""
        previous_crops: dict[int, np.ndarray] = {}
        pairs: list[tuple[float, float]] = []

        for frame in WINDOW:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, offset + frame)
            ok, image = self.cap.read()
            if not ok:
                continue
            for student_id, box in self.gt_boxes.get(frame, {}).items():
                x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
                x2 = min(image.shape[1], int(box[2]))
                y2 = min(image.shape[0], int(box[3]))
                crop = image[y1:y2, x1:x2]

                motion = 0.0
                if (student_id in previous_crops
                        and previous_crops[student_id].shape == crop.shape):
                    motion = float(np.mean(
                        cv2.absdiff(crop, previous_crops[student_id])) / 255.0)
                if crop.size > 0:
                    previous_crops[student_id] = crop.copy()

                reference = self.stored.get((student_id, frame))
                if reference is not None:
                    pairs.append((reference, motion))

        if not pairs:
            return 1.0, 0.0
        stored_values = np.array([p[0] for p in pairs], np.float64)
        video_values = np.array([p[1] for p in pairs], np.float64)
        error = float(np.mean(np.abs(stored_values - video_values)))
        if stored_values.std() < 1e-9 or video_values.std() < 1e-9:
            return error, 0.0
        correlation = float(np.corrcoef(stored_values, video_values)[0, 1])
        return error, correlation

    def find_offset(self, total_frames: int) -> tuple[int, float, float]:
        coarse: list[tuple[float, float, int]] = []
        for offset in range(0, total_frames - max(WINDOW), COARSE_STEP):
            error, correlation = self.score(offset)
            coarse.append((correlation, error, offset))
        coarse.sort(reverse=True)

        best = coarse[0]
        centre = best[2]
        fine: list[tuple[float, float, int]] = []
        for offset in range(max(0, centre - FINE_RADIUS),
                            min(total_frames - max(WINDOW), centre + FINE_RADIUS) + 1):
            error, correlation = self.score(offset)
            fine.append((correlation, error, offset))
        fine.sort(reverse=True)
        return fine[0][2], fine[0][1], fine[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=sorted(CLIPS), default=None,
                        help="Only locate one clip (default: all).")
    args = parser.parse_args()

    master = resolve_master()
    cap = cv2.VideoCapture(str(master))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 78)
    print("CLASSROOM CLIP LOCATOR")
    print("=" * 78)
    print(f"   master : {master}")
    print(f"   frames : {total_frames}")

    targets = [args.clip] if args.clip else list(CLIPS)
    results: dict[str, int] = {}

    # Validate on the clips whose offsets are already recorded in metadata.
    for name in ("clip_001", "clip_008"):
        if name not in targets:
            continue
        spec = CLIPS[name]
        matcher = MotionMatcher(
            master,
            parse_cvat_boxes(ANNO_DIR / spec["xml"]),
            load_stored_motion(DATA_DIR / spec["npz"]),
        )
        offset, error, correlation = matcher.find_offset(total_frames)
        matcher.close()
        expected = spec["expected_offset"]
        status = "MATCHES metadata" if offset == expected else f"MISMATCH (expected {expected})"
        print()
        print(f"   {name}: master@{offset}  err={error:.6f}  corr={correlation:.4f}  {status}")
        if offset != expected:
            raise SystemExit(
                f"Validation failed for {name}: located {offset}, expected {expected}. "
                "Refusing to report offsets from an unvalidated method.")

    # Only now locate the clip without metadata.
    for name in targets:
        if name in ("clip_001", "clip_008"):
            continue
        spec = CLIPS[name]
        matcher = MotionMatcher(
            master,
            parse_cvat_boxes(ANNO_DIR / spec["xml"]),
            load_stored_motion(DATA_DIR / spec["npz"]),
        )
        offset, error, correlation = matcher.find_offset(total_frames)
        matcher.close()
        print()
        print(f"   {name}: master@{offset}  err={error:.6f}  corr={correlation:.4f}")
        results[name] = offset

    print()
    print("=" * 78)
    print("OFFSET TABLE (use with extract_classroom_sequences.py --start-frame)")
    print("=" * 78)
    for name in ("clip_001", "clip_004", "clip_008"):
        offset = results.get(name, CLIPS[name]["expected_offset"])
        print(f"   {name}  --start-frame {offset}")


if __name__ == "__main__":
    main()
