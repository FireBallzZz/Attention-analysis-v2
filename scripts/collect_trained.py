"""Put every sample that was already trained into data/processed/trained.

Running this again does not duplicate a clip or image. Later training scripts
add new samples to the same folder.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.trained_store import (  # noqa: E402
    add_daisee_clips,
    add_l2cs_npz,
    add_scb_images,
    counts,
)

SCB_SRC = (
    ROOT
    / "data"
    / "raw"
    / "scb"
    / "SCB5-Handrise-Read-write"
    / "SCB5-Handrise-Read-write-2024-9-17"
)
GAZE_NPZ = ROOT / "data" / "processed" / "extra20" / "gaze_l2cs_20.npz"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    images = sorted((SCB_SRC / "images" / "train").glob("*.jpg"))
    if len(images) < 40:
        raise SystemExit(f"Need the SCB training images under {SCB_SRC}")
    september_scb = random.Random(1).sample(images, 40)
    october_scb = random.Random(20).sample(images, 20)
    add_scb_images(september_scb, source="benchmark 2026-09-22 seed 1")
    add_scb_images(october_scb, source="extra20 2026-10-04 seed 20")

    benchmark = _load("benchmark_smoke", ROOT / "scripts" / "benchmark_smoke.py")
    smoke = _load("smoke_test_train", ROOT / "scripts" / "smoke_test_train.py")
    september_clips = [path for path, _lab in benchmark._pick_daisee_clips(20, 1)]
    october_clips = [path for path, _lab in smoke._pick_daisee_clips(20, 20)]
    add_daisee_clips(september_clips, source="benchmark 2026-09-22 seed 1")
    add_daisee_clips(october_clips, source="extra20 2026-10-04 seed 20")

    if not GAZE_NPZ.exists():
        raise SystemExit(f"Missing L2CS training faces: {GAZE_NPZ}")
    add_l2cs_npz(GAZE_NPZ, source="extra20 2026-10-04")

    stored = counts()
    print(f"Trained data: {ROOT / 'data' / 'processed' / 'trained'}")
    print(f"  SCB images : {stored['scb']}")
    print(f"  DAiSEE clips: {stored['daisee']}")
    print(f"  L2CS faces : {stored['l2cs']}")


if __name__ == "__main__":
    main()
