"""Real-time Progress Monitor for Phase 6 Full DAiSEE Extraction."""

from __future__ import annotations

import datetime
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

TARGET_CLIPS = 8986
NPZ_PATH = WORKSPACE_ROOT / "data" / "processed" / "fusion_daisee_full_v4.npz"
START_TIME = datetime.datetime(2026, 10, 23, 15, 49, 32)


def main() -> None:
    now = datetime.datetime.now()
    elapsed_sec = (now - START_TIME).total_seconds()

    if not NPZ_PATH.exists():
        print(f"Archive not found: {NPZ_PATH}")
        return

    with np.load(NPZ_PATH) as d:
        completed = len(d["clip"])
        seqs = len(d["features"])
        splits = d["splits"]
        u_splits, s_counts = np.unique(splits, return_counts=True)
        size_mb = NPZ_PATH.stat().st_size / (1024**2)
        mtime = datetime.datetime.fromtimestamp(NPZ_PATH.stat().st_mtime)

    pct = (completed / TARGET_CLIPS) * 100.0
    rate = completed / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining_clips = TARGET_CLIPS - completed
    eta_sec = remaining_clips / rate if rate > 0 else 0.0
    eta_time = now + datetime.timedelta(seconds=eta_sec)

    bar_len = 35
    filled = int(bar_len * (completed / TARGET_CLIPS))
    bar = "=" * filled + "-" * (bar_len - filled)

    print("=" * 70)
    print(" LIVE EXTRACTION PROGRESS (fusion_daisee_full_v4.npz)")
    print("=" * 70)
    print(f"Progress Status   : [{bar}] {pct:.2f}%")
    print(f"Completed Clips   : {completed:,} / {TARGET_CLIPS:,} clips")
    print(f"Remaining Clips   : {remaining_clips:,} clips")
    print(f"Extracted Seqs    : {seqs:,} sequences (shape: {seqs:,} x 16 x 32)")
    print(f"Current File Size : {size_mb:.2f} MB")
    print(f"Last Saved Disk   : {mtime.strftime('%Y-%m-%d %I:%M:%S %p')}")
    print("-" * 70)
    print(f"Elapsed Time      : {elapsed_sec/60:.1f} minutes ({elapsed_sec/3600:.2f} hours)")
    print(f"Processing Speed  : {rate:.2f} clips/second ({rate*60:.1f} clips/minute)")
    print(f"Estimated Time Left: {eta_sec/3600:.2f} hours (Estimated Completion: ~{eta_time.strftime('%I:%M %p')})")
    print("-" * 70)
    print("Partition Sequence Counts:")
    for s, c in zip(u_splits, s_counts):
        print(f"  * {s:10s} : {c:,} sequences")
    print("=" * 70)


if __name__ == "__main__":
    main()
