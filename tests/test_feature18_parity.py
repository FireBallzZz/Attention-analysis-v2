"""Regression test suite for Feature 18 (crop-level optical motion) parity.

Verifies mathematical alignment between live inference (AttentionPipeline.process_frame)
and training/benchmark sequence extraction (extract_classroom_sequences.py).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.detect_track import Detection


def _make_pipeline() -> AttentionPipeline:
    cfg = {
        "device": "cpu",
        "face": {"backend": "stub"},
        "scb": {"conf": 0.25},
    }
    return AttentionPipeline(cfg, load_weights=False)


def test_feature18_exact_parity():
    """Verify bit-for-bit parity between pipeline and extract_classroom_sequences.py formula."""
    pipe = _make_pipeline()
    rng = np.random.default_rng(42)

    # Synthetic 400x400 BGR frames
    frame1 = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
    frame2 = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)

    bbox = np.array([60.0, 50.0, 180.0, 210.0], dtype=np.float32)

    # 1. Compute reference motion using extract_classroom_sequences.py (lines 181-190)
    x1_i, y1_i = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2_i, y2_i = min(frame1.shape[1], int(bbox[2])), min(frame1.shape[0], int(bbox[3]))
    crop1 = frame1[y1_i:y2_i, x1_i:x2_i]
    crop2 = frame2[y1_i:y2_i, x1_i:x2_i]
    diff = cv2.absdiff(crop2, crop1)
    expected_motion = float(np.mean(diff) / 255.0)

    # 2. Inject detection into pipeline
    det = Detection(
        bbox_xyxy=bbox,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=10,
    )
    pipe.detector.infer = lambda f: [det]

    # Frame 1: Cold start
    res1 = pipe.process_frame(timestamp=0.0, frame=frame1)
    assert res1["students"][0]["motion"] == 0.0

    # Frame 2: Motion computed
    res2 = pipe.process_frame(timestamp=0.1, frame=frame2)
    computed_motion = res2["students"][0]["motion"]
    feature_18_val = pipe.tracks[10].feature_window[-1][18]

    # Parity assertions within numerical precision (1e-7)
    assert abs(computed_motion - expected_motion) < 1e-7
    assert abs(feature_18_val - expected_motion) < 1e-7
    assert abs(feature_18_val - computed_motion) < 1e-7


def test_feature18_first_frame_zero():
    """Verify that any new track ID outputs motion == 0.0000 on its first frame."""
    pipe = _make_pipeline()
    frame = np.full((300, 300, 3), 128, dtype=np.uint8)

    det_new = Detection(
        bbox_xyxy=np.array([20.0, 20.0, 120.0, 120.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=42,
    )
    pipe.detector.infer = lambda f: [det_new]

    res = pipe.process_frame(timestamp=0.0, frame=frame)
    assert res["students"][0]["motion"] == 0.0
    assert pipe.tracks[42].feature_window[-1][18] == 0.0


def test_feature18_shape_mismatch_fallback():
    """Verify safe fallback (0.0) when bounding box changes shape between frames."""
    pipe = _make_pipeline()
    frame1 = np.full((300, 300, 3), 50, dtype=np.uint8)
    frame2 = np.full((300, 300, 3), 150, dtype=np.uint8)
    frame3 = np.full((300, 300, 3), 200, dtype=np.uint8)

    # Frame 1: 100x100 box
    det1 = Detection(
        bbox_xyxy=np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=7,
    )
    pipe.detector.infer = lambda f: [det1]
    pipe.process_frame(timestamp=0.0, frame=frame1)

    # Frame 2: 101x100 box (1-pixel width fluctuation) -> must not throw OpenCV exception
    det2 = Detection(
        bbox_xyxy=np.array([50.0, 50.0, 151.0, 150.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=7,
    )
    pipe.detector.infer = lambda f: [det2]
    res2 = pipe.process_frame(timestamp=0.1, frame=frame2)

    assert res2["students"][0]["motion"] == 0.0
    assert pipe.tracks[7].feature_window[-1][18] == 0.0

    # Frame 3: Box dimensions remain 101x100, pixels change -> motion resumes safely
    det3 = Detection(
        bbox_xyxy=np.array([50.0, 50.0, 151.0, 150.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=7,
    )
    pipe.detector.infer = lambda f: [det3]
    res3 = pipe.process_frame(timestamp=0.2, frame=frame3)

    expected_diff = (200 - 150) / 255.0
    assert abs(res3["students"][0]["motion"] - expected_diff) < 1e-7


def test_feature18_frame_gap_resilience():
    """Verify that a temporal gap (> 0.5s) suppresses spurious cross-gap motion."""
    pipe = _make_pipeline()
    frame1 = np.full((300, 300, 3), 10, dtype=np.uint8)
    frame2 = np.full((300, 300, 3), 240, dtype=np.uint8)
    frame3 = np.full((300, 300, 3), 250, dtype=np.uint8)

    det = Detection(
        bbox_xyxy=np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=3,
    )
    pipe.detector.infer = lambda f: [det]

    # Frame 1 at t = 1.0s
    pipe.process_frame(timestamp=1.0, frame=frame1)

    # Frame 2 at t = 3.0s (2.0s gap, e.g. severe occlusion / tracking drop)
    # Drastic pixel difference (240 vs 10) must NOT produce a motion spike
    res2 = pipe.process_frame(timestamp=3.0, frame=frame2)
    assert res2["students"][0]["motion"] == 0.0
    assert pipe.tracks[3].feature_window[-1][18] == 0.0

    # Frame 3 at t = 3.1s (normal continuous 0.1s interval)
    res3 = pipe.process_frame(timestamp=3.1, frame=frame3)
    expected_diff = (250 - 240) / 255.0
    assert abs(res3["students"][0]["motion"] - expected_diff) < 1e-7


def test_feature18_cache_pruning():
    """Verify that stale tracks older than 5.0s are pruned from _prev_crops to prevent leaks."""
    pipe = _make_pipeline()
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)

    # Populate 55 ephemeral tracks at timestamp = 1.0s
    for tid in range(1, 56):
        det = Detection(
            bbox_xyxy=np.array([10.0, 10.0, 80.0, 80.0], dtype=np.float32),
            keypoints=np.zeros((17, 3), dtype=np.float32),
            conf=0.9,
            track_id=tid,
        )
        pipe.detector.infer = lambda f, d=det: [d]
        pipe.process_frame(timestamp=1.0, frame=frame)

    assert len(pipe._prev_crops) == 55

    # Advance time to t = 7.0s (> 5.0s gap) with a new active track ID 999
    det_active = Detection(
        bbox_xyxy=np.array([20.0, 20.0, 90.0, 90.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=999,
    )
    pipe.detector.infer = lambda f: [det_active]
    pipe.process_frame(timestamp=7.0, frame=frame)

    # All tracks from t=1.0s must be pruned; only track 999 should remain
    assert len(pipe._prev_crops) == 1
    assert 999 in pipe._prev_crops
    for tid in range(1, 56):
        assert tid not in pipe._prev_crops
