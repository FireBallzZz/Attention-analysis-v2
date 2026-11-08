"""Regression test suite for Phase 5C: Warmup Window Alignment and Temporal Integrity.

Verifies:
1. Live inference waits until 16 valid sequential observations are available before emitting predictions.
2. Frame 0 is NOT repeated/padded; early predictions are suppressed as 'WARMUP'.
3. The first valid prediction occurs at frame index 15 (the 16th frame), matching offline evaluation.
4. Temporal feature order and shape match offline sequence extraction exactly.
5. Large timestamp gaps (>0.5s), missing observations, and timestamp resets trigger safe track window resets.
6. Drop detector and heuristic fallback behaviors remain robust and backward-compatible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.attention import AttentionScorer
from attention_pipeline.stages.detect_track import Detection
from attention_pipeline.types import TrackState


def _make_pipeline_with_mock_fusion() -> AttentionPipeline:
    cfg = {
        "device": "cpu",
        "face": {"backend": "stub"},
        "scb": {"conf": 0.25},
        "fusion": {"window": 16, "binary_threshold": 0.50, "max_gap": 0.5},
    }
    pipe = AttentionPipeline(cfg, load_weights=False)
    # Enable fusion scoring using initialized AttentionFusionNet model
    pipe.scorer.use_heuristic = False
    return pipe


def test_warmup_suppression_and_first_valid_prediction():
    """Verify that frames 0..14 emit 'WARMUP' and the first valid prediction occurs at frame index 15."""
    pipe = _make_pipeline_with_mock_fusion()
    frame = np.full((300, 300, 3), 128, dtype=np.uint8)

    det = Detection(
        bbox_xyxy=np.array([50.0, 50.0, 150.0, 150.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=1,
    )
    pipe.detector.infer = lambda f: [det]

    # Frames 0 through 14 (first 15 frames): Must be WARMUP
    for frame_idx in range(15):
        ts = frame_idx * 0.1  # 10 FPS
        res = pipe.process_frame(timestamp=ts, frame=frame)
        student = res["students"][0]

        assert student["band"] == "WARMUP", f"Frame {frame_idx} emitted {student['band']} instead of WARMUP"
        assert student["score"] is None, f"Frame {frame_idx} produced early score {student['score']}"
        assert student["fusion_score"] is None, f"Frame {frame_idx} produced early fusion_score"
        assert student["cue"] == "warmup"
        assert len(pipe.tracks[1].feature_window) == frame_idx + 1

    # Frame 15 (16th frame): Complete sequence available -> FIRST valid prediction
    res15 = pipe.process_frame(timestamp=1.5, frame=frame)
    student15 = res15["students"][0]

    assert student15["band"] in ("HIGH", "LOW"), f"Frame 15 failed to emit valid prediction: {student15['band']}"
    assert student15["score"] is not None
    assert isinstance(student15["score"], float)
    assert student15["fusion_score"] is not None
    assert isinstance(student15["fusion_score"], float)
    assert len(pipe.tracks[1].feature_window) == 16


def test_temporal_feature_order_exact_parity():
    """Verify that temporal window construction in AttentionScorer matches offline extraction sequence."""
    model = AttentionFusionNet(feat_dim=32, hidden=64)
    model.eval()
    scorer = AttentionScorer(model=model, device=torch.device("cpu"), window=16)
    scorer.use_heuristic = False

    track = TrackState(student_id=5)

    # 1. Simulate 20 unique observations with identifiable feature patterns
    raw_features = []
    for step in range(20):
        # Unique vector for each step
        feat = np.full(32, float(step + 1), dtype=np.float32)
        raw_features.append(feat)

    # Offline extraction logic from scripts/extract_classroom_sequences.py
    # buf = deque(maxlen=16); buf.append(feat); seq = np.stack(list(buf)[-16:], axis=0)
    offline_buf = []
    for step in range(20):
        offline_buf.append(raw_features[step])
        scorer.push(track, raw_features[step], timestamp=step * 0.1)

    # Compare 16-frame window at step 19 (after 20 steps)
    offline_window = np.stack(offline_buf[-16:], axis=0)
    live_window = np.stack(track.feature_window[-16:], axis=0)

    # Both must match bit-for-bit in shape and chronological order
    assert offline_window.shape == (16, 32)
    assert live_window.shape == (16, 32)
    assert np.array_equal(offline_window, live_window)

    # Slices: index 0 must be step 4 (value 5.0), index 15 must be step 19 (value 20.0)
    assert live_window[0, 0] == 5.0
    assert live_window[15, 0] == 20.0


def test_track_reset_on_timestamp_gap():
    """Verify that a temporal gap (> 0.5s) resets the sequence window and forces fresh warmup."""
    pipe = _make_pipeline_with_mock_fusion()
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)

    det = Detection(
        bbox_xyxy=np.array([20.0, 20.0, 100.0, 100.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=8,
    )
    pipe.detector.infer = lambda f: [det]

    # Warm up student for 16 frames: t = 0.0 to 1.5s
    for i in range(16):
        res = pipe.process_frame(timestamp=i * 0.1, frame=frame)

    assert res["students"][0]["band"] in ("HIGH", "LOW")
    assert len(pipe.tracks[8].feature_window) == 16

    # Temporal occlusion gap of 2.0s: next frame arrives at t = 3.5s (gap = 2.0s > 0.5s)
    res_gap = pipe.process_frame(timestamp=3.5, frame=frame)
    student_gap = res_gap["students"][0]

    # Continuity is broken: must reset and re-enter WARMUP
    assert student_gap["band"] == "WARMUP"
    assert student_gap["score"] is None
    assert len(pipe.tracks[8].feature_window) == 1

    # Feed 14 more frames (total 15 frames since gap): t = 3.6 to 4.9s -> still in WARMUP
    for i in range(1, 15):
        res_warm = pipe.process_frame(timestamp=3.5 + i * 0.1, frame=frame)
        assert res_warm["students"][0]["band"] == "WARMUP"

    # 16th frame since gap: t = 5.0s -> valid prediction resumes
    res_recov = pipe.process_frame(timestamp=5.0, frame=frame)
    assert res_recov["students"][0]["band"] in ("HIGH", "LOW")
    assert res_recov["students"][0]["score"] is not None
    assert len(pipe.tracks[8].feature_window) == 16


def test_backward_timestamp_or_tracking_reuse_reset():
    """Verify that out-of-order timestamps or tracking ID reuse safely reset window."""
    model = AttentionFusionNet(feat_dim=32, hidden=64)
    model.eval()
    scorer = AttentionScorer(model=model, device=torch.device("cpu"), window=16, max_gap=0.5)
    scorer.use_heuristic = False

    track = TrackState(student_id=3)
    feat = np.zeros(32, dtype=np.float32)

    # Push 16 frames at t = 10.0 .. 11.5
    for i in range(16):
        score, rubric, fs = scorer.push(track, feat, timestamp=10.0 + i * 0.1)

    assert fs is not None
    assert len(track.feature_window) == 16

    # Video loops or tracking ID reused: next timestamp jumps backwards to t = 1.0s
    score_reset, rubric_reset, fs_reset = scorer.push(track, feat, timestamp=1.0)
    assert fs_reset is None
    assert score_reset is None
    assert len(track.feature_window) == 1


def test_heuristic_fallback_unaffected_by_warmup():
    """Verify that pure heuristic mode (no neural model) produces immediate rubric predictions."""
    cfg = {
        "device": "cpu",
        "face": {"backend": "stub"},
        "scb": {"conf": 0.25},
    }
    pipe = AttentionPipeline(cfg, load_weights=False)
    # With load_weights=False, use_heuristic is True
    assert pipe.scorer.use_heuristic is True

    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    det = Detection(
        bbox_xyxy=np.array([20.0, 20.0, 100.0, 100.0], dtype=np.float32),
        keypoints=np.zeros((17, 3), dtype=np.float32),
        conf=0.9,
        track_id=1,
    )
    pipe.detector.infer = lambda f: [det]

    # Frame 0 should immediately produce rubric band without waiting for 16 frames
    res0 = pipe.process_frame(timestamp=0.0, frame=frame)
    student0 = res0["students"][0]
    assert student0["band"] in ("HIGH", "MODERATE", "LOW")
    assert student0["score"] is not None
    assert student0["fusion_score"] is None
