"""Unit test for Phase 6G-4B head roll calculation and continuity."""

from __future__ import annotations

import numpy as np
import pytest
from attention_pipeline.stages.modalities import FaceModalityExtractor


def test_upright_face_zero_roll():
    """Level face with leye and reye at same height must yield roll ~ 0.0 deg."""
    # OpenCV image coords: Origin top-left.
    # Anatomical Left Eye (kpts[1]): x=110, y=100 (camera-right)
    # Anatomical Right Eye (kpts[2]): x=90, y=100 (camera-left)
    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [100.0, 110.0, 0.9]  # nose
    kpts[1] = [110.0, 100.0, 0.9]  # leye
    kpts[2] = [90.0, 100.0, 0.9]   # reye
    kpts[3] = [120.0, 100.0, 0.9]  # lear
    kpts[4] = [80.0, 100.0, 0.9]   # rear

    pose = extractor._head_pose(landmarks=None, kpts=kpts)
    yaw, pitch, roll = pose[0], pose[1], pose[2]

    assert abs(roll) < 1e-4, f"Expected 0.0 deg for level face, got {roll}"
    # Normalized in features.py: roll / 180.0
    norm_roll = roll / 180.0
    assert abs(norm_roll) < 1e-6


def test_micro_jitter_continuity():
    """Sub-pixel jitter (+/- 0.01 px) must produce continuous, tiny variation (< 0.05 deg)."""
    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    kpts_plus = np.zeros((17, 3), dtype=np.float32)
    kpts_plus[0] = [100.0, 110.0, 0.9]
    kpts_plus[1] = [110.0, 100.01, 0.9]  # leye y = 100.01
    kpts_plus[2] = [90.0, 100.00, 0.9]   # reye y = 100.00
    kpts_plus[3] = [120.0, 100.0, 0.9]
    kpts_plus[4] = [80.0, 100.0, 0.9]

    kpts_minus = np.zeros((17, 3), dtype=np.float32)
    kpts_minus[0] = [100.0, 110.0, 0.9]
    kpts_minus[1] = [110.0, 99.99, 0.9]  # leye y = 99.99
    kpts_minus[2] = [90.0, 100.00, 0.9]  # reye y = 100.00
    kpts_minus[3] = [120.0, 100.0, 0.9]
    kpts_minus[4] = [80.0, 100.0, 0.9]

    pose_plus = extractor._head_pose(landmarks=None, kpts=kpts_plus)
    pose_minus = extractor._head_pose(landmarks=None, kpts=kpts_minus)

    roll_plus = pose_plus[2]
    roll_minus = pose_minus[2]

    # In the flawed implementation, this difference was ~360 deg (1.9996 normalized)
    delta = abs(roll_plus - roll_minus)
    assert delta < 0.1, f"Expected continuous delta < 0.1 deg, got {delta} deg"


def test_directional_tilt():
    """Verify left and right tilt produce expected sign."""
    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    # Tilt left: leye is lower (higher y in image)
    kpts_left = np.zeros((17, 3), dtype=np.float32)
    kpts_left[0] = [100.0, 110.0, 0.9]
    kpts_left[1] = [110.0, 105.0, 0.9]  # leye lower by 5 px
    kpts_left[2] = [90.0, 95.0, 0.9]    # reye higher by 5 px
    kpts_left[3] = [120.0, 105.0, 0.9]
    kpts_left[4] = [80.0, 95.0, 0.9]

    pose_left = extractor._head_pose(landmarks=None, kpts=kpts_left)
    assert pose_left[2] > 0.0, "Tilt left should produce positive roll"

    # Tilt right: leye is higher (lower y in image)
    kpts_right = np.zeros((17, 3), dtype=np.float32)
    kpts_right[0] = [100.0, 110.0, 0.9]
    kpts_right[1] = [110.0, 95.0, 0.9]
    kpts_right[2] = [90.0, 105.0, 0.9]
    kpts_right[3] = [120.0, 95.0, 0.9]
    kpts_right[4] = [80.0, 105.0, 0.9]

    pose_right = extractor._head_pose(landmarks=None, kpts=kpts_right)
    assert pose_right[2] < 0.0, "Tilt right should produce negative roll"


def test_missing_landmarks_safety():
    """Ensure missing keypoints return safe zero vector."""
    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    pose_none = extractor._head_pose(landmarks=None, kpts=None)
    assert np.all(pose_none == 0.0)

    pose_short = extractor._head_pose(landmarks=None, kpts=np.zeros((3, 3), dtype=np.float32))
    assert np.all(pose_short == 0.0)
