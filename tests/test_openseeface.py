"""OpenSeeFace pose conversion. The tracker weights are optional."""

from __future__ import annotations

import numpy as np

from attention_pipeline.stages.openseeface_backend import (
    eye_mouth_aspects,
    five_points,
    head_pose_from_euler,
    head_pose_from_landmarks,
    image_landmarks,
)


def test_euler_is_yaw_pitch_roll() -> None:
    pose = head_pose_from_euler([12.0, -30.0, 4.0])
    assert pose.shape == (3,)
    assert abs(float(pose[0]) - (-30.0)) < 1e-5
    assert abs(float(pose[1]) - 12.0) < 1e-5
    assert abs(float(pose[2]) - 4.0) < 1e-5


def test_landmarks_swap_y_x_into_image_xy() -> None:
    raw = np.zeros((66, 3), np.float32)
    raw[:, 0] = 10  # OpenSeeFace y
    raw[:, 1] = 20  # OpenSeeFace x
    xy = image_landmarks(raw)
    assert xy.shape == (66, 2)
    assert abs(float(xy[0, 0]) - 20.0) < 1e-5
    assert abs(float(xy[0, 1]) - 10.0) < 1e-5


def test_landmark_pose_follows_the_nose() -> None:
    raw = np.zeros((66, 3), np.float32)

    def put(index: int, x: float, y: float) -> None:
        raw[index, 0] = y
        raw[index, 1] = x

    for offset, index in enumerate(range(36, 42)):
        put(index, 10 + offset, 20)
    for offset, index in enumerate(range(42, 48)):
        put(index, 30 + offset, 20)
    put(30, 40, 28)
    yaw, pitch, _roll = head_pose_from_landmarks(raw)
    assert float(yaw) > 0
    assert float(pitch) > 0


def test_eye_and_mouth_aspects_are_in_range() -> None:
    pts = np.zeros((66, 2), np.float32)

    def put(index: int, x: float, y: float) -> None:
        pts[index] = [y, x]

    put(36, 0, 0)
    put(39, 20, 0)
    put(37, 6, -4)
    put(38, 14, -4)
    put(41, 6, 4)
    put(40, 14, 4)
    put(42, 40, 0)
    put(45, 60, 0)
    put(43, 46, -4)
    put(44, 54, -4)
    put(47, 46, 4)
    put(46, 54, 4)
    put(48, 10, 30)
    put(54, 40, 30)
    for index, x in ((59, 22), (60, 25), (61, 28)):
        put(index, x, 26)
    for index, x in ((63, 22), (64, 25), (65, 28)):
        put(index, x, 34)
    raw = np.concatenate([pts, np.ones((66, 1), np.float32)], axis=1)
    eye, mouth = eye_mouth_aspects(raw, eye_blink=[0.8, 0.6])
    assert abs(eye - 0.7) < 1e-5
    assert 0.0 < mouth < 1.0
    five = five_points(raw)
    assert five.shape == (5, 2)
