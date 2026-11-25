"""Camera-independent head/gaze features.

Head pose and gaze are measured relative to the CAMERA, but attention is about
looking at the teacher / board, whose direction from the camera differs in
every room. That makes the raw cues flip sign between classrooms.

`relative_to_baseline` re-expresses head yaw/pitch and gaze yaw/pitch as the
deviation from each student's own typical direction (median over the clip, or
over a trailing window at deployment). Most students face the front most of
the time, so the median approximates the direction of the board. No labels
are used.

Channels changed (see attention_pipeline/features.py for the layout):
  0 head yaw, 1 head pitch, 3 gaze yaw, 4 gaze pitch  -> deviation from baseline
  5..8 (unused emotion slots)                          -> |deviation| of the four
  21 facing-front score                                -> 1 - |yaw deviation| / 45 deg
"""

from __future__ import annotations

import numpy as np

DIR_CH = (0, 1, 3, 4)


def baselines(X: np.ndarray, groups: np.ndarray) -> dict:
    """Median direction per group (student-in-video) over all frames of all its windows."""
    out = {}
    for g in np.unique(groups):
        frames = X[groups == g][:, :, DIR_CH].reshape(-1, len(DIR_CH))
        out[g] = np.median(frames, axis=0)
    return out


def relative_to_baseline(X: np.ndarray, groups: np.ndarray, base: dict | None = None) -> np.ndarray:
    base = base or baselines(X, groups)
    Y = X.copy()
    for g, b in base.items():
        m = groups == g
        dev = X[m][:, :, DIR_CH] - b
        sub = Y[m]
        sub[:, :, DIR_CH] = dev
        sub[:, :, 5:9] = np.abs(dev)
        sub[:, :, 21] = np.clip(1.0 - np.abs(dev[:, :, 0]) * 180.0 / 45.0, 0.0, 1.0)
        Y[m] = sub
    return Y
