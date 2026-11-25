"""Box motion: the attention cue that transfers between classrooms.

Measured the same way in evaluation (scripts/motion_rule.py) and in the live
pipeline: mean absolute pixel change inside the student's CURRENT box between
two frames 0.1 s apart (the pipeline's 10 fps step), on a 64x64 grey crop.
Using the same box on both frames avoids the zero readings that box jitter
caused in the older feature.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SIZE = 64


def box_motion(prev_frame: np.ndarray | None, frame: np.ndarray, box) -> float | None:
    if prev_frame is None or prev_frame.shape != frame.shape:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    a = cv2.cvtColor(cv2.resize(prev_frame[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(a, b)) / 255.0)


class MotionRule:
    """P(HIGH) = sigmoid(a + b * z), z = log(mean motion over the window) minus the
    median of that quantity over the same class (video). Fitted by scripts/motion_rule.py.
    The decision threshold (chosen on held-out classrooms) maps to score 50."""

    def __init__(self, a: float, b: float, threshold: float, window: int = 16, **_: object) -> None:
        self.a, self.b, self.threshold, self.window = float(a), float(b), float(threshold), int(window)

    @classmethod
    def load(cls, path) -> "MotionRule":
        import json
        from pathlib import Path

        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def window_log(motions) -> float:
        return math.log(max(float(np.mean(motions)), 1e-4))

    def p_high(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.a + self.b * z)))

    def score(self, z: float) -> float:
        p, t = self.p_high(z), min(max(self.threshold, 1e-3), 1 - 1e-3)
        return 50.0 * p / t if p < t else 50.0 + 50.0 * (p - t) / (1.0 - t)


class LiveMotionScorer:
    """Per-video state for the live pipeline: per-student motion windows and the
    class's running median of window-level log motion (label-free normalisation)."""

    def __init__(self, rule: MotionRule, min_class: int = 30, history: int = 3000) -> None:
        from collections import deque

        self.rule, self.min_class = rule, min_class
        self._deque = deque
        self.hist: dict[int, "deque[float]"] = {}
        self.class_logs = deque(maxlen=history)

    def reset(self) -> None:
        self.hist.clear()
        self.class_logs.clear()

    def update(self, tid: int, motion: float | None) -> float | None:
        """Add one observation; return a 0-100 score or None while warming up."""
        h = self.hist.setdefault(tid, self._deque(maxlen=self.rule.window))
        if motion is not None:
            h.append(motion)
        if len(h) < self.rule.window:
            return None
        lw = MotionRule.window_log(h)
        self.class_logs.append(lw)
        if len(self.class_logs) < self.min_class:
            return None
        return self.rule.score(lw - float(np.median(self.class_logs)))
