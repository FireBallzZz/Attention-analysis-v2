"""Per-student head-pitch baseline.

The geometric head pitch from YOLO keypoints carries a large systematic
offset. A student sitting upright and facing the board measures about +25 to
+32 degrees in this coordinate system, not 0, because the pitch is derived from
the nose-versus-eye-midpoint offset in the image plane.

Scoring that absolute value against fixed thresholds (head-up <= 6, reading
6..42) therefore marked 80-90% of every classroom clip as "reading", so the
rubric could not tell a student looking down at notes apart from one looking
down because they are asleep.

Each student's own neutral reference is tracked over time instead, and the
rubric reasons about the difference, which is comparable across postures and
across students.
"""

from __future__ import annotations

import numpy as np

# Relative pitch bands, in degrees, measured against the student's own neutral.
HEAD_UP_DELTA = -4.0      # at or above their own neutral
READING_DELTA = 22.0      # below this the head is down on the desk


class PitchBaselineTracker:
    """Tracks each student's neutral head pitch over time."""

    def __init__(self, warmup: int = 24, capacity: int = 240) -> None:
        self.warmup = int(warmup)
        self.capacity = int(capacity)
        self._history: dict[int, list[float]] = {}

    def reset(self) -> None:
        """Clear per-video state."""
        self._history.clear()

    def update(self, student_id: int, pitch: float) -> None:
        """Record one observation of the student's head pitch."""
        if pitch is None or not np.isfinite(pitch):
            return
        history = self._history.setdefault(int(student_id), [])
        history.append(float(pitch))
        if len(history) > self.capacity:
            del history[: len(history) - self.capacity]

    def baseline(self, student_id: int) -> float | None:
        """The student's neutral pitch, or None while still warming up."""
        history = self._history.get(int(student_id))
        if not history or len(history) < self.warmup:
            return None
        return float(np.median(history))

    def delta(self, student_id: int, pitch: float) -> float | None:
        """Pitch relative to the student's own neutral, or None when unknown."""
        base = self.baseline(student_id)
        if base is None or pitch is None or not np.isfinite(pitch):
            return None
        return float(pitch) - base


class EyeBaselineTracker:
    """Tracks each student's normal eye openness so a closed eye is detectable.

    The absolute eye-opening measure varies a lot between students (different
    face shapes, glasses, lighting), so a fixed threshold marks everyone as
    either open or closed. Each student's own median opening is used instead.
    """

    def __init__(self, warmup: int = 24, capacity: int = 240) -> None:
        self.warmup = int(warmup)
        self.capacity = int(capacity)
        self._history: dict[int, list[float]] = {}

    def reset(self) -> None:
        """Clear per-video state."""
        self._history.clear()

    def update(self, student_id: int, openness: float | None) -> None:
        """Record one observation of the student's eye openness."""
        if openness is None or not np.isfinite(openness):
            return
        history = self._history.setdefault(int(student_id), [])
        history.append(float(openness))
        if len(history) > self.capacity:
            del history[: len(history) - self.capacity]

    def baseline(self, student_id: int) -> float | None:
        """The student's normal eye opening, or None while still warming up."""
        history = self._history.get(int(student_id))
        if not history or len(history) < self.warmup:
            return None
        return float(np.median(history))

    def is_closed(self, student_id: int, openness: float | None) -> bool | None:
        """Whether the eyes are closed, or None when the baseline is unknown."""
        if openness is None:
            return None
        base = self.baseline(student_id)
        if base is None:
            # An absolute floor still works before the baseline is warm.
            return bool(openness < 0.30)
        return bool(openness < base * 0.62)
