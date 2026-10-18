"""OpenSeeFace landmarks and head pose for students whose face is in frame.

The library is the GitHub tracker in third_party/OpenSeeFace. It is not on PyPI.
Head pose is yaw, pitch, roll in degrees. Landmarks are image x, y.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Right eye 36-41, left eye 42-47, mouth outer 48/54 and inner lips.
_RIGHT_EYE = (36, 39, 37, 38, 41, 40)
_LEFT_EYE = (42, 45, 43, 44, 47, 46)


def openseeface_root() -> Path | None:
    name = "OpenSeeFace"
    candidates = [
        Path("third_party") / name,
        Path(__file__).resolve().parents[3] / "third_party" / name,
    ]
    for path in candidates:
        if (path / "tracker.py").is_file() and (path / "models" / "lm_model3_opt.onnx").is_file():
            return path
    return None


def load_tracker_class():
    root = openseeface_root()
    if root is None:
        return None
    root_s = str(root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    spec = importlib.util.spec_from_file_location("openseeface_tracker", root / "tracker.py")
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Tracker


def image_landmarks(lms: np.ndarray) -> np.ndarray:
    """OpenSeeFace stores each point as (y, x, confidence). Return image (x, y)."""
    pts = np.asarray(lms, np.float32)
    count = min(66, pts.shape[0])
    return np.stack([pts[:count, 1], pts[:count, 0]], axis=1)


def head_pose_from_euler(euler) -> np.ndarray:
    """OpenSeeFace euler is pitch, yaw, roll. The pipeline stores yaw, pitch, roll."""
    angles = np.asarray(euler, np.float32).reshape(-1)
    if angles.size < 3 or not np.isfinite(angles[:3]).all():
        return np.zeros(3, np.float32)
    pitch, yaw, roll = (float(v) for v in angles[:3])
    return np.clip(np.array([yaw, pitch, roll], np.float32), -90.0, 90.0)


def head_pose_from_landmarks(lms: np.ndarray) -> np.ndarray:
    """Yaw, pitch, roll in degrees from OpenSeeFace landmarks.

    The tracker's own euler angles come out near 180 degrees for a normal
    seated face, so attention uses the same geometry as the keypoint fallback.
    """
    pts = image_landmarks(lms)
    if pts.shape[0] < 48:
        return np.zeros(3, np.float32)
    nose = pts[30]
    eye_a = pts[36:42].mean(axis=0)
    eye_b = pts[42:48].mean(axis=0)
    if eye_a[0] <= eye_b[0]:
        left_eye, right_eye = eye_a, eye_b
    else:
        left_eye, right_eye = eye_b, eye_a
    mid = (left_eye + right_eye) / 2.0
    span = abs(float(right_eye[0] - left_eye[0])) + 1e-6
    yaw = np.degrees(np.arctan2(float(nose[0] - mid[0]), span))
    pitch = np.degrees(np.arctan2(float(nose[1] - mid[1]), span))
    roll = np.degrees(np.arctan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0]) + 1e-6))
    return np.clip(np.array([yaw, pitch, roll], np.float32), -90.0, 90.0)


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pts[a] - pts[b]))


def _eye_aspect(pts: np.ndarray, corners: tuple[int, ...]) -> float:
    c1, c2, u1, u2, l1, l2 = corners
    horizontal = _dist(pts, c1, c2) + 1e-6
    vertical = _dist(pts, u1, l1) + _dist(pts, u2, l2)
    return vertical / (2.0 * horizontal)


def eye_mouth_aspects(lms: np.ndarray, eye_blink=None) -> tuple[float, float]:
    pts = image_landmarks(lms)
    if eye_blink is not None and len(eye_blink) >= 2:
        eye = float(np.clip(np.mean(np.asarray(eye_blink, np.float32)), 0.0, 1.0))
    elif pts.shape[0] > 47:
        eye = float(np.clip(0.5 * (_eye_aspect(pts, _RIGHT_EYE) + _eye_aspect(pts, _LEFT_EYE)), 0.0, 1.0))
    else:
        eye = 0.3
    if pts.shape[0] > 65:
        width = _dist(pts, 48, 54) + 1e-6
        upper = pts[[59, 60, 61]].mean(axis=0)
        lower = pts[[63, 64, 65]].mean(axis=0)
        mouth = float(np.clip(np.linalg.norm(upper - lower) / width, 0.0, 1.0))
    else:
        mouth = 0.3
    return eye, mouth


def five_points(lms: np.ndarray) -> np.ndarray:
    """Nose, right eye, left eye, and mouth corners in image pixels."""
    pts = image_landmarks(lms)
    if pts.shape[0] < 55:
        return pts[:5].astype(np.float32)
    right = pts[36:42].mean(axis=0)
    left = pts[42:48].mean(axis=0)
    return np.stack([pts[30], right, left, pts[48], pts[54]], axis=0).astype(np.float32)


def face_hint(det) -> tuple[int, int, int, int]:
    """A (x, y, w, h) face window for the OpenSeeFace tracker."""
    box = np.asarray(det.bbox_xyxy, np.float32)
    kpts = det.keypoints
    if kpts is not None and kpts.shape[0] >= 5 and float(kpts[0, 2] + kpts[1, 2]) > 0.2:
        eyes = kpts[[0, 1, 2], :2]
        cx, cy = eyes.mean(axis=0)
        span = float(np.linalg.norm(kpts[1, :2] - kpts[2, :2])) + 8.0
        side = max(span * 3.2, 48.0)
        return int(cx - side / 2.0), int(cy - side / 2.0), int(side), int(side)
    x1, y1, x2, y2 = (float(v) for v in box)
    return int(x1), int(y1), int(max(x2 - x1, 16)), int(max((y2 - y1) * 0.5, 16))


def _face_xyxy(face) -> np.ndarray | None:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    x, y, w, h = (float(v) for v in bbox)
    if w < 4 or h < 4:
        return None
    return np.array([x, y, x + w, y + h], np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def match_face(faces, det, used: set[int]):
    """Pick the OpenSeeFace track whose face sits on this student."""
    box = np.asarray(det.bbox_xyxy, np.float32)
    upper = box.copy()
    upper[3] = upper[1] + 0.6 * max(float(box[3] - box[1]), 1.0)
    best = None
    best_score = 0.05
    for face in faces:
        if id(face) in used:
            continue
        face_box = _face_xyxy(face)
        if face_box is None:
            continue
        cx = (float(face_box[0]) + float(face_box[2])) / 2.0
        cy = (float(face_box[1]) + float(face_box[3])) / 2.0
        inside = float(box[0]) <= cx <= float(box[2]) and float(box[1]) <= cy <= float(box[3])
        score = _iou(face_box, upper) + (0.35 if inside else 0.0)
        if score > best_score:
            best = face
            best_score = score
    return best


class OpenSeeFaceRunner:
    """One tracker for the frame. Student boxes tell it where the faces are."""

    def __init__(self) -> None:
        self._cls = None
        self._tracker = None
        self._size: tuple[int, int] | None = None
        self.faces: list = []
        self._load_error: str | None = None
        try:
            self._cls = load_tracker_class()
        except Exception as exc:
            self._cls = None
            self._load_error = str(exc)

    @property
    def available(self) -> bool:
        return self._cls is not None

    def prepare(self, frame: np.ndarray, detections) -> None:
        self.faces = []
        if not self.available:
            return
        height, width = frame.shape[:2]
        if self._tracker is None or self._size != (width, height):
            self._tracker = self._cls(
                width,
                height,
                model_type=3,
                max_faces=60,
                silent=True,
                scan_every=30,
                threshold=0.45,
                detection_threshold=0.4,
                max_threads=1,
                no_gaze=False,
                use_retinaface=False,
            )
            self._size = (width, height)
        hints = [face_hint(det) for det in detections]
        self.faces = list(self._tracker.predict(frame, additional_faces=hints) or [])
