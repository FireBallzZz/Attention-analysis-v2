"""Live face evidence: eye openness, mouth openness and mouth activity.

The rubric needs three things the old feature vector could not provide:

  * ``eye_closed``   - is the eye actually shut? (the sleep detector needs it)
  * ``mouth_open``   - how open is the mouth right now?
  * ``mouth_active`` - is the mouth moving frame to frame? (peer-talk needs it)

The previous implementation returned a hard-coded 0.3 for both aspects because
``_landmarks()`` only ever produced the five YOLO-pose keypoints, while the
aspect maths needs the 68-point OpenSeeFace layout. Rather than depend on an
optional third-party tracker we derive the evidence from the face crop that is
already available, using the eye and nose keypoints to place the regions of
interest.

Everything here is deliberately conservative: when the face crop is missing or
too small the functions report "unknown" (valid=False) instead of guessing, so
the rubric can fall back to behaviour-only cues rather than inventing evidence.
"""

from __future__ import annotations

import cv2
import numpy as np

# YOLO-pose17 landmark indices used to place the face regions.
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2

_MIN_FACE_PX = 24
_EYE_CLOSE_RATIO = 0.30  # absolute floor for "closed"
_EYE_CLOSE_FRACTION = 0.62  # closed when below this fraction of the student's own baseline
_MOUTH_ACTIVE_PX = 0.02  # illumination-normalized demeaned mouth change threshold
_MOUTH_CANON_W = 24
_MOUTH_CANON_H = 16
_MOUTH_SEARCH_PX = 2


def _eye_centres(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (left_eye_xy, right_eye_xy) from YOLO-pose keypoints."""
    if keypoints is None or keypoints.shape[0] < 3:
        return None
    if float(keypoints[NOSE, 2]) < 0.2:
        return None
    if float(keypoints[LEFT_EYE, 2]) < 0.2 or float(keypoints[RIGHT_EYE, 2]) < 0.2:
        return None
    return keypoints[LEFT_EYE, :2].astype(np.float32), keypoints[RIGHT_EYE, :2].astype(np.float32)


def eye_openness(
    crop: np.ndarray,
    keypoints: np.ndarray,
    origin: tuple[float, float],
) -> float | None:
    """Estimate how open the eyes are, in [0, 1].

    Uses the vertical dark-pixel extent inside each eye region relative to the
    region's width. Returns None when the face or the eye points are unusable.
    """
    centres = _eye_centres(keypoints)
    if centres is None or crop is None:
        return None
    h, w = crop.shape[:2]
    if h < _MIN_FACE_PX or w < _MIN_FACE_PX:
        return None

    span = _eye_span(keypoints)
    if span is None:
        return None

    ratios: list[float] = []
    for eye in centres:
        region = _crop_around(
            crop, _to_local(eye, origin), radius_x=0.55 * span, radius_y=0.35 * span
        )
        if region is None:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region_h, region_w = gray.shape[:2]
        if region_h < 3 or region_w < 3:
            continue
        # Dark pixels are iris/pupil/lash; their vertical spread measures openness.
        dark = gray < int(np.percentile(gray, 45))
        rows = np.where(dark.any(axis=1))[0]
        if rows.size == 0:
            continue
        vertical = float(rows[-1] - rows[0] + 1)
        ratios.append(vertical / region_w)

    if not ratios:
        return None
    return float(np.clip(np.mean(ratios), 0.0, 1.0))


def mouth_openness(
    crop: np.ndarray,
    keypoints: np.ndarray,
    origin: tuple[float, float],
    bbox: tuple[float, float, float, float] | np.ndarray | None = None,
) -> float | None:
    """Estimate how open the mouth is, in [0, 1]."""
    patch = mouth_patch(crop, keypoints, origin, bbox=bbox)
    if patch is None:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch_h, patch_w = gray.shape[:2]
    if patch_h < 3 or patch_w < 3:
        return None
    # Measure the darkest row band relative to the patch height. A closed
    # mouth is a thin dark line, an open mouth a taller dark region.
    dark = gray < int(np.percentile(gray, 40))
    row_counts = dark.sum(axis=1)
    if row_counts.size == 0:
        return 0.0
    darkest = float(row_counts.max())
    return float(np.clip(darkest / max(patch_h, 1) * 2.0, 0.0, 1.0))


def mouth_patch(
    crop: np.ndarray,
    keypoints: np.ndarray,
    origin: tuple[float, float],
    bbox: tuple[float, float, float, float] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract the mouth region as a patch.

    The patch is keyed off the nose and eye geometry (with a fallback to visible
    eye-to-nose vertical/person scale on turned profiles), so it follows the head
    without clipping on turned or downward-tilted faces.
    """
    if crop is None or keypoints is None or keypoints.shape[0] < 1:
        return None
    if float(keypoints[NOSE, 2]) < 0.2:
        return None
    nose = keypoints[NOSE, :2].astype(np.float32)
    visible_eyes = [
        keypoints[idx, :2].astype(np.float32)
        for idx in (LEFT_EYE, RIGHT_EYE)
        if keypoints.shape[0] > idx and float(keypoints[idx, 2]) >= 0.2
    ]
    if len(visible_eyes) == 2:
        horiz = float(np.linalg.norm(visible_eyes[0] - visible_eyes[1]))
        mid_eye = 0.5 * (visible_eyes[0] + visible_eyes[1])
    elif len(visible_eyes) == 1:
        # Side-profile turns often occlude the far eye; fall back to nose-to-visible-eye scale.
        horiz = float(np.linalg.norm(visible_eyes[0] - nose)) * 1.6
        mid_eye = visible_eyes[0]
    else:
        return None

    # Vertical nose-to-eye scale keeps mouth coverage intact when profile or
    # pitch collapses horizontal eye span.
    vert = max(float(nose[1] - mid_eye[1]), 0.0) * 2.1
    scale = max(horiz, vert)
    if bbox is not None and len(bbox) >= 4:
        scale = max(scale, 0.09 * max(float(bbox[3]) - float(bbox[1]), 0.0))
    if scale < 6.0:
        return None

    # Pull horizontal anchor slightly toward mid-eye so turned profile mouths do
    # not get half-clipped off the cheek onto background walls.
    cx = 0.65 * float(nose[0]) + 0.35 * float(mid_eye[0])
    cy = float(nose[1]) + 0.72 * scale
    centre = np.array([cx, cy], dtype=np.float32)
    return _crop_around(
        crop,
        _to_local(centre, origin),
        radius_x=0.44 * scale,
        radius_y=0.33 * scale,
    )


def canonical_mouth_patch(patch: np.ndarray | None) -> np.ndarray | None:
    """Resize a raw BGR/gray mouth patch to a smoothed canonical 24x16 float32 map."""
    if patch is None or patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        return None
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0.7)
    canon = cv2.resize(
        smoothed, (_MOUTH_CANON_W, _MOUTH_CANON_H), interpolation=cv2.INTER_AREA
    )
    return canon.astype(np.float32)


def mouth_activity(previous_patch: np.ndarray | None, current_patch: np.ndarray | None) -> float:
    """Illumination-compensated, subpixel-aligned mouth change normalised to [0, 1]."""
    if previous_patch is None or current_patch is None:
        return 0.0
    if previous_patch.size == 0 or current_patch.size == 0:
        return 0.0
    p0 = (
        previous_patch
        if (
            previous_patch.ndim == 2
            and previous_patch.shape == (_MOUTH_CANON_H, _MOUTH_CANON_W)
            and previous_patch.dtype == np.float32
        )
        else canonical_mouth_patch(previous_patch)
    )
    p1 = (
        current_patch
        if (
            current_patch.ndim == 2
            and current_patch.shape == (_MOUTH_CANON_H, _MOUTH_CANON_W)
            and current_patch.dtype == np.float32
        )
        else canonical_mouth_patch(current_patch)
    )
    if p0 is None or p1 is None or p0.shape != p1.shape:
        return 0.0

    sy = sx = _MOUTH_SEARCH_PX
    h, w = p1.shape
    ch, cw = h - 2 * sy, w - 2 * sx
    if ch < 2 or cw < 2:
        return 0.0

    c2 = p1[sy : sy + ch, sx : sx + cw]
    m2 = float(c2.mean())
    c2d = c2 - m2

    # Coarse integer search + half-pixel affine refinement cancel keypoint jitter
    # while demeaning removes global exposure/flicker shifts.
    best_err = 1e9
    best_dy, best_dx = 0, 0
    for dy in range(-sy, sy + 1):
        for dx in range(-sx, sx + 1):
            s1 = p0[sy + dy : sy + dy + ch, sx + dx : sx + dx + cw]
            err = float(np.mean(np.abs(c2d - (s1 - s1.mean()))))
            if err < best_err:
                best_err = err
                best_dy, best_dx = dy, dx

    for fdy in (-0.5, 0.0, 0.5):
        for fdx in (-0.5, 0.0, 0.5):
            if fdy == 0.0 and fdx == 0.0:
                continue
            ndy, ndx = best_dy + fdy, best_dx + fdx
            if abs(ndy) > sy or abs(ndx) > sx:
                continue
            mat = np.float32([[1.0, 0.0, -ndx], [0.0, 1.0, -ndy]])
            warped = cv2.warpAffine(
                p0,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            s1 = warped[sy : sy + ch, sx : sx + cw]
            err = float(np.mean(np.abs(c2d - (s1 - s1.mean()))))
            if err < best_err:
                best_err = err

    lum = max(max(m2, float(p0.mean())), 45.0)
    gain = float(np.clip((125.0 / lum) ** 0.50, 0.90, 1.50))
    return float(np.clip((best_err * gain) / 255.0, 0.0, 1.0))


def eye_is_closed(openness: float | None) -> bool:
    """Closed when the measured opening drops below the calibrated ratio."""
    if openness is None:
        return False
    return openness < _EYE_CLOSE_RATIO


def mouth_is_active(activity: float) -> bool:
    return activity >= _MOUTH_ACTIVE_PX


def _eye_span(keypoints: np.ndarray) -> float | None:
    centres = _eye_centres(keypoints)
    if centres is None:
        return None
    left, right = centres
    return float(np.linalg.norm(left - right))


def _to_local(point: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    """Convert a frame-coordinate point into crop-local coordinates."""
    return np.asarray(point, dtype=np.float32) - np.asarray(origin, dtype=np.float32)


def _crop_around(
    crop: np.ndarray,
    centre: np.ndarray,
    radius_x: float | None = None,
    radius_y: float | None = None,
) -> np.ndarray | None:
    """Crop a region around a point expressed in crop-local coordinates."""
    h, w = crop.shape[:2]
    if radius_x is None or radius_y is None:
        return None
    cx, cy = float(centre[0]), float(centre[1])
    x1 = int(max(cx - radius_x, 0))
    y1 = int(max(cy - radius_y, 0))
    x2 = int(min(cx + radius_x, w))
    y2 = int(min(cy + radius_y, h))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return crop[y1:y2, x1:x2]
