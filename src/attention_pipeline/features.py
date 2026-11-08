"""Pack pose / L2CS gaze / SCB behavior outputs into the 32-D fusion vector.

Feature Layout Specification (FEAT_DIM = 32):
  [0:3]   Head Pose (yaw, pitch, roll) in [-1, 1], normalized by 180.0 deg
  [3:5]   Eye Gaze (yaw, pitch) in [-1, 1], normalized by 180.0 deg
  [5:12]  Emotion Probabilities (7 classes: angry, disgust, fear, happy, sad, surprise, neutral) in [0, 1]
  [12:18] Posture / Classroom Behavior Probabilities (6 classes from SCB: hand-raising, reading, writing,
          using-phone, bowing-head, leaning-over-table) in [0, 1]
  [18]    Inter-frame Optical Motion in [0, 1]
  [19]    Eye Aspect Ratio (EAR) in [0, 1]
  [20]    Mouth Aspect Ratio (MAR) in [0, 1]
  [21]    Facing-Front Score (1.0 - |yaw| / 45.0) in [0, 1]
  [22]    Torso Lean Angle in [0, 1] (0 = upright, 1 = strongly leaned; confidence-gated with shoulder fallback)
  [23]    Upper-Body Keypoint Confidence in [0, 1] (mean confidence of nose, eyes, shoulders)
  [24:32] Reserved zero-padding (8 dimensions): reserved for future temporal kinematics.
          Preserved at FEAT_DIM=32 to maintain strict backwards compatibility with existing checkpoints.
"""

from __future__ import annotations

import numpy as np

from attention_pipeline.types import FEAT_DIM, StudentFrame

_EPS = 1e-6
_CONF_THRESH = 0.25
_UPPER_BODY_KPTS = (0, 1, 2, 5, 6)  # Nose (0), Left Eye (1), Right Eye (2), Left Shoulder (5), Right Shoulder (6)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def facing_front_score(yaw_deg: float) -> float:
    return _clip01(1.0 - abs(yaw_deg) / 45.0)


def _is_kpt_reliable(kpts: np.ndarray, idx: int, conf_thresh: float = _CONF_THRESH) -> bool:
    if kpts is None or kpts.shape[0] <= idx:
        return False
    if kpts.shape[1] >= 3:
        conf = float(kpts[idx, 2])
        if np.isnan(conf) or np.isinf(conf):
            return False
        return conf >= conf_thresh
    return bool(np.all(np.isfinite(kpts[idx, :2])))


def torso_lean(keypoints: np.ndarray | None, conf_thresh: float = _CONF_THRESH) -> float:
    """Torso lean angle in [0, 1] (0 = upright, 1 = strongly leaned).

    Confidence-aware gating strategy:
    - Case 1 (Reliable Hips & Shoulders): If at least one hip (11, 12) and at least one
      shoulder (5, 6) have confidence >= conf_thresh, calculates horizontal deflection of
      the shoulder-to-hip vector.
    - Case 2 (Occluded Hips Fallback): If both hips have confidence < conf_thresh (e.g. desk
      occlusion or webcam framing) but both shoulders (5, 6) have confidence >= conf_thresh,
      calculates lateral roll from the tilt of the inter-shoulder axis.
    - Case 3 (Explicit Missingness): If neither hips nor both shoulders are reliable,
      returns 0.0 without inventing an upright posture. Missingness is explicitly represented
      in conjunction with Channel 23 (Upper-Body Keypoint Confidence = 0.0).
    """
    if keypoints is None or keypoints.shape[0] < 13:
        return 0.0

    l_sh_ok = _is_kpt_reliable(keypoints, 5, conf_thresh)
    r_sh_ok = _is_kpt_reliable(keypoints, 6, conf_thresh)
    l_hp_ok = _is_kpt_reliable(keypoints, 11, conf_thresh)
    r_hp_ok = _is_kpt_reliable(keypoints, 12, conf_thresh)

    # Determine average shoulder position if available
    sh: np.ndarray | None = None
    if l_sh_ok and r_sh_ok:
        sh = keypoints[[5, 6], :2].mean(axis=0)
    elif l_sh_ok:
        sh = keypoints[5, :2]
    elif r_sh_ok:
        sh = keypoints[6, :2]

    # Determine average hip position if available
    hp: np.ndarray | None = None
    if l_hp_ok and r_hp_ok:
        hp = keypoints[[11, 12], :2].mean(axis=0)
    elif l_hp_ok:
        hp = keypoints[11, :2]
    elif r_hp_ok:
        hp = keypoints[12, :2]

    # Case 1: Both shoulder and hip anchors available
    if sh is not None and hp is not None:
        vec = sh - hp
        mag = float(np.linalg.norm(vec)) + _EPS
        horizontal = abs(float(vec[0])) / mag
        return _clip01(horizontal)

    # Case 2: Hips occluded, but both shoulders reliable -> inter-shoulder tilt fallback
    if l_sh_ok and r_sh_ok:
        vec_sh = keypoints[6, :2] - keypoints[5, :2]  # right shoulder - left shoulder
        mag_sh = float(np.linalg.norm(vec_sh)) + _EPS
        vertical = abs(float(vec_sh[1])) / mag_sh  # shoulder slope tilt indicates lateral lean
        return _clip01(vertical)

    # Case 3: Missingness (insufficient or unreliable keypoints)
    return 0.0


def upper_body_keypoint_confidence(keypoints: np.ndarray | None) -> float:
    """Channel 23: Mean normalized confidence of 5 core upper-body keypoints.

    Contributing keypoints:
      - 0: Nose
      - 1: Left Eye
      - 2: Right Eye
      - 5: Left Shoulder
      - 6: Right Shoulder

    Missingness handling:
      - If keypoints is None, empty, or has < 7 rows: returns 0.0.
      - If keypoints has < 3 columns (no confidence channel): returns 1.0 if finite, else 0.0.
      - Any NaN, Inf, or negative confidence is safely replaced with 0.0.
      - Output is arithmetic mean of the 5 confidences, strictly normalized to [0.0, 1.0].
    """
    if keypoints is None or keypoints.shape[0] < 7:
        return 0.0

    if keypoints.shape[1] < 3:
        valid = [
            bool(np.all(np.isfinite(keypoints[idx, :2])))
            for idx in _UPPER_BODY_KPTS
            if idx < keypoints.shape[0]
        ]
        return float(np.mean(valid)) if valid else 0.0

    confs = []
    for idx in _UPPER_BODY_KPTS:
        if idx < keypoints.shape[0]:
            c = float(keypoints[idx, 2])
            if np.isnan(c) or np.isinf(c) or c < 0.0:
                confs.append(0.0)
            else:
                confs.append(min(1.0, c))
        else:
            confs.append(0.0)

    return _clip01(float(np.mean(confs)))


def pack_features(
    head_pose: np.ndarray,
    gaze: np.ndarray,
    emotion_probs: np.ndarray,
    posture_probs: np.ndarray,
    motion: float,
    eye_aspect: float,
    mouth_aspect: float,
    keypoints: np.ndarray,
) -> np.ndarray:
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    feat[0:3] = np.clip(np.asarray(head_pose, dtype=np.float32) / 180.0, -1, 1)
    feat[3:5] = np.clip(np.asarray(gaze, dtype=np.float32) / 180.0, -1, 1)
    feat[5:12] = np.asarray(emotion_probs, dtype=np.float32).reshape(-1)[:7]
    feat[12:18] = np.asarray(posture_probs, dtype=np.float32).reshape(-1)[:6]
    feat[18] = _clip01(motion)
    feat[19] = _clip01(eye_aspect)
    feat[20] = _clip01(mouth_aspect)
    feat[21] = facing_front_score(float(head_pose[0]))
    feat[22] = torso_lean(keypoints)
    feat[23] = upper_body_keypoint_confidence(keypoints)
    # Channels 24..31 remain 0.0 reserved padding
    return feat


def heuristic_attention_score(feat: np.ndarray) -> float:
    """Rubric score from the 32-D vector when no StudentFrame is available."""
    from attention_pipeline.rubric import classify_from_features

    return classify_from_features(feat).score


def dominant_modality(feat: np.ndarray) -> str:
    """Heuristic: which cue most explains a low score for the LLM payload."""
    yaw = abs(feat[0] * 180.0)
    gaze = np.linalg.norm(feat[3:5] * 180.0)
    emotion_neg = float(feat[5] + feat[6] + feat[7] + feat[9])  # angry, disgust, fear, sad
    looking_down = float(feat[15] + feat[17])  # using-phone + leaning-over-table
    motion = float(feat[18])
    scores = {
        "head_pose": yaw / 45.0 + (1.0 - feat[21]),
        "gaze": gaze / 30.0,
        "emotion": emotion_neg,
        "posture": looking_down + feat[22],
        "facial_motion": motion,
    }
    return max(scores, key=scores.get)


def to_student_frame(
    student_id: int,
    timestamp: float,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    head_pose: np.ndarray,
    gaze: np.ndarray,
    emotion_probs: np.ndarray,
    posture_probs: np.ndarray,
    motion: float,
    eye_aspect: float,
    mouth_aspect: float,
    has_phone_obj: bool = False,
) -> StudentFrame:
    features = pack_features(
        head_pose,
        gaze,
        emotion_probs,
        posture_probs,
        motion,
        eye_aspect,
        mouth_aspect,
        keypoints,
    )
    return StudentFrame(
        student_id=student_id,
        timestamp=timestamp,
        bbox=np.asarray(bbox, dtype=np.float32),
        keypoints=np.asarray(keypoints, dtype=np.float32),
        head_pose=np.asarray(head_pose, dtype=np.float32),
        gaze=np.asarray(gaze, dtype=np.float32),
        emotion_probs=np.asarray(emotion_probs, dtype=np.float32),
        posture_probs=np.asarray(posture_probs, dtype=np.float32),
        motion=float(motion),
        eye_aspect=float(eye_aspect),
        mouth_aspect=float(mouth_aspect),
        features=features,
        has_phone_obj=bool(has_phone_obj),
    )
