"""Classroom attention rubric: HIGH / MODERATE / LOW from posture, face, and action cues."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attention_pipeline.types import POSTURE_LABELS, StudentFrame

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class RubricResult:
    band: str
    score: float
    cue: str


def _angle_diff(a: float, b: float) -> float:
    return abs(_signed_diff(a, b))


def _signed_diff(a: float, b: float) -> float:
    """Signed yaw gap in degrees. Positive means `a` is to the right of `b`."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def classify_attention(obs: StudentFrame, lesson_yaw: float | None = None) -> RubricResult:
    """Score attention relative to the lesson, not the camera.

    Side-classroom cameras make 'facing the teacher' look like a turned head.
    Reading and writing look like a lowered head; that is HIGH, not LOW.
    Chin-propping and open-eye slouched listening are distinguished from sleep.
    """
    yaw = float(obs.head_pose[0])
    pitch = float(obs.head_pose[1])
    motion = float(obs.motion)
    gaze = obs.gaze
    ear = float(obs.eye_aspect)
    posture = {name: float(obs.posture_probs[i]) for i, name in enumerate(POSTURE_LABELS)}
    has_phone_obj = getattr(obs, "has_phone_obj", False)
    cues = _pose_cues(
        obs.keypoints, obs.bbox, yaw, pitch, motion, posture, lesson_yaw,
        gaze=gaze, eye_aspect=ear, has_phone_obj=has_phone_obj,
    )
    disengaged = _disengaged(cues)
    if disengaged is not None:
        return disengaged
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    idx = int(np.argmax(obs.posture_probs)) if obs.posture_probs.size else 0
    conf = float(obs.posture_probs[idx]) if obs.posture_probs.size else 0.0

    if cues.get("phone"):
        return RubricResult("LOW", 22.0, "using phone")

    if conf >= 0.35:
        ov = scb_attention_override(POSTURE_LABELS[idx], cues=cues, conf=conf)
        if ov is not None:
            band, score, cue = ov
            return RubricResult(band, score, cue)
    return _decide(cues)


def classify_from_features(feat: np.ndarray, lesson_yaw: float | None = None) -> RubricResult:
    yaw = float(feat[0] * 180.0)
    pitch = float(feat[1] * 180.0)
    gaze_yaw = float(feat[3] * 180.0)
    gaze_pitch = float(feat[4] * 180.0)
    motion = float(feat[18])
    ear = float(feat[19]) if feat.shape[0] > 19 else 0.3
    posture = {name: float(feat[12 + i]) for i, name in enumerate(POSTURE_LABELS)}
    gaze = np.array([gaze_yaw, gaze_pitch], dtype=np.float32)
    cues = _pose_cues(
        None, None, yaw, pitch, motion, posture, lesson_yaw,
        gaze=gaze, eye_aspect=ear,
    )
    disengaged = _disengaged(cues)
    if disengaged is not None:
        return disengaged
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    idx = int(np.argmax([posture.get(n, 0.0) for n in POSTURE_LABELS]))
    conf = posture.get(POSTURE_LABELS[idx], 0.0)

    # Elevated phone check
    if posture.get("using-phone", 0.0) >= 0.30 and cues.get("phone"):
        return RubricResult("LOW", 22.0, "using phone")

    if conf >= 0.35:
        ov = scb_attention_override(POSTURE_LABELS[idx], cues=cues, conf=conf)
        if ov is not None:
            band, score, cue = ov
            return RubricResult(band, score, cue)
    return _decide(cues)


def _ok(kpts: np.ndarray, idx: int, thr: float = 0.2) -> bool:
    return kpts is not None and kpts.shape[0] > idx and float(kpts[idx, 2]) >= thr


def _pose_cues(
    kpts,
    bbox,
    yaw: float,
    pitch: float,
    motion: float,
    posture: dict[str, float],
    lesson_yaw: float | None,
    gaze: np.ndarray | None = None,
    eye_aspect: float = 0.3,
    has_phone_obj: bool = False,
) -> dict[str, bool]:
    # Camera-relative facing is only a fallback. Lesson direction is the class median yaw.
    # Positive signed yaw is toward image-right, relative to the lesson.
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    delta = abs(signed)
    facing_lesson = delta < 18
    glance = 18 <= delta < 34
    off_lesson = delta >= 34

    reading_tilt = 6 < pitch <= 42
    head_up = pitch <= 6
    collapsed = (pitch > 46 and not reading_tilt) or posture.get("leaning-over-table", 0) > 0.75
    on_desk = posture.get("leaning-over-table", 0) > 0.75
    hand_raised = posture.get("hand-raising", 0) > 0.5
    slouch = posture.get("leaning-over-table", 0) > 0.5
    upright = False
    standing = False
    nose_on_desk = False
    chin_rest = False
    writing = posture.get("writing", 0) > 0.45 or posture.get("reading", 0) > 0.45
    phone = False
    # Don't treat raw SCB "reading" as writing unless pose also shows desk work.
    if posture.get("reading", 0) > 0.45 and posture.get("writing", 0) < 0.45:
        writing = False
    if posture.get("bowing-head", 0) > 0.45 and pitch <= 32:
        reading_tilt = True

    usable = (
        kpts is not None
        and bbox is not None
        and float(np.max(kpts[:, :2])) > 2.0
        and _ok(kpts, NOSE)
        and _ok(kpts, L_SH)
        and _ok(kpts, R_SH)
    )
    if usable:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bh = max(y2 - y1, 1.0)
        bw = max(x2 - x1, 1.0)
        aspect = bh / bw
        nose = kpts[NOSE, :2]
        sh = (kpts[L_SH, :2] + kpts[R_SH, :2]) / 2.0
        hip = None
        if _ok(kpts, L_HIP) and _ok(kpts, R_HIP):
            hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2.0
        head_clearance = (sh[1] - nose[1]) / bh
        # Tighter desk-work band; side views often have mid clearance while watching.
        if 0.045 <= head_clearance <= 0.11 and pitch > 8:
            reading_tilt = True
        if head_clearance > 0.12:
            upright = True
            collapsed = False
            reading_tilt = False if pitch <= 8 else reading_tilt
        if head_clearance < 0.0:
            on_desk = True
            collapsed = True
            nose_on_desk = True
        if hip is not None:
            torso = (hip[1] - sh[1]) / bh
            if 0.16 <= torso < 0.24:
                slouch = True
            if torso < 0.12 and head_clearance < 0.04:
                on_desk = True
                collapsed = True
            # Standing: long vertical box + visible hips farther down than seated desk row.
            if aspect > 2.0 and torso > 0.28 and upright:
                standing = True
        elif aspect > 2.2 and upright:
            standing = True
        wr_l = kpts[L_WR] if _ok(kpts, L_WR) else None
        wr_r = kpts[R_WR] if _ok(kpts, R_WR) else None
        if wr_l is not None and wr_l[1] < sh[1] - 0.08 * bh:
            hand_raised = True
        if wr_r is not None and wr_r[1] < sh[1] - 0.08 * bh:
            hand_raised = True
        wrists = [w for w in (wr_l, wr_r) if w is not None]
        if wrists:
            mid_torso_y = (sh[1] + hip[1]) / 2.0 if hip is not None else (sh[1] + 0.35 * bh)
            near_desk = all(float(w[1]) > mid_torso_y for w in wrists)
            # Posture Check: Chin-Propping (The Thinker / palm supporting jaw or cheek)
            if _ok(kpts, NOSE):
                for w in wrists:
                    if np.linalg.norm(w[:2] - nose) < 0.24 * bh:
                        chin_rest = True
                        break

    # Gaze & Eye State Verification
    gaze_pitch = float(gaze[1]) if (gaze is not None and len(gaze) > 1) else pitch
    gaze_yaw = float(gaze[0]) if (gaze is not None and len(gaze) > 0) else yaw
    ear = float(eye_aspect)
    eyes_open = ear >= 0.18

    # Gaze Compensation: If head is tilted down (pitch > 18), but gaze looks upward towards the board
    looking_up_with_eyes = bool(pitch > 18.0 and gaze_pitch < -18.0)

    # Watching downwards (toward desk/lap/device):
    watching_phone = bool(
        pitch > 12.0
        and not looking_up_with_eyes
        and not chin_rest
    )

    # Hand-geometry phone grip:
    mobile_in_hand = False
    if usable and wrists:
        if wr_l is not None and wr_r is not None:
            dx = abs(float(wr_l[0]) - float(wr_r[0]))
            dy = abs(float(wr_l[1]) - float(wr_r[1]))
            if dx < 0.42 * bw and dy < 0.22 * bh and float(wr_l[1]) > sh[1] and float(wr_r[1]) > sh[1]:
                mobile_in_hand = True
        elif len(wrists) == 1:
            w = wrists[0]
            if float(w[1]) > sh[1] and abs(float(w[0]) - float(nose[0])) < 0.28 * bw:
                mobile_in_hand = True

    # Multi-Modal Phone Identification:
    # 1. Physical phone object detected on desk/hands + student looking towards device
    # 2. SCB neural classifier detects using-phone (>= 0.28) + student looking towards device
    phone_prob = posture.get("using-phone", 0.0)
    writing_prob = posture.get("writing", 0.0)
    scb_phone = bool(phone_prob >= 0.28 and watching_phone and writing_prob < 0.45 and not chin_rest)
    obj_phone = bool(has_phone_obj and watching_phone and not chin_rest)

    # Phone is physically confirmed by detected device or neural SCB behavior
    phone = bool((scb_phone or obj_phone) and not chin_rest and not (collapsed and pitch > 48.0))

    if usable and wrists:
        if near_desk and not collapsed and reading_tilt and not phone:
            writing = True

    # Desk work from wrists near desk + head tilt (not tilt alone).
    # Side cameras make "watching teacher" look like a small pitch; that is not writing.
    desk_work = bool(writing and reading_tilt and pitch <= 42)
    # A lowered head on notes stays classwork. Head-on-desk is a much deeper drop.
    # Chin-propped posture or open eyes gazing upward explicitly prevents false head-on-table!
    head_on_table = bool(
        (pitch > 48 or nose_on_desk)
        and not (desk_work and pitch <= 42 and not nose_on_desk)
        and not chin_rest
        and not (looking_up_with_eyes and eyes_open)
    )
    if (upright or chin_rest or (looking_up_with_eyes and eyes_open)) and pitch <= 25 and not nose_on_desk:
        head_on_table = False
        collapsed = False
    still = motion < 0.05
    # True sleep requires head on desk, motionless, not writing, not propping chin, and eyes closed/occluded.
    sleeping = bool(
        head_on_table
        and still
        and not writing
        and not chin_rest
        and not (eyes_open and looking_up_with_eyes)
    )
    # A clear turn away from the lesson. Reading with the head down is not "looking away".
    turned = off_lesson and (not desk_work or delta >= 55) and (
        head_up or not reading_tilt or delta >= 55
    )
    look_left = bool(turned and signed < 0)
    look_right = bool(turned and signed > 0)
    glance_left = bool(glance and signed < 0 and not desk_work and not head_on_table)
    glance_right = bool(glance and signed > 0 and not desk_work and not head_on_table)

    return {
        "facing_lesson": facing_lesson,
        "glance": glance,
        "off_lesson": off_lesson,
        "head_up": head_up,
        "reading_tilt": reading_tilt,
        "collapsed": collapsed,
        "on_desk": on_desk,
        "upright": upright,
        "slouch": slouch,
        "hand_raised": hand_raised,
        "writing": writing,
        "phone": phone,
        "standing": standing,
        "still": still,
        "active": motion >= 0.08,
        "look_left": look_left,
        "look_right": look_right,
        "glance_left": glance_left,
        "glance_right": glance_right,
        "head_on_table": head_on_table,
        "sleeping": sleeping,
        "chin_rest": chin_rest,
        "looking_up_with_eyes": looking_up_with_eyes,
        "eyes_open": eyes_open,
    }


def _disengaged(cues: dict[str, bool]) -> RubricResult | None:
    """Behaviors that are not classwork. These win over a reading/writing label."""
    if cues.get("phone"):
        return RubricResult("LOW", 22.0, "using phone")
    if cues.get("sleeping"):
        return RubricResult("LOW", 16.0, "sleeping")
    if cues.get("head_on_table"):
        return RubricResult("LOW", 20.0, "head on table")
    if cues.get("look_left"):
        return RubricResult("LOW", 30.0, "looking left")
    if cues.get("look_right"):
        return RubricResult("LOW", 30.0, "looking right")
    return None


def _decide(c: dict[str, bool]) -> RubricResult:
    # True disengagement only.
    if c["phone"]:
        return RubricResult("LOW", 22.0, "phone-like pose")
    if c.get("head_on_table") and c["still"]:
        return RubricResult("LOW", 16.0, "sleeping")
    if c.get("head_on_table"):
        return RubricResult("LOW", 20.0, "head on table")
    if c["off_lesson"] and not c["writing"] and not c["standing"]:
        side = "left" if c.get("look_left") else "right"
        return RubricResult("LOW", 32.0, f"looking {side}")

    # Active learning: notes, reading, facing the teacher, hand raise.
    if c["hand_raised"]:
        return RubricResult("HIGH", 92.0, "hand raised")
    if c["standing"] and c["facing_lesson"]:
        return RubricResult("HIGH", 88.0, "standing / presenting")
    if c["writing"] and not c["off_lesson"]:
        return RubricResult("HIGH", 88.0, "reading / writing")
    if c["facing_lesson"] and c["head_up"]:
        return RubricResult("HIGH", 86.0, "watching instructor")
    if c["facing_lesson"] and c["reading_tilt"]:
        return RubricResult("HIGH", 84.0, "reading / taking notes")
    if c["facing_lesson"] and not c["slouch"]:
        return RubricResult("HIGH", 80.0, "oriented to lesson")

    # Propped and slouched listening (awake engagement, distinguished from sleep)
    if c.get("chin_rest"):
        if c["facing_lesson"]:
            return RubricResult("MODERATE", 65.0, "listening, chin on hand")
        return RubricResult("MODERATE", 60.0, "thinking, chin on hand")
    if c.get("looking_up_with_eyes") and c["facing_lesson"]:
        return RubricResult("MODERATE", 66.0, "slouched, watching lesson")

    # Partial engagement.
    if c.get("glance_left"):
        return RubricResult("MODERATE", 52.0, "glancing left")
    if c.get("glance_right"):
        return RubricResult("MODERATE", 52.0, "glancing right")
    if c["glance"]:
        return RubricResult("MODERATE", 55.0, "glancing away")
    if c["slouch"] and c["facing_lesson"]:
        return RubricResult("MODERATE", 58.0, "listening, relaxed")
    if c["facing_lesson"]:
        return RubricResult("MODERATE", 60.0, "listening")
    return RubricResult("MODERATE", 52.0, "partially engaged")


_CLASSWORK_CUES = {
    "hand raised",
    "reading",
    "writing notes",
    "reading / writing",
    "reading / taking notes",
    "reading / head to desk",
    "watching instructor",
    "standing / presenting",
    "listening, chin on hand",
    "thinking, chin on hand",
    "slouched, watching lesson",
}


def apply_peer_cues(
    frames: list[StudentFrame],
    rubrics: list[RubricResult],
    lesson_yaw: float | None,
) -> list[RubricResult]:
    """Mark same-row students who turn toward each other or actively disturb neighbors.

    Turning toward the lesson is not talking or disturbing.
    Disturbing requires the active instigator to face the peer, reach across into the peer's
    personal desk space, and actively move. Innocent students facing the lesson or writing
    notes are NEVER marked as disturbing others.
    """
    updated = list(rubrics)
    talking: set[int] = set()
    disturbing: set[int] = set()
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            relation = _row_relation(frames[i], frames[j])
            if relation is None:
                continue
            side_i, close = relation
            i_looks = _faces_side(frames[i], lesson_yaw, side_i)
            j_looks = _faces_side(frames[j], lesson_yaw, -side_i)
            i_reaches = _reaches(frames[i], frames[j], side_i)
            j_reaches = _reaches(frames[j], frames[i], -side_i)
            i_active = float(frames[i].motion) >= 0.12
            j_active = float(frames[j].motion) >= 0.12
            i_speech = float(frames[i].mouth_aspect) >= 0.28 or (
                float(frames[i].mouth_aspect) >= 0.24 and float(frames[i].motion) >= 0.08
            )
            j_speech = float(frames[j].mouth_aspect) >= 0.28 or (
                float(frames[j].mouth_aspect) >= 0.24 and float(frames[j].motion) >= 0.08
            )

            # Talking logic:
            # 1. Mutual looking towards each other: both are engaged in conversation
            if i_looks and j_looks:
                talking.add(i)
                talking.add(j)
            # 2. One-sided talking initiation: student turns and speaks towards peer
            elif i_looks and i_speech:
                talking.add(i)
            elif j_looks and j_speech:
                talking.add(j)

            # Disturbing logic:
            # ONLY the active instigator who turns, reaches across into peer space,
            # and moves actively can be disturbing. The passive recipient/victim is NEVER marked.
            if i_looks and i_reaches and i_active:
                disturbing.add(i)
            if j_looks and j_reaches and j_active:
                disturbing.add(j)

    for index, rubric in enumerate(updated):
        if rubric.cue in {"sleeping", "head on table", "hand raised"}:
            continue
        if index in disturbing:
            updated[index] = RubricResult("LOW", 18.0, "disturbing others")
        elif index in talking:
            updated[index] = RubricResult("LOW", 26.0, "talking to peer")
    return updated


def _center(obs: StudentFrame) -> tuple[float, float]:
    kpts = obs.keypoints
    if kpts is not None and kpts.shape[0] > 0 and float(kpts[0, 2]) >= 0.25:
        return float(kpts[0, 0]), float(kpts[0, 1])
    box = obs.bbox
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _row_relation(a: StudentFrame, b: StudentFrame) -> tuple[float, bool] | None:
    ax, ay = _center(a)
    bx, by = _center(b)
    h_a = max(float(a.bbox[3] - a.bbox[1]), 1.0)
    h_b = max(float(b.bbox[3] - b.bbox[1]), 1.0)
    w_a = max(float(a.bbox[2] - a.bbox[0]), 1.0)
    w_b = max(float(b.bbox[2] - b.bbox[0]), 1.0)
    min_h = min(h_a, h_b)
    max_h = max(h_a, h_b)
    width = (w_a + w_b) / 2.0

    # Must be at similar depth/scale (same row in perspective projection)
    if (min_h / max_h) < 0.65:
        return None
    # Must be vertically aligned in the row
    if abs(by - ay) > 0.40 * min_h:
        return None
    # Must be within reasonable horizontal peer distance
    dx = bx - ax
    gap = abs(dx)
    if gap < 0.25 * width or gap > 2.2 * width:
        return None
    return (1.0 if dx > 0 else -1.0), gap < 1.3 * width


def _faces_side(obs: StudentFrame, lesson_yaw: float | None, side: float) -> bool:
    yaw = float(obs.head_pose[0])
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    min_deg = 14.0 if float(obs.mouth_aspect) >= 0.25 else 18.0
    return signed * side > 0 and abs(signed) >= min_deg


def _reaches(obs: StudentFrame, other: StudentFrame, side: float) -> bool:
    """Checks if obs actively extends their arm across into other's space."""
    kpts = obs.keypoints
    if kpts is None or kpts.shape[0] < 11:
        return False
    ax, _ = _center(obs)
    bx, _ = _center(other)
    x_mid = (ax + bx) / 2.0

    for idx in (7, 8, 9, 10):
        if float(kpts[idx, 2]) >= 0.35:
            x = float(kpts[idx, 0])
            if side > 0 and (x > x_mid or x >= float(other.bbox[0])):
                return True
            if side < 0 and (x < x_mid or x <= float(other.bbox[2])):
                return True
    return False
