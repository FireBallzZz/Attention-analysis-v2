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
        pitch_delta=getattr(obs, "pitch_delta", None),
        eye_closed=bool(getattr(obs, "eye_closed", False)),
        eye_state_valid=bool(getattr(obs, "aspects_valid", False)),
        body_motion=getattr(obs, "body_motion", None),
        rel_gaze_yaw=getattr(obs, "rel_gaze_yaw", None),
        pose_reliable=bool(getattr(obs, "pose_reliable", True)),
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
    pitch_delta: float | None = None,
    eye_closed: bool = False,
    eye_state_valid: bool = False,
    body_motion: float | None = None,
    rel_gaze_yaw: float | None = None,
    pose_reliable: bool = True,
) -> dict[str, bool]:
    # Camera-relative facing is only a fallback. Lesson direction is the class median yaw.
    # Positive signed yaw is toward image-right, relative to the lesson.
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    delta = abs(signed)
    facing_lesson = delta < 18
    glance = 18 <= delta < 34
    off_lesson = delta >= 34

    # Head pitch carries a large systematic offset (a student sitting upright
    # measures about +25 to +32 degrees here), so absolute thresholds cannot
    # separate "looking down at notes" from "head on desk". When the student's
    # own neutral is known, reason about the relative drop instead.
    if not pose_reliable:
        reading_tilt = False
        head_up = False
        collapsed = posture.get("leaning-over-table", 0) > 0.75
    elif pitch_delta is not None:
        reading_tilt = -5.0 < pitch_delta <= 22.0
        head_up = pitch_delta <= -5.0
        collapsed = pitch_delta > 26.0
    else:
        reading_tilt = 6 < pitch <= 42
        head_up = pitch <= 6
        collapsed = (pitch > 46 and not reading_tilt) or posture.get("leaning-over-table", 0) > 0.75
    eff_motion = float(motion) if body_motion is None else float(body_motion)
    collapsed = collapsed or posture.get("leaning-over-table", 0) > 0.75
    on_desk = posture.get("leaning-over-table", 0) > 0.75
    hand_raised = posture.get("hand-raising", 0) > 0.5
    slouch = posture.get("leaning-over-table", 0) > 0.5
    upright = False
    standing = False
    nose_on_desk = False
    chin_rest = False
    near_desk = False
    scb_desk_work = posture.get("writing", 0) >= 0.45 or posture.get("reading", 0) >= 0.60
    writing = posture.get("writing", 0) > 0.45 or posture.get("reading", 0) > 0.45
    phone = False
    # Don't treat raw SCB "reading" as writing unless pose also shows desk work.
    if posture.get("reading", 0) > 0.45 and posture.get("writing", 0) < 0.45:
        writing = False
    if pose_reliable and posture.get("bowing-head", 0) > 0.45 and pitch <= 32:
        reading_tilt = True

    usable = (
        kpts is not None
        and bbox is not None
        and float(np.max(kpts[:, :2])) > 2.0
        and _ok(kpts, NOSE)
        and _ok(kpts, L_SH)
        and _ok(kpts, R_SH)
    )
    wrists: list[np.ndarray] = []
    wr_l = wr_r = None
    mid_torso_y = 0.0
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
        # Desk-work band: elevated classroom cameras make a student leaning over
        # a notebook measure slightly negative head_clearance (-0.01..-0.04) even
        # while pitch stays in the normal reading range (~28-44 deg).
        is_writing_scb = posture.get("writing", 0) >= 0.40
        desk_hc_floor = -0.15 if is_writing_scb else -0.045
        if desk_hc_floor <= head_clearance <= 0.11 and pitch > 8:
            reading_tilt = True
        if head_clearance > 0.12:
            upright = True
            collapsed = False
            reading_tilt = False if pitch <= 8 else reading_tilt
        deep_drop = (
            head_clearance < -0.15
            or (head_clearance < -0.06 and not is_writing_scb)
            or (
                head_clearance < -0.015
                and (pitch > 48 or (pitch_delta is not None and pitch_delta > 22.0))
            )
        )
        if deep_drop:
            on_desk = True
            collapsed = True
            nose_on_desk = True
        ankle_ys = [float(kpts[idx, 1]) for idx in (15, 16) if _ok(kpts, idx, 0.30)]
        hip_in_box = False
        if hip is not None:
            hip_in_box = (y1 < float(hip[1]) <= y2 + 0.05 * bh) and ((float(hip[1]) - y1) <= 0.65 * bh)
            torso = (hip[1] - sh[1]) / bh
            if 0.16 <= torso < 0.24:
                slouch = True
            if torso < 0.12 and head_clearance < -0.02:
                on_desk = True
                collapsed = True
            # Standing: require hip inside the upper/middle box plus visible extended legs/ankles
            # so narrow side-profile or occluded-hip seated students do not trigger false "standing".
            knee_ys = [float(kpts[idx, 1]) for idx in (13, 14) if _ok(kpts, idx, 0.35)]
            legs_extended = bool(
                hip_in_box
                and (
                    (ankle_ys and (max(ankle_ys) - float(hip[1])) > 0.32 * bh)
                    or (
                        knee_ys
                        and (max(knee_ys) - float(hip[1])) > 0.22 * bh
                        and max(knee_ys) <= y2 + 0.08 * bh
                    )
                )
            )
            if upright and hip_in_box and (
                (legs_extended and aspect > 2.0 and torso > 0.28)
                or (aspect > 2.45 and torso > 0.34 and bool(knee_ys or ankle_ys))
            ):
                standing = True
        elif upright and aspect > 2.45 and ankle_ys:
            standing = True
        wr_l = kpts[L_WR] if _ok(kpts, L_WR) else None
        wr_r = kpts[R_WR] if _ok(kpts, R_WR) else None
        wrists = [w for w in (wr_l, wr_r) if w is not None]
        mid_torso_y = (sh[1] + hip[1]) / 2.0 if hip is not None else (sh[1] + 0.35 * bh)

        torso_ref = abs(float(hip[1]) - float(sh[1])) if hip is not None else 0.45 * bh

        def _is_chin_wrist(w: np.ndarray) -> bool:
            if not _ok(kpts, NOSE):
                return False
            dist = float(np.linalg.norm(w[:2] - nose))
            above_chest = float(w[1]) <= min(
                float(sh[1]) + max(0.11 * bh, 0.08 * torso_ref),
                mid_torso_y - 0.02 * bh,
            )
            under_chin = (
                abs(float(w[0]) - float(nose[0])) <= 0.26 * bw
                and float(nose[1]) - 0.08 * bh <= float(w[1]) <= float(sh[1]) + 0.06 * bh
                and dist <= 0.40 * bh
            )
            return bool(
                (dist < 0.26 * bh and float(w[1]) >= float(nose[1]) - 0.08 * bh and above_chest)
                or (under_chin and posture.get("writing", 0) < 0.45)
            )

        if wrists:
            near_desk = all(float(w[1]) > mid_torso_y for w in wrists)
            # Posture Check: Chin-Propping (palm/knuckles supporting jaw/cheek) evaluated BEFORE hand_raised
            chin_rest = any(_is_chin_wrist(w) for w in wrists)
            non_chin_raised = any(
                float(w[1]) < sh[1] - 0.08 * bh
                and (not _ok(kpts, NOSE) or float(w[1]) <= float(nose[1]) + 0.12 * torso_ref)
                and not _is_chin_wrist(w)
                for w in wrists
            )
            if non_chin_raised:
                hand_raised = True
            elif chin_rest:
                hand_raised = False

    # Gaze & Eye State Verification
    gaze_pitch = float(gaze[1]) if (gaze is not None and len(gaze) > 1) else pitch
    gaze_yaw = float(gaze[0]) if (gaze is not None and len(gaze) > 0) else yaw
    ear = float(eye_aspect)
    # The landmark-based aspect is a constant 0.3 whenever no 68-point tracker
    # is present, so it can never confirm open eyes. When a live measurement
    # exists it takes precedence; otherwise fall back to the aspect but treat
    # it as unverified.
    if eye_state_valid:
        eyes_open = not eye_closed
    else:
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
    elevated_phone_grip = False
    if usable and wrists:
        if wr_l is not None and wr_r is not None:
            dx = abs(float(wr_l[0]) - float(wr_r[0]))
            dy = abs(float(wr_l[1]) - float(wr_r[1]))
            if dx < 0.42 * bw and dy < 0.22 * bh and float(wr_l[1]) > sh[1] and float(wr_r[1]) > sh[1]:
                mobile_in_hand = True
                if dx < 0.36 * bw and dy < 0.16 * bh and float(wr_l[1]) <= mid_torso_y and float(wr_r[1]) <= mid_torso_y:
                    elevated_phone_grip = True
        elif len(wrists) == 1:
            w = wrists[0]
            if float(w[1]) > sh[1] and abs(float(w[0]) - float(nose[0])) < 0.28 * bw:
                mobile_in_hand = True

    # Multi-Modal Phone Identification:
    # 1. Physical phone object detected on desk/hands + student looking towards device
    # 2. SCB neural classifier detects using-phone (>= 0.28) + student looking towards device
    # Keep wrist-only grip as a guard against false SCB reading override, not as a standalone LOW override.
    phone_prob = posture.get("using-phone", 0.0)
    writing_prob = posture.get("writing", 0.0)
    scb_phone = bool(phone_prob >= 0.28 and watching_phone and writing_prob < 0.45 and not chin_rest)
    obj_phone = bool(has_phone_obj and watching_phone and not chin_rest)
    pose_phone = False

    # Phone is confirmed by physical object or SCB using-phone class
    phone = bool((scb_phone or obj_phone) and not chin_rest and not (collapsed and pitch > 48.0))

    if usable and wrists:
        if (
            near_desk
            and reading_tilt
            and (
                delta < 20.0
                or (
                    delta < 30.0
                    and pitch >= 24.0
                    and (rel_gaze_yaw is None or abs(float(rel_gaze_yaw)) < 22.0)
                )
            )
            and not nose_on_desk
            and not phone
            and not elevated_phone_grip
            and not chin_rest
            and pitch <= 45
            and (pitch_delta is None or pitch_delta <= 25.0)
        ):
            collapsed = False
            writing = True

    if chin_rest and writing_prob < 0.45:
        writing = False

    if (not pose_reliable) and on_desk and not hand_raised and posture.get("writing", 0) < 0.45:
        nose_on_desk = True
        collapsed = True

    # Desk work from wrists near desk + head tilt (not tilt alone).
    # Side cameras make "watching teacher" look like a small pitch; that is not writing.
    desk_work = bool(writing and reading_tilt and pitch <= 45)
    # A lowered head on notes stays classwork. Head-on-desk is a much deeper drop.
    # Chin-propped posture or open eyes gazing upward explicitly prevents false head-on-table!
    head_on_table = bool(
        (pitch > 48 or nose_on_desk)
        and not (desk_work and pitch <= 45 and not nose_on_desk)
        and not chin_rest
        and not (looking_up_with_eyes and eyes_open)
    )
    if (
        (upright and not on_desk and not nose_on_desk)
        or ((chin_rest or (looking_up_with_eyes and eyes_open)) and pitch <= 25 and not nose_on_desk)
    ):
        head_on_table = False
        collapsed = False
    still = eff_motion < 0.05
    # True sleep requires head on desk, motionless, not writing, not propping
    # chin, and eyes closed/occluded. When a live eye measurement exists it is
    # mandatory: an open eye rules out sleep regardless of head angle, which is
    # what stopped the rule from firing on students looking down at notes.
    if eye_state_valid:
        sleeping = bool(
            head_on_table
            and still
            and not writing
            and not chin_rest
            and eye_closed
            and not looking_up_with_eyes
        )
    else:
        # Without a verifiable eye state, require the stronger geometric
        # evidence of the head resting on the desk rather than merely lowered.
        sleeping = bool(
            head_on_table
            and still
            and not writing
            and not chin_rest
            and collapsed
            and not (eyes_open and looking_up_with_eyes)
        )
    # A clear turn away from the lesson (do not mask 34..54 deg sideways turns under neutral reading tilt).
    gaze_off = bool(
        rel_gaze_yaw is not None
        and glance
        and abs(float(rel_gaze_yaw)) >= 28.0
        and (float(rel_gaze_yaw) * signed > 0)
        and not scb_desk_work
    )
    turned = bool(
        (
            off_lesson
            and (not scb_desk_work or delta >= 48)
            and not (looking_up_with_eyes and delta < 48)
        )
        or gaze_off
    )
    look_left = bool(turned and signed < 0)
    look_right = bool(turned and signed > 0)
    glance_left = bool(glance and signed < 0 and not look_left and not desk_work and not head_on_table)
    glance_right = bool(glance and signed > 0 and not look_right and not desk_work and not head_on_table)

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
        "obj_phone": obj_phone,
        "scb_phone": scb_phone,
        "pose_phone": pose_phone,
        "mobile_in_hand": mobile_in_hand,
        "elevated_phone_grip": elevated_phone_grip,
        "standing": standing,
        "still": still,
        "active": eff_motion >= 0.08,
        "look_left": look_left,
        "look_right": look_right,
        "glance_left": glance_left,
        "glance_right": glance_right,
        "signed_left": signed < 0,
        "head_on_table": head_on_table,
        "sleeping": sleeping,
        "chin_rest": chin_rest,
        "looking_up_with_eyes": looking_up_with_eyes,
        "eyes_open": eyes_open,
    }


def _disengaged(cues: dict[str, bool]) -> RubricResult | None:
    """Behaviors that are not classwork. These win over a reading/writing label."""
    if cues.get("phone"):
        if cues.get("obj_phone") or cues.get("scb_phone"):
            return RubricResult("LOW", 22.0, "using phone")
        return RubricResult("LOW", 24.0, "phone-like pose")
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
        if c.get("obj_phone") or c.get("scb_phone"):
            return RubricResult("LOW", 22.0, "using phone")
        return RubricResult("LOW", 24.0, "phone-like pose")
    if c.get("sleeping"):
        return RubricResult("LOW", 16.0, "sleeping")
    if c.get("head_on_table"):
        return RubricResult("LOW", 20.0, "head on table")
    if c["off_lesson"] and not c["writing"] and not c["standing"]:
        side = "left" if (c.get("look_left") or c.get("signed_left")) else "right"
        return RubricResult("LOW", 32.0, f"looking {side}")

    # Active learning: notes, reading, facing the teacher, hand raise.
    if c["hand_raised"]:
        return RubricResult("HIGH", 92.0, "hand raised")
    if c["standing"] and c["facing_lesson"]:
        return RubricResult("HIGH", 88.0, "standing / presenting")
    if c["writing"] and not c["off_lesson"] and not c.get("chin_rest"):
        return RubricResult("HIGH", 88.0, "reading / writing")
    if c.get("looking_up_with_eyes") and c["facing_lesson"]:
        return RubricResult("MODERATE", 66.0, "slouched, watching lesson")
    if c["facing_lesson"] and c["head_up"]:
        return RubricResult("HIGH", 86.0, "watching instructor")
    if c["facing_lesson"] and c["reading_tilt"]:
        return RubricResult("HIGH", 84.0, "reading / taking notes")
    if c.get("chin_rest"):
        if c["facing_lesson"]:
            return RubricResult("MODERATE", 65.0, "listening, chin on hand")
        return RubricResult("MODERATE", 60.0, "thinking, chin on hand")
    if c["facing_lesson"] and not c["slouch"]:
        return RubricResult("HIGH", 80.0, "oriented to lesson")

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


class SpeechPersistenceTracker:
    """Tracks how long a student has shown real speech evidence.

    A single noisy frame must not produce a "talking to peer" verdict. The
    evidence has to persist across several frames before it counts, which is
    what separates a real conversation from tracking jitter.
    """

    def __init__(self, required: int = 3, decay: int = 1) -> None:
        self.required = int(required)
        self.decay = int(decay)
        self._hits: dict[int, int] = {}

    def reset(self) -> None:
        """Clear per-video state."""
        self._hits.clear()

    def update(self, student_id: int, speaking: bool) -> None:
        """Record one frame of speech evidence for a student."""
        current = self._hits.get(int(student_id), 0)
        current = current + 1 if speaking else max(0, current - self.decay)
        self._hits[int(student_id)] = current

    def sustained(self, student_id: int) -> bool:
        """Whether the student has shown speech evidence long enough."""
        return self._hits.get(int(student_id), 0) >= self.required


_PROTECTED_PEER_CUES = {
    "sleeping",
    "sleeping / head on desk",
    "head on table",
    "hand raised",
    "using phone",
    "phone-like pose",
}


def apply_peer_cues(
    frames: list[StudentFrame],
    rubrics: list[RubricResult],
    lesson_yaw: float | None,
    speech_tracker: SpeechPersistenceTracker | None = None,
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
            if relation is None or _has_intervening(frames, i, j):
                continue
            side_i, close = relation
            i_looks = _faces_side(frames[i], lesson_yaw, side_i)
            j_looks = _faces_side(frames[j], lesson_yaw, -side_i)
            i_looks_close = close and _faces_side(frames[i], lesson_yaw, side_i, min_deg=14.0)
            j_looks_close = close and _faces_side(frames[j], lesson_yaw, -side_i, min_deg=14.0)
            i_reaches = _reaches(frames[i], frames[j], side_i)
            j_reaches = _reaches(frames[j], frames[i], -side_i)
            bm_i = getattr(frames[i], "body_motion", None)
            bm_j = getattr(frames[j], "body_motion", None)
            i_active = float(frames[i].motion if bm_i is None else bm_i) >= 0.12
            j_active = float(frames[j].motion if bm_j is None else bm_j) >= 0.12
            # Speech evidence must persist over several frames before it counts.
            if speech_tracker is None:
                i_speech = j_speech = False
            else:
                i_speech = speech_tracker.sustained(frames[i].student_id)
                j_speech = speech_tracker.sustained(frames[j].student_id)

            ba, bb = frames[i].bbox, frames[j].bbox
            ha = max(float(ba[3] - ba[1]), 1.0)
            hb = max(float(bb[3] - bb[1]), 1.0)
            min_h, max_h = min(ha, hb), max(ha, hb)
            same_depth = (min_h / max_h) >= 0.72 and abs(float(ba[3]) - float(bb[3])) / min_h <= 0.55
            uni_thr = 22.0 if close else 26.0
            i_looks_uni = _faces_side(frames[i], lesson_yaw, side_i, min_deg=uni_thr)
            j_looks_uni = _faces_side(frames[j], lesson_yaw, -side_i, min_deg=uni_thr)
            # If neighbor turns in the exact same direction while silent, both are
            # looking toward the side presenter/board rather than conversing.
            i_neighbor_same_way = _faces_side(frames[j], lesson_yaw, side_i, min_deg=14.0)
            j_neighbor_same_way = _faces_side(frames[i], lesson_yaw, -side_i, min_deg=14.0)

            # Talking logic. Merely turning toward a neighbour is not talking:
            # a student checking the board or a neighbour's notes looks the
            # same way. Speech evidence is required in every branch.
            if i_looks and j_looks and (i_speech or j_speech):
                talking.add(i)
                talking.add(j)
            elif close and i_looks_close and j_looks_close and (i_speech or j_speech):
                talking.add(i)
                talking.add(j)
            elif close and i_speech and j_speech and (i_looks_close or j_looks_close):
                talking.add(i)
                talking.add(j)
            elif same_depth and i_looks_uni and i_speech and (not i_neighbor_same_way or j_speech):
                talking.add(i)
            elif same_depth and j_looks_uni and j_speech and (not j_neighbor_same_way or i_speech):
                talking.add(j)

            # Disturbing logic:
            # ONLY the active instigator who turns, reaches across into peer space,
            # and moves actively can be disturbing. The passive recipient/victim is NEVER marked.
            if i_looks and i_reaches and i_active:
                disturbing.add(i)
            if j_looks and j_reaches and j_active:
                disturbing.add(j)

    for index, rubric in enumerate(updated):
        if rubric.cue in _PROTECTED_PEER_CUES:
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


def _shoulder_y(obs: StudentFrame) -> float | None:
    kpts = obs.keypoints
    if kpts is not None and kpts.shape[0] > R_SH and _ok(kpts, L_SH) and _ok(kpts, R_SH):
        return (float(kpts[L_SH, 1]) + float(kpts[R_SH, 1])) / 2.0
    return None


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
    h_ratio = min_h / max_h

    # Must be at similar depth/scale (same row in perspective projection)
    if h_ratio < 0.65:
        return None
    # Must be within reasonable horizontal peer distance
    dx = bx - ax
    gap = abs(dx)
    if gap < 0.25 * width or gap > 2.2 * width:
        return None
    # Guard against front-behind depth stacking across different classroom rows
    ix1 = max(float(a.bbox[0]), float(b.bbox[0]))
    iy1 = max(float(a.bbox[1]), float(b.bbox[1]))
    ix2 = min(float(a.bbox[2]), float(b.bbox[2]))
    iy2 = min(float(a.bbox[3]), float(b.bbox[3]))
    iomin = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
        / max(min(h_a * w_a, h_b * w_b), 1.0)
    )
    if iomin >= 0.30 and (
        gap < 0.60 * width
        or abs(by - ay) > 0.25 * min_h
        or abs(float(a.bbox[3]) - float(b.bbox[3])) > 0.22 * min_h
    ):
        return None

    sh_a = _shoulder_y(a)
    sh_b = _shoulder_y(b)
    dy_sh = abs(sh_b - sh_a) / min_h if (sh_a is not None and sh_b is not None) else None
    dy_nose = abs(by - ay) / min_h

    # Allow same-row horizontal peers as well as close leaning whisperers whose
    # heads differ in vertical height while shoulders align in the same row.
    same_row = (dy_nose <= 0.40 and (dy_sh is None or dy_sh <= 0.55)) or (
        gap < 1.45 * width
        and h_ratio >= 0.72
        and dy_sh is not None
        and dy_sh <= 0.45
        and dy_nose <= 0.95
    )
    if not same_row:
        return None
    return (1.0 if dx > 0 else -1.0), gap < 1.3 * width


def _has_intervening(frames: list[StudentFrame], i: int, j: int) -> bool:
    """Whether another same-row student sits directly between `i` and `j`."""
    ax, ay = _center(frames[i])
    bx, by = _center(frames[j])
    lo_x, hi_x = min(ax, bx), max(ax, bx)
    span_x = hi_x - lo_x
    ha = max(float(frames[i].bbox[3] - frames[i].bbox[1]), 1.0)
    hb = max(float(frames[j].bbox[3] - frames[j].bbox[1]), 1.0)
    min_h = min(ha, hb)
    lo_y = min(ay, by) - 0.5 * min_h
    hi_y = max(ay, by) + 0.5 * min_h
    for k, obs in enumerate(frames):
        if k == i or k == j:
            continue
        kx, ky = _center(obs)
        hk = max(float(obs.bbox[3] - obs.bbox[1]), 1.0)
        if hk < 0.70 * min_h:
            continue
        if (lo_x + 0.25 * span_x) < kx < (hi_x - 0.25 * span_x) and lo_y <= ky <= hi_y:
            return True
    return False


def _faces_side(
    obs: StudentFrame,
    lesson_yaw: float | None,
    side: float,
    min_deg: float | None = None,
) -> bool:
    yaw = float(obs.head_pose[0])
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    # The mouth aspect used to relax this threshold, but it is a constant 0.3
    # without a landmark tracker, so it always relaxed it. Use the live
    # speech signal instead when it is available.
    if min_deg is None:
        min_deg = 18.0 if obs.mouth_active else 22.0
    return signed * side > 0 and abs(signed) >= min_deg


def _speaks(obs: StudentFrame) -> bool:
    """Whether the student has verifiable speech evidence.

    A live mouth-motion measurement is authoritative. The landmark-based
    mouth aspect is a constant 0.3 in this pipeline, so it carries no
    information and must never be treated as positive evidence on its own.
    """
    if getattr(obs, "aspects_valid", False):
        return bool(obs.mouth_active)
    # No live measurement: the constant aspect cannot prove speech, so the
    # honest answer is "no evidence" rather than a guess.
    return False



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
