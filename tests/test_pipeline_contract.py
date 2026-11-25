from __future__ import annotations

from pathlib import Path

import numpy as np

from attention_pipeline.features import heuristic_attention_score, pack_features
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.types import FEAT_DIM, TrackState


def _feat(yaw: float, looking_down: bool) -> np.ndarray:
    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    posture = np.zeros(6, np.float32)
    posture[1 if looking_down else 0] = 1.0
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 1.0
    return pack_features(
        head_pose=np.array([yaw, 0.0, 0.0], np.float32),
        gaze=np.array([0.0, -30.0 if looking_down else 0.0], np.float32),
        emotion_probs=emotion,
        posture_probs=posture,
        motion=0.5 if looking_down else 0.1,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        keypoints=kpts,
    )


def test_feature_dim() -> None:
    f = _feat(0, False)
    assert f.shape == (FEAT_DIM,)
    assert 80 <= heuristic_attention_score(f) <= 100
    reading = _feat(10, True)
    assert heuristic_attention_score(reading) >= 70
    asleep = pack_features(
        head_pose=np.array([0.0, 58.0, 0.0], np.float32),
        gaze=np.array([0.0, 58.0], np.float32),
        emotion_probs=np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
        posture_probs=np.array([0, 0, 0, 0, 0, 1], np.float32),
        motion=0.01,
        eye_aspect=0.2,
        mouth_aspect=0.2,
        keypoints=np.zeros((17, 3), np.float32),
    )
    assert heuristic_attention_score(asleep) < 40


def test_scb_behavior_match_and_override() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector, SCBDet
    from attention_pipeline.types import SCB_BEHAVIOR_LABELS

    det = SCBBehaviorDetector(weights=None)
    box = np.array([10, 10, 110, 210], np.float32)
    dets = [
        SCBDet(bbox_xyxy=box, cls=1, name="reading", conf=0.9),
        SCBDet(
            bbox_xyxy=np.array([500, 500, 600, 700], np.float32),
            cls=3,
            name="using-phone",
            conf=0.99,
        ),
    ]
    probs = det.match_probs(box, dets)
    assert probs is not None
    assert SCB_BEHAVIOR_LABELS[int(np.argmax(probs))] == "reading"

    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 1.0

    def frame(idx: int, pitch: float = 16.0):
        posture = np.zeros(6, np.float32)
        posture[idx] = 1.0
        return to_student_frame(
            1,
            0.0,
            bbox=np.array([0, 0, 100, 200], np.float32),
            keypoints=kpts,
            head_pose=np.array([8.0, pitch, 0], np.float32),
            gaze=np.array([8.0, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=0.1,
            eye_aspect=0.3,
            mouth_aspect=0.2,
        )

    assert classify_attention(frame(1)).band == "HIGH"
    assert classify_attention(frame(2)).band == "HIGH"
    assert classify_attention(frame(3)).band == "LOW"
    assert classify_attention(frame(3)).cue == "using phone"


def test_classroom_rubric() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention

    def frame(yaw, pitch, posture_idx, motion=0.1):
        emotion = np.zeros(7, np.float32)
        emotion[6] = 1.0
        posture = np.zeros(6, np.float32)
        posture[posture_idx] = 1.0
        kpts = np.zeros((17, 3), np.float32)
        kpts[:, 2] = 1.0
        return to_student_frame(
            1,
            0.0,
            bbox=np.array([0, 0, 100, 200], np.float32),
            keypoints=kpts,
            head_pose=np.array([yaw, pitch, 0], np.float32),
            gaze=np.array([yaw, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=motion,
            eye_aspect=0.3,
            mouth_aspect=0.2,
        )

    high = classify_attention(frame(4, 0, 0))
    assert high.band == "HIGH"
    reading = classify_attention(frame(8, 16, 0))
    assert reading.band == "HIGH"
    watching = classify_attention(frame(40, 2, 0), lesson_yaw=42.0)
    assert watching.band == "HIGH"
    low = classify_attention(frame(55, 40, 5, motion=0.02), lesson_yaw=0.0)
    assert low.band == "LOW"


def test_looking_sleeping_and_peer_talk() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import (
        SpeechPersistenceTracker,
        apply_peer_cues,
        classify_attention,
    )

    def frame(yaw, pitch, motion=0.1, bbox=None, keypoints=None, speaking=False,
              eye_closed=False):

        emotion = np.zeros(7, np.float32)
        emotion[6] = 1.0
        posture = np.full(6, 0.1, np.float32)
        kpts = np.zeros((17, 3), np.float32) if keypoints is None else keypoints
        box = np.array([0, 0, 100, 180], np.float32) if bbox is None else bbox
        return to_student_frame(
            1,
            0.0,
            bbox=box,
            keypoints=kpts,
            head_pose=np.array([yaw, pitch, 0], np.float32),
            gaze=np.array([yaw, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=motion,
            eye_aspect=0.3,
            mouth_aspect=0.2,
            mouth_motion=0.05 if speaking else 0.0,
            mouth_active=speaking,
            eye_closed=eye_closed,
            aspects_valid=True,
        )


    assert classify_attention(frame(-50, 2), lesson_yaw=0.0).cue == "looking left"
    assert classify_attention(frame(50, 2), lesson_yaw=0.0).cue == "looking right"
    # A "sleeping" claim needs a verified closed eye, not just a head angle.
    assert classify_attention(
        frame(0, 58, motion=0.01, eye_closed=True), lesson_yaw=0.0
    ).cue == "sleeping"
    # With the eyes verifiably open the same pose is only "head on table".
    assert classify_attention(
        frame(0, 58, motion=0.01, eye_closed=False), lesson_yaw=0.0
    ).cue == "head on table"
    assert classify_attention(frame(0, 55, motion=0.2), lesson_yaw=0.0).cue == "head on table"

    # Facing the lesson, even if the class looks to the right, is not talking.
    lesson = frame(30, 2, bbox=np.array([0, 40, 80, 200], np.float32))
    neighbor = frame(30, 2, bbox=np.array([120, 40, 200, 200], np.float32))
    kept = apply_peer_cues(
        [lesson, neighbor],
        [classify_attention(lesson, lesson_yaw=30.0), classify_attention(neighbor, lesson_yaw=30.0)],
        30.0,
    )
    assert all(item.cue != "talking" for item in kept)

    # Turning toward a neighbour is not talking on its own: a student reading
    # a neighbour's notes looks exactly the same way. Real speech evidence is
    # required, and it must persist across frames.
    tracker = SpeechPersistenceTracker(required=3)
    left = frame(40, 2, bbox=np.array([0, 40, 90, 210], np.float32))
    right = frame(-40, 2, bbox=np.array([110, 40, 200, 210], np.float32))
    turned_only = apply_peer_cues(
        [left, right],
        [classify_attention(left, lesson_yaw=0.0), classify_attention(right, lesson_yaw=0.0)],
        0.0,
        tracker,
    )
    assert all(item.cue not in {"talking", "talking to peer"} for item in turned_only)

    # With sustained speech evidence the same pair is marked as talking.
    speaking_left = frame(40, 2, bbox=np.array([0, 40, 90, 210], np.float32), speaking=True)
    speaking_right = frame(-40, 2, bbox=np.array([110, 40, 200, 210], np.float32), speaking=True)
    tracker = SpeechPersistenceTracker(required=3)
    for _ in range(3):
        tracker.update(speaking_left.student_id, True)
        tracker.update(speaking_right.student_id, True)
    talked = apply_peer_cues(
        [speaking_left, speaking_right],
        [
            classify_attention(speaking_left, lesson_yaw=0.0),
            classify_attention(speaking_right, lesson_yaw=0.0),
        ],
        0.0,
        tracker,
    )
    assert talked[0].cue in {"talking", "talking to peer"}
    assert talked[1].cue in {"talking", "talking to peer"}


    kpts = np.zeros((17, 3), np.float32)
    kpts[9] = [150, 80, 1.0]  # wrist inside the neighbor box
    kpts[0, 2] = 0.0
    reacher = frame(40, 2, motion=0.2, bbox=np.array([0, 40, 90, 210], np.float32), keypoints=kpts)
    quiet = frame(0, 2, bbox=np.array([110, 40, 200, 210], np.float32))
    disturbed = apply_peer_cues(
        [reacher, quiet],
        [classify_attention(reacher, lesson_yaw=0.0), classify_attention(quiet, lesson_yaw=0.0)],
        0.0,
    )
    assert disturbed[0].cue == "disturbing others"
    assert disturbed[1].cue != "disturbing others"


def test_video_accuracy_is_a_new_file() -> None:
    from attention_pipeline.video_accuracy import measure_rows, unique_report_path, write_video_accuracy

    high = {
        "band": "HIGH",
        "head_pose": [0.0, 10.0, 0.0],
        "posture": [0.0, 0.9, 0.05, 0.0, 0.0, 0.05],
    }
    low = {
        "band": "LOW",
        "head_pose": [0.0, 60.0, 0.0],
        "posture": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    }
    mismatch = {
        "band": "HIGH",
        "head_pose": [0.0, 70.0, 0.0],
        "posture": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    }
    metrics = measure_rows([{"students": [high, low, mismatch]}])
    assert metrics["support"] == 3
    assert abs(metrics["accuracy"] - (2 / 3)) < 1e-6
    assert metrics["per_band"]["LOW"]["support"] == 2

    folder = Path("runs") / "_accuracy_test"
    folder.mkdir(parents=True, exist_ok=True)
    video = Path("video") / "Classroom_video.mp4"
    first = unique_report_path(video, folder)
    first.write_text("keep\n", encoding="utf-8")
    second = unique_report_path(video, folder)
    assert first != second
    assert first.read_text(encoding="utf-8") == "keep\n"
    scores = folder / "scores.jsonl"
    scores.write_text(
        '{"students": ['
        + '{"band": "HIGH", "head_pose": [0, 10, 0], "posture": [0, 0.9, 0, 0, 0, 0.1]}'
        + "]}\n",
        encoding="utf-8",
    )
    written = write_video_accuracy(video, Path("runs/overlay.mp4"), scores, folder)
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "COMPONENT BENCHMARK SCORES" in text
    assert "VIDEO ACCURACY" in text
    assert "GROUND TRUTH ACCURACY" in text
    assert first.read_text(encoding="utf-8") == "keep\n"


def test_geometric_posture_keypoints() -> None:
    from attention_pipeline.stages.modalities import geometric_posture

    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 1.0
    k[:, 0] = np.linspace(10, 90, 17)
    k[:, 1] = np.linspace(20, 180, 17)
    probs = geometric_posture(k)
    assert probs.shape == (6,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_fusion_forward() -> None:
    net = AttentionFusionNet()
    x = torch_window()
    score, aux = net(x)
    assert score.shape == (2,)
    assert aux.shape == (2, 4)
    assert float(score.detach().min()) >= 0 and float(score.detach().max()) <= 100


def torch_window():
    import torch

    return torch.zeros(2, 16, FEAT_DIM)


def test_drop_sustained() -> None:
    det = AttentionDropDetector(
        DropConfig(low_threshold=40, sustain_seconds=3, cooldown_seconds=0, ema_alpha=1.0)
    )
    track = TrackState(student_id=7)
    feat = _feat(70, True)
    event = None
    for i in range(40):
        track.scores.append(20.0)
        track.timestamps.append(i * 0.1)
        event = det.update(track, feat)
    assert event is not None
    assert event.kind == "sustained"
    assert event.student_id == 7


def test_attention_bands_and_overlay() -> None:
    import numpy as np

    from attention_pipeline.stages.visualize import attention_band, draw_attention

    assert attention_band(82) == "HIGH"
    assert attention_band(55) == "MODERATE"
    assert attention_band(20) == "LOW"
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = draw_attention(
        frame,
        {
            "timestamp": 1.2,
            "students": [
                {"student_id": 1, "bbox": [40, 40, 160, 280], "score": 81.0},
                {"student_id": 2, "bbox": [200, 50, 320, 300], "score": 22.0},
            ],
            "events": [],
        },
    )
    assert out.shape == frame.shape
    assert out.sum() > 0


def _person(x: float, head_y: float, thigh: float) -> tuple:
    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [x, head_y + 28, 1.0]
    kpts[5] = [x - 18, head_y + 70, 1.0]
    kpts[6] = [x + 18, head_y + 70, 1.0]
    hip = head_y + 150
    kpts[11] = [x - 12, hip, 1.0]
    kpts[12] = [x + 12, hip, 1.0]
    kpts[13] = [x - 12, hip + thigh, 1.0]
    kpts[14] = [x + 12, hip + thigh, 1.0]
    box = np.array([x - 36, head_y, x + 36, head_y + 130], np.float32)
    return box, 0.8, kpts


def test_standing_instructor_is_not_marked() -> None:
    from attention_pipeline.stages.detect_track import _drop_instructors

    students = [_person(120 + i * 90, 280, 36) for i in range(8)]
    instructor = _person(1500, 90, 120)
    kept, remembered, misses = _drop_instructors(students + [instructor], 1080, None, 0)
    assert len(kept) == 8
    assert remembered is not None
    assert remembered[1] < 150
    assert misses == 0
    # A seated person with long visible legs stays, because their head is in the rows.
    front = _person(400, 520, 90)
    kept2, _, _ = _drop_instructors(students + [front], 1080, None, 0)
    assert len(kept2) == 9


def test_nested_boxes_count_as_one_student() -> None:
    from attention_pipeline.stages.detect_track import _suppress_duplicate_boxes

    outer = np.array([100.0, 80.0, 280.0, 320.0], np.float32)
    inner = np.array([140.0, 100.0, 220.0, 220.0], np.float32)
    k_outer = np.zeros((17, 3), np.float32)
    k_outer[0] = [190, 120, 0.9]
    k_outer[1] = [175, 115, 0.8]
    k_outer[2] = [205, 115, 0.8]
    k_inner = np.zeros((17, 3), np.float32)
    people = [(outer, 0.7, k_outer), (inner, 0.4, k_inner)]
    kept = _suppress_duplicate_boxes(people, frame_w=1920)
    assert len(kept) == 1


def _face_kpts(cx: float, cy: float, eye_dx: float = 10.0) -> np.ndarray:
    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [cx, cy + 5.0, 0.95]
    kpts[1] = [cx - eye_dx, cy, 0.95]
    kpts[2] = [cx + eye_dx, cy, 0.95]
    kpts[5] = [cx - 24.0, cy + 36.0, 0.9]
    kpts[6] = [cx + 24.0, cy + 36.0, 0.9]
    return kpts


def test_overlapping_adjacent_and_behind_row_students_stay_separate() -> None:
    from attention_pipeline.stages.detect_track import (
        DetectorTracker,
        _cluster_by_head,
        _suppress_duplicate_boxes,
    )

    # Real 1920x1080 multi-row geometry (e.g. runs/11.mp4): distinct students sit behind/beside
    # front-row students with high iomin (0.37 - 0.68) yet separate eye landmarks.
    front = (np.array([369.0, 513.0, 573.0, 727.0], np.float32), 0.60, _face_kpts(471, 566, 14.4))
    behind_shoulder = (np.array([462.0, 494.0, 578.0, 679.0], np.float32), 0.58, _face_kpts(519, 548, 11.7))
    front_center = (np.array([1050.0, 497.0, 1182.0, 697.0], np.float32), 0.88, _face_kpts(1116, 548, 13.3))
    behind_boy = (np.array([1006.0, 454.0, 1105.0, 608.0], np.float32), 0.65, _face_kpts(1070, 500, 9.6))
    back_left_a = (np.array([273.0, 452.0, 338.0, 602.0], np.float32), 0.86, _face_kpts(291, 496, 9.4))
    back_left_b = (np.array([194.0, 462.0, 305.0, 632.0], np.float32), 0.61, _face_kpts(250, 508, 11.2))
    # True duplicate on front_center (same face shifted by ~1.5 px) must still collapse.
    dup_front = (np.array([1048.0, 496.0, 1180.0, 696.0], np.float32), 0.35, _face_kpts(1117, 547, 13.3))

    clustered = _cluster_by_head(
        [front, behind_shoulder, front_center, behind_boy, back_left_a, back_left_b, dup_front],
        frame_w=1920,
    )
    assert len(clustered) == 6
    suppressed = _suppress_duplicate_boxes(clustered, frame_w=1920)
    assert len(suppressed) == 6

    # Tracking Stages 1, 4, and 6 in infer() must also preserve all 6 distinct students.
    det = DetectorTracker(n_init=1, auto_max_students=False)
    det._lazy_load = lambda: None
    det._detect_people = lambda _frame: list(suppressed)
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tracked = det.infer(dummy)
    assert len(tracked) == 6


def test_head_matched_box_expansion_keeps_track_and_suppresses_coasting_ghost() -> None:
    from attention_pipeline.stages.detect_track import DetectorTracker

    det = DetectorTracker(n_init=1, auto_max_students=False, max_age=30)
    det._lazy_load = lambda: None
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

    tight_face = (np.array([588.0, 440.0, 642.0, 506.0], np.float32), 0.85, _face_kpts(600.0, 447.0, 9.0))
    shoulder_kpts = np.zeros((17, 3), np.float32)
    shoulder_kpts[2] = [616.0, 443.0, 0.34]  # outside shoulder box
    shoulder_kpts[4] = [600.0, 444.0, 0.97]  # outside shoulder box
    shoulder_kpts[5] = [517.0, 500.0, 0.95]
    shoulder_kpts[6] = [566.0, 536.0, 0.99]
    shoulder_frag = (np.array([510.0, 454.0, 573.0, 537.0], np.float32), 0.82, shoulder_kpts)
    expanded = (np.array([543.0, 401.0, 631.0, 542.0], np.float32), 0.86, _face_kpts(588.0, 443.0, 9.0))

    frames = [
        [tight_face, shoulder_frag],
        [tight_face, shoulder_frag],
        [tight_face],
        [tight_face],
        [tight_face],
        [tight_face],
        [tight_face],
        [expanded],
    ]
    idx = {"i": 0}

    def fake_detect(_frame: np.ndarray):
        out = frames[idx["i"]]
        idx["i"] += 1
        return list(out)

    det._detect_people = fake_detect
    for _ in range(7):
        det.infer(dummy)
    tracked = det.infer(dummy)
    assert len(tracked) == 1
    assert tracked[0].track_id == 1

    # Also verify a coasting (age >= 4) face box overlapping a larger fresh box (iomin >= 0.55,
    # head dist < 28 px) does not linger alongside it as a second track.
    det2 = DetectorTracker(n_init=1, auto_max_students=False, max_age=30, smooth_box=False)
    det2._lazy_load = lambda: None
    wide = (np.array([543.0, 401.0, 631.0, 542.0], np.float32), 0.86, _face_kpts(578.0, 443.0, 9.0))
    shifted_face = (np.array([588.0, 440.0, 642.0, 506.0], np.float32), 0.85, _face_kpts(600.0, 447.0, 9.0))
    seq = [
        [wide, shifted_face],  # 2 separate tracks initially (dist ~22 px; face_kpts small eye_dx=9)
        [wide, shifted_face],
        [wide],
        [wide],
        [wide],
        [wide],  # shifted_face reaches age = 4 and overlaps wide with iomin > 0.55, dist ~22 px
    ]
    idx2 = {"i": 0}
    det2._detect_people = lambda _f: list(seq[idx2.__setitem__("i", idx2["i"] + 1) or (idx2["i"] - 1)])
    for _ in range(5):
        det2.infer(dummy)
    tracked2 = det2.infer(dummy)
    assert len(tracked2) == 1
    assert tracked2[0].track_id == 1

    # An isolated sleeping/head-down student with occluded face landmarks must still coast.
    det3 = DetectorTracker(n_init=1, auto_max_students=False, max_age=30)
    det3._lazy_load = lambda: None
    sleep_kpts = np.zeros((17, 3), np.float32)
    sleep_kpts[5] = [770.0, 420.0, 0.9]
    sleep_kpts[6] = [820.0, 420.0, 0.9]
    sleeping = (np.array([760.0, 390.0, 840.0, 490.0], np.float32), 0.75, sleep_kpts)
    peer = (np.array([100.0, 350.0, 180.0, 460.0], np.float32), 0.90, _face_kpts(140.0, 380.0, 10.0))
    seq3 = [[peer, sleeping], [peer, sleeping], [peer], [peer], [peer], [peer], [peer]]
    idx3 = {"i": 0}
    det3._detect_people = lambda _f: list(seq3[idx3.__setitem__("i", idx3["i"] + 1) or (idx3["i"] - 1)])
    for _ in range(6):
        det3.infer(dummy)
    tracked3 = det3.infer(dummy)
    assert len(tracked3) == 2
    assert {d.track_id for d in tracked3} == {1, 2}


def test_keep_real_students_drops_faceless_and_caps_count() -> None:
    from attention_pipeline.stages.detect_track import DetectorTracker

    det = DetectorTracker(require_face=True, min_det_conf=0.25, max_people=2)
    face = np.zeros((17, 3), np.float32)
    face[0] = [10, 10, 0.9]
    face[1] = [8, 9, 0.8]
    face[2] = [12, 9, 0.8]
    blank = np.zeros((17, 3), np.float32)
    people = [
        (np.array([0, 0, 40, 60], np.float32), 0.9, face),
        (np.array([50, 0, 90, 60], np.float32), 0.8, face.copy()),
        (np.array([100, 0, 140, 60], np.float32), 0.7, face.copy()),
        (np.array([150, 0, 190, 60], np.float32), 0.95, blank),
        (np.array([200, 0, 240, 60], np.float32), 0.1, face.copy()),
    ]
    kept = det._keep_real_students(people)
    assert len(kept) == 2
    assert all(_c >= 0.25 for _b, _c, _k in kept)


def test_writing_posture_is_not_misclassified_as_head_on_table() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.stages.modalities import geometric_posture

    # Writing posture from an elevated camera: nose 3 px below shoulder line
    # (head_clearance = -0.015), occluded hips, wrists resting on desk.
    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [50.0, 103.0, 0.9]   # nose (just below shoulders due to perspective)
    kpts[5] = [30.0, 100.0, 0.9]   # left shoulder
    kpts[6] = [70.0, 100.0, 0.9]   # right shoulder
    kpts[9] = [38.0, 175.0, 0.9]   # left wrist on desk
    kpts[10] = [62.0, 175.0, 0.9]  # right wrist on desk
    # hips remain conf=0.0 (desk occlusion)
    gp = geometric_posture(kpts)
    assert gp[5] == 0.0
    assert gp[1] == 1.0  # classified as reading, not leaning-over-table

    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    obs = to_student_frame(
        14,
        5.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([2.0, 38.0, 0.0], np.float32),
        gaze=np.array([2.0, 38.0], np.float32),
        emotion_probs=emotion,
        posture_probs=gp,
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=-1.0,
    )
    res = classify_attention(obs, lesson_yaw=4.0)
    assert res.band == "HIGH"
    assert res.cue in {"reading", "reading / writing", "writing notes"}

    # Even with uniform posture probs (no SCB override), pose cues alone yield desk work.
    obs_uniform = to_student_frame(
        16,
        5.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([2.0, 38.0, 0.0], np.float32),
        gaze=np.array([2.0, 38.0], np.float32),
        emotion_probs=emotion,
        posture_probs=np.full(6, 1.0 / 6.0, np.float32),
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=-1.0,
    )
    res_uniform = classify_attention(obs_uniform, lesson_yaw=4.0)
    assert res_uniform.band == "HIGH"
    assert res_uniform.cue == "reading / writing"

    # Upright student facing instructor with hands on desk stays "watching instructor".
    kpts_up = kpts.copy()
    kpts_up[0] = [50.0, 68.0, 0.9]  # head_clearance = (100 - 68)/200 = +0.16 (upright)
    obs_up = to_student_frame(
        1,
        5.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts_up,
        head_pose=np.array([2.0, 24.0, 0.0], np.float32),
        gaze=np.array([2.0, 24.0], np.float32),
        emotion_probs=emotion,
        posture_probs=np.full(6, 1.0 / 6.0, np.float32),
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=-6.0,
    )
    res_up = classify_attention(obs_up, lesson_yaw=4.0)
    assert res_up.cue == "watching instructor"

    # True deep head drop still registers as leaning-over-table / head on table.
    kpts_drop = kpts.copy()
    kpts_drop[0] = [50.0, 130.0, 0.9]
    assert geometric_posture(kpts_drop)[5] == 1.0

    # Deep overhead lean with SCB writing confirmation and normal face pitch stays writing.
    kpts_steep = kpts.copy()
    kpts_steep[0] = [50.0, 124.0, 0.9]  # head_clearance = -0.12
    scb_write_posture = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.5], np.float32)
    obs_steep = to_student_frame(
        12,
        2.2,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts_steep,
        head_pose=np.array([2.0, 27.2, 0.0], np.float32),
        gaze=np.array([2.0, 27.2], np.float32),
        emotion_probs=emotion,
        posture_probs=scb_write_posture,
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=-1.5,
    )
    res_steep = classify_attention(obs_steep, lesson_yaw=4.0)
    assert res_steep.band == "HIGH"
    assert res_steep.cue in {"reading / writing", "writing notes"}


def test_scb_match_all_containment_fallback() -> None:
    from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector, SCBDet
    from attention_pipeline.types import SCB_BEHAVIOR_LABELS

    det = SCBBehaviorDetector(weights=None)
    # Upper-torso pose bbox nested inside full-body + desk SCB writing box (IoU ~0.19)
    student_boxes = [np.array([1064.0, 398.0, 1145.0, 526.0], np.float32)]
    scb_dets = [
        SCBDet(
            bbox_xyxy=np.array([974.0, 376.0, 1147.0, 692.0], np.float32),
            cls=2,
            name="writing",
            conf=0.67,
        )
    ]
    matched = det.match_all(student_boxes, scb_dets, min_iou=0.4)
    assert matched[0] is not None
    assert SCB_BEHAVIOR_LABELS[int(np.argmax(matched[0]))] == "writing"


def test_chin_rest_does_not_trigger_hand_raised_or_head_on_table() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.stages.modalities import geometric_posture
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [60.0, 75.0, 0.9]   # nose
    kpts[5] = [40.0, 120.0, 0.9]  # left shoulder (head_clearance = (120-75)/200 = +0.225 upright)
    kpts[6] = [80.0, 120.0, 0.9]  # right shoulder
    kpts[9] = [68.0, 92.0, 0.9]   # wrist propping chin/cheek just below nose
    gp = geometric_posture(kpts)
    assert gp[0] < 0.25  # must remain uniform neutral, never hand-raising (1.0)
    k_hand_up = kpts.copy()
    k_hand_up[9] = [30.0, 45.0, 0.9]
    assert geometric_posture(k_hand_up)[0] == 1.0

    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    uniform = np.full(6, 1.0 / 6.0, np.float32)
    # Upright student touching chin/cheek with profile gimbal-lock pitch=52 must stay off head-on-table.
    obs_profile = to_student_frame(
        18,
        5.0,
        bbox=np.array([10.0, 40.0, 110.0, 240.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([48.0, 52.0, 0.0], np.float32),
        gaze=np.array([48.0, 52.0], np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.05,
        eye_aspect=0.3,
        mouth_aspect=0.2,
    )
    res_profile = classify_attention(obs_profile, lesson_yaw=10.0)
    assert res_profile.cue == "looking right"
    assert res_profile.band == "LOW"

    # Even if SCB classifier predicts hand-raising with high confidence, chin_rest cancels it.
    assert scb_attention_override("hand-raising", cues={"chin_rest": True}, conf=0.9) is None

    # Attentive student facing lesson and reading with chin propped stays HIGH reading/taking notes.
    obs_study = to_student_frame(
        13,
        5.0,
        bbox=np.array([10.0, 40.0, 110.0, 240.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([12.0, 30.0, 0.0], np.float32),
        gaze=np.array([12.0, 30.0], np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.03,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=5.0,
    )
    res_study = classify_attention(obs_study, lesson_yaw=10.0)
    assert res_study.band == "HIGH"
    assert res_study.cue == "reading / taking notes"


def test_sideways_turn_is_not_masked_by_desk_work_or_neutral_tilt() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [50.0, 82.0, 0.9]
    kpts[5] = [30.0, 100.0, 0.9]
    kpts[6] = [70.0, 100.0, 0.9]
    kpts[9] = [38.0, 175.0, 0.9]
    kpts[10] = [62.0, 175.0, 0.9]
    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    uniform = np.full(6, 1.0 / 6.0, np.float32)

    # Clear sideways turn (delta = 40 deg >= 34) with normal reading pitch is still looking right (LOW).
    obs_turn = to_student_frame(
        18,
        3.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([50.0, 30.0, 0.0], np.float32),
        gaze=np.array([50.0, 30.0], np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.03,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=0.0,
    )
    res_turn = classify_attention(obs_turn, lesson_yaw=10.0)
    assert res_turn.band == "LOW"
    assert res_turn.cue == "looking right"
    assert scb_attention_override("reading", cues={"off_lesson": True, "look_right": True}, conf=0.95) is None

    # Side-staring student (delta = 23 deg >= 20) with hands near desk must not inflate to HIGH reading / writing.
    obs_side = to_student_frame(
        4,
        3.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([33.0, 33.0, 0.0], np.float32),
        gaze=np.array([48.0, 20.0], np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.03,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=1.0,
        rel_gaze_yaw=32.0,
    )
    res_side = classify_attention(obs_side, lesson_yaw=10.0)
    assert res_side.band == "LOW"
    assert res_side.cue == "looking right"

    # Slightly off-center student genuinely working on notes (delta = 19 deg < 20) stays HIGH reading / writing.
    obs_desk = to_student_frame(
        5,
        3.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([23.0, 36.0, 0.0], np.float32),
        gaze=np.array([22.0, 36.0], np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=4.0,
    )
    res_desk = classify_attention(obs_desk, lesson_yaw=4.0)
    assert res_desk.band == "HIGH"
    assert res_desk.cue == "reading / writing"


def test_seated_narrow_profile_with_extrapolated_hips_is_not_standing() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention

    # Tight upper-body crop (45x98, aspect=2.18) with extrapolated hips outside y2 and no visible knees/ankles.
    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [427.0, 397.0, 0.85]
    kpts[5] = [426.0, 422.0, 0.80]
    kpts[6] = [380.0, 425.0, 0.95]
    kpts[11] = [418.0, 518.0, 0.51]
    kpts[12] = [385.0, 521.0, 0.81]
    kpts[13] = [414.0, 542.0, 0.05]
    kpts[14] = [378.0, 543.0, 0.12]
    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    obs = to_student_frame(
        10,
        3.0,
        bbox=np.array([395.0, 365.0, 440.0, 463.0], np.float32),
        keypoints=kpts,
        head_pose=np.array([10.0, 15.0, 0.0], np.float32),
        gaze=np.array([10.0, 15.0], np.float32),
        emotion_probs=emotion,
        posture_probs=np.full(6, 1.0 / 6.0, np.float32),
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=-6.0,
    )
    res = classify_attention(obs, lesson_yaw=8.0)
    assert res.cue != "standing / presenting"


def test_phone_verification_suppresses_bottles_and_mouth_pens() -> None:
    import torch

    from attention_pipeline.pipeline import _phone_min_y, _verified_phone_boxes

    class _FakeBox:
        def __init__(self, xyxy, cls_id: int, conf: float) -> None:
            self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
            self.cls = torch.tensor([cls_id], dtype=torch.float32)
            self.conf = torch.tensor([conf], dtype=torch.float32)

    class _FakeRes:
        def __init__(self, boxes) -> None:
            self.boxes = boxes

    # Bottle overlapping cell phone with higher/competitive conf suppresses the false phone hit.
    res_bottle = _FakeRes([
        _FakeBox([100, 200, 135, 290], 67, 0.40),
        _FakeBox([101, 198, 136, 292], 39, 0.51),
    ])
    assert _verified_phone_boxes(res_bottle) == []

    # True phone without competing bottle/cup/book stays verified.
    res_phone = _FakeRes([
        _FakeBox([200, 250, 240, 295], 67, 0.78),
    ])
    kept = _verified_phone_boxes(res_phone)
    assert len(kept) == 1 and abs(kept[0][1] - 0.78) < 1e-5

    # Pen held up near lips (y=130) sits above shoulder floor_y, while lap phone (y=260) is below.
    kpts = np.zeros((17, 3), np.float32)
    kpts[5] = [100.0, 160.0, 0.9]
    kpts[6] = [140.0, 160.0, 0.9]
    floor_y = _phone_min_y(kpts, sy1=100.0, sbh=180.0)
    assert 130.0 < floor_y < 260.0


def test_mouth_activity_resists_jitter_and_flicker_while_detecting_speech() -> None:
    from attention_pipeline.stages import face_evidence

    canvas = np.full((64, 64, 3), 130, dtype=np.uint8)
    # Draw facial texture + a closed dark horizontal mouth line
    canvas[20:46, 18:46] = (135, 145, 160)
    canvas[34:36, 25:39] = (45, 40, 50)

    p0 = canvas[22:46, 20:44].copy()
    # Jittered crop (shifted 1 px) + uniform brightness shift (+14): not speech.
    p_jitter = np.clip(canvas[23:47, 21:45].astype(np.int16) + 14, 0, 255).astype(np.uint8)
    act_jitter = face_evidence.mouth_activity(p0, p_jitter)
    assert not face_evidence.mouth_is_active(act_jitter), f"Expected quiet, got {act_jitter}"

    # Open mouth (darker taller oval inside the same crop): active speech.
    canvas_open = canvas.copy()
    canvas_open[31:39, 25:39] = (30, 25, 35)
    p_open = canvas_open[22:46, 20:44].copy()
    act_open = face_evidence.mouth_activity(p0, p_open)
    assert face_evidence.mouth_is_active(act_open), f"Expected speech, got {act_open}"


def test_face_extractor_profile_pose_mouth_unclipped_and_turn_gated() -> None:
    from types import SimpleNamespace
    import cv2

    from attention_pipeline.stages.modalities import FaceModalityExtractor

    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")

    # Occluded left eye at [0, 0] (conf=0.01) must not pull mid_eye to origin (no bogus +90 yaw).
    kpts_occ = np.zeros((17, 3), dtype=np.float32)
    kpts_occ[0] = [320.0, 220.0, 0.90]
    kpts_occ[1] = [0.0, 0.0, 0.01]
    kpts_occ[2] = [308.0, 206.0, 0.85]
    pose_occ = extractor._head_pose(landmarks=None, kpts=kpts_occ)
    assert 0.0 < abs(float(pose_occ[0])) < 60.0, f"Unexpected occluded-eye yaw: {pose_occ}"

    # Turned/tilted profile near face crop edge still extracts unclipped full-frame mouth patch.
    frame = np.full((400, 500, 3), 120, dtype=np.uint8)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [260.0, 205.0, 0.90]
    kpts[1] = [252.0, 190.0, 0.85]
    kpts[2] = [248.0, 191.0, 0.25]
    frame[180:250, 225:280] = (140, 150, 165)
    cv2.rectangle(frame, (246, 217), (264, 221), (40, 35, 45), -1)
    det = SimpleNamespace(
        bbox_xyxy=np.array([210.0, 165.0, 310.0, 345.0], dtype=np.float32),
        keypoints=kpts,
    )
    sig0 = extractor.extract(frame, det, student_id=7)
    assert sig0.mouth_openness is not None

    # Open lips while head remains steady -> detects mouth motion >= 0.02.
    frame_talk = frame.copy()
    cv2.rectangle(frame_talk, (246, 214), (264, 227), (25, 22, 30), -1)
    sig1 = extractor.extract(frame_talk, det, student_id=7)
    assert sig1.mouth_motion >= 0.02, f"Expected lip motion >= 0.02, got {sig1.mouth_motion}"

    # Rapid head rotation jump (> 5.5 deg) suppresses rigid turn artifact.
    kpts_turn = kpts.copy()
    kpts_turn[0, 0] = 275.0
    det_turn = SimpleNamespace(bbox_xyxy=det.bbox_xyxy, keypoints=kpts_turn)
    sig_turn = extractor.extract(frame_talk, det_turn, student_id=7)
    assert sig_turn.mouth_motion == 0.0


def test_peer_talk_leaning_whisper_row_guards_and_phone_priority() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import (
        SpeechPersistenceTracker,
        apply_peer_cues,
        classify_attention,
    )

    def _student(
        sid: int,
        yaw: float,
        bbox: list[float],
        nose_xy: tuple[float, float],
        sh_y: float,
        speaking: bool = False,
        phone_obj: bool = False,
        pitch: float = 16.0,
    ):
        kpts = np.zeros((17, 3), dtype=np.float32)
        nx, ny = nose_xy
        cx = 0.5 * (bbox[0] + bbox[2])
        kpts[0] = [nx, ny, 0.9]
        kpts[5] = [cx - 20.0, sh_y, 0.9]
        kpts[6] = [cx + 20.0, sh_y, 0.9]
        return to_student_frame(
            sid,
            1.0,
            bbox=np.array(bbox, dtype=np.float32),
            keypoints=kpts,
            head_pose=np.array([yaw, pitch, 0.0], dtype=np.float32),
            gaze=np.array([yaw, pitch], dtype=np.float32),
            emotion_probs=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            posture_probs=np.full(6, 1.0 / 6.0, dtype=np.float32),
            motion=0.04,
            eye_aspect=0.3,
            mouth_aspect=0.2,
            mouth_motion=0.04 if speaking else 0.0,
            mouth_active=speaking,
            aspects_valid=True,
            has_phone_obj=phone_obj,
        )

    # 1. Close leaning whisperers (one upright, one leaning forward > 0.40*h lower at nose
    # without collapsing onto the table, same shoulder row, subtle +/-16 deg reciprocal turns
    # with sustained speech) must both trigger "talking to peer".
    a = _student(1, 16.0, [100, 90, 200, 270], (165, 120), sh_y=190.0, speaking=True)
    b = _student(2, -15.0, [215, 130, 315, 310], (250, 212), sh_y=215.0, speaking=True)
    tracker = SpeechPersistenceTracker(required=3)
    for _ in range(3):
        tracker.update(1, True)
        tracker.update(2, True)
    res_whisper = apply_peer_cues(
        [a, b],
        [classify_attention(a, 0.0), classify_attention(b, 0.0)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert res_whisper[0].cue == "talking to peer"
    assert res_whisper[1].cue == "talking to peer"

    # 2. Front-behind depth stacking (iomin >= 0.30 + shifted bottom edge) must NOT pair.
    front = _student(3, 28.0, [100, 180, 220, 380], (160, 220), sh_y=270.0, speaking=True)
    behind = _student(4, -26.0, [150, 120, 260, 300], (210, 155), sh_y=210.0, speaking=False)
    res_depth = apply_peer_cues(
        [front, behind],
        [classify_attention(front, 0.0), classify_attention(behind, 0.0)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert all(r.cue != "talking to peer" for r in res_depth)

    # 3. Intervening quiet student sitting between two outer students blocks pairing across them.
    left = _student(11, 19.0, [0, 100, 100, 280], (50, 130), sh_y=185.0, speaking=True)
    mid = _student(12, 0.0, [105, 100, 195, 280], (150, 130), sh_y=185.0, speaking=False)
    right = _student(13, -23.0, [200, 100, 300, 280], (250, 130), sh_y=185.0, speaking=False)
    for _ in range(3):
        tracker.update(11, True)
    res_pair_direct = apply_peer_cues(
        [left, right],
        [classify_attention(x, 0.0) for x in (left, right)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert all(r.cue == "talking to peer" for r in res_pair_direct)
    res_mid = apply_peer_cues(
        [left, mid, right],
        [classify_attention(x, 0.0) for x in (left, mid, right)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert all(r.cue != "talking to peer" for r in res_mid)

    # 4. Phone priority: phone user stays "using phone" even if turning & moving lips near peer.
    phone_user = _student(
        21, 24.0, [100, 100, 200, 280], (150, 130), sh_y=185.0, speaking=True, phone_obj=True, pitch=26.0
    )
    peer = _student(22, -24.0, [210, 100, 310, 280], (260, 130), sh_y=185.0, speaking=True)
    for _ in range(3):
        tracker.update(21, True)
        tracker.update(22, True)
    res_prio = apply_peer_cues(
        [phone_user, peer],
        [classify_attention(phone_user, 0.0), classify_attention(peer, 0.0)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert res_prio[0].cue == "using phone"
    assert res_prio[1].cue == "talking to peer"


def test_pose_reliable_false_slumped_on_desk_never_watching_instructor() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention

    # Slumped sleeper whose face keypoints fell below conf threshold (pose_valid=False -> [0,0,0] fallback)
    # with SCB leaning-over-table posture must stay LOW sleeping/head on table, never "watching instructor".
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = [100.0, 200.0, 0.9]
    kpts[6] = [160.0, 200.0, 0.9]
    posture = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.92], dtype=np.float32)
    obs = to_student_frame(
        12,
        4.0,
        bbox=np.array([70.0, 170.0, 190.0, 310.0], dtype=np.float32),
        keypoints=kpts,
        head_pose=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        gaze=np.array([0.0, 0.0], dtype=np.float32),
        emotion_probs=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        posture_probs=posture,
        motion=0.01,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pose_reliable=False,
        pitch_delta=None,
    )
    res = classify_attention(obs, lesson_yaw=0.0)
    assert res.band == "LOW"
    assert res.cue in {"sleeping", "head on table"}


def test_keep_real_students_retains_deep_desk_sleeper_with_valid_shoulders() -> None:
    from attention_pipeline.stages.detect_track import DetectorTracker

    det = DetectorTracker(require_face=False, min_det_conf=0.25)
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Box [320, 420, 440, 560] -> bw=120, bh=140; shoulders at y=460, nose dropped deep at y=505 (>= sh_y + 0.15*bh = 481)
    kpts[0] = [380.0, 505.0, 0.65]
    kpts[5] = [345.0, 460.0, 0.98]
    kpts[6] = [415.0, 460.0, 0.98]
    sleeper = (np.array([320.0, 420.0, 440.0, 560.0], dtype=np.float32), 0.89, kpts)
    kept = det._keep_real_students([sleeper], frame_w=1024)
    assert len(kept) == 1


def test_drop_instructors_catches_576p_and_720p_standing_instructors() -> None:
    from attention_pipeline.stages.detect_track import _drop_instructors

    # 576p classroom with standing instructor ~22 px above 20th-percentile student head band
    students_576 = [_person(100 + i * 80, 232.0, 20.0) for i in range(6)]
    instructor_576 = _person(750.0, 210.0, 90.0)
    kept_576, remembered, misses = _drop_instructors(students_576 + [instructor_576], 576, None, 0)
    assert len(kept_576) == 6
    assert remembered is not None and misses == 0

    # Seated student well below the band stays tracked
    seated = _person(320.0, 286.0, 20.0)
    kept_seated, _, _ = _drop_instructors(students_576 + [seated], 576, remembered, misses)
    assert len(kept_seated) == 7

    # 720p standing instructor ~43 px above band is also dropped
    students_720 = [_person(120 + i * 90, 200.0, 24.0) for i in range(6)]
    instructor_720 = _person(900.0, 157.0, 110.0)
    kept_720, _, _ = _drop_instructors(students_720 + [instructor_720], 720, None, 0)
    assert len(kept_720) == 6


def test_verified_phone_boxes_rejects_pen_slivers_and_bottom_strips_while_keeping_real_phones() -> None:
    import torch

    from attention_pipeline.pipeline import _verified_phone_boxes

    class _FakeBox:
        def __init__(self, xyxy, cls_id: int, conf: float) -> None:
            self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
            self.cls = torch.tensor([cls_id], dtype=torch.float32)
            self.conf = torch.tensor([conf], dtype=torch.float32)

    class _FakeRes:
        def __init__(self, boxes) -> None:
            self.boxes = boxes

    # Pen sliver (22x10, max side < 25.5) and extreme aspect strip (45x9) are rejected.
    slivers = _FakeRes([
        _FakeBox([100, 150, 122, 160], 67, 0.72),
        _FakeBox([150, 200, 195, 209], 67, 0.80),
        # Frame-bottom clipped paper strip (by2 = 175 + offset_y 400 = 575 >= 576 - 3, bh=14 < 20)
        _FakeBox([210, 161, 240, 175], 67, 0.75),
    ])
    assert _verified_phone_boxes(slivers, frame_h=576, offset_y=400.0) == []

    # Slim genuine smartphone (31x11, hi=31 >= 25.5, lo=11 >= 9.5, ratio=2.82 <= 4.2) stays verified.
    real_phone = _FakeRes([
        _FakeBox([200, 100, 231, 111], 67, 0.65),
    ])
    kept = _verified_phone_boxes(real_phone, frame_h=576, offset_y=200.0)
    assert len(kept) == 1 and abs(kept[0][1] - 0.65) < 1e-5


def test_peer_talk_guards_same_way_presenter_look_and_walker_occlusion() -> None:
    from types import SimpleNamespace
    import cv2

    from attention_pipeline.features import to_student_frame
    from attention_pipeline.pipeline import _mouth_occluded
    from attention_pipeline.rubric import (
        SpeechPersistenceTracker,
        apply_peer_cues,
        classify_attention,
    )
    from attention_pipeline.stages.modalities import FaceModalityExtractor

    # 1. Both neighbors look the SAME direction (toward a side presenter) while one happens
    # to have mouth activity -> must NOT mark unilateral "talking to peer".
    def _s(sid: int, yaw: float, x0: float, speaking: bool):
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [x0 + 50.0, 140.0, 0.9]
        kpts[5] = [x0 + 25.0, 185.0, 0.9]
        kpts[6] = [x0 + 75.0, 185.0, 0.9]
        return to_student_frame(
            sid,
            2.0,
            bbox=np.array([x0, 100.0, x0 + 100.0, 280.0], dtype=np.float32),
            keypoints=kpts,
            head_pose=np.array([yaw, 18.0, 0.0], dtype=np.float32),
            gaze=np.array([yaw, 18.0], dtype=np.float32),
            emotion_probs=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            posture_probs=np.full(6, 1.0 / 6.0, dtype=np.float32),
            motion=0.04,
            eye_aspect=0.3,
            mouth_aspect=0.2,
            mouth_motion=0.05 if speaking else 0.0,
            mouth_active=speaking,
            aspects_valid=True,
        )

    s_a = _s(10, -28.0, 100.0, speaking=True)   # looking image-left (toward presenter), silent neighbor also looking left
    s_b = _s(20, -24.0, 0.0, speaking=False)
    tracker = SpeechPersistenceTracker(required=3)
    for _ in range(3):
        tracker.update(10, True)
    res = apply_peer_cues(
        [s_a, s_b],
        [classify_attention(s_a, 0.0), classify_attention(s_b, 0.0)],
        lesson_yaw=0.0,
        speech_tracker=tracker,
    )
    assert all(r.cue != "talking to peer" for r in res)

    # 2. Foreground walker crossing seated student triggers _mouth_occluded and suppresses mouth_motion.
    seated_kpts = _face_kpts(250.0, 200.0, 10.0)
    seated_box = np.array([200.0, 160.0, 300.0, 320.0], dtype=np.float32)
    walker_box = np.array([210.0, 110.0, 320.0, 360.0], dtype=np.float32)
    det_seated = SimpleNamespace(bbox_xyxy=seated_box, keypoints=seated_kpts)
    assert _mouth_occluded(det_seated, [seated_box, walker_box], self_idx=0) is True

    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    f0 = np.full((400, 400, 3), 120, dtype=np.uint8)
    f1 = f0.copy()
    cv2.rectangle(f1, (220, 190), (280, 260), (20, 20, 220), -1)
    extractor.extract(f0, det_seated, student_id=6, suppress_mouth=False)
    sig_occ = extractor.extract(f1, det_seated, student_id=6, suppress_mouth=True)
    assert sig_occ.mouth_motion == 0.0


def test_side_parallax_desk_work_stays_reading_writing_while_side_gaze_stays_low() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [50.0, 82.0, 0.9]
    kpts[5] = [30.0, 100.0, 0.9]
    kpts[6] = [70.0, 100.0, 0.9]
    kpts[9] = [38.0, 175.0, 0.9]
    kpts[10] = [62.0, 175.0, 0.9]
    emotion = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    uniform = np.full(6, 1.0 / 6.0, dtype=np.float32)

    # Outer desk column parallax (delta=25 deg < 30, pitch=36 >= 24, rel_gaze_yaw=5 deg < 22)
    obs_parallax = to_student_frame(
        9,
        4.0,
        bbox=np.array([0.0, 50.0, 100.0, 250.0], dtype=np.float32),
        keypoints=kpts,
        head_pose=np.array([29.0, 36.0, 0.0], dtype=np.float32),
        gaze=np.array([9.0, 36.0], dtype=np.float32),
        emotion_probs=emotion,
        posture_probs=uniform,
        motion=0.02,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        pitch_delta=2.0,
        rel_gaze_yaw=5.0,
    )
    res_parallax = classify_attention(obs_parallax, lesson_yaw=4.0)
    assert res_parallax.band == "HIGH"
    assert res_parallax.cue == "reading / writing"




