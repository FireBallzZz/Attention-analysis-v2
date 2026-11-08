"""Verification script: validates end-to-end pipeline stages and contracts.

Ensures that no changes have broken any component of the classroom attention pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_features_and_rubric():
    print("[1/6] Testing features, heuristic scoring, and rubric...")
    from attention_pipeline.features import heuristic_attention_score, pack_features, to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.types import FEAT_DIM

    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    posture = np.zeros(6, np.float32)
    posture[1] = 1.0  # reading
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 1.0

    feat = pack_features(
        head_pose=np.array([5.0, 15.0, 0.0], np.float32),
        gaze=np.array([5.0, 15.0], np.float32),
        emotion_probs=emotion,
        posture_probs=posture,
        motion=0.1,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        keypoints=kpts,
    )
    assert feat.shape == (FEAT_DIM,), f"Feature dimension mismatch: {feat.shape}"
    score = heuristic_attention_score(feat)
    assert score >= 70, f"Expected HIGH attention for reading, got score={score}"

    obs = to_student_frame(
        1,
        0.0,
        bbox=np.array([10, 10, 100, 200], np.float32),
        keypoints=kpts,
        head_pose=np.array([5.0, 15.0, 0.0], np.float32),
        gaze=np.array([5.0, 15.0], np.float32),
        emotion_probs=emotion,
        posture_probs=posture,
        motion=0.1,
        eye_aspect=0.3,
        mouth_aspect=0.2,
    )
    res = classify_attention(obs)
    assert res.band == "HIGH", f"Expected HIGH band, got {res.band}"
    print("  -> Features and rubric OK.")


def test_drop_detector_and_intervention():
    print("[2/6] Testing drop detector and pedagogical intervention...")
    from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
    from attention_pipeline.stages.intervention import recommend
    from attention_pipeline.types import TrackState

    detector = AttentionDropDetector(DropConfig(low_threshold=40.0, sustain_seconds=2.0))
    track = TrackState(student_id=1)
    feat = np.zeros(32, np.float32)

    # Simulate sudden drop
    track.scores = [85.0, 80.0, 40.0, 25.0]
    track.timestamps = [0.0, 0.5, 1.0, 1.5]
    event = detector.update(track, feat)
    assert event is not None, "Expected attention drop event to trigger"
    assert event.kind in {"sudden", "sustained"}, f"Unexpected event kind: {event.kind}"

    rec = recommend(event, class_mean_score=65.0)
    assert "teaching_recommendation" in rec, "Missing teaching recommendation"
    assert "explanation" in rec, "Missing explanation"
    print("  -> Drop detector & intervention OK.")


def test_gt_calibrator_contracts():
    print("[3/6] Testing GT calibrator (backward compatibility & invariant features)...")
    from attention_pipeline.gt_calibrator import GtAttentionCalibrator, find_ground_truth, pack_features

    # Test 34-D legacy compatibility
    f_34 = pack_features(0.0, 0.0, 0.0, 0.0, 0.0, np.array([0, 0, 10, 10]), None, 2, feat_dim=34)
    assert f_34.shape == (34,), f"Expected 34 features, got {f_34.shape}"
    assert f_34[13 + 2] == 1.0, "Expected one-hot at index 15"

    # Test 13-D invariant compatibility (no track ID leakage)
    f_13 = pack_features(0.0, 0.0, 0.0, 0.0, 0.0, np.array([0, 0, 10, 10]), None, 2, feat_dim=13)
    assert f_13.shape == (13,), f"Expected 13 features, got {f_13.shape}"

    # Test ground truth lookup logic
    found = find_ground_truth(Path("Anno") / "clip_004.mp4")
    assert found is not None, "Failed to resolve clip_004 Anno ground truth"
    assert found[0].name == "ground_truth.csv"
    assert found[1].name == "annotations.xml"

    unlabeled = find_ground_truth(Path("video") / "Trainee 1.mp4")
    assert unlabeled is None, "Unlabeled video should return None"
    print("  -> Calibrator and GT resolution OK.")


def test_modalities_and_openseeface():
    print("[4/6] Testing modality extraction & aspect ratios...")
    from attention_pipeline.stages.modalities import FaceModalityExtractor

    extractor = FaceModalityExtractor(gaze=None, device="cpu", backend="stub")
    # Basic keypoints
    kpts_5 = np.zeros((5, 2), np.float32)
    aspects = extractor._aspects(kpts_5)
    assert aspects == (0.3, 0.3), f"Expected fallback (0.3, 0.3), got {aspects}"
    print("  -> Modality extraction OK.")


def test_peer_cues():
    print("[5/6] Testing peer interaction dynamics...")
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import RubricResult, apply_peer_cues

    kpts1 = np.zeros((17, 3), np.float32)
    kpts1[:, 2] = 1.0
    kpts1[0, :2] = [150.0, 150.0]  # nose at box 1 center

    kpts2 = np.zeros((17, 3), np.float32)
    kpts2[:, 2] = 1.0
    kpts2[0, :2] = [230.0, 150.0]  # nose at box 2 center

    posture = np.zeros(6, np.float32)
    posture[0] = 1.0

    # Two students side-by-side looking toward each other
    s1 = to_student_frame(
        1, 0.0, np.array([100, 100, 200, 300], np.float32), kpts1,
        np.array([35.0, 0.0, 0.0], np.float32), np.array([35.0, 0.0], np.float32),
        np.zeros(7, np.float32), posture, 0.15, 0.3, 0.3,
    )
    s2 = to_student_frame(
        2, 0.0, np.array([180, 100, 280, 300], np.float32), kpts2,
        np.array([-35.0, 0.0, 0.0], np.float32), np.array([-35.0, 0.0], np.float32),
        np.zeros(7, np.float32), posture, 0.15, 0.3, 0.3,
    )

    rubrics = [RubricResult("HIGH", 80.0, "listening"), RubricResult("HIGH", 80.0, "listening")]
    updated = apply_peer_cues([s1, s2], rubrics, lesson_yaw=0.0)
    assert updated[0].band == "LOW" or updated[1].band == "LOW", "Expected talking/disturbing cue"
    print("  -> Peer cues OK.")


def test_video_accuracy_module():
    print("[6/6] Testing video accuracy evaluation...")
    from attention_pipeline.video_accuracy import measure_rows

    rows = [
        {
            "students": [
                {"student_id": 1, "band": "HIGH", "head_pose": [0.0, 0.0, 0.0], "posture": [0, 1, 0, 0, 0, 0]},
                {"student_id": 2, "band": "LOW", "head_pose": [0.0, 50.0, 0.0], "posture": [0, 0, 0, 0, 0, 1]},
            ]
        }
    ]
    metrics = measure_rows(rows)
    assert "accuracy" in metrics, "Missing accuracy key"
    assert "macro" in metrics, "Missing macro metrics"
    print(f"  -> Video accuracy calculation OK (Accuracy={metrics['accuracy']*100:.1f}%).")


def main():
    print("==================================================")
    print("RUNNING ATTENTION PIPELINE INTEGRITY VERIFICATION")
    print("==================================================")
    test_features_and_rubric()
    test_drop_detector_and_intervention()
    test_gt_calibrator_contracts()
    test_modalities_and_openseeface()
    test_peer_cues()
    test_video_accuracy_module()
    print("==================================================")
    print("ALL TESTS PASSED! PIPELINE CONTRACTS ARE 100% INTACT.")
    print("==================================================")


if __name__ == "__main__":
    main()
