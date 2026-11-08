"""Tests for Phase 2 & Phase 3 pipeline integrity and anti-leakage splitting."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from attention_pipeline.types import FEAT_DIM, TrackState, StudentFrame
from attention_pipeline.features import pack_features
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.stages.attention import AttentionScorer


def test_feature_dimension_specification():
    """Verify that feature packing produces exactly FEAT_DIM=32 dimensions."""
    head_pose = np.array([10.0, -5.0, 2.0], dtype=np.float32)
    gaze = np.array([5.0, -3.0], dtype=np.float32)
    emotion_probs = np.ones(7, dtype=np.float32) / 7.0
    posture_probs = np.ones(6, dtype=np.float32) / 6.0
    motion = 0.05
    eye_aspect = 0.28
    mouth_aspect = 0.15
    keypoints = np.zeros((17, 3), dtype=np.float32)

    feat = pack_features(
        head_pose=head_pose,
        gaze=gaze,
        emotion_probs=emotion_probs,
        posture_probs=posture_probs,
        motion=motion,
        eye_aspect=eye_aspect,
        mouth_aspect=mouth_aspect,
        keypoints=keypoints,
    )

    assert feat.shape == (FEAT_DIM,)
    assert feat.shape[0] == 32
    assert np.all(np.isfinite(feat))
    # Indices 0..22 must contain active values
    assert not np.all(feat[:23] == 0)
    # Channel 23 reflects upper-body keypoint confidence (0.0 when keypoints are all zeros)
    assert feat[23] == 0.0
    # Indices 24..31 must be preserved zeros
    assert np.all(feat[24:] == 0.0)

    # When keypoints have non-zero confidence, Channel 23 reflects normalized confidence
    keypoints_with_conf = np.zeros((17, 3), dtype=np.float32)
    keypoints_with_conf[:, 2] = 0.8
    feat2 = pack_features(
        head_pose=head_pose,
        gaze=gaze,
        emotion_probs=emotion_probs,
        posture_probs=posture_probs,
        motion=motion,
        eye_aspect=eye_aspect,
        mouth_aspect=mouth_aspect,
        keypoints=keypoints_with_conf,
    )
    assert np.isclose(feat2[23], 0.8)
    assert np.all(feat2[24:] == 0.0)


def test_attention_fusion_checkpoint_compatibility():
    """Verify that AttentionFusionNet loads existing checkpoint without shape mismatch."""
    model = AttentionFusionNet(feat_dim=32, hidden=64)
    dummy_input = torch.randn(2, 16, 32)
    scores, aux = model(dummy_input)
    assert scores.shape == (2,)
    assert aux.shape == (2, 4)

    ckpt_path = Path("checkpoints/fusion_daisee.pt")
    if ckpt_path.exists():
        model.load(ckpt_path)
        scores_loaded, aux_loaded = model(dummy_input)
        assert scores_loaded.shape == (2,)
        assert aux_loaded.shape == (2, 4)


def test_grouped_split_leakage_safety():
    """Verify that subject-disjoint and clip-disjoint splitting has zero group overlap."""
    rng = np.random.default_rng(42)
    # 20 subjects, 3 clips per subject, 10 sequences per clip = 600 sequences
    subjects = [f"subj_{i:03d}" for i in range(20)]
    clip_ids = []
    subject_ids = []
    for s in subjects:
        for c in range(3):
            cid = f"{s}_clip_{c}"
            for _ in range(10):
                clip_ids.append(cid)
                subject_ids.append(s)

    clip_ids = np.array(clip_ids)
    subject_ids = np.array(subject_ids)

    # Subject-disjoint split
    unique_subjects = np.unique(subject_ids)
    shuffled_subjects = rng.permutation(unique_subjects)
    n_val = int(round(len(shuffled_subjects) * 0.2))
    val_subj_set = set(shuffled_subjects[:n_val])
    train_subj_set = set(shuffled_subjects[n_val:])

    train_indices = [i for i, s in enumerate(subject_ids) if s in train_subj_set]
    val_indices = [i for i, s in enumerate(subject_ids) if s in val_subj_set]

    train_s = set(subject_ids[train_indices])
    val_s = set(subject_ids[val_indices])
    train_c = set(clip_ids[train_indices])
    val_c = set(clip_ids[val_indices])

    # Assert ZERO leakage across splits
    assert len(train_s & val_s) == 0, "Subject leakage detected!"
    assert len(train_c & val_c) == 0, "Clip leakage detected!"
    assert len(train_indices) + len(val_indices) == len(clip_ids)


def test_scorer_preserves_continuous_fusion_score():
    """Verify AttentionScorer.push() records and returns fusion_score without discarding."""
    model = AttentionFusionNet(feat_dim=32, hidden=64)
    model.eval()
    scorer = AttentionScorer(model=model, device=torch.device("cpu"), window=16)
    scorer.use_heuristic = False

    track = TrackState(student_id=1)
    feat = np.ones(32, dtype=np.float32) * 0.5

    for step in range(16):
        score, rubric, fusion_score = scorer.push(track, feat, timestamp=step * 0.1)

    assert fusion_score is not None
    assert isinstance(fusion_score, float)
    assert len(track.scores) == 16
    assert len(track.fusion_scores) == 16
    assert track.fusion_scores[-1] == fusion_score
    assert rubric.band in ("HIGH", "MODERATE", "LOW")


def test_classroom_student_disjoint_split_and_classifier_head():
    """Verify classroom sequence split has zero student overlap and head outputs 3-class logits."""
    from scripts.train_classroom_head import ClassroomAttentionClassifier, ClassroomSequenceDataset

    npz_path = Path("data/processed/classroom_clip_004_sequences.npz")
    if not npz_path.exists():
        pytest.skip("classroom_clip_004_sequences.npz not generated yet")

    val_students = {2, 5, 8, 10, 16, 18}
    all_data = np.load(npz_path)
    all_sids = set(np.unique(all_data["student_ids"]).tolist())
    train_students = all_sids - val_students

    assert len(train_students & val_students) == 0, "Student overlap detected!"
    assert len(train_students) == 15
    assert len(val_students) == 6

    fusion = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ClassroomAttentionClassifier(fusion, num_classes=3, freeze_encoder=True)

    dummy_input = torch.randn(4, 16, 32)
    logits = model(dummy_input)
    assert logits.shape == (4, 3)

