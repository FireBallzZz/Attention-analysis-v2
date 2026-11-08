"""Phase 6B: Feature Extraction Hardening and Regression Test Suite.

Verifies:
1. Confidence-aware hip gating (c >= 0.25) in torso_lean().
2. Safe shoulder-tilt fallback when hips are occluded (desk occlusion / webcam framing).
3. Explicit missingness representation when upper-body keypoints are unreliable.
4. Channel 23 upper-body keypoint confidence semantics, normalization, and bounds.
5. Strict 32-D feature vector dimension and range bounds.
6. 100% numerical parity between offline extraction and live inference.
7. End-to-end gradient flow in the proposed trainable frame encoder.
8. Student/recording split-isolation on public training data (DAiSEE v2) with zero leakage.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from attention_pipeline.types import FEAT_DIM, StudentFrame
from attention_pipeline.features import (
    pack_features,
    torso_lean,
    upper_body_keypoint_confidence,
    to_student_frame,
    facing_front_score,
)
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier


# ==============================================================================
# 1. Torso Lean: Reliable Hips and Shoulders (Case 1)
# ==============================================================================
def test_torso_lean_case1_reliable_hips_and_shoulders():
    """Verify torso lean when both hips and shoulders are visible (c >= 0.25)."""
    kpts = np.zeros((17, 3), dtype=np.float32)

    # Perfectly upright: Shoulders at y=100, Hips at y=300, vertically aligned
    kpts[5] = [100.0, 100.0, 0.9]  # Left shoulder
    kpts[6] = [200.0, 100.0, 0.9]  # Right shoulder
    kpts[11] = [100.0, 300.0, 0.9]  # Left hip
    kpts[12] = [200.0, 300.0, 0.9]  # Right hip

    lean_upright = torso_lean(kpts)
    assert np.isclose(lean_upright, 0.0, atol=1e-4), f"Expected 0.0 for upright pose, got {lean_upright}"

    # Leaned pose: Shoulders shifted horizontally by 100px
    kpts[5, 0] += 100.0
    kpts[6, 0] += 100.0
    lean_leaned = torso_lean(kpts)
    assert lean_leaned > 0.3, f"Expected noticeable lean angle, got {lean_leaned}"
    assert lean_leaned <= 1.0

    # Single hip visible (e.g. body partially turned): Left hip occluded, right hip visible
    kpts[11, 2] = 0.05  # occluded
    kpts[12, 2] = 0.95  # visible
    lean_single_hip = torso_lean(kpts)
    assert 0.0 <= lean_single_hip <= 1.0


# ==============================================================================
# 2. Torso Lean: Occluded Hips Shoulder Fallback (Case 2)
# ==============================================================================
def test_torso_lean_case2_occluded_hips_shoulder_fallback():
    """Verify safe shoulder tilt fallback when hips are occluded by desk/framing (c < 0.25)."""
    kpts = np.zeros((17, 3), dtype=np.float32)

    # Hips occluded (desk occlusion / webcam framing)
    kpts[11] = [100.0, 300.0, 0.02]  # Left hip occluded
    kpts[12] = [200.0, 300.0, 0.04]  # Right hip occluded

    # Shoulders clearly visible and horizontal (upright desk posture)
    kpts[5] = [100.0, 100.0, 0.95]  # Left shoulder
    kpts[6] = [200.0, 100.0, 0.95]  # Right shoulder

    lean_upright = torso_lean(kpts)
    assert np.isclose(lean_upright, 0.0, atol=1e-4), f"Expected 0.0 for horizontal shoulders, got {lean_upright}"

    # Lateral lean behind desk: shoulder line tilts sideways
    kpts[5] = [100.0, 130.0, 0.95]  # Left shoulder dropped
    kpts[6] = [200.0, 70.0, 0.95]  # Right shoulder raised
    lean_tilted = torso_lean(kpts)
    assert lean_tilted > 0.4, f"Expected significant tilt fallback, got {lean_tilted}"
    assert lean_tilted <= 1.0


# ==============================================================================
# 3. Torso Lean: Explicit Missingness (Case 3)
# ==============================================================================
def test_torso_lean_case3_explicit_missingness():
    """Verify explicit missingness handling when fallback keypoints are also unreliable."""
    # None input
    assert torso_lean(None) == 0.0

    # Truncated input (< 13 keypoints)
    short_kpts = np.zeros((8, 3), dtype=np.float32)
    assert torso_lean(short_kpts) == 0.0

    # All keypoints zero / occluded
    zero_kpts = np.zeros((17, 3), dtype=np.float32)
    assert torso_lean(zero_kpts) == 0.0

    # Hips occluded AND only one shoulder visible (cannot compute shoulder tilt)
    one_sh_kpts = np.zeros((17, 3), dtype=np.float32)
    one_sh_kpts[5] = [100.0, 100.0, 0.9]  # Left shoulder visible
    one_sh_kpts[6] = [200.0, 100.0, 0.1]  # Right shoulder occluded
    assert torso_lean(one_sh_kpts) == 0.0


# ==============================================================================
# 4. Channel 23: Upper-Body Keypoint Confidence Semantics and Normalization
# ==============================================================================
def test_channel23_confidence_semantics_and_normalization():
    """Verify Channel 23 anchors, normalization to [0, 1], and robustness."""
    # Anchors: 0 (Nose), 1 (Left Eye), 2 (Right Eye), 5 (Left Shoulder), 6 (Right Shoulder)
    kpts = np.zeros((17, 3), dtype=np.float32)

    # 1. Full confidence
    for idx in (0, 1, 2, 5, 6):
        kpts[idx, 2] = 1.0
    assert np.isclose(upper_body_keypoint_confidence(kpts), 1.0)

    # 2. Known arithmetic mean: [0.9, 0.8, 0.7, 0.6, 0.5] -> mean = 0.70
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    for idx, c in zip((0, 1, 2, 5, 6), confs):
        kpts[idx, 2] = c
    assert np.isclose(upper_body_keypoint_confidence(kpts), 0.70)

    # 3. Robustness: Out-of-bounds clipping, NaN, Inf, and negatives
    kpts[0, 2] = 1.5  # clipped to 1.0
    kpts[1, 2] = -0.5  # treated as 0.0
    kpts[2, 2] = np.nan  # treated as 0.0
    kpts[5, 2] = np.inf  # treated as 0.0
    kpts[6, 2] = 0.5  # 0.5
    # Expected mean: (1.0 + 0.0 + 0.0 + 0.0 + 0.5) / 5 = 1.5 / 5 = 0.30
    assert np.isclose(upper_body_keypoint_confidence(kpts), 0.30)

    # 4. Missingness
    assert upper_body_keypoint_confidence(None) == 0.0
    assert upper_body_keypoint_confidence(np.zeros((5, 3), dtype=np.float32)) == 0.0

    # 5. 2-column keypoints without confidence channel
    kpts_2d = np.ones((17, 2), dtype=np.float32) * 50.0
    assert np.isclose(upper_body_keypoint_confidence(kpts_2d), 1.0)


# ==============================================================================
# 5. 32-D Feature Vector Dimensions and Range Bounds
# ==============================================================================
def test_32d_feature_vector_dimensions_and_ranges():
    """Verify that pack_features produces exactly 32-D within supported range bounds."""
    rng = np.random.default_rng(123)

    head_pose = np.array([25.0, -15.0, 5.0], dtype=np.float32)
    gaze = np.array([-10.0, 8.0], dtype=np.float32)
    emotion_probs = np.array([0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.6], dtype=np.float32)
    posture_probs = np.array([0.1, 0.5, 0.2, 0.1, 0.05, 0.05], dtype=np.float32)
    motion = 0.25
    eye_aspect = 0.30
    mouth_aspect = 0.20

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = rng.uniform(50, 200, size=17)
    kpts[:, 1] = rng.uniform(50, 400, size=17)
    kpts[:, 2] = rng.uniform(0.3, 0.95, size=17)

    feat = pack_features(
        head_pose=head_pose,
        gaze=gaze,
        emotion_probs=emotion_probs,
        posture_probs=posture_probs,
        motion=motion,
        eye_aspect=eye_aspect,
        mouth_aspect=mouth_aspect,
        keypoints=kpts,
    )

    assert feat.shape == (FEAT_DIM,)
    assert feat.shape[0] == 32
    assert feat.dtype == np.float32

    # Angles normalized by 180 in [-1.0, 1.0]
    assert np.all(feat[0:3] >= -1.0) and np.all(feat[0:3] <= 1.0)
    assert np.all(feat[3:5] >= -1.0) and np.all(feat[3:5] <= 1.0)

    # Probabilities and ratios in [0.0, 1.0]
    assert np.all(feat[5:12] >= 0.0) and np.all(feat[5:12] <= 1.0)
    assert np.all(feat[12:18] >= 0.0) and np.all(feat[12:18] <= 1.0)
    assert 0.0 <= feat[18] <= 1.0
    assert 0.0 <= feat[19] <= 1.0
    assert 0.0 <= feat[20] <= 1.0
    assert 0.0 <= feat[21] <= 1.0
    assert 0.0 <= feat[22] <= 1.0
    assert 0.0 <= feat[23] <= 1.0

    # Reserved zero-padding: channels 24 to 31 MUST be strictly 0.0
    assert np.all(feat[24:32] == 0.0), f"Channels 24-31 must be strictly 0.0, got: {feat[24:32]}"


# ==============================================================================
# 6. Offline Extraction and Live Inference Feature Parity
# ==============================================================================
def test_offline_and_live_feature_parity():
    """Ensure pack_features (offline) and to_student_frame (live) produce identical vectors."""
    head_pose = np.array([-18.5, 12.0, -3.2], dtype=np.float32)
    gaze = np.array([5.4, -8.1], dtype=np.float32)
    emotion_probs = np.array([0.02, 0.01, 0.01, 0.05, 0.01, 0.05, 0.85], dtype=np.float32)
    posture_probs = np.array([0.0, 0.8, 0.1, 0.05, 0.05, 0.0], dtype=np.float32)
    motion = 0.123
    eye_aspect = 0.300
    mouth_aspect = 0.300

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = [120.0, 150.0, 0.85]
    kpts[6] = [210.0, 145.0, 0.88]
    kpts[11] = [125.0, 320.0, 0.10]  # desk-occluded hip
    kpts[12] = [205.0, 325.0, 0.05]  # desk-occluded hip
    kpts[0] = [165.0, 100.0, 0.95]
    kpts[1] = [150.0, 90.0, 0.92]
    kpts[2] = [180.0, 90.0, 0.90]

    # Offline extraction path
    feat_offline = pack_features(
        head_pose=head_pose,
        gaze=gaze,
        emotion_probs=emotion_probs,
        posture_probs=posture_probs,
        motion=motion,
        eye_aspect=eye_aspect,
        mouth_aspect=mouth_aspect,
        keypoints=kpts,
    )

    # Live inference path
    bbox = np.array([100.0, 80.0, 240.0, 360.0], dtype=np.float32)
    student_frame = to_student_frame(
        student_id=42,
        timestamp=1.25,
        bbox=bbox,
        keypoints=kpts,
        head_pose=head_pose,
        gaze=gaze,
        emotion_probs=emotion_probs,
        posture_probs=posture_probs,
        motion=motion,
        eye_aspect=eye_aspect,
        mouth_aspect=mouth_aspect,
    )
    feat_live = student_frame.features

    # Strict numerical identity across all 32 channels
    assert np.array_equal(feat_offline, feat_live), "Discrepancy detected between offline and live feature vectors!"
    assert np.max(np.abs(feat_offline - feat_live)) == 0.0


# ==============================================================================
# 7. Trainable Frame Encoder: Gradient Flow Verification
# ==============================================================================
def test_trainable_frame_encoder_gradient_flow():
    """Verify that unfreezing the frame encoder flows non-zero gradients end-to-end."""
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)

    # 1. Unfrozen model (trainable frame encoder)
    model_trainable = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=False)
    for p in model_trainable.frame_encoder.parameters():
        assert p.requires_grad, "Frame encoder parameters must require grad when unfreezing"

    x = torch.randn(4, 16, 32, requires_grad=True)
    logits = model_trainable(x)
    loss = logits.sum()
    loss.backward()

    # Frame encoder weights must receive non-zero gradient
    fe_weight_grad = model_trainable.frame_encoder[0].weight.grad
    assert fe_weight_grad is not None, "Frame encoder weight grad must not be None"
    assert torch.norm(fe_weight_grad) > 0.0, "Frame encoder weight grad norm must be > 0"

    # Temporal backbone and head must also receive gradient
    gru_grad = model_trainable.temporal.weight_ih_l0.grad
    assert gru_grad is not None and torch.norm(gru_grad) > 0.0

    head_grad = model_trainable.head[0].weight.grad
    assert head_grad is not None and torch.norm(head_grad) > 0.0

    # Gradient must flow all the way back to input tensor x
    assert x.grad is not None and torch.norm(x.grad) > 0.0

    # 2. Frozen model (baseline verification)
    base_net2 = AttentionFusionNet(feat_dim=32, hidden=64)
    model_frozen = ManualBinaryAttentionClassifier(base_net2, freeze_frame_encoder=True)
    for p in model_frozen.frame_encoder.parameters():
        assert not p.requires_grad, "Frame encoder parameters must not require grad when frozen"

    x2 = torch.randn(4, 16, 32)
    logits2 = model_frozen(x2)
    loss2 = logits2.sum()
    loss2.backward()
    assert model_frozen.frame_encoder[0].weight.grad is None, "Frozen encoder must have None grad"
    assert model_frozen.temporal.weight_ih_l0.grad is not None


# ==============================================================================
# 8. Split Isolation: Public Dataset Only (DAiSEE v2)
# ==============================================================================
def test_public_dataset_split_isolation_daisee():
    """Verify subject-disjoint and clip-disjoint split isolation on public DAiSEE v2 dataset."""
    daisee_path = Path("data/processed/fusion_daisee_v2.npz")
    if not daisee_path.exists():
        pytest.skip("Public dataset fusion_daisee_v2.npz not found")

    data = np.load(daisee_path)
    subject_ids = data["subject_ids"]
    clip_ids = data["clip_ids"]
    n_total = len(subject_ids)

    unique_subjects = np.unique(subject_ids)
    assert len(unique_subjects) >= 5, "Public dataset must contain multiple subjects"

    # Deterministic subject-disjoint partition (80% train, 20% val)
    rng = np.random.default_rng(42)
    shuffled_subjects = rng.permutation(unique_subjects)
    n_val_subj = max(1, int(round(len(shuffled_subjects) * 0.2)))
    val_subj_set = set(shuffled_subjects[:n_val_subj])
    train_subj_set = set(shuffled_subjects[n_val_subj:])

    train_mask = np.isin(subject_ids, list(train_subj_set))
    val_mask = np.isin(subject_ids, list(val_subj_set))

    # Verification 1: Zero subject overlap
    train_subs = set(subject_ids[train_mask])
    val_subs = set(subject_ids[val_mask])
    assert len(train_subs & val_subs) == 0, f"Subject leakage detected: {train_subs & val_subs}"

    # Verification 2: Zero clip overlap
    train_clips = set(clip_ids[train_mask])
    val_clips = set(clip_ids[val_mask])
    assert len(train_clips & val_clips) == 0, f"Clip leakage detected: {train_clips & val_clips}"

    # Verification 3: Complete sample partition
    assert (train_mask.sum() + val_mask.sum()) == n_total, "Samples leaked or dropped during partition"


# ==============================================================================
# 9. Hardened Dataset Integrity: DAiSEE v3 Hardened
# ==============================================================================
def test_hardened_public_dataset_v3_integrity():
    """Verify data/processed/fusion_daisee_v3_hardened.npz features and metadata."""
    v3_path = Path("data/processed/fusion_daisee_v3_hardened.npz")
    if not v3_path.exists():
        pytest.skip("fusion_daisee_v3_hardened.npz not yet generated")

    data = np.load(v3_path)
    feats = data["features"]
    score = data["score"]
    aux = data["aux"]
    frame_indices = data["frame_indices"]
    clip_ids = data["clip_ids"]
    subject_ids = data["subject_ids"]

    # 1. Dimensions and finite checks
    assert feats.shape == (23661, 16, 32)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # 2. Channel 23 active confidence
    ch23 = feats[:, :, 23]
    assert ch23.min() >= 0.0 and ch23.max() <= 1.0
    assert ch23.mean() > 0.5, "Channel 23 upper-body confidence must be actively populated"
    assert (ch23 > 0.0).sum() / ch23.size > 0.95

    # 3. Channel 22 torso lean with shoulder fallback
    ch22 = feats[:, :, 22]
    assert ch22.min() >= 0.0 and ch22.max() <= 1.0
    assert 0.01 <= ch22.mean() <= 0.10

    # 4. Channels 24-31 zero-padding
    assert np.all(feats[:, :, 24:] == 0.0)

    # 5. Metadata alignment
    assert len(score) == 23661
    assert len(aux) == 23661
    assert len(clip_ids) == 23661
    assert len(subject_ids) == 23661
    assert frame_indices.shape == (23661, 16)
