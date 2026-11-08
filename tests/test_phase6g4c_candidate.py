"""Unit and regression tests for Phase 6G-4C candidate model and training components."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch import nn

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier
from scripts.train_phase6g4c_candidate import ClassBalancedFocalLoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"


def test_active_checkpoint_hash_invariance():
    """Verify active production checkpoint SHA-256 remains 100% invariant."""
    assert ACTIVE_CHECKPOINT_PATH.exists(), f"Active checkpoint missing: {ACTIVE_CHECKPOINT_PATH}"
    with open(ACTIVE_CHECKPOINT_PATH, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest().upper()
    assert h == EXPECTED_ACTIVE_SHA256, f"Checkpoint hash modified! Got {h}"


def test_binary_class_mapping_and_order():
    """Verify binary mapping: Class 0 = DISENGAGED (Low=2), Class 1 = ENGAGED (High=0, Mod=1)."""
    p1 = PROJECT_ROOT / "data" / "processed" / "classroom_clip_001_v3_hardened.npz"
    p8 = PROJECT_ROOT / "data" / "processed" / "classroom_clip_008_v3_hardened.npz"

    for p in [p1, p8]:
        d = np.load(p, allow_pickle=True)
        raw = d["targets_raw"]
        bin_t = d["targets_binary"]
        b_raw = d["bands_raw"]
        b_bin = d["bands_binary"]

        assert np.all((bin_t == 1) == (raw <= 1))
        assert np.all((bin_t == 0) == (raw == 2))
        assert np.all((bin_t == 1) == (b_bin == "ENGAGED"))
        assert np.all((bin_t == 0) == (b_bin == "DISENGAGED"))
        assert np.all((raw == 0) == (b_raw == "HIGH"))
        assert np.all((raw == 1) == (b_raw == "MODERATE"))
        assert np.all((raw == 2) == (b_raw == "LOW"))


def test_train_val_track_disjointness():
    """Verify zero overlap between training tracks and validation tracks [2, 5, 8, 11]."""
    p1 = PROJECT_ROOT / "data" / "processed" / "classroom_clip_001_v3_hardened.npz"
    p8 = PROJECT_ROOT / "data" / "processed" / "classroom_clip_008_v3_hardened.npz"
    d1 = np.load(p1, allow_pickle=True)
    d8 = np.load(p8, allow_pickle=True)

    sids = np.concatenate([d1["student_ids"], d8["student_ids"]])
    val_tracks = [2, 5, 8, 11]
    is_val = np.isin(sids, val_tracks)
    is_train = ~is_val

    train_sids = set(np.unique(sids[is_train]).tolist())
    val_sids = set(np.unique(sids[is_val]).tolist())

    assert train_sids.intersection(val_sids) == set()
    assert len(train_sids) == 17
    assert len(val_sids) == 4
    assert int(np.sum(is_train)) == 8065
    assert int(np.sum(is_val)) == 2280


def test_model_output_shape_and_valid_logits():
    """Verify candidate architecture produces (B, 2) finite logits from (B, 16, 32) input."""
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True)
    model.eval()

    dummy_input = torch.randn(8, 16, 32, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)
        scores = model.score(dummy_input)

    assert logits.shape == (8, 2)
    assert probs.shape == (8, 2)
    assert scores.shape == (8,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(probs))
    # Probabilities sum to 1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_correct_loss_weight_indexing():
    """Verify loss weight index 0 scales DISENGAGED (Class 0) and index 1 scales ENGAGED (Class 1)."""
    weights = torch.tensor([1.6265, 0.3735], dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="none")
    focal_loss = ClassBalancedFocalLoss(weight=weights, gamma=2.0, reduction="none")

    # Symmetric logits: [0.0, 0.0] gives equal prob 0.5 for both classes
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    ce_vals = ce_loss(logits, targets)
    focal_vals = focal_loss(logits, targets)

    # Ratio of loss for target 0 vs target 1 must match ratio of weights
    expected_ratio = float(weights[0] / weights[1])
    assert pytest.approx(float(ce_vals[0] / ce_vals[1]), rel=1e-4) == expected_ratio
    assert pytest.approx(float(focal_vals[0] / focal_vals[1]), rel=1e-4) == expected_ratio


def test_checkpoint_save_load_roundtrip():
    """Verify candidate model state_dict roundtrips cleanly without parameter alteration."""
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True)
    model.eval()

    dummy_x = torch.randn(4, 16, 32, dtype=torch.float32)
    with torch.no_grad():
        orig_logits = model(dummy_x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save({"model": model.state_dict(), "class_names": ["DISENGAGED", "ENGAGED"]}, tmp_path)

        # New instance
        new_base = AttentionFusionNet(feat_dim=32, hidden=64)
        new_model = ManualBinaryAttentionClassifier(new_base, freeze_frame_encoder=True)
        ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
        new_model.load_state_dict(ckpt["model"])
        new_model.eval()

        with torch.no_grad():
            new_logits = new_model(dummy_x)

        assert torch.allclose(orig_logits, new_logits, atol=1e-6)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_saved_candidate_checkpoints():
    """Verify saved candidate checkpoints exist, load cleanly, and produce valid inference outputs."""
    p_primary = PROJECT_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
    p_focal = PROJECT_ROOT / "checkpoints" / "candidates" / "phase6g4c_focal_loss_ablation_candidate.pt"

    assert p_primary.exists(), f"Primary candidate missing: {p_primary}"
    assert p_focal.exists(), f"Focal loss candidate missing: {p_focal}"

    dummy_x = torch.randn(2, 16, 32, dtype=torch.float32)

    for p, expected_loss in [(p_primary, "weighted_ce"), (p_focal, "focal_loss")]:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        assert ckpt["loss_type"] == expected_loss
        assert ckpt["class_names"] == ["DISENGAGED", "ENGAGED"]
        assert ckpt["val_tracks"] == [2, 5, 8, 11]
        assert "val_metrics" in ckpt
        assert "model" in ckpt

        base = AttentionFusionNet(feat_dim=32, hidden=64)
        model = ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True)
        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            out = model(dummy_x)
            probs = model.predict_proba(dummy_x)
            scores = model.score(dummy_x)

        assert out.shape == (2, 2)
        assert probs.shape == (2, 2)
        assert scores.shape == (2,)
        assert torch.all(torch.isfinite(out))

