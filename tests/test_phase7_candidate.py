"""Tests for Phase 7 Public-Dataset Candidate Attention Classifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase7_public_trained_binary_candidate.pt"
ACTIVE_BASELINE_PATH = WORKSPACE_ROOT / "checkpoints" / "classroom_manual_binary_model_v2_clean.pt"
EXPECTED_ACTIVE_SHA256 = "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E"
CANDIDATE_6G4C_PATH = WORKSPACE_ROOT / "checkpoints" / "candidates" / "phase6g4c_engaged_disengaged_candidate.pt"
EXPECTED_6G4C_SHA256 = "E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E"


def get_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def test_protected_checkpoints_invariance():
    assert ACTIVE_BASELINE_PATH.exists()
    assert CANDIDATE_6G4C_PATH.exists()
    assert get_sha256(ACTIVE_BASELINE_PATH) == EXPECTED_ACTIVE_SHA256
    assert get_sha256(CANDIDATE_6G4C_PATH) == EXPECTED_6G4C_SHA256


def test_phase7_candidate_checkpoint_integrity():
    assert CANDIDATE_PATH.exists()
    ckpt = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "epoch" in ckpt
    assert "calibrated_threshold" in ckpt
    assert "class_weights" in ckpt

    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

    base = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base, freeze_frame_encoder=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Forward pass on dummy batch (B=4, T=16, D=32)
    dummy = torch.randn(4, 16, 32)
    logits = model(dummy)
    assert logits.shape == (4, 2)
    assert torch.all(torch.isfinite(logits))

    probs = model.predict_proba(dummy)
    assert probs.shape == (4, 2)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(4), atol=1e-5)

    scores = model.score(dummy)
    assert scores.shape == (4,)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 100.0)
