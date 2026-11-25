"""Tests for the corrected metrics, final checkpoint and leakage-free splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from attention_pipeline.metrics import binary_report
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "checkpoints" / "final_human_v1.pt"


def test_macro_f1_is_not_high_class_f1():
    # Clip 001 confusion matrix from the old report: the model never finds LOW.
    r = binary_report(tn=0, fp=51, fn=1, tp=1015)
    assert r["high_f1"] == pytest.approx(97.50, abs=0.01)
    assert r["low_f1"] == 0.0
    assert r["macro_f1"] == pytest.approx(48.75, abs=0.01)
    assert r["balanced_accuracy"] == pytest.approx(49.95, abs=0.01)
    # ...and it is worse than always answering HIGH
    assert r["accuracy"] < r["always_high_accuracy"]


def test_binary_report_perfect_and_counts():
    r = binary_report(tn=10, fp=0, fn=0, tp=30)
    assert r["accuracy"] == r["balanced_accuracy"] == r["macro_f1"] == 100.0
    assert r["n"] == 40 and r["always_high_accuracy"] == 75.0


def _model():
    return ManualBinaryAttentionClassifier(AttentionFusionNet(feat_dim=32, hidden=64))


def test_default_mask_and_threshold_keep_old_behaviour():
    m = _model().eval()
    assert torch.all(m.input_mask == 1) and m.threshold == 0.5
    x = torch.randn(3, 16, 32)
    p = m.predict_proba(x)[:, 1] * 100
    assert torch.allclose(m.score(x), p, atol=1e-5)


def test_score_rescale_puts_threshold_at_50():
    m = _model().eval()
    m.threshold = 0.8
    x = torch.randn(64, 16, 32)
    p = m.predict_proba(x)[:, 1]
    s = m.score(x)
    assert torch.all((s >= 50) == (p >= 0.8))
    assert float(s.min()) >= 0.0 and float(s.max()) <= 100.0


def test_masked_channels_are_ignored():
    m = _model().eval()
    m.input_mask[[0, 1, 2, 3, 4, 21]] = 0
    x = torch.randn(4, 16, 32)
    y = x.clone()
    y[:, :, [0, 1, 2, 3, 4, 21]] = torch.randn(4, 16, 6) * 10
    assert torch.allclose(m.predict_proba(x), m.predict_proba(y), atol=1e-6)


@pytest.mark.skipif(not FINAL.exists(), reason="final checkpoint not trained")
def test_final_checkpoint_loads_with_mask_and_threshold():
    m = _model().load(FINAL)
    ck = torch.load(FINAL, map_location="cpu", weights_only=False)
    assert 0.0 < m.threshold < 1.0 and m.threshold == pytest.approx(ck["calibrated_threshold"])
    assert m.input_mask[[0, 1, 2, 3, 4, 21]].sum() == 0 and m.input_mask.sum() == 26
    assert ck["label_source"].startswith("human")


@pytest.mark.skipif(not (ROOT / "data/processed/human_test_8.npz").exists(), reason="human data missing")
def test_rooms_share_no_person_and_test6_excluded():
    from evaluate_cross_classroom import load_all

    X, y, room, person, clip, info = load_all(include_test6=False)
    assert "room_D" not in info
    for p in np.unique(person):
        assert len(set(room[person == p])) == 1
    assert set(np.unique(y)) == {0, 1}


def test_final_config_points_at_final_model():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["models"]["manual_binary_weights"].endswith("final_human_v1.pt")
    assert cfg["fusion"]["binary_threshold"] == 0.5
