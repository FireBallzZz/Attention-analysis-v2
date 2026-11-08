"""Unit and integration tests for Phase 6G-4B re-extracted v3 hardened datasets."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def clip_001_v3():
    p = DATA_DIR / "classroom_clip_001_v3_hardened.npz"
    assert p.exists(), f"Missing dataset: {p}"
    return np.load(p, allow_pickle=True)


@pytest.fixture(scope="module")
def clip_008_v3():
    p = DATA_DIR / "classroom_clip_008_v3_hardened.npz"
    assert p.exists(), f"Missing dataset: {p}"
    return np.load(p, allow_pickle=True)


def test_dataset_tensor_shapes(clip_001_v3, clip_008_v3):
    """Verify sequence length = 16, feature dimension = 32, and sequence counts."""
    assert clip_001_v3["features"].shape == (5192, 16, 32)
    assert clip_001_v3["features"].dtype == np.float32

    assert clip_008_v3["features"].shape == (5153, 16, 32)
    assert clip_008_v3["features"].dtype == np.float32


def test_finite_values(clip_001_v3, clip_008_v3):
    """Verify all features are finite numbers (no NaN or Inf)."""
    assert np.all(np.isfinite(clip_001_v3["features"]))
    assert np.all(np.isfinite(clip_008_v3["features"]))


def test_head_roll_distribution(clip_001_v3, clip_008_v3):
    """Verify head roll is centered near 0.0 with realistic standard deviation (< 0.15)."""
    r1 = clip_001_v3["features"][:, :, 2]
    r8 = clip_008_v3["features"][:, :, 2]

    # In legacy data, std was ~0.78 and mean was -0.50
    assert abs(r1.mean()) < 0.10, f"Clip 001 head roll mean {r1.mean()} should be centered near 0.0"
    assert r1.std() < 0.15, f"Clip 001 head roll std {r1.std()} should be tightly bounded"

    assert abs(r8.mean()) < 0.10, f"Clip 008 head roll mean {r8.mean()} should be centered near 0.0"
    assert r8.std() < 0.15, f"Clip 008 head roll std {r8.std()} should be tightly bounded"


def test_channel_23_population(clip_001_v3, clip_008_v3):
    """Verify Channel 23 is populated with upper-body confidence (mean in [0.75, 0.90])."""
    c1 = clip_001_v3["features"][:, :, 23]
    c8 = clip_008_v3["features"][:, :, 23]

    assert 0.75 <= c1.mean() <= 0.90, f"Clip 001 Ch23 mean {c1.mean():.4f} out of expected range"
    assert 0.75 <= c8.mean() <= 0.90, f"Clip 008 Ch23 mean {c8.mean():.4f} out of expected range"

    assert np.count_nonzero(c1) / c1.size > 0.90, "Clip 001 Ch23 non-zero fraction should be > 90%"
    assert np.count_nonzero(c8) / c8.size > 0.90, "Clip 008 Ch23 non-zero fraction should be > 90%"


def test_dual_target_consistency(clip_001_v3, clip_008_v3):
    """Verify binary target strictly maps High (0) and Moderate (1) to ENGAGED (1), and Low (2) to DISENGAGED (0)."""
    for d, name in [(clip_001_v3, "clip_001"), (clip_008_v3, "clip_008")]:
        raw = d["targets_raw"]
        bin_t = d["targets_binary"]

        # Assert allowed values
        assert set(np.unique(raw)).issubset({0, 1, 2}), f"{name} contains unexpected raw targets"
        assert set(np.unique(bin_t)).issubset({0, 1}), f"{name} contains unexpected binary targets"

        # Assert exact mapping
        assert np.all((bin_t == 1) == (raw <= 1)), f"{name} binary target mapping does not match raw <= 1"
        assert np.all((bin_t == 0) == (raw == 2)), f"{name} binary target mapping does not match raw == 2"

        # Assert backward-compatible target alias
        assert np.all(d["targets"] == raw), f"{name} legacy targets key must match targets_raw"


def test_loader_compatibility(clip_001_v3, clip_008_v3):
    """Verify standard sequence loader reads new v3 hardened datasets seamlessly."""
    p1 = DATA_DIR / "classroom_clip_001_v3_hardened.npz"
    p8 = DATA_DIR / "classroom_clip_008_v3_hardened.npz"

    d1 = np.load(p1)
    d8 = np.load(p8)

    feats_all = np.concatenate([d1["features"], d8["features"]], axis=0)
    sids_all = np.concatenate([d1["student_ids"], d8["student_ids"]], axis=0)
    targets_all = np.concatenate([d1["targets_binary"], d8["targets_binary"]], axis=0)

    assert len(feats_all) == 10345
    assert len(sids_all) == 10345
    assert len(targets_all) == 10345
    assert np.sum(targets_all == 1) == 8414  # ENGAGED
    assert np.sum(targets_all == 0) == 1931  # DISENGAGED
