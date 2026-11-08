"""Unit tests for Phase 8A rebuild: manifest integrity, atomic save, and extraction contract."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def test_master_manifest_integrity():
    manifest_path = WORKSPACE_ROOT / "data" / "manifests" / "daisee_master_manifest_v1.csv"
    assert manifest_path.exists(), f"Missing master manifest: {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    assert len(rows) == 8986, f"Expected 8,986 master rows, found {len(rows)}"
    
    # Check required columns
    expected_cols = {"clip_id", "subject_id", "split", "video_path", "engagement", "boredom", "confusion", "frustration", "binary_target", "binary_label", "label_source"}
    assert expected_cols.issubset(set(rows[0].keys()))
    
    # Check binary label contract
    for r in rows:
        eng = int(r["engagement"])
        b_target = int(r["binary_target"])
        b_label = r["binary_label"]
        if eng in (0, 1):
            assert b_target == 0
            assert b_label == "NON-ATTENTIVE"
        else:
            assert b_target == 1
            assert b_label == "ATTENTIVE"


def test_quarantine_manifest_integrity():
    q_path = WORKSPACE_ROOT / "data" / "manifests" / "daisee_quarantine_manifest_v1.csv"
    assert q_path.exists(), f"Missing quarantine manifest: {q_path}"
    
    with open(q_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    assert len(rows) == 82, f"Expected 82 quarantined rows, found {len(rows)}"
    for r in rows:
        assert r["subject_id"] == "998826"
        assert r["quarantine_reason"] == "UNANNOTATED_NO_OFFICIAL_LABEL"


def test_split_subject_disjointness():
    manifest_path = WORKSPACE_ROOT / "data" / "manifests" / "daisee_master_manifest_v1.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    train_subs = set(r["subject_id"] for r in rows if r["split"] == "Train")
    val_subs = set(r["subject_id"] for r in rows if r["split"] == "Validation")
    test_subs = set(r["subject_id"] for r in rows if r["split"] == "Test")
    
    assert len(train_subs) == 70
    assert len(val_subs) == 22
    assert len(test_subs) == 21
    
    assert len(train_subs & val_subs) == 0, f"Train and Val overlap: {train_subs & val_subs}"
    assert len(train_subs & test_subs) == 0, f"Train and Test overlap: {train_subs & test_subs}"
    assert len(val_subs & test_subs) == 0, f"Val and Test overlap: {val_subs & test_subs}"


def test_atomic_save_mechanism():
    """Verify that writing to .tmp and replacing preserves file integrity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = Path(tmp_dir) / "test_archive.npz"
        tmp_path = dest_path.with_suffix(".tmp.npz")
        
        arr = np.ones((10, 16, 32), dtype=np.float32)
        np.savez_compressed(tmp_path, features=arr)
        assert tmp_path.exists()
        assert not dest_path.exists()
        
        tmp_path.replace(dest_path)
        assert dest_path.exists()
        assert not tmp_path.exists()
        
        with np.load(dest_path) as loaded:
            assert "features" in loaded
            assert loaded["features"].shape == (10, 16, 32)


def test_protected_checkpoints_invariant():
    """Verify that all 3 protected checkpoints remain bit-for-bit unchanged."""
    expected_hashes = {
        "checkpoints/classroom_manual_binary_model_v2_clean.pt": "2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E",
        "checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt": "E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E",
        "checkpoints/candidates/phase7_public_trained_binary_candidate.pt": "1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5",
    }
    
    for rel_path, exp_h in expected_hashes.items():
        p = WORKSPACE_ROOT / rel_path
        assert p.exists(), f"Protected file missing: {p}"
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        assert h == exp_h, f"Hash mismatch on {rel_path}!"


def test_smoke_archive_schema_and_downstream_model():
    """Verify that the smoke test fusion archive is structurally valid and downstream compatible."""
    import torch
    import sys
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier

    smoke_npz = WORKSPACE_ROOT / "data" / "processed" / "smoke_test_fusion.npz"
    assert smoke_npz.exists(), f"Smoke archive missing: {smoke_npz}"

    with np.load(smoke_npz) as data:
        assert "features" in data
        assert "score" in data
        assert "aux" in data
        assert "clip_ids" in data
        assert "subject_ids" in data
        assert "splits" in data
        assert "binary_targets" in data
        assert "frame_indices" in data
        assert "clip" in data

        features = data["features"]
        assert features.ndim == 3
        assert features.shape[1] == 16
        assert features.shape[2] == 32
        assert features.dtype == np.float32
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

        # Test downstream model forward pass
        base_net = AttentionFusionNet(feat_dim=32)
        model = ManualBinaryAttentionClassifier(base_net=base_net)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            probs = model.predict_proba(x)
            assert probs.shape == (features.shape[0], 2)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(features.shape[0]), atol=1e-5)
