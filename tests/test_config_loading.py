"""Config-loading integrity tests.

Two defects motivated these tests:

1. `project_path("")` resolved to the project root, which exists, so a config
   that intended to DISABLE a modality would try to load a directory as a
   checkpoint. An empty weight path must resolve to None.

2. `_maybe_load` propagated a checkpoint/architecture mismatch as an
   exception, which aborted the whole pipeline even though the caller had a
   documented fallback chain. A mismatched checkpoint must return False so the
   fallback can run.

Every config in `configs/` must also build a pipeline without raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attention_pipeline.utils import project_path

ROOT = Path(__file__).resolve().parents[1]


def test_empty_weight_path_disables_the_module():
    """An empty string means 'disabled' and must not resolve to a real path."""
    assert project_path("") is None
    assert project_path("   ") is None
    assert project_path(None) is None


def test_relative_weight_path_resolves_against_project_root():
    """A real relative path still resolves to the project root."""
    resolved = project_path("checkpoints/scb_yolo.pt")
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.name == "scb_yolo.pt"


def test_incompatible_checkpoint_does_not_abort_the_pipeline():
    """A checkpoint with the wrong architecture must return False, not raise."""
    import torch
    from torch import nn

    from attention_pipeline.pipeline import AttentionPipeline

    class WrongShape(nn.Module):
        def load(self, path, map_location="cpu"):
            raise RuntimeError("Missing key(s) in state_dict: 'head.0.weight'")

    pipe = AttentionPipeline.__new__(AttentionPipeline)
    pipe.device = torch.device("cpu")
    assert pipe._maybe_load(WrongShape(), "checkpoints/fusion_merged.pt") is False


def test_missing_checkpoint_returns_false():
    """A path that does not exist is a clean 'not loaded', not an error."""
    import torch
    from torch import nn

    from attention_pipeline.pipeline import AttentionPipeline

    class NeverCalled(nn.Module):
        def load(self, path, map_location="cpu"):  # pragma: no cover
            raise AssertionError("load must not be called for a missing file")

    pipe = AttentionPipeline.__new__(AttentionPipeline)
    pipe.device = torch.device("cpu")
    assert pipe._maybe_load(NeverCalled(), "checkpoints/does_not_exist.pt") is False
    assert pipe._maybe_load(NeverCalled(), "") is False


@pytest.mark.parametrize(
    "cfg_name",
    ["pipeline.yaml", "pipeline_merged.yaml", "pipeline_trained20.yaml", "pipeline_smoke.yaml"],
)
def test_every_config_builds_a_pipeline(cfg_name):
    """Each shipped config must construct the pipeline without raising."""
    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.utils import load_yaml

    cfg_path = ROOT / "configs" / cfg_name
    if not cfg_path.exists():
        pytest.skip(f"{cfg_name} not present")

    cfg = load_yaml(str(cfg_path))
    pipe = AttentionPipeline(cfg, load_weights=True)
    assert pipe.scorer is not None
    assert pipe.detector is not None
    # A disabled modality must report itself as off.
    if not (cfg.get("models") or {}).get("posture_weights"):
        assert pipe.posture_x.use_learned is False
