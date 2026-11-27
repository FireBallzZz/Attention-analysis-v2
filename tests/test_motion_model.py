"""Tests for the motion attention model (attention_pipeline/motion.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from attention_pipeline.motion import LiveMotionScorer, MotionRule, box_motion

ROOT = Path(__file__).resolve().parent.parent
RULE = ROOT / "checkpoints" / "motion_rule_v1.json"


def test_box_motion_same_frame_is_zero_and_change_is_positive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    b = a.copy()
    b[20:60, 20:60] = 255 - b[20:60, 20:60]
    box = (10, 10, 80, 80)
    assert box_motion(a, a, box) == 0.0
    assert box_motion(a, b, box) > 0.02
    assert box_motion(None, a, box) is None            # no previous frame
    assert box_motion(a, a, (5, 5, 7, 7)) is None      # degenerate box


def test_rule_threshold_maps_to_50_and_more_motion_scores_lower():
    rule = MotionRule(a=0.0, b=-1.0, threshold=0.5)
    assert rule.score(0.0) == pytest.approx(50.0)
    assert rule.score(1.0) < 50.0 < rule.score(-1.0)
    assert 0.0 <= rule.score(10.0) <= 100.0 and 0.0 <= rule.score(-10.0) <= 100.0


def test_live_scorer_warms_up_then_ranks_movers_low():
    sc = LiveMotionScorer(MotionRule(a=0.0, b=-1.0, threshold=0.5, window=4), min_class=4)
    out = {}
    for _ in range(10):
        out[1] = sc.update(1, 0.002)   # still student
        out[2] = sc.update(2, 0.004)   # still student
        out[3] = sc.update(3, 0.05)    # moving student
    assert None not in out.values()
    assert out[1] > 50 > out[3]        # judged relative to the class median
    sc.reset()
    assert sc.update(1, 0.01) is None  # warm-up again after reset


@pytest.mark.skipif(not RULE.exists(), reason="motion rule not fitted")
def test_fitted_rule_says_more_motion_means_low():
    rule = json.loads(RULE.read_text())
    assert rule["b"] < 0
    assert 0.0 < rule["threshold"] < 1.0
    assert rule["expected_unseen_classroom"]["balacc"] > 0.5


def test_config_uses_motion_rule():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["models"]["motion_rule"].endswith("motion_rule_v1.json")


def test_phone_crop_rule_configured_and_disabled_path_is_safe():
    import yaml

    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.utils import load_yaml

    cfg = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["fusion"]["phone_crop"] == pytest.approx(0.25)
    assert cfg["fusion"]["phone_crop_cap"] == pytest.approx(30.0)
    c = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    c["fusion"]["phone_crop"] = 0.0
    pipe = AttentionPipeline(c, load_weights=False)
    assert pipe._phone_on_crops(np.zeros((100, 100, 3), np.uint8), []) == {}
