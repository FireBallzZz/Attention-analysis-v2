"""Regression tests for the score-combination policy and the decision threshold.

The live pipeline used to blend the trained model's score with the heuristic
rubric score (min / 0.4f+0.6h / f). Measured against the human CVAT
annotation across all three classroom clips, the rubric's score is INVERTED
(point-biserial r = -0.35) while the model's score is well ordered
(r = +0.61), so the blend destroyed the model's signal.

These tests pin the corrected behaviour:

  * the heuristic may only FLOOR the model score for verified hard evidence
    of disengagement (phone, sleep, head on desk, peer talk),
  * for every other cue the model score is used unchanged, even when the
    rubric is confidently "engaged",
  * the deployed threshold is the value validated on the human annotation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.rubric import RubricResult
from attention_pipeline.utils import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def _pipeline(combine_mode: str, threshold: float = 0.45) -> AttentionPipeline:
    cfg = load_yaml(str(CONFIG_PATH))
    cfg["fusion"]["combine_mode"] = combine_mode
    cfg["fusion"]["binary_threshold"] = threshold
    # Weights are irrelevant to the combination logic under test.
    return AttentionPipeline(cfg, load_weights=False)


def _rubric(band: str, score: float, cue: str) -> RubricResult:
    return RubricResult(band, score, cue)


@pytest.fixture(scope="module")
def pipe() -> AttentionPipeline:
    return _pipeline("model_hard")


def test_model_score_used_unchanged_for_soft_cues(pipe: AttentionPipeline) -> None:
    """Neutral / soft cues must not alter the model's score."""
    soft_cues = [
        ("HIGH", 86.0, "watching instructor"),
        ("MODERATE", 60.0, "listening"),
        ("MODERATE", 52.0, "partially engaged"),
        ("MODERATE", 52.0, "glancing left"),
        ("MODERATE", 52.0, "glancing right"),
        ("MODERATE", 60.0, "thinking, chin on hand"),
    ]
    for band, h_score, cue in soft_cues:
        rubric = _rubric(band, h_score, cue)
        assert pipe._combine_scores(70.0, rubric, None) == 70.0, cue
        assert pipe._combine_scores(20.0, rubric, None) == 20.0, cue


def test_active_study_cues_protect_desk_work(pipe: AttentionPipeline) -> None:
    """Active study (reading, writing, hand-raising) must not be penalized by downward head pitch."""
    study_cues = [
        ("HIGH", 88.0, "reading / writing"),
        ("HIGH", 84.0, "reading / taking notes"),
        ("HIGH", 92.0, "hand raised"),
    ]
    for band, h_score, cue in study_cues:
        rubric = _rubric(band, h_score, cue)
        # Even if neural BiGRU outputs a low 30.0 due to downward pitch, active study lifts it to h_score:
        assert pipe._combine_scores(30.0, rubric, None) == h_score, cue
        # If neural score is already higher (e.g. 95.0), it preserves the higher score:
        assert pipe._combine_scores(95.0, rubric, None) == 95.0, cue


def test_hard_evidence_floors_the_model_score(pipe: AttentionPipeline) -> None:
    """Verified hard evidence may lower the model score."""
    hard_cues = [
        ("LOW", 22.0, "using phone"),
        ("LOW", 24.0, "phone-like pose"),
        ("LOW", 16.0, "sleeping"),
        ("LOW", 20.0, "head on table"),
        ("LOW", 26.0, "talking to peer"),
        ("LOW", 18.0, "disturbing others"),
        ("LOW", 30.0, "looking left"),
        ("LOW", 30.0, "looking right"),
        ("LOW", 30.0, "looking away"),
    ]
    for band, h_score, cue in hard_cues:
        rubric = _rubric(band, h_score, cue)
        # Model 70, rubric h_score -> the rubric floors it to h_score.
        assert pipe._combine_scores(70.0, rubric, None) == h_score, cue
        # Model 10 is already below the rubric, so it stays 10.
        assert pipe._combine_scores(10.0, rubric, None) == 10.0, cue


def test_model_hard_neutral_cues_never_raise_score(pipe: AttentionPipeline) -> None:
    """Non-academic cues must not artificially push a LOW model score up."""
    rubric = _rubric("HIGH", 86.0, "watching instructor")
    assert pipe._combine_scores(30.0, rubric, None) == 30.0


def test_model_mode_ignores_the_rubric_entirely() -> None:
    pipe = _pipeline("model")
    rubric = _rubric("LOW", 16.0, "sleeping")
    assert pipe._combine_scores(70.0, rubric, None) == 70.0


def test_blended_mode_still_floors_on_hard_evidence() -> None:
    """The legacy policy must remain available for comparison."""
    pipe = _pipeline("blended")
    rubric = _rubric("LOW", 16.0, "sleeping")
    assert pipe._combine_scores(70.0, rubric, None) == 16.0


def test_deployed_threshold_matches_the_validated_value() -> None:
    """The old 0.45 was tuned on the test annotation. The final checkpoint
    carries its own threshold (chosen on held-out persons) and rescales its
    score so that threshold sits at 50, so the pipeline cut-off is 0.50."""
    cfg = load_yaml(str(CONFIG_PATH))
    assert float(cfg["fusion"]["binary_threshold"]) == pytest.approx(0.50)


def test_deployed_combine_mode_is_model_floor() -> None:
    """The rubric's 'active study' lift reversed the motion model on human labels
    (reports/final_model/pipeline_vs_human.json), so only the hard-evidence floor is kept."""
    cfg = load_yaml(str(CONFIG_PATH))
    assert str(cfg["fusion"]["combine_mode"]) == "model_floor"


def test_study_protection_keeps_confident_studying_students_at_or_above_50() -> None:
    from types import SimpleNamespace

    pipe = _pipeline("model_floor")
    pipe.study_protect = 0.3
    writing = SimpleNamespace(posture_probs=np.array([0.05, 0.1, 0.6, 0.1, 0.1, 0.05]))
    unclear = SimpleNamespace(posture_probs=np.full(6, 1 / 6))
    neutral = _rubric("HIGH", 70.0, "watching instructor")
    assert pipe._combine_scores(20.0, neutral, writing) == 50.0      # protected
    assert pipe._combine_scores(80.0, neutral, writing) == 80.0      # never lowered
    assert pipe._combine_scores(20.0, neutral, unclear) == 20.0      # no confident SCB -> unchanged
    # hard evidence still wins over study protection
    assert pipe._combine_scores(70.0, _rubric("LOW", 15.0, "using phone"), writing) == 15.0


def test_deployed_study_protection_enabled() -> None:
    cfg = load_yaml(str(CONFIG_PATH))
    assert float(cfg["fusion"]["study_protect"]) == pytest.approx(0.3)


def test_model_floor_lowers_on_hard_evidence_but_never_lifts() -> None:
    pipe = _pipeline("model_floor")
    pipe.study_protect = 0.0
    assert pipe._combine_scores(70.0, _rubric("LOW", 16.0, "sleeping"), None) == 16.0
    assert pipe._combine_scores(30.0, _rubric("HIGH", 90.0, "reading / writing"), None) == 30.0
    assert pipe._combine_scores(55.0, _rubric("HIGH", 80.0, "watching instructor"), None) == 55.0
