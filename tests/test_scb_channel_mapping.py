"""The SCB detector must map its own class names onto the posture channels.

The deployed checkpoint resolves three classes and the merged checkpoint
four, and a checkpoint may name its classes differently from the pipeline
taxonomy. If the posture channel were resolved by the checkpoint's class
index, a class name mismatch would silently write a behaviour into the
WRONG channel - for example a "reading" box into the "using-phone" slot,
which the rubric then treats as hard evidence of disengagement.

These tests pin the name-based resolution.
"""

from __future__ import annotations

import numpy as np

from attention_pipeline.stages.scb_behavior import (
    SCBBehaviorDetector,
    SCBDet,
    _canon_name,
    _probs_from_det,
)
from attention_pipeline.types import SCB_BEHAVIOR_LABELS

CH = {name: i for i, name in enumerate(SCB_BEHAVIOR_LABELS)}


def test_canon_name_maps_upstream_spellings() -> None:
    assert _canon_name("hand-raising") == "hand-raising"
    assert _canon_name("read") == "reading"
    assert _canon_name("write") == "writing"
    assert _canon_name("handraising") == "hand-raising"
    assert _canon_name("BowHead") == "bowing-head"
    assert _canon_name("bow-head") == "bowing-head"
    assert _canon_name("TurnHead") == "turnhead"
    assert _canon_name("using-a-phone") == "using-phone"
    assert _canon_name("leaning-on-the-desk") == "leaning-over-table"
    assert _canon_name("leaning-over-the-table") == "leaning-over-table"


def test_channel_map_is_built_from_names_not_indices() -> None:
    det = SCBBehaviorDetector(weights=None)
    # A checkpoint whose class order does NOT match the pipeline taxonomy.
    det.names = ["bowing-head", "writing", "reading", "hand-raising"]
    det._rebuild_channel_map()

    assert det._channel_of[0] == CH["bowing-head"]
    assert det._channel_of[1] == CH["writing"]
    assert det._channel_of[2] == CH["reading"]
    assert det._channel_of[3] == CH["hand-raising"]


def test_unknown_class_gets_no_channel() -> None:
    det = SCBBehaviorDetector(weights=None)
    det.names = ["hand-raising", "reading", "writing", "turnhead"]
    det._rebuild_channel_map()
    # "turnhead" is not part of the taxonomy, so it must not claim a channel.
    assert 3 not in det._channel_of
    assert set(det._channel_of.values()) == {
        CH["hand-raising"],
        CH["reading"],
        CH["writing"],
    }


def test_probs_land_in_the_right_channel() -> None:
    det = SCBBehaviorDetector(weights=None)
    det.names = ["bowing-head", "writing", "reading", "hand-raising"]
    det._rebuild_channel_map()

    # A detection whose NAME is known wins over the class index.
    probs = _probs_from_det(
        SCBDet(np.array([0, 0, 10, 10], np.float32), cls=1, name="reading", conf=0.8),
        det._channel_of,
    )
    assert probs is not None
    assert int(np.argmax(probs)) == CH["reading"]
    # The vector is normalised to sum to one, so a single detection is 1.0.
    assert probs[CH["reading"]] == 1.0
    assert float(probs.sum()) == 1.0

    # A detection with an unknown name falls back to the mapped class index.
    probs = _probs_from_det(
        SCBDet(np.array([0, 0, 10, 10], np.float32), cls=0, name="turnhead", conf=0.6),
        det._channel_of,
    )
    assert probs is not None
    assert int(np.argmax(probs)) == CH["bowing-head"]


def test_merged_four_class_vocabulary_fits_the_taxonomy() -> None:
    """The merged training set must stay a subset of the 6-slot taxonomy."""
    det = SCBBehaviorDetector(weights=None)
    det.names = ["hand-raising", "reading", "writing", "bowing-head"]
    det._rebuild_channel_map()
    assert len(det._channel_of) == 4
    assert all(0 <= ch < len(SCB_BEHAVIOR_LABELS) for ch in det._channel_of.values())
    # using-phone and leaning-over-table stay unreachable, never misrouted.
    assert CH["using-phone"] not in det._channel_of.values()
    assert CH["leaning-over-table"] not in det._channel_of.values()
