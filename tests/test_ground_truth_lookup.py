"""Ground truth files are resolved per video, not only for clip_004."""

from __future__ import annotations

import json
from pathlib import Path

from attention_pipeline.gt_calibrator import find_ground_truth
from attention_pipeline.video_accuracy import score_ground_truth


def test_ground_truth_beside_any_video(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "ground_truth.csv"
    xml_path = tmp_path / "annotations.xml"
    csv_path.write_text("frame,student_id,attention\n0,0,Attentive\n", encoding="utf-8")
    xml_path.write_text(
        "<annotations><track id='0'>"
        "<box frame='0' outside='0' xtl='0' ytl='0' xbr='10' ybr='10'/>"
        "</track></annotations>",
        encoding="utf-8",
    )
    found = find_ground_truth(video)
    assert found == (csv_path, xml_path)

    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"frame": 0, "students": [{"bbox": [0, 0, 10, 10], "band": "HIGH"}]}) + "\n",
        encoding="utf-8",
    )
    metrics = score_ground_truth(scores, csv_path, xml_path)
    assert metrics["support"] == 1
    assert abs(metrics["accuracy"] - 1.0) < 1e-6


def test_clip_004_uses_anno_files() -> None:
    video = Path("video") / "clip_004.mp4"
    if not video.exists():
        return
    found = find_ground_truth(video)
    assert found is not None
    csv_path, xml_path = found
    assert csv_path.name == "ground_truth.csv"
    assert xml_path.name == "annotations.xml"


def test_unlabeled_video_has_no_ground_truth() -> None:
    found = find_ground_truth(Path("video") / "Trainee 1.mp4")
    assert found is None
