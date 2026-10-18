"""Trained samples accumulate in one folder and are not stored twice."""

from pathlib import Path

import numpy as np

from attention_pipeline.trained_store import add_l2cs_faces, add_scb_images, counts


def test_new_images_are_added_and_repeats_stay_put(tmp_path: Path) -> None:
    image_dir = tmp_path / "raw" / "images" / "train"
    image_dir.mkdir(parents=True)
    label_dir = tmp_path / "raw" / "labels" / "train"
    label_dir.mkdir(parents=True)
    first = image_dir / "student_a.jpg"
    second = image_dir / "student_b.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    (label_dir / "student_a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    store = tmp_path / "trained"

    assert add_scb_images([first], source="run-1", root=store) == 1
    assert add_scb_images([first, second], source="run-2", root=store) == 1
    assert counts(store)["scb"] == 2
    assert (store / "scb" / "images" / "student_a.jpg").exists()
    assert (store / "scb" / "labels" / "student_a.txt").exists()
    manifest = (store / "manifest.json").read_text(encoding="utf-8")
    assert "run-1" in manifest
    assert manifest.count('"id": "student_a"') == 1


def test_l2cs_face_is_saved_once(tmp_path: Path) -> None:
    store = tmp_path / "trained"
    face = np.zeros((3, 8, 8), np.uint8)
    face[0] = 255
    sample = {"id": "clip9", "image": face, "clip": "clip9.avi", "yaw": 1.5, "pitch": -2.0}
    assert add_l2cs_faces([sample], source="run-1", root=store) == 1
    assert add_l2cs_faces([sample], source="run-2", root=store) == 0
    assert counts(store)["l2cs"] == 1
    assert (store / "l2cs" / "faces" / "clip9.jpg").exists()
