"""One folder for every SCB image, DAiSEE clip, and L2CS face that has been trained.

New training appends samples. A sample already stored is left as it is.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from attention_pipeline.utils import PROJECT_ROOT

STORE = PROJECT_ROOT / "data" / "processed" / "trained"


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else STORE


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def load_manifest(root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = _manifest_path(_root(root))
    if not path.exists():
        return {"scb": [], "daisee": [], "l2cs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "scb": list(data.get("scb") or []),
        "daisee": list(data.get("daisee") or []),
        "l2cs": list(data.get("l2cs") or []),
    }


def _save_manifest(root: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "counts": {key: len(data[key]) for key in ("scb", "daisee", "l2cs")},
        "scb": data["scb"],
        "daisee": data["daisee"],
        "l2cs": data["l2cs"],
    }
    path = _manifest_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _place(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _label_for(image: Path) -> Path | None:
    parts = list(image.parts)
    if "images" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    candidate = Path(*parts).with_suffix(".txt")
    return candidate if candidate.exists() else None


def _known(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows}


def add_scb_images(images: list[Path], source: str, root: Path | None = None) -> int:
    """Copy SCB images and their labels into the store. Returns how many were new."""
    folder = _root(root)
    data = load_manifest(folder)
    known = _known(data["scb"])
    added = 0
    for image in images:
        image = Path(image)
        stem = image.stem
        if stem in known or not image.exists():
            continue
        dest = folder / "scb" / "images" / image.name
        _place(image, dest)
        label = _label_for(image)
        if label is not None:
            _place(label, folder / "scb" / "labels" / f"{stem}.txt")
        data["scb"].append(
            {"id": stem, "file": dest.relative_to(folder).as_posix(), "source": source}
        )
        known.add(stem)
        added += 1
    if added:
        _save_manifest(folder, data)
    return added


def add_daisee_clips(clips: list[Path], source: str, root: Path | None = None) -> int:
    """Copy DAiSEE clips into the store. Returns how many were new."""
    folder = _root(root)
    data = load_manifest(folder)
    known = _known(data["daisee"])
    added = 0
    for clip in clips:
        clip = Path(clip)
        stem = clip.stem
        if stem in known or not clip.exists():
            continue
        dest = folder / "daisee" / clip.name
        _place(clip, dest)
        data["daisee"].append(
            {"id": stem, "file": dest.relative_to(folder).as_posix(), "source": source}
        )
        known.add(stem)
        added += 1
    if added:
        _save_manifest(folder, data)
    return added


def add_l2cs_faces(faces: list[dict[str, Any]], source: str, root: Path | None = None) -> int:
    """Save L2CS face images. Each face needs id and an RGB array shaped (3, H, W) or (H, W, 3)."""
    import cv2
    import numpy as np

    folder = _root(root)
    data = load_manifest(folder)
    known = _known(data["l2cs"])
    added = 0
    face_dir = folder / "l2cs" / "faces"
    face_dir.mkdir(parents=True, exist_ok=True)
    for face in faces:
        stem = str(face["id"])
        if stem in known:
            continue
        image = np.asarray(face["image"])
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dest = face_dir / f"{stem}.jpg"
        cv2.imwrite(str(dest), bgr)
        row: dict[str, Any] = {
            "id": stem,
            "file": dest.relative_to(folder).as_posix(),
            "source": source,
            "clip": str(face.get("clip") or stem),
        }
        if face.get("yaw") is not None:
            row["yaw"] = float(face["yaw"])
        if face.get("pitch") is not None:
            row["pitch"] = float(face["pitch"])
        data["l2cs"].append(row)
        known.add(stem)
        added += 1
    if added:
        _save_manifest(folder, data)
    return added


def add_l2cs_npz(path: Path, source: str, root: Path | None = None) -> int:
    """Add faces from a training npz. Clip names are used when there is one per face."""
    import numpy as np

    archive = np.load(path, allow_pickle=True)
    images = archive["images"]
    clips = [str(item) for item in archive["clip"]] if "clip" in archive.files else []
    yaw = archive["yaw"] if "yaw" in archive.files else None
    pitch = archive["pitch"] if "pitch" in archive.files else None
    aligned = len(clips) == len(images)
    faces = []
    for index, image in enumerate(images):
        clip = Path(clips[index]).stem if aligned else f"face_{index:06d}"
        face_id = clip
        if any(item["id"] == face_id for item in faces):
            face_id = f"{clip}_{index:04d}"
        faces.append(
            {
                "id": face_id,
                "image": image,
                "clip": clip,
                "yaw": None if yaw is None or len(yaw) != len(images) else float(yaw[index]),
                "pitch": None if pitch is None or len(pitch) != len(images) else float(pitch[index]),
            }
        )
    return add_l2cs_faces(faces, source, root)


def counts(root: Path | None = None) -> dict[str, int]:
    data = load_manifest(root)
    return {key: len(data[key]) for key in ("scb", "daisee", "l2cs")}
