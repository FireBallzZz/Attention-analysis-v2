"""Train on 20 SCB images, 20 DAiSEE clips, and 20 L2CS face images.

Official checkpoints/scb_yolo.pt is not replaced.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 20
SEED = 20
SCB_OUT = ROOT / "checkpoints" / "scb_yolo11n_20.pt"
GAZE_NPZ = ROOT / "data" / "processed" / "extra20" / "gaze_l2cs_20.npz"
GAZE_OUT = ROOT / "checkpoints" / "l2cs_finetuned_20.pt"
FUSION_OUT = ROOT / "checkpoints" / "fusion_daisee_20.pt"


def _smoke():
    path = ROOT / "scripts" / "smoke_test_train.py"
    spec = importlib.util.spec_from_file_location("smoke_test_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def _store():
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline import trained_store

    return trained_store


def _remember_scb(data_yaml: Path) -> None:
    images = list((data_yaml.parent / "images").rglob("*.jpg"))
    added = _store().add_scb_images(images, source=f"extra20 seed {SEED}")
    print(f"Trained store SCB +{added} -> data/processed/trained")


def _remember_daisee(clips: list[Path]) -> None:
    added = _store().add_daisee_clips(clips, source=f"extra20 seed {SEED}")
    print(f"Trained store DAiSEE +{added} -> data/processed/trained")


def _remember_l2cs(npz: Path) -> None:
    added = _store().add_l2cs_npz(npz, source=f"extra20 seed {SEED + 1}")
    print(f"Trained store L2CS +{added} -> data/processed/trained")


def _build_l2cs_faces(clips: list[tuple[Path, dict]], out: Path, n_images: int) -> int:
    """One face image from each clip until n_images faces are saved."""
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.detect_track import DetectorTracker

    out.parent.mkdir(parents=True, exist_ok=True)
    detector = DetectorTracker(
        weights="yolo11n-pose.pt", conf=0.25, imgsz=640, tiles=False, device="cpu"
    )
    images: list[np.ndarray] = []
    pitch: list[float] = []
    yaw: list[float] = []
    used: list[str] = []

    def angles(kpts: np.ndarray) -> tuple[float, float] | None:
        if kpts is None or kpts.shape[0] < 3:
            return None
        nose, leye, reye = kpts[0], kpts[1], kpts[2]
        if float(nose[2] + leye[2] + reye[2]) < 0.6:
            return None
        mid = (leye[:2] + reye[:2]) / 2.0
        eye_w = abs(float(reye[0] - leye[0])) + 1e-6
        y = float(np.degrees(np.arctan2(nose[0] - mid[0], eye_w)))
        p = float(np.degrees(np.arctan2(nose[1] - mid[1], eye_w)))
        return y, p

    for path, _lab in clips:
        if len(images) >= n_images:
            break
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        saved = False
        for det in detector.infer(frame):
            ang = angles(det.keypoints)
            if ang is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            h, w = frame.shape[:2]
            y2 = min(y1 + int(max(y2 - y1, 1) * 0.45), h)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, w)
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            images.append(np.transpose(crop, (2, 0, 1)))
            yaw.append(ang[0])
            pitch.append(ang[1])
            used.append(path.name)
            saved = True
            break
        if saved:
            print(f"L2CS face {len(images)}/{n_images}  {path.name}")
    if len(images) < n_images:
        raise SystemExit(f"Only found {len(images)} L2CS faces, need {n_images}.")
    np.savez(
        out,
        images=np.stack(images[:n_images]).astype(np.uint8),
        pitch=np.asarray(pitch[:n_images], np.float32),
        yaw=np.asarray(yaw[:n_images], np.float32),
        clip=np.asarray(used[:n_images]),
    )
    print(f"Wrote {len(images[:n_images])} L2CS faces -> {out}")
    return len(images[:n_images])


def main() -> None:
    smoke = _smoke()
    print(f"Extra training: {N} SCB images, {N} DAiSEE clips, {N} L2CS faces")
    data_yaml = smoke._copy_scb_subset(n_images=N, seed=SEED)
    _run(
        "train_scb_yolo.py",
        "--data",
        str(data_yaml),
        "--model",
        "yolo11n.pt",
        "--epochs",
        "5",
        "--out",
        str(SCB_OUT),
        "--fresh",
    )
    _remember_scb(Path(data_yaml))
    clips = smoke._pick_daisee_clips(n_clips=N, seed=SEED)
    print(f"DAiSEE clips: {len(clips)}")
    npz = smoke._build_fusion_npz(n_clips=N, seconds=2.0, seed=SEED)
    _run(
        "train_fusion.py",
        "--npz",
        str(npz),
        "--epochs",
        "8",
        "--batch",
        "4",
        "--out",
        str(FUSION_OUT),
        "--fresh",
    )
    _remember_daisee([path for path, _lab in clips])
    face_clips = smoke._pick_daisee_clips(n_clips=N * 3, seed=SEED + 1)
    _build_l2cs_faces(face_clips, GAZE_NPZ, N)
    _run(
        "train_gaze.py",
        "--npz",
        str(GAZE_NPZ),
        "--epochs",
        "5",
        "--batch",
        "4",
        "--out",
        str(GAZE_OUT),
        "--fresh",
    )
    _remember_l2cs(GAZE_NPZ)
    print("\nExtra training finished.")
    print(f"  SCB YOLO11 : {SCB_OUT}")
    print(f"  DAiSEE fusion: {FUSION_OUT}")
    print(f"  L2CS faces : {GAZE_OUT}")
    print("Official checkpoints/scb_yolo.pt was not replaced.")


if __name__ == "__main__":
    main()
