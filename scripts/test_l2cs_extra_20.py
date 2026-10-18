"""Collect 20 new L2CS faces, outside the training 20, and test the fine-tuned model."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
TRAIN_NPZ = ROOT / "data" / "processed" / "extra20" / "gaze_l2cs_20.npz"
OUT = ROOT / "data" / "processed" / "extra20" / "l2cs_test_20"
WEIGHTS = ROOT / "checkpoints" / "l2cs_finetuned_20.pt"
N = 20


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angles(kpts: np.ndarray) -> tuple[float, float] | None:
    if kpts is None or kpts.shape[0] < 3:
        return None
    nose, leye, reye = kpts[0], kpts[1], kpts[2]
    if float(nose[2] + leye[2] + reye[2]) < 0.6:
        return None
    mid = (leye[:2] + reye[:2]) / 2.0
    eye_w = abs(float(reye[0] - leye[0])) + 1e-6
    yaw = float(np.degrees(np.arctan2(nose[0] - mid[0], eye_w)))
    pitch = float(np.degrees(np.arctan2(nose[1] - mid[1], eye_w)))
    return yaw, pitch


def _angular(py: float, pp: float, gy: float, gp: float) -> float:
    pred = np.array(
        [
            np.cos(np.radians(pp)) * np.sin(np.radians(py)),
            np.sin(np.radians(pp)),
            np.cos(np.radians(pp)) * np.cos(np.radians(py)),
        ]
    )
    gt = np.array(
        [
            np.cos(np.radians(gp)) * np.sin(np.radians(gy)),
            np.sin(np.radians(gp)),
            np.cos(np.radians(gp)) * np.cos(np.radians(gy)),
        ]
    )
    cos = float(np.clip(np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt) + 1e-8), -1, 1))
    return float(np.degrees(np.arccos(cos)))


def collect(used: set[str]) -> list[dict]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.detect_track import DetectorTracker

    smoke = _load("smoke_test_train", ROOT / "scripts" / "smoke_test_train.py")
    clips = smoke._pick_daisee_clips(n_clips=120, seed=40)
    detector = DetectorTracker(
        weights="yolo11n-pose.pt", conf=0.25, imgsz=640, tiles=False, device="cpu"
    )
    faces = OUT / "faces"
    faces.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path, _lab in clips:
        if path.name in used or path.stem in used:
            continue
        if len(rows) >= N:
            break
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        for det in detector.infer(frame):
            ang = _angles(det.keypoints)
            if ang is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            h, w = frame.shape[:2]
            y2 = min(y1 + int(max(y2 - y1, 1) * 0.45), h)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, w)
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            name = f"{len(rows):04d}.jpg"
            cv2.imwrite(str(faces / name), frame[y1:y2, x1:x2])
            rows.append({"file": name, "yaw": ang[0], "pitch": ang[1], "source": path.name})
            print(f"test face {len(rows)}/{N}  {path.name}")
            break
    if len(rows) < N:
        raise SystemExit(f"Only collected {len(rows)} new L2CS test faces.")
    (OUT / "labels.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def evaluate(rows: list[dict]) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    bench = _load("benchmark_smoke", ROOT / "scripts" / "benchmark_smoke.py")
    from attention_pipeline.models.l2cs_net import L2CSNet
    from attention_pipeline.utils import resolve_device

    device = resolve_device("auto")
    model = L2CSNet().to(device).eval()
    model.load(WEIGHTS, map_location=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    yaw_err, pitch_err, angular = [], [], []
    y_true, y_pred = [], []
    for row in rows:
        img = cv2.imread(str(OUT / "faces" / row["file"]))
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).unsqueeze(0).to(device)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            pred = model.predict_degrees(tensor).squeeze(0).cpu().numpy()
        pyaw, ppitch = float(pred[0]), float(pred[1])
        gyaw, gpitch = float(row["yaw"]), float(row["pitch"])
        yaw_err.append(abs(pyaw - gyaw))
        pitch_err.append(abs(ppitch - gpitch))
        angular.append(_angular(pyaw, ppitch, gyaw, gpitch))
        y_true.append(bench._yaw_bin(gyaw) * 3 + bench._pitch_bin(gpitch))
        y_pred.append(bench._yaw_bin(pyaw) * 3 + bench._pitch_bin(ppitch))
    cls = bench._prf1_acc(y_true, y_pred, n_classes=9)
    cls["angular_mae_deg"] = float(np.mean(angular))
    cls["yaw_mae_deg"] = float(np.mean(yaw_err))
    cls["pitch_mae_deg"] = float(np.mean(pitch_err))
    cls["support"] = len(y_true)
    return cls


def write_report(metrics: dict) -> Path:
    folder = ROOT / "runs" / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"l2cs_test_20_{stamp}_accuracy.txt"
    lines = [
        "=" * 72,
        "L2CS-NET TEST  (20 new images)",
        "=" * 72,
        "",
        "These 20 face images were not in the 20-image L2CS training set.",
        f"Weights : {WEIGHTS}",
        f"Images  : {OUT / 'faces'}",
        f"Support : {metrics['support']}",
        "",
        f"Precision : {metrics['precision']*100:.2f}%",
        f"Recall    : {metrics['recall']*100:.2f}%",
        f"F1        : {metrics['f1']*100:.2f}%",
        f"Accuracy  : {metrics['accuracy']*100:.2f}%",
        "",
        f"Yaw error    : {metrics['yaw_mae_deg']:.2f} deg",
        f"Pitch error  : {metrics['pitch_mae_deg']:.2f} deg",
        f"Angular error: {metrics['angular_mae_deg']:.2f} deg",
        "=" * 72,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    trained = np.load(TRAIN_NPZ)
    used = set(str(x) for x in trained["clip"])
    print(f"Holding out {len(used)} training clips")
    rows = collect(used)
    metrics = evaluate(rows)
    path = write_report(metrics)
    print(
        f"L2CS test accuracy {metrics['accuracy']*100:.2f}%  "
        f"P {metrics['precision']*100:.2f}%  R {metrics['recall']*100:.2f}%  "
        f"F1 {metrics['f1']*100:.2f}%"
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
