"""Score the SCB images, DAiSEE clips, and L2CS faces in data/processed/trained.

Uses the merged checkpoints. Writes a new report under runs/reports.
Does not change runs/smoke_metrics.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "processed" / "trained"
SCB_WEIGHTS = ROOT / "checkpoints" / "scb_yolo11n_merged.pt"
GAZE_WEIGHTS = ROOT / "checkpoints" / "l2cs_merged.pt"
FUSION_WEIGHTS = ROOT / "checkpoints" / "fusion_merged.pt"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench():
    return _load("benchmark_smoke", ROOT / "scripts" / "benchmark_smoke.py")


def _manifest() -> dict:
    path = STORE / "manifest.json"
    if not path.exists():
        raise SystemExit("Missing data/processed/trained. Run: python scripts/collect_trained.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(metrics: dict) -> str:
    return (
        f"P {metrics['precision']*100:.2f}%  "
        f"R {metrics['recall']*100:.2f}%  "
        f"F1 {metrics['f1']*100:.2f}%  "
        f"Acc {metrics['accuracy']*100:.2f}%"
    )


def test_scb(bench) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector

    images = STORE / "scb" / "images"
    labels = STORE / "scb" / "labels"
    detector = SCBBehaviorDetector(weights=SCB_WEIGHTS, device="cpu", conf=0.25)
    y_true: list[int] = []
    y_pred: list[int] = []
    tp = fp = fn = 0
    paths = sorted(images.glob("*.jpg"))
    print(f"SCB images: {len(paths)}")
    for index, img_path in enumerate(paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        gts = bench._load_yolo_labels(labels / f"{img_path.stem}.txt", width, height)
        pred_boxes = []
        for pred in detector.infer(frame):
            if pred.name == "reading":
                cls = 1
            elif pred.name == "writing":
                cls = 2
            elif pred.name == "hand-raising":
                cls = 0
            else:
                cls = pred.cls if pred.cls < 3 else 0
            pred_boxes.append((cls, pred.bbox_xyxy))
        matched = set()
        for g_cls, g_box in gts:
            best_i, best_iou = -1, 0.0
            for i, (p_cls, p_box) in enumerate(pred_boxes):
                if i in matched:
                    continue
                iou = bench._iou(g_box, p_box)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= 0.5:
                matched.add(best_i)
                p_cls = pred_boxes[best_i][0]
                y_true.append(g_cls)
                y_pred.append(p_cls)
                if p_cls == g_cls:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                fn += 1
                y_true.append(g_cls)
                y_pred.append(-1)
        fp += len(pred_boxes) - len(matched)
        if index % 10 == 0 or index == len(paths):
            print(f"  SCB {index}/{len(paths)}")
    det_p = tp / (tp + fp) if (tp + fp) else 0.0
    det_r = tp / (tp + fn) if (tp + fn) else 0.0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) else 0.0
    matched_true = [t for t, p in zip(y_true, y_pred) if p >= 0]
    matched_pred = [p for p in y_pred if p >= 0]
    cls = bench._prf1_acc(matched_true, matched_pred, n_classes=3) if matched_true else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "support": 0
    }
    return {
        "precision": det_p,
        "recall": det_r,
        "f1": det_f1,
        "accuracy": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "support": len(paths),
        "class_accuracy": cls["accuracy"],
    }


def test_l2cs(bench) -> dict:
    import torch
    from torchvision.transforms import functional as TF

    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.models.l2cs_net import L2CSNet
    from attention_pipeline.utils import resolve_device

    rows = [row for row in _manifest()["l2cs"] if "yaw" in row and "pitch" in row]
    device = resolve_device("auto")
    model = L2CSNet().to(device).eval()
    model.load(GAZE_WEIGHTS, map_location=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    yaw_err, pitch_err, angular = [], [], []
    y_true, y_pred = [], []
    print(f"L2CS faces: {len(rows)}")
    for row in rows:
        image = cv2.imread(str(STORE / row["file"]))
        rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
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
    metrics = bench._prf1_acc(y_true, y_pred, n_classes=9)
    metrics["yaw_mae_deg"] = float(np.mean(yaw_err)) if yaw_err else 0.0
    metrics["pitch_mae_deg"] = float(np.mean(pitch_err)) if pitch_err else 0.0
    metrics["angular_mae_deg"] = float(np.mean(angular)) if angular else 0.0
    return metrics


def _angular(pyaw: float, ppitch: float, gyaw: float, gpitch: float) -> float:
    pred = np.array(
        [
            np.cos(np.radians(ppitch)) * np.sin(np.radians(pyaw)),
            np.sin(np.radians(ppitch)),
            np.cos(np.radians(ppitch)) * np.cos(np.radians(pyaw)),
        ]
    )
    truth = np.array(
        [
            np.cos(np.radians(gpitch)) * np.sin(np.radians(gyaw)),
            np.sin(np.radians(gpitch)),
            np.cos(np.radians(gpitch)) * np.cos(np.radians(gyaw)),
        ]
    )
    cosine = float(np.clip(np.dot(pred, truth), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def test_daisee(bench) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.types import FEAT_DIM
    from attention_pipeline.utils import resolve_device

    smoke = _load("smoke_test_train", ROOT / "scripts" / "smoke_test_train.py")
    labels = smoke._daisee_label_map()
    clips = sorted((STORE / "daisee").glob("*.avi")) + sorted((STORE / "daisee").glob("*.mp4"))
    print(f"DAiSEE clips: {len(clips)}")
    pipe = AttentionPipeline.from_yaml(ROOT / "configs" / "pipeline.yaml", load_weights=True)
    window = int(pipe.cfg.get("fusion", {}).get("window", 16))
    features, scores, auxs = [], [], []
    for index, path in enumerate(clips, start=1):
        lab = labels.get(path.stem)
        if lab is None:
            print(f"  skip {path.name} (no DAiSEE label)")
            continue
        pipe.tracks.clear()
        pipe._lesson_yaw = None
        score = float(np.clip(float(lab["engagement"]) / 3.0 * 100.0, 0, 100))
        aux = np.array(
            [lab["engagement"], lab["boredom"], lab["confusion"], lab["frustration"]],
            np.float32,
        )
        collected: list[np.ndarray] = []
        for ts, frame in smoke._iter_clip(path, max_seconds=2.0, target_fps=5.0):
            pipe.process_frame(ts, frame)
            for track in pipe.tracks.values():
                if track.feature_window:
                    collected = list(track.feature_window[-window:])
                    break
            if len(collected) >= window:
                break
        if not collected:
            collected = [np.zeros(FEAT_DIM, np.float32) for _ in range(window)]
        while len(collected) < window:
            collected = [collected[0]] + collected
        features.append(np.stack(collected[-window:], axis=0).astype(np.float32))
        scores.append(score)
        auxs.append(aux)
        print(f"  DAiSEE {index}/{len(clips)}  {path.name}")
    device = resolve_device("auto")
    model = AttentionFusionNet().to(device).eval()
    model.load(FUSION_WEIGHTS, map_location=device)
    with torch.no_grad():
        pred_s, pred_a = model(torch.from_numpy(np.stack(features)).to(device))
        pred_s = pred_s.cpu().numpy()
        pred_a = pred_a.cpu().numpy()
    aux_arr = np.stack(auxs)
    y_true = [int(round(float(row[0]))) for row in aux_arr]
    y_pred = [int(np.clip(round(float(score) / 100.0 * 3.0), 0, 3)) for score in pred_s]
    metrics = bench._prf1_acc(y_true, y_pred, n_classes=4)
    metrics["score_mae"] = float(np.mean(np.abs(pred_s - np.asarray(scores, np.float32))))
    metrics["engagement_mae"] = float(np.mean(np.abs(pred_a[:, 0] - aux_arr[:, 0])))
    return metrics


def write_report(scb: dict, gaze: dict, fusion: dict) -> Path:
    folder = ROOT / "runs" / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"trained_{stamp}_accuracy.txt"
    lines = [
        "=" * 72,
        "TRAINED-SET TEST",
        "=" * 72,
        "",
        "Scored on the clips and images already used for training.",
        f"Folder  : {STORE}",
        f"SCB     : {SCB_WEIGHTS.name}",
        f"DAiSEE  : {FUSION_WEIGHTS.name}",
        f"L2CS    : {GAZE_WEIGHTS.name}",
        "",
        f"SCB images   support {scb['support']}",
        f"  {_pct(scb)}",
        "",
        f"DAiSEE clips support {fusion['support']}",
        f"  {_pct(fusion)}",
        f"  score error {fusion['score_mae']:.2f}   engagement error {fusion['engagement_mae']:.2f}",
        "",
        f"L2CS faces   support {gaze['support']}",
        f"  {_pct(gaze)}",
        f"  yaw {gaze['yaw_mae_deg']:.2f} deg   pitch {gaze['pitch_mae_deg']:.2f} deg   "
        f"angular {gaze['angular_mae_deg']:.2f} deg",
        "",
        "This file is a new report. runs/smoke_metrics.json was not changed.",
        "=" * 72,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    for path in (SCB_WEIGHTS, GAZE_WEIGHTS, FUSION_WEIGHTS):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python scripts/merge_checkpoints.py")
    bench = _bench()
    scb = test_scb(bench)
    print("SCB", _pct(scb))
    gaze = test_l2cs(bench)
    print("L2CS", _pct(gaze))
    fusion = test_daisee(bench)
    print("DAiSEE", _pct(fusion))
    report = write_report(scb, gaze, fusion)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
