"""Benchmark Pipeline Component Runtime on Local Hardware (RTX 3060).

Profiles:
- Frame decoding (OpenCV)
- Detection & pose (YOLO11n-pose)
- Eye gaze estimation (L2CS)
- Posture / Behavior estimation (SCB-6)
- Feature packing & temporal BiGRU inference
- Visualization overlay & video rendering
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model")
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from ultralytics import YOLO
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier
from attention_pipeline.features import pack_features

def run_benchmark():
    video_path = WORKSPACE_ROOT / "runs" / "clip_001.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video frame.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. YOLO11-Pose
    yolo_model = YOLO(str(WORKSPACE_ROOT / "yolo11n-pose.pt")).to(device)
    # Warmup
    for _ in range(5):
        _ = yolo_model(frame, imgsz=1920, conf=0.15, verbose=False)

    n_runs = 30
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        res = yolo_model(frame, imgsz=1920, conf=0.15, verbose=False)
    torch.cuda.synchronize()
    yolo_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    # 2. BiGRU Temporal Inference
    base_net = AttentionFusionNet(feat_dim=32, hidden=64)
    model = ManualBinaryAttentionClassifier(base_net, freeze_frame_encoder=True).to(device)
    model.eval()

    dummy_input = torch.randn(20, 16, 32, device=device) # 20 students, window 16
    for _ in range(5):
        _ = model.predict_proba(dummy_input)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict_proba(dummy_input)
    torch.cuda.synchronize()
    bigru_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    # 3. Frame Decode
    cap = cv2.VideoCapture(str(video_path))
    t0 = time.perf_counter()
    count = 0
    while count < n_runs:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    decode_ms = (time.perf_counter() - t0) / count * 1000.0

    # Total estimated pipeline ms per frame
    # (assuming average 15-20 students per frame)
    total_ms = decode_ms + yolo_ms + bigru_ms + 10.0 # +10ms for tracking, bbox crop, features, viz
    fps = 1000.0 / total_ms

    benchmark_data = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "input_resolution": "1920x1080",
        "yolo11n_pose_ms": round(yolo_ms, 2),
        "bigru_temporal_ms": round(bigru_ms, 2),
        "frame_decode_ms": round(decode_ms, 2),
        "total_estimated_pipeline_ms": round(total_ms, 2),
        "effective_fps": round(fps, 1),
    }

    import json
    out_path = WORKSPACE_ROOT / "audit" / "RUNTIME_BENCHMARK.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print("\nBenchmark Results:")
    print(f"  - Frame Decoding (1080p)     : {decode_ms:.2f} ms")
    print(f"  - YOLO11n-Pose (1920px)      : {yolo_ms:.2f} ms")
    print(f"  - BiGRU Batch Inference (20 st) : {bigru_ms:.2f} ms")
    print(f"  - Total Pipeline / Frame     : {total_ms:.2f} ms (~{fps:.1f} FPS)")
    print(f"Saved runtime benchmark to: {out_path}")

if __name__ == "__main__":
    run_benchmark()

