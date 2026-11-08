# Standalone Reproducibility Script & Execution Commands

**Project:** Hybrid Student Attention Analysis Thesis Framework (v2.4 Production Engine)  
**Audit Date:** 2026-11-08  
**Environment Requirements:** Python 3.10+, PyTorch 2.0+, CUDA-enabled GPU (RTX 3060 tested), Ultralytics, OpenCV, Scikit-Learn, ReportLab.

---

## 1. Quick Reproduction Commands

From the workspace root (`C:\Users\User\Downloads\Thesis\Model\Attention-analysis-model`):

### 1.1 Complete Metric Audit & Recalculation (All Benchmarks)
Executes independent metric recalculation across Clip 001, Clip 007, Video Project 3, Test 4, and Held-Out Validation:
```bash
python audit/run_metric_audit.py
```
*Outputs:*
- `audit/METRIC_RECALCULATION.json`
- `audit/METRIC_RECALCULATION.csv`

### 1.2 Hardware & Component Runtime Profiling
Benchmarks video decoding, YOLO11-Pose, L2CS gaze, and BiGRU temporal inference on the local GPU:
```bash
python audit/benchmark_runtime.py
```
*Outputs:*
- `audit/RUNTIME_BENCHMARK.json`

### 1.3 Repository & Checkpoint Inventory Audit
Computes cryptographic SHA-256 hashes and verifies all files:
```bash
python audit/generate_inventory.py
```
*Outputs:*
- `audit/FILE_INVENTORY.json`

---

## 2. Production Pipeline Execution Commands

### 2.1 Video Clip 001 (10s Benchmark with User CVAT Annotations)
```bash
python run_production_pipeline.py --video runs/clip_001.mp4 --out-video runs/clip_001_overlay.mp4 --out-jsonl runs/clip_001_scores.jsonl --fps 10
python scripts/analyze_clip_001.py
```

### 2.2 Video Clip 007 (10s Benchmark with CVAT Annotations)
```bash
python run_production_pipeline.py --video runs/clip_007.mp4 --out-video runs/clip_007_overlay.mp4 --out-jsonl runs/clip_007_scores.jsonl --fps 10
python scripts/analyze_clip_007.py
```

### 2.3 Video Project 3 (23s Large Lecture Hall Ecological Stress Test)
```bash
python run_production_pipeline.py --video "runs/Video Project 3.mp4" --out-video runs/video_project_3_overlay.mp4 --out-jsonl runs/video_project_3_scores.jsonl --fps 10
python scripts/analyze_video_project_3.py
```

### 2.4 Test 4 (60s Multi-Row Classroom Video with Middle-Row Occlusion Fix)
```bash
python run_production_pipeline.py --video "runs/test 4.mp4" --out-video runs/test_4_overlay.mp4 --out-jsonl runs/test_4_scores.jsonl --fps 10
python scripts/analyze_test_4.py
```

---

## 3. Retraining and Scientific Hyperparameter Tuning

To reproduce the systematic grid search and train the champion checkpoint:
```bash
python scripts/tune_classroom_model_scientific.py
```
*Saves champion model to:* `checkpoints/classroom_manual_binary_unquarantined.pt`

---

## 4. Protected Asset Invariant Note
Under no circumstances should the restricted asset `PRE DEFENCE/YOLO Test.mp4` be executed, modified, or evaluated. The file remains strictly excluded from all audit and inference workflows.

