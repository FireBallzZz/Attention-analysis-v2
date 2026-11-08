# Thesis Project Inventory & Artifact Audit

**Project Title:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Active Production Model:** v2.4 Production Engine  
**Audit Date:** 2026-11-08  
**Hardware Environment:** NVIDIA GeForce RTX 3060 (12GB VRAM), CUDA 12.8, PyTorch 2.6.0, Python 3.14  

---

## 1. Executive Summary

This inventory documents all core model weights, source scripts, ground-truth annotations, processed dataset archives, configurations, and evaluation reports in the thesis repository. Each artifact has been inspected for existence, purpose, cryptographic hash integrity (SHA-256), and reproduction dependency.

### Active Champion Checkpoint Identification
- **Active Champion Checkpoint:** `checkpoints/classroom_manual_binary_unquarantined.pt`
- **SHA-256:** `46451d72a16ce1e26f3d06ce90f3adc484a1370c8fba1058b8260a8c973a8799`
- **File Size:** 468,218 bytes
- **Architecture:** `ManualBinaryAttentionClassifier` (Frame Encoder + Bidirectional GRU + Multi-Pooling [Last + Mean + Max = 384-D] + 2-layer MLP Head)
- **Tuning Configuration:** `Config_C_Focal_Loss` ($\gamma = 2.0, \alpha = 0.45$, LR temporal = 1e-4, LR head = 1.5e-3, best epoch 6, seed 42)
- **Calibration Status:** Evaluated on 4,602 student-disjoint sequences; default threshold $\tau = 0.50$, calibrated threshold $\tau^* = 0.50$.

---

## 2. Checkpoint & Model Weight Inventory

| Checkpoint Path | Purpose / Architecture | Status | Required for Reproduction | SHA-256 Hash |
| :--- | :--- | :---: | :---: | :--- |
| `checkpoints/classroom_manual_binary_unquarantined.pt` | Champion v2.4 production classifier (Focal Loss, Multi-Pool 384-D) | Present | **YES** | `46451d72a16ce1e26f3d06ce90f3adc484a1370c8fba1058b8260a8c973a8799` |
| `checkpoints/scb_yolo.pt` | SCB-6 classroom behavioral category detector (YOLOv8 fine-tuned) | Present | **YES** | `1f93f151286b21c89338fb8e229b218c5b1d061cc7361c042aca6d416a415857` |
| `checkpoints/l2cs_finetuned.pt` | L2CS-Net gaze direction estimator (yaw, pitch) | Present | **YES** | `6524d7d53e26688cf4d7dca310c65e0fb8847bab60a6fc8614e9b643aef55528` |
| `checkpoints/baseline/fusion_daisee.pt` | DAiSEE pre-trained temporal BiGRU feature backbone | Present | **YES** | `b1d978ef12beda220b53c1ce14c1054ee50e9be636d5f336c7caaaa70b0116e7` |
| `yolo11n-pose.pt` | YOLO11-Pose detector (17 human keypoints, 1920px resolution) | Present | **YES** | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |
| `checkpoints/classroom_manual_binary_model_v1.pt` | Legacy binary classifier (standard CE loss) | Present | No (Archived) | `f0a2e3...` |
| `checkpoints/classroom_manual_binary_model_v2_clean.pt` | Legacy binary classifier (unweighted clean) | Present | No (Archived) | `c9704e...` |

---

## 3. Core Source Code Pipeline

| File Path | Description & Functional Responsibility | Matches Claimed v2.4 |
| :--- | :--- | :---: |
| `run_production_pipeline.py` | Full inference entrypoint: video ingestion, detection, tracking, feature extraction, scoring, JSONL logging, video rendering | **YES** |
| `src/attention_pipeline/pipeline.py` | Orchestrator connecting detector, pose, gaze, rubric, temporal BiGRU, drop alerts, and LLM hooks | **YES** |
| `src/attention_pipeline/stages/detect_track.py` | YOLO11 pose extraction, ByteTrack integration, anti-jitter EMA box smoothing, tiered head-separation filtering | **YES** |
| `src/attention_pipeline/features.py` | 32-D multimodal feature packer: head pose (3), gaze (2), emotion (7), SCB-6 (6), motion (1), EAR (1), MAR (1), facing front (1), torso lean (1), keypoint conf (1), zero padding (8) | **YES** |
| `src/attention_pipeline/rubric.py` | Rule-based classroom behavioral rubric: desk study cues, disengagement penalties, and peer-facing heuristics | **YES** |
| `src/attention_pipeline/models/attention_fusion.py` | Base temporal network: Frame linear projection ($32 \rightarrow 64$) and bidirectional GRU ($64 \rightarrow 128$) | **YES** |
| `src/attention_pipeline/models/manual_binary.py` | Multi-pool wrapper: Concat(Last 128, Mean 128, Max 128) = 384-D, 2-layer MLP head ($384 \rightarrow 128 \rightarrow 64 \rightarrow 2$) | **YES** |
| `src/attention_pipeline/video_accuracy.py` | Empirical heuristic check-band accuracy module | **YES** |
| `configs/pipeline.yaml` | Master pipeline configuration (FPS=10, imgsz=1920, conf=0.15, window=16, $\tau=0.50$, auto_max_students=false) | **YES** |

---

## 4. Ground-Truth Annotations & Dataset Archives

| Artifact Path | Format | Description / Provenance | Verified Status |
| :--- | :--- | :--- | :---: |
| `runs/Annoted file for clip 1/annotations.xml` | CVAT XML | Independent human bounding box and attention annotations for `clip_001.mp4` (20 student tracks, 5,980 boxes) | Verified |
| `Anno/annotations.xml` | CVAT XML | Manual bounding box annotations for classroom clips 004, 007, 008 | Verified |
| `Anno/ground_truth.csv` | CSV | Frame-by-frame 3-class attention labels (Attentive, Moderate, Non-Attentive) | Verified |
| `data/processed/classroom_clip_001_v3_hardened.npz` | NumPy NPZ | 5,463 extracted 16-frame feature sequences for Clip 001 | Verified |
| `data/processed/classroom_clip_004_v3_hardened.npz` | NumPy NPZ | 5,593 extracted 16-frame feature sequences for Clip 004 | Verified |
| `data/processed/classroom_clip_008_v3_hardened.npz` | NumPy NPZ | 5,274 extracted 16-frame feature sequences for Clip 008 | Verified |
| `PRE DEFENCE/YOLO Test.mp4` | MP4 | **RESTRICTED ASSET** — Forbidden per audit rules; provenance unverified. Kept completely untouched. | **EXCLUDED** |

---

## 5. Experimental Logs & Prior Supervisor Reports

| File Path | Description | Audited Role |
| :--- | :--- | :--- |
| `runs/reports/Student_Attention_Model_Supervisor_Report.pdf` | Prior 4-page PDF supervisor report submitted for review | **Subject of Audit** (contains discrepancies detailed in Section 6) |
| `runs/reports/tuning_scientific_audit.json` | Full hyperparameter grid search logs across 5 candidate architectures | Ground-truth log for champion model selection |
| `runs/reports/classroom_unquarantined_binary_metrics.json` | Training and evaluation log for unquarantined classroom run | Identifies the $N=4,602$ vs $N=4,386$ mismatch |
| `runs/reports/clip_001_metrics.json` | Evaluation log for Clip 001 at IoU $\ge 0.30$ | Documents that reported Macro-F1 (97.50%) is positive-class F1 |
| `runs/reports/clip_007_metrics.json` | Evaluation log for Clip 007 at IoU $\ge 0.30$ | Documents that reported Macro-F1 (89.25%) is positive-class F1 |
| `runs/reports/video_project_3_metrics.json` | Analysis log for Video Project 3 (52 detected tracks) | Confirms evaluation was against heuristic check band, not CVAT GT |
| `runs/reports/test_4_metrics.json` | Analysis log for test 4 multi-row classroom (23 students tracked) | Confirms evaluation was against heuristic check band, not CVAT GT |

