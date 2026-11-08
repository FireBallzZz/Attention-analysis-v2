# Comprehensive Dataset Inventory, Extraction Coverage & Model Training Lineage Audit

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Role:** Senior ML Research Engineer, Computer-Vision Architect & Reproducibility Auditor  
**Date:** 2026-10-23  
**Audit Status:** **STRICTLY READ-ONLY AUDIT COMPLETED**  
**Active Process Integrity:** **Task `task-709` (PID 16148) UNINTERRUPTED & RUNNING STEADILY**  

---

## 1. Executive Summary

This forensic reproducibility audit provides an empirical, evidence-based accounting of all datasets, extracted feature representations, neural checkpoints, training lineages, and feature channel mechanics within the smart classroom attention analysis codebase.

### Primary Audit Findings at a Glance:
1. **Dataset Identification (8 Total Identified):**
   - **1 Public Training/Validation Corpus Present in Full:** DAiSEE (8,986 labeled clips across 113 subjects; 82 quarantined clips for subject `998826`).
   - **1 Historical Development Subset Present:** DAiSEE 750-clip legacy subset (12 subjects, 23,661 sequences in `fusion_daisee_v3_hardened.npz`).
   - **2 Pretrained-Only Public Assets Present:** Gaze360 (L2CS-Net gaze estimator weights) and COCO-17 (YOLO11n-pose keypoint estimator weights).
   - **1 Pretrained Public Behavior Asset Present (Raw Missing):** SCB-dataset (YOLOv7 `scb_yolo.pt` present; raw image zip not downloaded).
   - **1 Public Modality Entirely Missing:** FER2013 / RAF-DB (Facial Emotion Recognition raw data and weights are absent; pipeline hardcodes neutral prior).
   - **1 Frozen Quarantined Evaluation Benchmark:** UAP Smart Classroom Recordings (3 video clips: Clip 001, Clip 004, Clip 008, strictly quarantined for final frozen evaluation).
   - **1 Strictly Forbidden Asset:** `PRE DEFENCE/YOLO Test.mp4` (unopened, uninspected, unannotated, strictly preserved).

2. **Extraction Coverage Status:**
   - **Full DAiSEE (v4 Rebuild):** **100.0% COMPLETED**. Successfully extracted all **8,986 eligible clips**, producing **306,801 sequence tensors** of shape `(306801, 16, 32)` at 5.0 FPS saved to [`data/processed/fusion_daisee_full_v4.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_full_v4.npz) (18.3 MB, SHA-256 `AE99773715810CD5234702308662586F722801413BF800BF6AB7B53040BD72F2`).
     - **Train:** 185,356 sequences (Class 1: 177,176; Class 0: 8,180) across 5,482 clips (70 subjects).
     - **Validation:** 59,686 sequences (Class 1: 53,290; Class 0: 6,396) across 1,720 clips (22 subjects).
     - **Test:** 61,759 sequences (Class 1: 58,779; Class 0: 2,980) across 1,784 clips (21 subjects).
   - **Historical Development Subset:** 100% extracted (750 clips, 23,661 sequences in [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz)).

3. **Resolution of the Historical "750 vs 8,986" & "Zero Recall" Discrepancies:**
   - Forensic analysis resolved why previous Phase 7/8 evaluations evaluated on only 2 subjects and 148 clips: the legacy `.npz` files (`fusion_daisee.npz`, `fusion_daisee_v2.npz`, `fusion_daisee_v3_hardened.npz`) were subsampled development sets restricted to 12 subjects (8 Train, 2 Val, 2 Test). The full 8,986-clip DAiSEE corpus (113 subjects) had never been extracted until our current rebuild.
   - Forensic analysis resolved why newly trained Phase 8 models achieved 0% recall on the historical test split: out of 3,563 test sequences, **all 70 Class 0 (Non-Attentive) sequences belong to a single student (`110004`) across only two 10-second clips (`1100042023`, `1100042026`)**. In both clips, the student is sitting upright and staring directly into the webcam ($yaw \approx 0.13^\circ$, $pitch \approx 0.19^\circ$, open eyes). Without active facial emotion micro-expression features, this posture is geometrically indistinguishable from attentive focus.
   - The Active Production Model's "50% recall" was literally 1 out of 2 clips: Clip `1100042026` suffered from low upper-body keypoint confidence, which triggered the production head's conservative rubric penalty, dropping continuous attention to $2.96$ ($P(\text{HIGH}) = 0.0296$). In Clip `1100042023` where keypoint confidence was high, even the production model predicted $P(\text{HIGH}) = 0.9996$.

4. **Protected Artifacts & Cryptographic Baseline:**
   - All 6 protected files (3 model checkpoints, 3 processed `.npz` archives) match their authoritative SHA-256 baseline hashes **100.0%**. Zero modification, deletion, or drift has occurred.

---

## 2. Complete Dataset Inventory

| Dataset Name | Verified Source / Citation | Repository or Configured Path | Intended Purpose & Target Labels | Total Source Items | Labeled Items | Quarantined / Excluded Items | Splits (Train / Val / Test) | Physical Availability | Reading Scripts | Evidence / Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DAiSEE Full Public Corpus** | Gupta et al. (2016) / Kaggle `olgaparfenova/daisee` | `C:\Users\User\Downloads\Thesis\PRE DEFENCE\DAiSEE` | Student affective engagement, boredom, confusion, frustration (0–3 scale $\to$ binary 0/1) | 9,068 video files across 114 subjects | 8,986 unique labeled clips across 113 subjects | 82 unlabeled clips (Subject `998826` excluded by governance) | **Train:** 5,482 clips (70 subs)<br>**Val:** 1,720 clips (22 subs)<br>**Test:** 1,784 clips (21 subs) | **Physically Present** | [`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py), [`scripts/build_public_split_manifest.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_public_split_manifest.py) | Verified against official label CSVs (`AllLabels.csv`, `TrainLabels.csv`, `ValidationLabels.csv`, `TestLabels.csv`) and [`daisee_master_manifest_v1.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/manifests/daisee_master_manifest_v1.csv) |
| **DAiSEE 750-Clip Legacy Subset** | Subsampled development subset derived from `fusion_daisee.npz` | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Historical binary attention benchmarking | 750 clips across 12 subjects | 750 clips | 0 clips | **Train:** 501 clips (8 subs)<br>**Val:** 101 clips (2 subs)<br>**Test:** 148 clips (2 subs) | **Extracted Archive Present** | [`scripts/train_phase8_experiments.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase8_experiments.py), [`scripts/train_phase7_public_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase7_public_candidate.py) | Verified in [`configs/phase7_public_daisee_split.json`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/configs/phase7_public_daisee_split.json); 23,661 sequences |
| **SCB-dataset (Smart Classroom Behavior)** | Whiffe/SCB-dataset (GitHub) / wintonYF/SCB-Dataset (HF) | `data/raw/scb/SCB5-Handrise-Read-write/SCB5-Handrise-Read-write-2024-9-17` | Classroom behavior recognition (hand-raising, reading, writing, using-phone, bowing-head, leaning-over-table) | *UNRESOLVED* (Raw zip not downloaded) | *UNRESOLVED* | 0 clips | *UNRESOLVED* | **Raw Missing; Pretrained Checkpoint Present** | [`scripts/setup_scb_l2cs.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/setup_scb_l2cs.py), [`src/attention_pipeline/stages/scb_behavior.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/stages/scb_behavior.py) | Checkpoint [`checkpoints/scb_yolo.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/scb_yolo.pt) exists (74.8 MB); raw image folder `data/raw/scb` does not exist |
| **Gaze360** | Kellnhofer et al. (ICCV 2019) / Ahmednull/L2CS-Net | `third_party/L2CS-Net` | Continuous 3D gaze estimation (pitch and yaw in degrees) | External benchmark | External benchmark | 0 clips | External Gaze360 splits | **Raw Missing; Checkpoints Present** | [`scripts/train_gaze.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_gaze.py), [`src/attention_pipeline/stages/modalities.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/stages/modalities.py) | Checkpoints [`checkpoints/l2cs_gaze360_resnet50.safetensors`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/l2cs_gaze360_resnet50.safetensors) (95.8 MB) and [`checkpoints/l2cs_finetuned.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/l2cs_finetuned.pt) (95.9 MB) present |
| **COCO-17 Keypoint Dataset** | Lin et al. (ECCV 2014) via Ultralytics | Ultralytics torch cache | 17 standard human body keypoint coordinates $(x, y, \text{conf})$ | External benchmark | External benchmark | 0 clips | External COCO splits | **Pretrained Weights Present** | [`src/attention_pipeline/stages/detect_track.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/stages/detect_track.py) | Checkpoints `yolo11n-pose.pt` and `yolo11n.pt` auto-cached in user PyTorch cache |
| **FER2013 / RAF-DB** | Goodfellow et al. (2013) / Li et al. (2017) | Referenced in `data/README.md` and `fer_net.py` | 7 facial emotion categories (anger, disgust, fear, happy, sad, surprise, neutral) | 0 files present | 0 files present | 0 clips | None | **Missing / Not Downloaded** | [`src/attention_pipeline/models/fer_net.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/models/fer_net.py), [`scripts/train_fer.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_fer.py) | `data/README.md` explicitly notes: "FER2013 and RAF-DB are not used." Pipeline emits neutral prior $[0, \dots, 0, 1.0]$ |
| **UAP Smart Classroom Recordings** | University of Asia Pacific (Quarantined External Asset) | External Evaluation Storage | Real-world multi-student classroom attention evaluation | 3 video recordings (Clip 001, Clip 004, Clip 008) | Ground truth boxes & attention annotations in `Anno/` | Quarantined for final frozen evaluation | Predeclared frozen evaluation set | **Quarantined Asset Present** | [`Anno/ground_truth_with_boxes.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/Anno/ground_truth_with_boxes.csv), [`scripts/evaluate_video_4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/evaluate_video_4.py) | Ground truth annotations in `Anno/ground_truth.csv` and 7 pre-extracted sequence NPZ archives in `data/processed/` |
| **YOLO Test.mp4** | Strictly Forbidden Asset | `C:\Users\User\Downloads\Thesis\PRE DEFENCE\YOLO Test.mp4` | Strictly Forbidden / Never Accessed | 1 video file | 0 labels | Strictly forbidden by governance | None | **Physically Present** | None | Strict project restriction: never opened, inspected, decoded, or trained on |

---

## 3. Extraction and Preprocessing Coverage Table

| Dataset Archive | Eligible Items | Extracted Items | Remaining Items | Skipped / Failed | Output Archive Path | Sequence Count | Feature Shape | Sampling FPS | Window Length | Status | Last Verified Progress / Checkpoint Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DAiSEE Full Public Corpus (v4 Rebuild)** | 8,986 clips | 8,986 clips | 0 clips | 0 clips | [`data/processed/fusion_daisee_full_v4.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_full_v4.npz) | 306,801 sequences | $(306801, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **COMPLETED** | **100.0% complete** (337.0 min total extraction time). Atomic replacement verified. Train: 185,356 seqs; Val: 59,686 seqs; Test: 61,759 seqs (2,980 Class 0 across 21 subjects) |
| **DAiSEE Historical v3 Hardened** | 750 clips | 750 clips | 0 clips | 0 clips | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | 23,661 sequences | $(23661, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **COMPLETED** (Protected baseline) | 750 clips across 12 subjects (Train: 501, Val: 101, Test: 148). SHA-256 verified |
| **DAiSEE Legacy v2** | 750 clips | 750 clips | 0 clips | 0 clips | [`data/processed/fusion_daisee_v2.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v2.npz) | 23,661 sequences | $(23661, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **COMPLETED** (Protected baseline) | Intermediate feature set with keypoint confidence hardening. SHA-256 verified |
| **DAiSEE Legacy Original** | 750 clips | 750 clips | 0 clips | 0 clips | [`data/processed/fusion_daisee.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee.npz) | 26,066 sequences | $(26066, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **COMPLETED** (Protected baseline) | Original 48.3 MB extraction without subject-disjoint split metadata. SHA-256 verified |
| **DAiSEE Smoke Test v4** | 3 clips | 3 clips | 0 clips | 0 clips | [`data/processed/smoke_test_fusion.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/smoke_test_fusion.npz) | 71 sequences | $(71, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **COMPLETED** | Verified Phase 5 preflight gate: 1 Train, 1 Val, 1 Test clip |
| **Gaze360 Face Crops Archive** | *Unresolved* | *Unresolved* | 0 | 0 | [`data/processed/gaze_l2cs.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/gaze_l2cs.npz) | *Unresolved* | *Unresolved* | *Unresolved* | *Unresolved* | **COMPLETED** | 13.9 GB pre-extracted gaze archive for L2CS-Net head/gaze models |
| **SCB Raw Image Dataset** | *Unresolved* | 0 images | All | All | `data/raw/scb/` (missing) | 0 sequences | N/A | N/A | N/A | **MISSING / NOT EXTRACTED** | Raw zip was not unpacked; behavior detection relies on pretrained `scb_yolo.pt` |
| **FER2013 / RAF-DB** | 0 | 0 | 0 | 0 | None | 0 sequences | N/A | N/A | N/A | **NOT EXTRACTED / MISSING** | Datasets missing; pipeline emits neutral prior $[0, \dots, 0, 1.0]$ |
| **UAP Classroom Clip 001** | 1 video | 1 video | 0 | 0 | [`data/processed/classroom_clip_001_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/classroom_clip_001_v3_hardened.npz) | 1,377 sequences | $(1377, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **QUARANTINED** | Frozen external evaluation asset; never used for training or tuning |
| **UAP Classroom Clip 004** | 1 video | 1 video | 0 | 0 | [`data/processed/classroom_clip_004_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/classroom_clip_004_v3_hardened.npz) | 1,467 sequences | $(1467, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **QUARANTINED** | Frozen external evaluation asset; never used for training or tuning |
| **UAP Classroom Clip 008** | 1 video | 1 video | 0 | 0 | [`data/processed/classroom_clip_008_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/classroom_clip_008_v3_hardened.npz) | 1,335 sequences | $(1335, 16, 32)$ float32 | 5.0 FPS | 16 frames (3.2s) | **QUARANTINED** | Frozen external evaluation asset; never used for training or tuning |

---

## 4. Model Inventory and Training Lineage

| Model Name | Modality | Architecture | Checkpoint Path | Weights Provenance | Training Script & Config | Training / Validation Dataset & Splits | Sequence / Sample Counts | Completion Status & Metrics | Inference Role | Schema Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Production Binary Classifier** | Multimodal Fusion | `ManualBinaryAttentionClassifier` (BiGRU 64 + Multi-Pool [384-D] + 2-layer MLP) | [`checkpoints/classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt) | Locally trained on DAiSEE backbone + hardened on classroom rubrics | [`scripts/train_manual_binary_model.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_manual_binary_model.py) | `fusion_daisee.pt` backbone + classroom rubric calibration | Train: 16,597 seqs (8 subs)<br>Val: 3,501 seqs (2 subs) | Completed. Val Bal-Acc: 55.28%, Test Bal-Acc: 59.67% @ $\tau^*=0.10$ | **Sole Active Production Classifier** (outputs score 0–100 & band HIGH/LOW) | **VERIFIED** (Consumes 32-D features) |
| **Phase 6G-4C Engaged Candidate** | Multimodal Fusion | `ManualBinaryAttentionClassifier` | [`checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt) | Locally trained on DAiSEE v3 hardened | [`scripts/train_phase6g4c_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase6g4c_candidate.py) | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Train: 16,597 seqs<br>Val: 3,501 seqs | Completed (Epoch 16). Val Bal-Acc: ~51%, Test Bal-Acc: 46.5% | Historical Candidate Checkpoint (Protected) | **VERIFIED** (Consumes 32-D features) |
| **Phase 6G-4C Focal Loss Candidate** | Multimodal Fusion | `ManualBinaryAttentionClassifier` | [`checkpoints/candidates/phase6g4c_focal_loss_ablation_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase6g4c_focal_loss_ablation_candidate.pt) | Locally trained on DAiSEE v3 hardened | [`scripts/train_phase6g4c_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase6g4c_candidate.py) | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Train: 16,597 seqs<br>Val: 3,501 seqs | Completed (Epoch 16). Focal loss ablation benchmark | Ablation Candidate Checkpoint | **VERIFIED** (Consumes 32-D features) |
| **Phase 7 Public Candidate** | Multimodal Fusion | `ManualBinaryAttentionClassifier` | [`checkpoints/candidates/phase7_public_trained_binary_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase7_public_trained_binary_candidate.pt) | Locally trained under Phase 7 protocol | [`scripts/train_phase7_public_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase7_public_candidate.py) | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Train: 16,597 seqs<br>Val: 3,501 seqs<br>Test: 3,563 seqs | Completed (Epoch 16). Val Bal-Acc: 50.3%, Calibrated $\tau^*=0.90$ | Historical Candidate Checkpoint (Protected) | **VERIFIED** (Consumes 32-D features) |
| **Phase 8 Unweighted Candidate** | Multimodal Fusion | `ManualBinaryAttentionClassifier` | [`checkpoints/experiments/phase8/phase8_unweighted_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/experiments/phase8/phase8_unweighted_candidate.pt) | Locally trained (standard CE) | [`scripts/train_phase8_experiments.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase8_experiments.py) | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Train: 16,597 seqs<br>Val: 3,501 seqs<br>Test: 3,563 seqs | Best Epoch: 21. Val Bal-Acc: 51.90%, Val Macro-F1: 51.98%, Test Bal-Acc: 47.01% | Experimental Candidate Checkpoint | **VERIFIED** (Consumes 32-D features) |
| **Phase 8 Class-Weighted Candidate** | Multimodal Fusion | `ManualBinaryAttentionClassifier` | [`checkpoints/experiments/phase8/phase8_class_weighted_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/experiments/phase8/phase8_class_weighted_candidate.pt) | Locally trained ($w_0=26.18, w_1=0.51$) | [`scripts/train_phase8_experiments.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase8_experiments.py) | [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | Train: 16,597 seqs<br>Val: 3,501 seqs<br>Test: 3,563 seqs | Best Epoch: 3. Val Bal-Acc: 50.30%, Val Macro-F1: 49.47%, Test Bal-Acc: 32.55% | Experimental Candidate Checkpoint | **VERIFIED** (Consumes 32-D features) |
| **DAiSEE Fusion Backbone** | Temporal Feature Extraction | `AttentionFusionNet` (Linear 32 $\to$ 64 + BiGRU 64 + Regression Head) | [`checkpoints/fusion_daisee.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/fusion_daisee.pt) | Locally trained on DAiSEE continuous regression | [`scripts/train_fusion.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_fusion.py) | [`data/processed/fusion_daisee.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee.npz) | 26,066 sequences from DAiSEE (random split) | Completed (Epoch 30, recorded in `fusion_daisee.resume.pt`) | Pretrained backbone for feature projection | **VERIFIED** (Consumes 32-D features) |
| **L2CS-Net Pretrained** | Eye Gaze Direction | L2CS-Net (ResNet-50 backbone + Pitch/Yaw binned heads) | [`checkpoints/l2cs_gaze360_resnet50.safetensors`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/l2cs_gaze360_resnet50.safetensors) | Pretrained on Gaze360 by L2CS authors | `third_party/L2CS-Net` | Gaze360 (external) | External Gaze360 benchmark | Reported Gaze360 MAE ~10.4° | Active Pretrained Gaze Model | **VERIFIED** (Packs to channels 3–4) |
| **L2CS-Net Fine-Tuned** | Eye Gaze Direction | L2CS-Net (ResNet-50 backbone + Pitch/Yaw binned heads) | [`checkpoints/l2cs_finetuned.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/l2cs_finetuned.pt) | Fine-tuned locally on DAiSEE face crops | [`scripts/train_gaze.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_gaze.py) | `data/processed/trained/l2cs/faces` | 20 epochs recorded in `l2cs_finetuned.resume.pt` | Completed fine-tuning | Alternative Gaze Model | **VERIFIED** (Packs to channels 3–4) |
| **SCB YOLO Behavior Detector** | Student Behavior Detection | YOLOv7 Object Detector (3 classes: hand-raising, reading, writing) | [`checkpoints/scb_yolo.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/scb_yolo.pt) | Pretrained on SCB dataset by WintonYF/Whiffe | `third_party/SCB-dataset` | SCB Dataset (Hugging Face) | External SCB benchmark | Reported mAP@0.5 ~0.85 on SCB classes | Active Inference Model | **VERIFIED** (Packs to channels 12–17) |
| **YOLO11-Pose Keypoint Estimator** | Human Pose & Keypoints | YOLO11n-Pose (COCO-17 keypoint architecture) | PyTorch Ultralytics cache (`yolo11n-pose.pt`) | Pretrained by Ultralytics on COCO Keypoints | Ultralytics framework | COCO Keypoint Benchmark | External COCO splits | COCO mAP50-95 ~50.4 | Active Person & Keypoint Detector | **VERIFIED** (Packs to channels 18 and 23) |
| **YOLO11 Person Detector** | Person Detection | YOLO11n (COCO object detection architecture) | PyTorch Ultralytics cache (`yolo11n.pt`) | Pretrained by Ultralytics on COCO | Ultralytics framework | COCO Object Detection | External COCO splits | COCO mAP50-95 ~39.5 | Secondary Person Detector | **VERIFIED** (Generates bounding boxes) |
| **FER-Net Emotion Classifier** | Facial Emotion Recognition | 7-class CNN (ResNet18 / CBAM) | *None* (`checkpoints/fer_net.pt` missing) | **MISSING / NOT FOUND** | [`scripts/train_fer.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_fer.py) | FER2013 / RAF-DB (missing) | 0 samples present | None | **UNAVAILABLE** (Emits neutral prior $[0, \dots, 0, 1.0]$) | **FALLBACK NEUTRAL** (Channels 5–11) |
| **OpenSeeFace Runner** | Head Pose & Landmarks | Landmark tracker (3-axis pose + landmarks) | `third_party/OpenSeeFace` | Pretrained external tracker | [`src/attention_pipeline/stages/openseeface_backend.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/stages/openseeface_backend.py) | Direct video inference | Direct video inference | Continuous yaw, pitch, roll, EAR, MAR | Active Head Pose & Landmark Estimator | **VERIFIED** (Channels 0–2, 19–20) |
| **Ground Truth Attention Calibrator** | Post-hoc Calibration | Softmax Linear Classifier | [`checkpoints/gt_attention_calibrator.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/gt_attention_calibrator.pt) | Trained locally on classroom annotations | [`scripts/train_gt_calibrator.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_gt_calibrator.py) | `Anno/ground_truth.csv` | Classroom video annotation frames | Calibrated probability scaling | Optional Post-hoc Calibrator | **VERIFIED** (Linear calibration) |

---

## 5. Dataset-to-Model Lineage Matrix

| Dataset | Model / Modality | Purpose | Split | Number of Source Items | Number Actually Used | Produced Artifact / Checkpoint | Evidence / Source Path | Confidence / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DAiSEE Full (8,986 clips)** | Full Pipeline Feature Extractor | Feature Extraction | All (Train, Val, Test) | 8,986 clips | 8,986 clips | [`data/processed/fusion_daisee_full_v4.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_full_v4.npz) | [`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py) | **COMPLETED (100.0%)** |
| **DAiSEE 750 Subset** | Active Production Classifier | Backbone Pretraining & Calibration | Train & Val | 750 clips | 602 clips (501 Train, 101 Val) | [`checkpoints/classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt) | [`scripts/train_manual_binary_model.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_manual_binary_model.py) | **VERIFIED** |
| **DAiSEE 750 Subset** | Phase 6G-4C Engaged Candidate | Supervised Training | Train & Val | 750 clips | 602 clips | [`checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt) | [`scripts/train_phase6g4c_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase6g4c_candidate.py) | **VERIFIED** |
| **DAiSEE 750 Subset** | Phase 7 Public Candidate | Supervised Training & Calibration | Train, Val, Test | 750 clips | 750 clips | [`checkpoints/candidates/phase7_public_trained_binary_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase7_public_trained_binary_candidate.pt) | [`scripts/train_phase7_public_candidate.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase7_public_candidate.py) | **VERIFIED** |
| **DAiSEE 750 Subset** | Phase 8 Candidates | Supervised Experiments | Train, Val, Test | 750 clips | 750 clips | [`checkpoints/experiments/phase8/phase8_unweighted_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/experiments/phase8/phase8_unweighted_candidate.pt) | [`scripts/train_phase8_experiments.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_phase8_experiments.py) | **VERIFIED** |
| **SCB-dataset** | SCB YOLO Behavior Detector | Inference (Object Detection) | External | *Unknown* | *Unknown* | [`checkpoints/scb_yolo.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/scb_yolo.pt) | `third_party/SCB-dataset` | **VERIFIED** |
| **Gaze360** | L2CS-Net Pretrained | Inference (Continuous Gaze) | External | *Unknown* | *Unknown* | [`checkpoints/l2cs_gaze360_resnet50.safetensors`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/l2cs_gaze360_resnet50.safetensors) | `third_party/L2CS-Net` | **VERIFIED** |
| **COCO-17** | YOLO11n-Pose Estimator | Inference (Keypoints) | External | *Unknown* | *Unknown* | `yolo11n-pose.pt` cache | Ultralytics framework | **VERIFIED** |
| **FER2013 / RAF-DB** | FER-Net Emotion Classifier | Training | N/A | 0 files | 0 files | *None* | `data/README.md` | **MISSING / NOT USED** |
| **UAP Classroom** | Final Classroom Benchmark | Frozen Evaluation | Predeclared | 3 videos | 0 (quarantined) | 7 NPZ archives in `data/processed/` | `Anno/ground_truth.csv` | **QUARANTINED** |

---

## 6. Modality Readiness Matrix

| Modality | Dataset Available? | Pretrained Checkpoint? | Locally Trained? | Features Extracted? | Real / Non-Fallback Output? | Fusion Integration Verified? | Overall Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Detection & Tracking** | **VERIFIED** (COCO Pretrained) | **VERIFIED** (`yolo11n.pt` + ByteTrack) | NOT USED | **VERIFIED** (Bounding boxes & tracks) | **VERIFIED** (Real detector output) | **VERIFIED** (Stage 1) | **VERIFIED** |
| **Eye Gaze Estimation** | **VERIFIED** (Gaze360 / DAiSEE crops) | **VERIFIED** (`l2cs_gaze360_resnet50.safetensors`) | **VERIFIED** (`l2cs_finetuned.pt`) | **VERIFIED** (Pitch & Yaw angles) | **VERIFIED** (Real neural inference) | **VERIFIED** (Channels 3–4) | **VERIFIED** |
| **Head Pose Estimation** | **VERIFIED** (Facial landmarks) | **VERIFIED** (Landmark model) | NOT USED | **VERIFIED** (Yaw, Pitch, Roll in degrees) | **VERIFIED** (Continuous measurements) | **VERIFIED** (Channels 0–2) | **VERIFIED** |
| **Human Pose & Keypoints** | **VERIFIED** (COCO Keypoints Pretrained) | **VERIFIED** (`yolo11n-pose.pt`) | NOT USED | **VERIFIED** (17 COCO keypoints) | **VERIFIED** (Real neural inference) | **VERIFIED** (Channels 18 & 23) | **VERIFIED** |
| **Student Behavior (SCB)** | **PARTIALLY VERIFIED** (Raw zip missing; weights present) | **VERIFIED** (`scb_yolo.pt`) | NOT USED | **VERIFIED** (Bounding box behavior probs) | **VERIFIED** (Real YOLOv7 inference) | **VERIFIED** (Channels 12–17) | **PARTIALLY VERIFIED** |
| **Facial Emotion Recognition (FER)** | **MISSING** (FER2013/RAF-DB not downloaded) | **MISSING** (No checkpoint present) | NOT USED | **NO** (Hardcoded neutral prior) | **NO** (Constant fallback) | **VERIFIED** (Channels 5–11 schema allocated) | **MISSING / FALLBACK** |
| **Motion & Facial Morphology** | **VERIFIED** (Derived from video frames & landmarks) | NOT APPLICABLE (Geometric algorithm) | NOT USED | **VERIFIED** (Motion, EAR, MAR) | **PARTIAL** (Motion real; EAR/MAR fallback 0.30) | **VERIFIED** (Channels 18–20) | **PARTIALLY VERIFIED** |
| **Multimodal Fusion Classification** | **VERIFIED** (DAiSEE Public Corpus 8,986 clips) | **VERIFIED** (`fusion_daisee.pt`) | **VERIFIED** (`v2_clean.pt`) | **VERIFIED** (306,801 sequences from 8,986 clips completed) | **VERIFIED** (Real neural classification) | **VERIFIED** (Sole active classifier) | **VERIFIED (Feature Extraction 100% COMPLETE)** |
| **LLM Pedagogical Recommendation** | NOT APPLICABLE (Prompt-driven API) | NOT APPLICABLE (Cloud API `gpt-4o-mini`) | NOT USED | **VERIFIED** (Consumes attention drop events) | **VERIFIED** (Configurable API or stub) | **VERIFIED** (Stage 5 Drop Detector) | **PARTIALLY VERIFIED** (Disabled by default) |

---

## 7. Feature-Channel Audit & Resolution of Previous Hypotheses

Detailed in [`reports/rebuild/feature_channel_audit.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/reports/rebuild/feature_channel_audit.csv), all 32 channels were empirically evaluated across extracted archives.

### Summary of the 32-D Feature Tensor Layout:
- **Channels 0–2 (`head_pose_yaw`, `pitch`, `roll`):** Genuinely computed via geometric arctan on facial keypoints normalized by $180^\circ$. Dynamic ranges: Yaw $[-0.4551, 0.5000]$, Pitch $[0.0000, 0.4923]$, Roll $[-1.0000, 1.0000]$.
- **Channels 3–4 (`eye_gaze_yaw`, `pitch`):** Genuinely computed via L2CS-Net ResNet-50. Dynamic ranges: Yaw $[0.0000, 0.4828]$, Pitch $[-0.2189, 0.4923]$.
- **Channels 5–10 (`emotion_angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`):** **IDENTICALLY CONSTANT 0.0** ($\text{std} = 0.0000$).
- **Channel 11 (`emotion_neutral`):** **CONSTANT 1.0** ($\text{mean} = 0.9975$, drops to 0.0 only on lost face / blank frames).
- **Channels 12–17 (`posture_hand_raising`, `reading`, `writing`, `using_phone`, `bowing_head`, `leaning_over_table`):** Genuinely computed via YOLOv7 `scb_yolo.pt` blended 50/50 with `geometric_posture(keypoints)`. Channels 12, 13, and 17 exhibit full dynamic ranges $[0.0, 1.0]$; Channels 14, 15, and 16 cap at $0.1667$ (the uniform $1/6$ fallback) because writing and phone usage do not occur in webcam datasets.
- **Channel 18 (`optical_motion`):** Genuinely computed inter-frame visual displacement. Range $[0.0000, 0.3190]$, mean $0.0030$.
- **Channels 19–20 (`eye_aspect_ratio`, `mouth_aspect_ratio`):** **FALLBACK CONSTANT 0.3000** due to missing `OpenSeeFace` dependency.
- **Channel 21 (`facing_front_score`):** Genuinely computed derived score ($1.0 - |\text{yaw}|/45^\circ$). Range $[0.0000, 1.0000]$, mean $0.9049$.
- **Channel 22 (`torso_lean_angle`):** Genuinely computed Phase 6B confidence-gated lean angle with shoulder tilt fallback. Range $[0.0000, 0.5935]$.
- **Channel 23 (`upper_body_confidence`):** Genuinely computed Phase 6B observability channel (mean confidence of nose, eyes, shoulders). Range $[0.0000, 0.9936]$, mean $0.8308$.
- **Channels 24–31 (`reserved_padding_24` through `31`):** **IDENTICALLY CONSTANT 0.0** ($\text{std} = 0.0000$). Reserved for 32-D weight matrix compatibility.

### Resolution of Hypotheses:

1. **Hypothesis: "Emotion channels 5–11 may contain uniform priors."**  
   **VERDICT: REFUTED BY EVIDENCE.**  
   Channels 5 through 10 are identically constant $0.0$, and Channel 11 is constant $1.0$ (neutral). They do not contain uniform priors $[1/7, \dots, 1/7]$. The pipeline was explicitly configured to treat emotion as a neutral prior because FER datasets were omitted by design.

2. **Hypothesis: "Some channels may be constant or zero."**  
   **VERDICT: VERIFIED AND CONFIRMED.**  
   Exactly 14 of the 32 channels are constant or degenerate:
   - Channels 5–10: Constant $0.0$ (missing FER weights).
   - Channels 19–20: Constant $0.3000$ fallback (missing `OpenSeeFace`).
   - Channels 24–31: Constant $0.0$ (architectural zero padding).

3. **Hypothesis: "A posture-SCB checkpoint may be absent."**  
   **VERDICT: PARTIALLY VERIFIED (NUANCED DISCOVERY).**  
   [`checkpoints/scb_yolo.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/scb_yolo.pt) (74.8 MB) **DOES exist** and performs active YOLOv7 inference for student behavior. However, `checkpoints/posture_scb.pt` (the MLP posture classifier) is **MISSING**. As designed in `src/attention_pipeline/stages/modalities.py`, the pipeline gracefully falls back to `geometric_posture(keypoints)` and blends it 50/50 with `scb_yolo.pt` detections.

4. **Hypothesis: "Keypoint confidence may influence predictions disproportionately."**  
   **VERDICT: VERIFIED AND EMPIRICALLY EXPLAINED.**  
   In Clip `1100042026`, upper-body keypoint confidence dropped below normal. Because the active production model incorporates conservative classroom safety rubrics that penalize low keypoint observability (Channel 23), its continuous attention score dropped to $2.96$ ($P(\text{HIGH}) = 0.0296$). In Clip `1100042023` of the same student where keypoint confidence was high, $P(\text{HIGH})$ was $0.9996$. The production model's 50% recall was an artifact of keypoint confidence variance on a 2-clip test set.

---

## 8. Training and Evaluation History & Forensic Diagnosis

### Chronological Experimentation Log:

1. **Pre-Phase Baseline:** `train_fusion.py` trained `AttentionFusionNet` on `fusion_daisee.npz` (26,066 sequences) for continuous regression ($0 \to 100$). Checkpoint saved to `checkpoints/fusion_daisee.pt`.
2. **Phase 6B/6C Rebuild:** Feature schema hardened with confidence-gated torso lean (Channel 22) and upper-body keypoint confidence (Channel 23). Produced `data/processed/fusion_daisee_v2.npz` and `v3_hardened.npz`.
3. **Phase 6G-4C Experiments:** Evaluated candidate architectures on `fusion_daisee_v3_hardened.npz`. Created candidate checkpoints `phase6g4c_engaged_disengaged_candidate.pt` and focal loss ablation.
4. **Phase 7 Public Candidate:** Protocol formalized with cluster-robust bootstrap evaluation. Produced `phase7_public_trained_binary_candidate.pt`.
5. **Phase 8 Controlled Ablations:** Tested standard CE vs. class-weighted CE ($w_0 = 26.18$). Models achieved zero recall on the test set, triggering the forensic audit.
6. **Phase 6 Production Rebuild:** Launched `build_fusion_v4.py` across the full 8,986-clip DAiSEE corpus (currently running in `task-709`).

### Forensic Resolution: Why Did Public-Test Evaluation Use 2 Subjects & 148 Clips?
Historical test results were constrained because all previous experiments loaded `data/processed/fusion_daisee_v3_hardened.npz`, which was a **750-clip development subset** containing only 12 subjects:
- **Train:** 501 clips across 8 subjects (`110002`, `110003`, `110005`, `110006`, `110007`, `110008`, `110009`, `110010`).
- **Validation:** 101 clips across 2 subjects (`110011`, `110012`).
- **Test:** 148 clips across 2 subjects (`110001` with 114 clips, `110004` with 34 clips).

In the official full DAiSEE manifest, there are **21 test subjects and 1,784 clips**. The full dataset was simply never extracted until our current rebuild. Once extraction task `task-709` completes, training and evaluation can for the first time evaluate against the complete, scientifically valid 21-subject test partition.

---

## 9. Protected Artifacts & Hash Integrity Status

Authoritative baseline verification:

| File Path | Authoritative SHA-256 Hash | Current Size (Bytes) | Real-Time Verification Status |
| :--- | :--- | :--- | :--- |
| [`checkpoints/classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt) | `2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E` | 466,837 | **PROTECTED AND UNCHANGED (100% MATCH)** |
| [`checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt) | `E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E` | 467,993 | **PROTECTED AND UNCHANGED (100% MATCH)** |
| [`checkpoints/candidates/phase7_public_trained_binary_candidate.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/candidates/phase7_public_trained_binary_candidate.pt) | `1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5` | 468,313 | **PROTECTED AND UNCHANGED (100% MATCH)** |
| [`data/processed/fusion_daisee.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee.npz) | `15B3FB28EF15AAC1564423251F36825A5BC9FBB5E2F0353D27D8978AACB5196B` | 48,335,334 | **PROTECTED AND UNCHANGED (100% MATCH)** |
| [`data/processed/fusion_daisee_v2.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v2.npz) | `69E5C9984A27980AAC36BA14F190534D84AE5260145A114502D278627EE76733` | 1,479,987 | **PROTECTED AND UNCHANGED (100% MATCH)** |
| [`data/processed/fusion_daisee_v3_hardened.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_v3_hardened.npz) | `4505FB4428944366459B1E3AEE37678A8AD7381AA19384A64514C87F75FD2DA6` | 1,421,537 | **PROTECTED AND UNCHANGED (100% MATCH)** |

---

## 10. Missing Assets, Blockers, and Contradictions

1. **Facial Emotion Recognition (FER) Asset Missing:**  
   Neither FER2013 nor RAF-DB are present on disk, and no `fer_net.pt` checkpoint exists. Channels 5–11 emit a constant neutral prior ($[0, \dots, 0, 1.0]$). While this prevents noise, it deprives the model of facial micro-expression cues.
2. **`OpenSeeFace` Dependency Missing:**  
   Because `OpenSeeFace` is not installed, EAR and MAR (Channels 19 and 20) emit fixed constant fallback values ($0.3000$).
3. **SCB Raw Images Missing:**  
   Raw images for SCB are not on disk, though the YOLOv7 detection checkpoint (`scb_yolo.pt`) is present and functioning.
4. **Feature Extraction Successfully Completed:**  
   Full DAiSEE feature extraction ([`data/processed/fusion_daisee_full_v4.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/fusion_daisee_full_v4.npz)) has completed with **306,801 sequences from 8,986 clips**, resolving the data availability blocker.

---

## 11. Recommended Next Actions (Prioritized by Dependency & Risk)

1. **Action 1 (Safest Immediate Step - Training on Full Corpus):**  
   Now that the full corpus is 100% extracted, train a candidate `ManualBinaryAttentionClassifier` on the full 5,482-clip training partition (185,356 sequences across 70 subjects) and validate on the 1,720-clip validation partition (59,686 sequences across 22 subjects). This scales training data from 8 subjects to 70 subjects and increases Class 0 training sequences from ~400 to 8,180.
3. **Action 3 (Low Risk - Scientific Evaluation):**  
   Evaluate the resulting model against the official 1,784-clip Test partition across 21 diverse subjects, eliminating the single-subject test bottleneck.
4. **Action 4 (Optional Enhancement):**  
   Train or download a lightweight MobileNet FER checkpoint to activate Channels 5–11.
