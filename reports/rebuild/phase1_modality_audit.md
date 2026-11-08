# Phase 1: Model and Feature-Extractor Forensic Audit Report

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Audit Timestamp:** 2026-10-23T15:20:00+06:00  
**Audit Phase:** Phase 1 — Modality & Model Audit  

---

## 1. Executive Summary

Every modality contributing to the 32-dimensional feature vector has been forensically audited to establish its true mathematical implementation, weight provenance, failure modes, and fallback behaviors.

### Summary Classification Table:
- **Category A (Pretrained Model Verified & Ready):** YOLO11n-Pose (`yolo11n-pose.pt`), L2CS-Net ResNet-50 (`checkpoints/l2cs_finetuned.pt`).
- **Category B (Project-Trained Model Verified & Ready):** SCB-YOLO Behavior Detector (`checkpoints/scb_yolo.pt`).
- **Category C (Usable Geometric Baseline / Heuristic):** Head Pose (Yaw, Pitch, Roll), Optical Motion (absdiff), Facing Front Score, Torso Lean Angle (Hardened), Upper-Body Keypoint Confidence (Hardened), Reserved Padding (0.0).
- **Category D (Missing Weights / Omitted by Design):** PostureMLP (`checkpoints/posture_scb.pt` missing -> falls back to `geometric_posture()`), FER Emotion Classifier (omitted -> neutral prior `[0,0,0,0,0,0,1]`).
- **Category E (Blocked by Missing Dependency):** OpenSeeFace (EAR/MAR -> fixed fallback `0.3000`).

---

## 2. Modality Deep-Dive Audits

### 2.1 Eye Gaze Estimation (Channels 3–4)
- **Model:** L2CS-Net (ResNet-50 backbone) predicting continuous pitch and yaw angles.
- **Weights:** `checkpoints/l2cs_finetuned.pt` (95,853,565 bytes, SHA-256: `6524D7D53E26688CF4D7DCA310C65E0FB8847BAB60A6FC8614E9B643AEF55528`).
- **Input Pipeline:** Cropped student face box resized to 224 x 224 RGB, normalized with ImageNet mean/std.
- **Output:** Pitch and Yaw in degrees, normalized by 180.0 deg into [-1.0, 1.0].
- **Fallback Behavior:** If face crop is smaller than 32 x 32 pixels or face is undetected, falls back to head pose pitch and yaw.
- **Scientific Honesty:** Screen-facing gaze direction is not equivalent to cognitive engagement; gaze provides an operational orientation feature.

### 2.2 Head Pose Estimation (Channels 0–2, 21)
- **Implementation:** Geometric trigonometric calculation from 5 facial keypoints of YOLO11n-Pose:
  - Nose (0), Left Eye (1), Right Eye (2), Left Ear (3), Right Ear (4).
  - Yaw = degrees(arctan2(x_nose - x_mid_eye, |x_reye - x_leye| + eps))
  - Pitch = degrees(arctan2(y_nose - y_mid_eye, |x_reye - x_leye| + eps))
  - Roll = degrees(arctan2(y_leye - y_reye, x_leye - x_reye + eps))
- **Facing Front Score (Channel 21):** 1.0 - |Yaw| / 45.0 deg clipped to [0.0, 1.0].
- **Classification:** Category C (Geometric rule-based baseline).

### 2.3 Body Posture & Classroom Behaviors (Channels 12–17, 22–23)
- **Object Detection:** `checkpoints/scb_yolo.pt` (74,825,045 bytes, SHA-256: `1F93F151286B21C89338FB8E229B218C5B1D061CC7361C042ACA6D416A415857`) detects 3 actions (`hand-raising`, `reading`, `writing`).
- **Learned Classifier Status:** The secondary MLP `checkpoints/posture_scb.pt` is **missing** from disk.
- **Heuristic Blending:** In `modalities.py:L173-210`, `geometric_posture(keypoints)` derives 6 posture probabilities from COCO-17 keypoint relative distances. In `pipeline.py:L165-168`, SCB-YOLO probabilities are blended (50% / 50%) with `geometric_posture()`.
- **Torso Lean (Channel 22):** Hardened confidence-gated calculation (c >= 0.25) with inter-shoulder tilt fallback when student hips are occluded by classroom desks.
- **Upper-Body Confidence (Channel 23):** Arithmetic mean confidence of 5 core anatomical keypoints (Nose, Left Eye, Right Eye, Left Shoulder, Right Shoulder).

### 2.4 Facial Emotion Recognition (Channels 5–11)
- **Status:** **Omitted by design**. FER2013/RAF-DB models are not instantiated.
- **Representation:** Constant neutral prior: Channels 5–10 are constant 0.0, Channel 11 is constant 1.0.
- **Audit Requirement:** Must never be represented as active emotion classification.

### 2.5 Optical Motion and Facial Aspect Ratios (Channels 18, 19–20)
- **Optical Motion (Channel 18):** Inter-frame pixel difference `cv2.absdiff(crop, prev_crop)` normalized by 255.0 into [0.0, 1.0].
- **EAR & MAR (Channels 19–20):** Fixed fallback 0.3000 due to absence of OpenSeeFace library.
