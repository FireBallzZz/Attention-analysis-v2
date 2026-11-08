# Phase 0: Repository Inventory & Baseline Audit Report

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Audit Timestamp:** 2026-10-23T15:19:00+06:00  
**Audit Phase:** Phase 0 — Baseline & Inventory  

---

## 1. Executive Summary

This document establishes the verified, non-destructive baseline of the thesis attention-analysis repository prior to any rebuild actions. Every listed checkpoint, script, dataset, dependency, and test result has been directly verified on local hardware.

### Key Invariants Established:
1. **Cryptographic Immutability:** All protected model checkpoints (classroom_manual_binary_model_v2_clean.pt, phase6g4c_engaged_disengaged_candidate.pt, phase7_public_trained_binary_candidate.pt) match their pre-rebuild SHA-256 hashes bit-for-bit.
2. **Dataset Safety & Isolation:** The unauthorized video PRE DEFENCE/YOLO Test.mp4 is strictly quarantined and unaccessed. Physical classroom recordings (UAP Clips 001, 004, 008) remain strictly isolated for final frozen evaluation.
3. **Verified Regression Test Suite:** The entire unit test suite of 67 collected tests executes in 4.84 seconds with **66 passed, 1 skipped, 0 failed**.
4. **Binary Label Contract:** Class 0 = NON-ATTENTIVE (DAiSEE engagement 0–1), Class 1 = ATTENTIVE (DAiSEE engagement 2–3).

---

## 2. Hardware and Environment Specification

| Component | Verified Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Pro (10.0.26200-SP0) |
| **Python Environment** | Python 3.14.4 (64-bit AMD64) |
| **PyTorch Version** | 2.14.1+cu126 |
| **CUDA Driver / Runtime** | CUDA 12.6, 	orch.cuda.is_available() == True |
| **Dedicated GPU** | NVIDIA GeForce RTX 3060 (12.0 GB VRAM) |
| **System Memory** | 15.2 GB Total RAM (4.1 GB Available) |
| **Disk Storage** | Drive C:\: 930.4 GB Total, 400.3 GB Free Space |
| **OpenCV** | 5.0.0 |
| **Ultralytics YOLO** | 8.4.182 |

---

## 3. Protected Asset Cryptographic Inventory

| Checkpoint Path | Expected SHA-256 Hash | Observed Pre-Rebuild SHA-256 Hash | Status |
| :--- | :---: | :---: | :---: |
| checkpoints/classroom_manual_binary_model_v2_clean.pt | 2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E | 2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E | **VERIFIED (Bit-for-Bit)** |
| checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt | E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E | E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E | **VERIFIED (Bit-for-Bit)** |
| checkpoints/candidates/phase7_public_trained_binary_candidate.pt | 1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5 | 1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5 | **VERIFIED (Bit-for-Bit)** |

---

## 4. End-to-End Data Flow Architecture

`
[Raw Video Source: MP4/AVI @ 5 FPS]
               │
               ▼
   [Stage 1: YOLO11n-Pose & Person Detection]
     - Detections: Bounding Boxes (x1, y1, x2, y2)
     - Keypoints: 17 COCO Anatomy Points (x, y, conf)
               │
               ▼
   [Stage 2: Multimodal Feature Assembly (32-D Vector)]
     - Channels 0-2   : Head Pose (Yaw, Pitch, Roll) via arctan of eye/ear/nose keypoints
     - Channels 3-4   : Eye Gaze (Yaw, Pitch) via L2CS-Net ResNet-50 face crop
     - Channels 5-11  : Emotion Probabilities (7-D neutral prior: [0,0,0,0,0,0,1])
     - Channels 12-17 : Posture/Behavior (SCB-YOLO blended with geometric keypoint rules)
     - Channel 18     : Optical Motion (Normalized inter-frame absdiff)
     - Channels 19-20 : EAR / MAR (Fixed fallback prior 0.3000)
     - Channel 21     : Facing-Front Score (1.0 - |yaw| / 45.0)
     - Channel 22     : Torso Lean (Confidence-gated angle with inter-shoulder tilt fallback)
     - Channel 23     : Upper-Body Keypoint Confidence (Mean confidence of 5 facial/shoulder points)
     - Channels 24-31 : Reserved Padding (0.0000)
               │
               ▼
   [Stage 3: Sliding Temporal Windowing]
     - Window size T = 16 steps (3.2 seconds at 5 FPS)
     - Shape per track: (16, 32) float32
               │
               ▼
   [Stage 4: Binary Attention Classification]
     - Backbone: AttentionFusionNet (Linear 32->64, 2-layer BiGRU 64->128)
     - Pooling: Concat [Last (128) + Mean (128) + Max (128)] = 384-D
     - Head: MLP (384 -> 128 -> 64 -> 2)
     - Output: Softmax Probabilities [P(NON-ATTENTIVE), P(ATTENTIVE)]
     - Decision Rule: P(ATTENTIVE) >= threshold -> ATTENTIVE, else NON-ATTENTIVE
`

---

## 5. Verified Test Suite Execution

Execution Command: python -m pytest tests/ -v  
Summary: **66 passed, 1 skipped, 0 failed in 4.84s**
