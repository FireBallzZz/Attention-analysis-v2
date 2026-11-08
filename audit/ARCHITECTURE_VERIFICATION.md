# Model Architecture & Configuration Verification

**Audit Date:** 2026-11-08  
**Champion Checkpoint:** `checkpoints/classroom_manual_binary_unquarantined.pt`  
**Pipeline Orchestrator:** `src/attention_pipeline/pipeline.py`  
**Configuration File:** `configs/pipeline.yaml`  

---

## 1. Detection and Multi-Object Tracking Pipeline

### 1.1 YOLO11-Pose Detector Configuration
- **Model Variant:** `yolo11n-pose.pt` (Ultralytics YOLO11 Nano-Pose)
- **Input Resolution:** $1920 \times 1080$ pixels (`imgsz: 1920`)
- **Confidence Threshold:** `conf: 0.15`
- **NMS IoU Threshold:** `iou: 0.45`
- **Extracted Keypoints:** 17 COCO human body landmarks (including nose, eyes, ears, shoulders, elbows, wrists, hips)
- **Execution Rate:** Downsampled to 10 FPS (`fps: 10`) for optimal throughput-latency trade-off

### 1.2 Multi-Student Tracking & Box Smoothing
- **Tracker Algorithm:** ByteTrack with custom spatial and temporal enhancements
- **Track Persistence:** `max_age: 45` frames (4.5 seconds at 10 FPS); `n_init: 1`
- **IoU Association Threshold:** `max_iou_distance: 0.7`
- **Temporal Box Smoothing:** Exponential Moving Average (EMA) smoothing enabled (`smooth_box: true`, `box_ema_alpha: 0.7`) to eliminate jitter and flickering:
  $$\text{box}_t = 0.7 \cdot \text{box}_{\text{det}} + 0.3 \cdot \text{box}_{t-1}$$
- **Area Expansion Damping:** `max_area_ratio: 1.5` prevents momentary false bounding-box spikes.
- **Capacity Constraint:** `auto_max_students: false`. Previously, dynamic auto-calibration capped class tracking at 23 students based on the initial 30 frames. Disabling this allows tracking full multi-row lecture halls (30–50+ students).

### 1.3 Tier-Aware Duplicate Suppression & Occlusion Handling
In multi-row classroom settings, students in front rows occlude the desks and lower bodies of students in the middle and back rows. Standard Non-Maximum Suppression (NMS) or naive box distance checks frequently suppressed rear-row heads when overlapping with front-row shoulders.
- **Fix Implemented:** Vertical head-separation filtering in `src/attention_pipeline/stages/detect_track.py`:
  If two candidate detections exhibit significant vertical head separation ($\Delta y_{\text{head}} \ge 18$ px) and center distance ($\text{dist} \ge 35$ px), they are confirmed as distinct students in consecutive tiers and are preserved.

---

## 2. 32-Dimensional Multimodal Feature Vector Verification

The multimodal feature extractor in `src/attention_pipeline/features.py` compiles a 32-dimensional feature vector per student per frame (`FEAT_DIM = 32`).

### 2.1 Complete 32-Dimensional Feature Representation Table

| Feature Component | Dims | Slice Index | Extraction Method / Source Model | Normalization Range | Active Implementation Status |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Head Pose (Yaw, Pitch, Roll)** | 3 | `[0:3]` | OpenSeeFace / Facial Landmark PnP | $[-1.0, 1.0]$ (divided by $180^\circ$) | **Active** |
| **Eye Gaze (Yaw, Pitch)** | 2 | `[3:5]` | L2CS-Net ResNet-50 (`checkpoints/l2cs_finetuned.pt`) | $[-1.0, 1.0]$ (divided by $180^\circ$) | **Active** |
| **Facial Emotion Distribution** | 7 | `[5:12]` | 7 emotion classes (angry, disgust, fear, happy, sad, surprise, neutral) | $[0.0, 1.0]$ | **Placeholder Neutral** (`[0,0,0,0,0,0,1]`; FER2013/RAF-DB excluded) |
| **Classroom Behavior (SCB-6)** | 6 | `[12:18]` | Fine-tuned YOLOv8 (`checkpoints/scb_yolo.pt`) | $[0.0, 1.0]$ (Softmax probabilities) | **Active (Verified SCB-6)** |
| **Inter-Frame Optical Motion** | 1 | `[18]` | Bounding-box displacement and keypoint flow | $[0.0, 1.0]$ (Clipped) | **Active** |
| **Eye Aspect Ratio (EAR)** | 1 | `[19]` | Eye landmark aspect ratio for blink/fatigue | $[0.0, 1.0]$ (Clipped) | **Active** |
| **Mouth Aspect Ratio (MAR)** | 1 | `[20]` | Lip landmark aspect ratio for yawning/talking | $[0.0, 1.0]$ (Clipped) | **Active** |
| **Facing-Front Score** | 1 | `[21]` | Angular deflection: $\max(0, 1.0 - \|\text{yaw}\| / 45^\circ)$ | $[0.0, 1.0]$ | **Active** |
| **Torso Lean Angle** | 1 | `[22]` | Shoulder-to-hip angle; fallback to inter-shoulder tilt | $[0.0, 1.0]$ ($0 = \text{upright}, 1 = \text{leaned}$) | **Active (Confidence-gated)** |
| **Upper-Body Keypoint Conf.** | 1 | `[23]` | Mean confidence of nose, eyes, and shoulders | $[0.0, 1.0]$ | **Active** |
| **Reserved Zero-Padding** | 8 | `[24:32]` | Fixed zeros reserved for future kinematics | Fixed $0.0$ | **Inactive Padding** (Maintains 32-D checkpoint compatibility) |
| **TOTAL DIMENSIONS** | **32** | — | — | — | **24 Active / 8 Padded** |

### 2.2 Critical Architecture Audit Findings
1. **SCB-6 vs SCB-5 Resolution:** The behavior model is confirmed to be **SCB-6**, outputting 6 behavioral categories:
   1. Hand-raising
   2. Reading
   3. Writing
   4. Using-phone
   5. Bowing-head
   6. Leaning-over-table  
   *Correction:* The dataset folder was historically labeled `SCB5-Handrise-Read-write-2024-9-17`, but the trained model in `checkpoints/scb_yolo.pt` outputs 6 logits.
2. **Emotion Modality:** The 7 emotion dimensions `[5:12]` are populated with a static neutral one-hot vector (`[0, 0, 0, 0, 0, 0, 1.0]`). Standalone facial emotion networks (FER2013 / RAF-DB) are explicitly disabled in `configs/pipeline.yaml` (lines 30) due to severe facial occlusion in multi-student classrooms.
3. **8-Dimensional Padding:** Channels 24 to 31 are hardcoded to $0.0$. This ensures exact tensor dimension compatibility with the pre-trained DAiSEE fusion backbone (`fusion_daisee.pt`).

---

## 3. Temporal Neural Architecture (BiGRU + Multi-Pooling)

### 3.1 Network Topology
The temporal attention classification model (`ManualBinaryAttentionClassifier` in `src/attention_pipeline/models/manual_binary.py`) processes sliding windows of 16 consecutive frames ($T = 16$, representing 1.6 seconds at 10 FPS):

```
Input: Tensor (Batch, T=16, Feat_Dim=32)
  │
  ▼
Frame Encoder (Linear 32 → 64, ReLU, LayerNorm 64, Linear 64 → 64, ReLU) [FROZEN]
  │
  ▼
Bidirectional GRU (Input 64, Hidden 64, Layers 1, Bidirectional=True)
  │
  ├── Last Hidden State: (Batch, 128)
  ├── Temporal Mean-Pool: (Batch, 128)
  └── Temporal Max-Pool:  (Batch, 128)
  │
  ▼
Multi-Pooled Representation: Concat(Last, Mean, Max) = 384 Dimensions
  │
  ▼
Classification Head:
  Linear(384 → 128) → BatchNorm1d(128) → GELU() → Dropout(0.2)
  Linear(128 → 64)  → GELU() → Dropout(0.1)
  Linear(64 → 2)    → Softmax()
  │
  ▼
Output Probabilities: [P(Class 0: LOW), P(Class 1: HIGH)]
```

### 3.2 Output Semantics & Thresholding
- **Class 0:** Disengaged / Non-Attentive (LOW)
- **Class 1:** Engaged / Attentive (HIGH)
- **Decision Threshold:** $\tau^* = 0.50$. Student is classified as HIGH if $P(\text{Class 1}) \ge \tau^*$.
- **Continuous Score:** $\text{Score} = P(\text{Class 1}) \times 100.0 \in [0.0, 100.0]$.

---

## 4. Action-Aware Hybrid Neural / Rule-Based Scoring Formula

In `src/attention_pipeline/pipeline.py` (lines 206–227), the raw temporal neural score is blended with behavioral priors to account for classroom desk work:

```python
if fusion_score is not None:
    h_score = float(rubric.score)
    cue_str = str(rubric.cue).lower()
    is_active_study = (
        h_score >= 70.0
        and any(w in cue_str for w in ("reading", "writing", "hand"))
        and not any(w in cue_str for w in ("phone", "sleep", "table", "left", "right"))
    )
    if is_active_study:
        # Blend temporal neural score with verified active study action prior
        effective_score = float(0.4 * fusion_score + 0.6 * h_score)
    elif h_score <= 35.0 and any(w in cue_str for w in ("phone", "sleep", "table")):
        # Grounded disengagement penalty
        effective_score = min(float(fusion_score), h_score)
    else:
        effective_score = float(fusion_score)

    band = "HIGH" if (effective_score / 100.0) >= self.binary_threshold else "LOW"
```

### 4.1 Audit Analysis of the Hybrid Blending Mechanism
1. **Mathematical Blending Formula:**
   $$\text{Final Score} = 0.4 \times \text{Score}_{\text{BiGRU}} + 0.6 \times \text{Score}_{\text{Rubric}}$$
   This formula is applied **conditionally**, strictly when `is_active_study` is satisfied ($\text{Score}_{\text{Rubric}} \ge 70.0$ and cues match reading, writing, or hand-raising without distraction cues).
2. **Impact on Classification:**
   During active reading or writing, students tilt their heads downward ($\text{pitch} \le -20^\circ$). A pure gaze/head-pose neural classifier would penalize this downward pitch as inattention. The $0.6$ rubric weight elevates the score above the $50.0$ threshold (e.g., $0.4 \times 35.0 + 0.6 \times 85.0 = 65.0 \ge 50.0$), successfully preventing false-negative disengagement classifications.
3. **Weight Derivation Limitation:**
   The $0.4 / 0.6$ blending weights are **heuristically defined domain priors**, not parameters learned end-to-end via gradient descent.
4. **Formal Classification:**
   The v2.4 production pipeline should be scientifically characterized as a **Hybrid Neural / Rule-Assisted Attention Scoring Architecture**, rather than an end-to-end differentiable neural network.

