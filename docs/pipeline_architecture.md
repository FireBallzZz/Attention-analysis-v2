# Classroom Attention Analysis Pipeline Architecture

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms.  
**Version:** Rebuilt v4.0 (Phase 8 Public Baseline & Production Hardening)  

---

## 1. System Overview

The pipeline extracts multimodal student behavioral cues from classroom or webcam video, aligns them into normalized 32-D feature vectors at 5.0 FPS, pools them into 16-frame sliding windows (representing 3.2-second temporal contexts with 50% overlap), and classifies student attention state using a temporal bidirectional recurrent neural network.

```mermaid
flowchart TD
    Video[Classroom / Public Video Stream] --> Detection[Stage 1: YOLO Person Detection & ByteTrack Tracking]
    Detection --> Crops[Student Spatial Crops & Temporal Trajectories]
    
    Crops --> Face[Face & Gaze Modalities: L2CS-Net + OpenSeeFace]
    Crops --> Pose[Body Posture & Keypoints: YOLO11-Pose]
    Crops --> Emotion[Facial Emotion Expression Prior: 7-D Uniform / FER]
    Crops --> Action[Student Action Detection: SCB Behavior Detector]
    
    Face --> Packing[32-D Multimodal Feature Packing]
    Pose --> Packing
    Emotion --> Packing
    Action --> Packing
    
    Packing --> SlidingWindow[16-Step Temporal Sliding Window (3.2s, 50% stride)]
    SlidingWindow --> TemporalModel[Stage 4: AttentionFusionNet / ManualBinaryAttentionClassifier]
    
    TemporalModel --> Score[Continuous Attention Score: 0.0 - 100.0]
    TemporalModel --> Binary[Binary Attention Classification: 0: NON-ATTENTIVE, 1: ATTENTIVE]
    
    Score --> DropDetector[Stage 5: Attention Drop Detection & Trigger]
    DropDetector --> LLM[LLM Pedagogical Recommendation Engine]
    Score --> Overlay[Stage 6: Real-Time Bounding Box & Video Overlay]
```

---

## 2. Pipeline Stages

### Stage 1: Person Detection & Tracking
- **Detector:** `yolo11n-pose.pt` (tiled multi-scale detection, confidence threshold: 0.08, IoU threshold: 0.45).
- **Tracker:** ByteTrack with exponential moving average (EMA) bounding box stabilization ($\alpha = 0.70$) and maximum area-expansion ratio damping (1.5x) to eliminate detection jitter across frames.
- **Student Capacity:** Dynamic auto-calibration based on persistent stable tracks.

### Stage 2: Multimodal Extraction
- **Gaze Direction & Eye State:** Pretrained L2CS-Net (ResNet-50 backbone) returning pitch and yaw gaze angles in degrees. Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) computed from facial landmark geometry.
- **Head Pose:** 3-axis continuous rotation (yaw, pitch, roll in degrees) with frontalization reference.
- **Body Posture & Upper-Body Keypoints:** 17 COCO keypoints with normalized coordinates, confidence thresholds, and upper-body keypoint validity weighting.
- **Classroom Behaviors:** SCB-dataset detector covering student actions (reading, writing, hand-raising, phone use, leaning, head bowing).

### Stage 3: Feature Packing Contract (32-D Schema)
All modalities are packed into a rigid, schema-validated 32-dimensional feature vector:
- Channels `0..2`: Head Pose (Yaw, Pitch, Roll in degrees).
- Channels `3..4`: Gaze Angles (Pitch, Yaw in degrees).
- Channels `5..11`: Facial Emotion Probabilities (7-class prior).
- Channels `12..17`: Posture Probabilities (6-class distribution).
- Channel `18`: Normalized Inter-frame Motion Energy $[0, 1]$.
- Channel `19`: Eye Aspect Ratio (EAR).
- Channel `20`: Mouth Aspect Ratio (MAR).
- Channels `21..22`: Front-Facing Gaze and Face Alignment Indicators.
- Channel `23`: Upper-Body Keypoint Confidence Weight $[0, 1]$.
- Channels `24..31`: Reserved Schema-Preserved Channels (Fixed Zero-Padded).

### Stage 4: Temporal Fusion Architecture
- **Backbone (`AttentionFusionNet`):**
  - Linear frame projection layer: $32 \to 64$ with LayerNorm and ReLU.
  - Bidirectional GRU layer: 1 layer, hidden size 64 (bidirectional output: 128 dimensions).
- **Pooling & Head (`ManualBinaryAttentionClassifier`):**
  - Temporal pooling: Concatenation of [Last-Step (128) + Mean-Pool (128) + Max-Pool (128)] = 384 dimensions.
  - Multi-Layer Perceptron (MLP):
    - Linear(384 $\to$ 128) + BatchNorm1d + GELU + Dropout(0.20)
    - Linear(128 $\to$ 64) + GELU + Dropout(0.10)
    - Linear(64 $\to$ 2 logits)
- **Output Classes:**
  - Class Index `0` = `NON-ATTENTIVE` (DAiSEE engagement levels 0–1).
  - Class Index `1` = `ATTENTIVE` (DAiSEE engagement levels 2–3).
  - Attention Score: $P(\text{ATTENTIVE}) \times 100.0$.
  - Temporal Warmup: The model requires 16 consecutive frames (3.2 seconds). During frames 1–15, the classifier safely reports `WARMUP` with score `None`, preventing fabricated predictions.

### Stage 5: Attention Drop Detection & LLM Recommendations
- Detects sustained low attention ($< 40.0$ for $\ge 3.0$ seconds) or sudden drops ($\Delta \ge 20.0$ drop over 2.0 seconds).
- Triggers contextual pedagogical interventions via LLM (GPT-4o-mini or local model).

### Stage 6: Annotated Video Visualization
- Real-time overlay bounding boxes:
  - **GREEN**: High Attention ($P(\text{HIGH}) \ge \tau$)
  - **RED**: Low Attention / Distraction ($P(\text{HIGH}) < \tau$)
  - **GRAY**: Temporal Warmup (insufficient history)
- Displays student track ID, attention status, behavioral cue, and attention progress bar.
