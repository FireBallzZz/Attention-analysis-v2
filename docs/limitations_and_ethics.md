# Scientific Limitations, Risk Analysis, and Ethical Governance

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis in Smart Classrooms.  

---

## 1. Scientific & Methodological Limitations

### 1.1 Operational Classification Surrogate vs. Cognitive Internal State
- **Critical Caveat:** The binary classifier predicts an *operational surrogate target* based on observable behavioral cues (gaze angle, head pose, posture, and facial expression) calibrated against external rater engagement labels in public webcam datasets.
- **Cognitive Independence:** An observable behavioral state (e.g., staring towards the front of a classroom) does not guarantee active internal cognitive information processing or comprehension. Conversely, a student looking down while taking notes or reflecting may be highly attentive while exhibiting non-frontal head angles.
- **Pedagogical Recommendation:** Automated computer vision attention scores must never be treated as direct mental telepathy or definitive psychometric measurements. They should only serve as supportive, aggregate pedagogical indicators for instructors to assess overall lesson engagement.

### 1.2 Extreme Class Imbalance in Public DAiSEE Corpus
- **Observed Distribution:**
  - Public DAiSEE exhibits severe class imbalance: ~98.0% Attentive (Class 1) vs. ~2.0% Non-Attentive (Class 0).
  - In the held-out test split (3,563 sequences across subjects 110001 and 110004), only 70 sequences belong to Class 0.
- **Consequences:** Standard neural training with unweighted cross-entropy collapses toward predicting Class 1 almost exclusively.
- **Mitigation & Uncertainty:**
  - We use class-weighted loss functions ($w_0 = 26.18$, $w_1 = 0.51$).
  - We evaluate using Balanced Accuracy and Macro-F1 rather than raw accuracy.
  - We compute cluster-robust bootstrap confidence intervals (resampling physical video clips rather than overlapping sequences) to report realistic uncertainty.

### 1.3 Modality Readiness Limitations
- **Facial Emotion Expression (FER):** Pretrained weights for 7-class FER are currently missing. Channels 5..11 emit a schema-compatible uniform prior $[1/7, \dots, 1/7]$ with zero fabricated confidence.
- **SCB Posture Classification:** COCO-17 keypoint geometry is used with a geometric baseline; it is honestly documented as geometric heuristic rather than a learned classifier until labeled classroom posture sets are trained.

### 1.4 Domain Shift: Public Webcam vs. Real Classroom
- **Public DAiSEE:** Single student seated close to a webcam, frontal perspective, direct gaze at monitor.
- **Smart Classroom:** Multiple students, wide camera field-of-view, oblique perspective, significant physical occlusion, varying desk heights, peer interactions.

---

## 2. Ethical Guidelines & Data Privacy

1. **Classroom Video Access Control:** Classroom recordings must remain strictly local, encrypted, and access-controlled. No raw video footage may be transmitted to cloud APIs or public servers.
2. **LLM Anonymization:** Prompts sent to LLM pedagogical recommendation engines contain only aggregated behavioral summaries and anonymized track IDs (e.g., "Student 3 showed head-drop for 4 seconds"), with zero biometric or facial crop data transmitted.
3. **Non-Punitive Deployment:** Attention scores must never be used for punitive grading, behavioral disciplinary sanctions, or student surveillance. Their sole authorized purpose is pedagogical reflection and teaching feedback.
