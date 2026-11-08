# Forensic Diagnostic Report: Investigation of Candidate Zero-Recall on Public Test Data and Bootstrap CI Methodology

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis in Smart Classrooms.  
**Auditor:** Principal Computer Vision Researcher, ML Systems Architect & Reproducibility Auditor  
**Date:** 2026-10-23  
**Status:** **INVESTIGATION COMPLETED (EVIDENCE-BASED FORENSIC AUDIT)**  

---

## 1. Executive Summary

During the Phase 8 controlled evaluation on the held-out public DAiSEE test split (3,563 sequences across subjects `110001` and `110004`), an apparent paradox emerged:
1. The **Active Production Model** ([`classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt)) achieved **50.00% recall** on Class 0 (Non-Attentive) at default $\tau=0.50$ (identifying 35/70 sequences) and **59.67% Balanced Accuracy** at calibrated $\tau^* = 0.10$.
2. In contrast, the newly trained **Phase 8 Candidate Models** (both unweighted and class-weighted cross-entropy with $w_0 = 26.18$) achieved **0.00% recall** on Class 0 (0/70 sequences identified), predicting $P(\text{HIGH}) \ge 0.96$ on all 70 test sequences.

In accordance with the Master Directive, this diagnostic review was conducted using **Train and Validation partitions only** for diagnostic modeling. The historical test metrics remain frozen and unperturbed.

---

## 2. Root-Cause Forensic Findings

### Finding 1: The "Single-Subject" Test Bottleneck
A dataset provenance audit of the held-out test split revealed that the 70 Class 0 sequences do not represent a diverse population of non-attentive students:
- Total sequences in Test: **3,563 sequences** across 148 physical clips.
- Total Class 0 sequences in Test: **70 sequences** (1.96% of the test set).
- Total Class 0 clips in Test: **Exactly 2 physical clips** (`1100042023` and `1100042026`).
- Total Class 0 subjects in Test: **Exactly 1 student** (Subject `110004`).
  - Subject `110001` has 114 clips $\to$ **100.0% Class 1** (0 Non-Attentive sequences).
  - Subject `110004` has 34 clips $\to$ **32 Class 1 clips, 2 Class 0 clips**.

**Conclusion:** The entire held-out public test evaluation of the Non-Attentive class depends exclusively on **two 10-second clips of a single individual**.

---

### Finding 2: The Physical & Visual Signature in Clips `1100042023` and `1100042026`
Inspection of the raw video metadata and extracted 32-D multimodal feature tensors revealed why the neural models predicted $P(\text{HIGH}) \approx 1.000$:

```
Clip 1100042023 (35 sequences):
  Raw DAiSEE Rater Scores: Engagement=1, Boredom=3, Confusion=1, Frustration=0
  Mean Head Pose (Yaw, Pitch, Roll): [0.13°, 0.21°, -0.01°]  (Head directly upright and facing screen)
  Mean Gaze Angles (Pitch, Yaw)    : [0.31°, 0.13°]           (Eyes fixated directly on monitor)
  Motion Energy                    : 0.0051                   (Seated stationary)
  Eye Aspect Ratio (EAR)           : 0.3000                   (Eyes fully open, no blinks or drowsy closure)

Clip 1100042026 (35 sequences):
  Raw DAiSEE Rater Scores: Engagement=1, Boredom=2, Confusion=1, Frustration=0
  Mean Head Pose (Yaw, Pitch, Roll): [0.13°, 0.18°, -0.06°]  (Head directly upright and facing screen)
  Mean Gaze Angles (Pitch, Yaw)    : [0.13°, 0.00°]           (Direct on-axis gaze)
  Motion Energy                    : 0.0019                   (Stationary)
  Eye Aspect Ratio (EAR)           : 0.3000                   (Eyes open)
```

**Scientific Mechanism:**
In both clips, the student is **staring directly at the webcam with open eyes, an upright head, and zero physical restlessness**. The external annotator labeled this student as "bored=3, engagement=1". However, because:
1. Facial Emotion Expression (FER) weights are currently missing (yielding an uninformative uniform prior $[1/7, \dots, 1/7]$ across channels 5..11);
2. 98.09% of all public training samples with frontal head pose and direct gaze belong to Class 1 (`ATTENTIVE`);
Any statistical neural classifier operating on geometric head pose, gaze vectors, and motion **must conclude that a student staring directly at the screen with open eyes is attentive**. Without subtle facial micro-expression features, a "bored student staring at the screen" and an "attentive student staring at the screen" have identical geometric feature representations.

---

### Finding 3: Cross-Subject Feature Inversion in Public Webcam Video
Analysis of feature discriminability across subjects in Train vs. Validation demonstrated severe cross-subject distribution shift:

| Feature Channel | Train Cohen's $d$ (Effect Size) | Validation Cohen's $d$ (Effect Size) | Phenomenon |
| :--- | :---: | :---: | :--- |
| **`Head_Roll` (Channel 2)** | **$-1.1818$** | **$+0.5466$** | **Complete Inversion across Splits** |
| **`Head_Pitch` (Channel 1)** | **$-0.8465$** | **$-0.2401$** | Strong Attenuation across Splits |
| **`Head_Yaw` (Channel 0)** | **$-0.3722$** | **$+0.3260$** | **Sign Inversion across Splits** |
| **`Post_5_Lean` (Channel 17)** | **$-0.4592$** | **$+0.2200$** | Inversion across Splits |

**Interpretation:**
Because public DAiSEE consists of students seated at arbitrary desk heights with differing laptop camera angles, individual postural idiosyncrasies dominate. A student tilting their head left in Subject A was annotated as disengaged, while in Subject B, a rightward tilt was annotated as engaged. Neural models fit to training subject idiosyncrasies fail to transfer these geometric rules to unseen test subjects.

---

### Finding 4: Deconstruction of the Production Model's "50% Recall"
When the predictions of [`classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt) were audited clip-by-clip on the 70 test sequences, a binary split was discovered:

- **Clip `1100042023` (35 seqs):** Mean $P(\text{HIGH}) = 0.9996 \implies$ **0 / 35 identified as Non-Attentive (0% recall)**.
- **Clip `1100042026` (35 seqs):** Mean $P(\text{HIGH}) = 0.0296 \implies$ **35 / 35 identified as Non-Attentive (100% recall)**.

**Why did Clip `1100042026` drop to $P(\text{HIGH}) = 0.03$?**
In Clip `1100042026`, the upper-body keypoint detector experienced low confidence, causing Channel 23 (`Keypoint_Conf`) to drop. The production baseline—which was calibrated with conservative classroom safety rubrics that heavily penalize unconfident or obscured upper-body postures—dropped its continuous score to $2.96$. In contrast, for Clip `1100042023` where keypoints were confident, the production model also predicted $P(\text{HIGH}) = 0.9996$.
Thus, the production model's 50% recall was literally **1 out of 2 clips**, triggered by keypoint confidence rather than learned semantic distinction.

---

## 3. Audit of Bootstrap Confidence Interval Methodology

The Phase 7/8 evaluation protocol used cluster-robust bootstrap resampling over physical video clips ($N=148$ resamples with replacement, 1,000 iterations).

### Methodological Soundness:
1. **Window-Level Clustering:** Because 16-frame sliding windows overlap by 50% within a clip, sequences within the same clip are statistically dependent. Resampling clips rather than individual sequences correctly prevents artificial deflation of uncertainty estimates.

### Critical Limitations Exposed by Finite-Sample Cluster Deficiency:
1. **$K=2$ Minority Cluster Bottleneck:** Because the test set contains only 2 clips belonging to Class 0, the bootstrap resampling distribution for Class 0 recall is governed by:
   $$\text{Recall}_{\text{bootstrap}} = \frac{35 \cdot k_{26}}{35 \cdot k_{23} + 35 \cdot k_{26}} = \frac{k_{26}}{k_{23} + k_{26}}$$
   where $k_{23}$ and $k_{26}$ are the resampled frequencies of clips `1100042023` and `1100042026`.
2. **Poisson Ratio Artifact:** In any resample where $k_{26} > 0$ and $k_{23} = 0$, recall is 100%. In any resample where $k_{26} = 0$ and $k_{23} > 0$, recall is 0%.
3. **Misleading CI Width:** The wide 95% confidence interval reported for Balanced Accuracy (`[21.26%, 76.99%]`) does **not** reflect population variance across diverse classrooms. It is entirely a mathematical artifact of the discrete binomial sampling distribution between two clips from a single student.

---

## 4. Synthesis & Recommendations

1. **Class Imbalance vs. Visual Indistinguishability:** While severe class imbalance (~98%:2%) causes gradients to favor Class 1, the deeper cause of candidate zero-recall is **visual indistinguishability**: in public webcam video, a bored student looking at the screen exhibits identical geometric head and eye poses to an attentive student.
2. **Path Forward for Classroom Deployment:** Genuine attention classification in smart classrooms must not rely exclusively on webcam facial geometry. It must leverage:
   - Multi-person spatial context (e.g., student gaze relative to teacher position and whiteboard);
   - Student action cues from YOLO/SCB (hand-raising, writing in notebook, reading, phone use);
   - Peer group synchrony (whether the student is oriented toward the same classroom focal point as peers).
