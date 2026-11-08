# Claim Verification & Scientific Classification Matrix

**Audit Date:** 2026-11-08  
**Auditor:** Senior ML Research Engineer, Computer Vision Specialist & Academic Thesis Reviewer  
**Audited Document:** `Student_Attention_Model_Supervisor_Report.pdf` & Thesis Project Claims  

---

## 1. Classification Taxonomy

Each claim is assigned one of four rigorous academic classifications:
- **Class A (Empirically Verified):** Directly verified from source code, active model checkpoints, real dataset arrays, and reproducible execution.
- **Class B (Derived / Reconciled):** Computationally reproducible, but required forensic reconciliation due to typographic errors, hardcoded template labels, or mislabeled metrics (e.g., Positive-Class F1 mislabeled as Macro-F1).
- **Class C (Unverifiable / Overstated):** Metric or claim lacks ground-truth provenance or makes assertions unsupported by empirical evidence (e.g., claiming supervised CVAT accuracy on an unannotated video).
- **Class D (Disproven / Invalid):** Factually refuted by code inspection or empirical calculation (e.g., claiming active facial emotion modeling when it is stubbed to neutral).

---

## 2. Claim-by-Claim Verification Matrix

| Claim ID | Audited Statement in Previous Report | Empirical Finding | Classification | Auditor Verdict & Correction |
| :---: | :--- | :--- | :---: | :--- |
| **CLM-01** | *Clip 001 achieves 95.13% Accuracy on Human Ground Truth (N=1,067).* | Exactly reproduced at $\text{IoU} \ge 0.30$ against `runs/Annoted file for clip 1/annotations.xml`. $\text{TN}=0, \text{FP}=51, \text{FN}=1, \text{TP}=1015$. | **Class A** | **VERIFIED.** Exact match. High accuracy is driven by 99.9% High recall on an attentive-dominated classroom. |
| **CLM-02** | *Clip 001 achieves 97.50% Macro-F1 Score.* | Mislabeled metric. $97.50\%$ is the **Positive-Class F1** ($\text{F1}_{\text{HIGH}}$). True $\text{F1}_{\text{LOW}} = 0.0\%$. True Macro-F1 is $\mathbf{48.75\%}$. | **Class B** | **RECONCILED.** Positive-class F1 was misreported as Macro-F1 due to class imbalance ($\text{TN}=0$). Balanced Acc is $49.95\%$. |
| **CLM-03** | *Clip 007 achieves 80.65% Accuracy and 89.25% Macro-F1 (N=863).* | Accuracy (80.65%) verified against `Anno/ground_truth.csv` ($\text{TN}=3, \text{FP}=165, \text{FN}=2, \text{TP}=693$). However, $89.25\%$ is $\text{F1}_{\text{HIGH}}$. True Macro-F1 is $\mathbf{46.36\%}$. | **Class B** | **RECONCILED.** Accuracy is verified. Macro-F1 was mislabeled; true Macro-F1 is $46.36\%$, Balanced Accuracy is $50.75\%$. |
| **CLM-04** | *Student-Disjoint Validation achieves 87.22% Accuracy on 4,386 sequences (TN=290, FP=332, FN=256, TP=3724).* | Arithmetic discrepancy: $290 + 332 + 256 + 3724 = \mathbf{4,602}$, NOT 4,386. The $N=4,386$ was a hardcoded title in an older script template. | **Class B** | **RECONCILED.** The true sample size is $N = 4,602$ across 6 held-out students. Active champion model achieves $\mathbf{87.74\%}$ Accuracy and $\mathbf{72.06\%}$ True Macro-F1 ($\text{TN}=295, \text{FP}=327, \text{FN}=237, \text{TP}=3743$). |
| **CLM-05** | *Video Project 3 evaluates 39 students with 90.22% Accuracy against Ground Truth (N=818, TN=65, FP=24, FN=56, TP=673).* | **NO HUMAN GROUND-TRUTH EXISTS FOR VIDEO PROJECT 3.** The evaluation was an internal agreement check against an automated heuristic rubric on a partial run. Full video tracks 52 students. | **Class C** | **OVERSTATED.** Must not be reported as a supervised CVAT benchmark. It is an unsupervised consistency stress test ($88.35\%$ to $90.22\%$ agreement with heuristic rubric). |
| **CLM-06** | *The model uses a 32-D feature vector integrating Facial Emotion (FER2013 / RAF-DB).* | In `src/attention_pipeline/features.py`, channels `[5:12]` are populated with a static neutral one-hot vector (`[0,0,0,0,0,0,1]`). FER2013 and RAF-DB are explicitly disabled. | **Class D** | **DISPROVEN.** Standalone emotion CNNs are not active. The 32-D layout contains 24 active multimodal dimensions and 8 reserved zero-padded dimensions. |
| **CLM-07** | *The behavior module detects SCB-5 categories.* | The trained model in `checkpoints/scb_yolo.pt` outputs 6 distinct classroom behaviors (hand-raising, reading, writing, using-phone, bowing-head, leaning-over-table). | **Class B** | **RECONCILED.** The system actually uses **SCB-6**, not SCB-5. |
| **CLM-08** | *Action-Aware Scoring blends BiGRU (0.4) and Rubric (0.6).* | Verified in `src/attention_pipeline/pipeline.py` (lines 206–227). Blending is applied conditionally during active desk study. | **Class A** | **VERIFIED.** Implemented exactly as stated. The weights (0.4/0.6) are heuristic domain priors, characterizing the system as a hybrid neural/rule-assisted model. |
| **CLM-09** | *Real-Time Inference operates at ~30 FPS on RTX 3060.* | Profiled on local RTX 3060: YOLO11-Pose (20.83 ms) + BiGRU (0.47 ms) + Decode/Overlay (11.36 ms) = 32.66 ms / frame ($30.6\text{ FPS}$). | **Class A** | **VERIFIED.** Real-time capability on 1080p video is mathematically and empirically confirmed. |
| **CLM-10** | *IoU sensitivity improves from 93.69% to 95.74% due to tracking robustness.* | Numerically verified, but the cause was misattributed. The increase from 93.69% ($N=1331$) to 95.74% ($N=868$) is caused by **selection bias** discarding occluded/hard instances. | **Class B** | **RECONCILED.** Performance is reproducible, but must be scientifically documented as selection bias rather than intrinsic tracker robustness. |

---

## 3. Discrepancy Reconciliation Summary

### Discrepancy 1: Sample Size $N = 4,386$ vs Confusion Matrix Sum $= 4,602$
- **Original Claim:** The validation summary stated $N = 4,386$, while its confusion matrix summed to $290 + 332 + 256 + 3724 = 4,602$.
- **Audit Discovery:** In `scripts/train_classroom_binary_unquarantined.py`, line 358 had a hardcoded markdown header template `## 1. Validation Performance Summary (4,386 Held-Out Sequences)` left over from an earlier 2-clip experiment. The author of the supervisor report copied "4,386" from this header text, while copying the actual 4,602 confusion matrix entries from the JSON results.
- **Audited Correct Value:** $N = 4,602$ validation sequences ($622$ Disengaged, $3,980$ Engaged). Under the active tuned checkpoint, the confusion matrix is $\text{TN}=295, \text{FP}=327, \text{FN}=237, \text{TP}=3743$, yielding **87.74% Accuracy** and **72.06% True Macro-F1**.

### Discrepancy 2: Positive-Class F1 Misreported as Macro-F1
- **Original Claim:** Macro-F1 was reported as $97.50\%$ (Clip 001) and $89.25\%$ (Clip 007).
- **Audit Discovery:** In `scripts/analyze_clip_001.py` (lines 121–122) and `run_production_pipeline.py` (line 250), the variable `f1 = 2 * prec * rec / (prec + rec)` was computed solely using positive-class precision and recall ($\text{TP} / (\text{TP} + \text{FP})$ and $\text{TP} / (\text{TP} + \text{FN})$), and then output under the key `"macro_f1"`.
- **Audited Correct Value:** True Macro-F1 is $\frac{\text{F1}_{\text{LOW}} + \text{F1}_{\text{HIGH}}}{2}$:
  - Clip 001: True Macro-F1 = $\mathbf{48.75\%}$ ($\text{F1}_{\text{LOW}} = 0.0\%, \text{F1}_{\text{HIGH}} = 97.50\%$).
  - Clip 007: True Macro-F1 = $\mathbf{46.36\%}$ ($\text{F1}_{\text{LOW}} = 3.47\%, \text{F1}_{\text{HIGH}} = 89.25\%$).
  - Held-Out Validation: True Macro-F1 = $\mathbf{72.06\%}$ ($\text{F1}_{\text{LOW}} = 51.13\%, \text{F1}_{\text{HIGH}} = 92.99\%$).

### Discrepancy 3: Video Project 3 Ground Truth Status
- **Original Claim:** Presented in the supervisor report table alongside Clip 001 and Clip 007 as an evaluated benchmark with $N=818, \text{TN}=65, \text{FP}=24, \text{FN}=56, \text{TP}=673$.
- **Audit Discovery:** No CVAT XML or CSV ground-truth files exist for `Video Project 3.mp4`. The $818$ comparisons were evaluated against an internal heuristic rule rubric (`video_accuracy.py`).
- **Audited Correct Value:** Video Project 3 must be clearly labeled as an **Unsupervised Consistency Stress Test** (Heuristic Agreement: $88.35\%$ to $90.22\%$, 52 detected tracks), NOT a supervised human ground-truth benchmark.

