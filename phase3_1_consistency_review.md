# Phase 3.1: Final Consistency and Evidence Audit

**Document Role:** Formal Review and Rigorous Evidence Reconciliation of Phase 3 Generalization Audit  
**Target Architecture:** Canonical `ManualBinaryAttentionClassifier`  
**Active Checkpoint:** [`checkpoints/classroom_manual_binary_model_v2_clean.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/classroom_manual_binary_model_v2_clean.pt) (SHA-256: `2bfa5af8e190b5d4...`)  
**Preserved Baseline Checkpoint:** [`checkpoints/baseline/classroom_manual_binary_model_v1.pt`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/checkpoints/baseline/classroom_manual_binary_model_v1.pt) (SHA-256: `52617140804aefb5...`)  
**Frozen Evaluation Source:** [`runs/4_manual_ground_truth_aligned.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/runs/4_manual_ground_truth_aligned.csv)  
**Date:** 2026-10-19  

---

## 1. Prior Observation Reconciliation: `runs/4.mp4`

### Issue Flagged:
Earlier drafts contained an apparent contradiction: stating that `runs/4.mp4` was "not observed during V2 selection" while also documenting that it was "observed during earlier V1 checkpoint selection".

### Exact Reconciliation:
1. **Historical V1 Development (Test Leakage Flaw):**  
   During baseline V1 development, every training epoch was evaluated against `runs/4.mp4` manual ground truth, and epoch 16 was selected explicitly because it attained the highest test accuracy ($70.44\%$) on `runs/4.mp4`. This was an undeniable test-set leakage in checkpoint selection.
2. **V2 Clean Development (Execution Isolation):**  
   In training `v2_clean` ([`scripts/train_manual_binary_model.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/train_manual_binary_model.py)), `runs/4.mp4` was strictly excluded from all training steps, loss computation, validation checks, epoch selection (epoch 24), and threshold calibration ($\tau^* = 0.40$).
3. **Epistemic Conclusion (Defensible Phrasing):**  
   While `runs/4.mp4` was not directly used to select `v2_clean` parameters, **it cannot be described as an untouched, virgin, or independent test set**. The multimodal feature extractors, spatial bounding box annotations, and baseline architecture in the repository were previously observed and developed with knowledge of `runs/4.mp4`.  
   **Thesis Mandate:** `runs/4.mp4` must be designated as a **previously observed benchmark recording**, and performance on it must be reported as a cross-recording benchmark, not as proof of untouched out-of-domain generalization.

---

## 2. Pipeline Parity Audit: Model Parity vs. End-to-End Pipeline Parity

### Issue Flagged:
Earlier text referred to "100% pipeline parity" between live inference and offline evaluation. This claim is inaccurate and conflates model weight parity with end-to-end execution equivalence.

### Detailed Distinction:

| Execution Dimension | Live Runtime Pipeline (`src/attention_pipeline/cli.py`) | Offline Benchmark Harness (`scripts/evaluate_manual_ground_truth_with_boxes.py`) | Parity Status |
| :--- | :--- | :--- | :---: |
| **Model Class** | `ManualBinaryAttentionClassifier` | `ManualBinaryAttentionClassifier` | **100% Identical** |
| **Loaded Weights** | `checkpoints/classroom_manual_binary_model_v2_clean.pt` | `checkpoints/classroom_manual_binary_model_v2_clean.pt` | **100% Identical** |
| **Multimodal Features** | 32-D feature vector ($T=16$ temporal window) | 32-D feature vector ($T=16$ temporal window) | **100% Identical** |
| **Class Index Order** | Index 0 = LOW, Index 1 = HIGH | Index 0 = LOW, Index 1 = HIGH | **100% Identical** |
| **Decision Rule** | $P(\text{HIGH}) \ge 0.50 \to \text{HIGH}$, else $\text{LOW}$ | $P(\text{HIGH}) \ge 0.40$ (calibrated) & $\ge 0.50$ (default) | **Configurable Boundary** |
| **Sampling Frequency** | **$10\text{ Hz}$ downsampled** (`infer_fps = 10`) | **$29.97\text{ Hz}$ native frame rate** | **DIVERGENT** |
| **Processed Frames** | **100 frames** (across 10.01 seconds) | **300 frames** (all native video frames) | **DIVERGENT** |
| **Total Observations** | **1,978 student observations** | **5,700 evaluated sequences** (+ 300 warmup) | **DIVERGENT** |
| **Tracking Mechanism** | Dynamic YOLO11-pose + ByteTrack ($24$ track IDs) | Fixed manual bounding boxes from CVAT ($20$ IDs) | **DIVERGENT** |
| **Warmup Handling** | Frame 0 repeated to pad window (scores all 100 frames) | Frames 0–14 suppressed as `"WARMUP"` ($300$ rows) | **DIVERGENT** |
| **Student Selection** | Unfiltered (detects all visible persons in room) | Constrained strictly to 20 ground-truth students | **DIVERGENT** |

### Rigorous Conclusion:
- **Model / Checkpoint Parity:** **CONFIRMED (100%).** The exact same weights, architecture, and feature definitions run in both environments.
- **End-to-End Pipeline Parity:** **NOT EQUIVALENT.** The live runtime applies temporal downsampling ($10\text{ Hz}$ vs $30\text{ Hz}$), real-time unconstrained tracking, window padding, and open-set student detection. The aggregate counts ($1,978$ vs $5,700$) are mathematically and operationally distinct.

---

## 3. Direct Recomputation from Frozen Aligned CSV

We audited [`runs/4_manual_ground_truth_aligned.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/runs/4_manual_ground_truth_aligned.csv) directly using independent scripts. Zero missing values or inferred entries exist.

### 3.1 Global Dataset Integrity Check
- **Total CSV Rows:** Exactly $6,000$ rows ($300\text{ frames} \times 20\text{ students}$).
- **Warmup Rows:** Exactly $300$ rows (Frames 0–14 $\times$ 20 students; `model_prediction = "WARMUP"`, `is_match = null`).
- **Evaluated Rows:** Exactly $5,700$ rows (Frames 15–299 $\times$ 20 students; zero nulls in any feature or prediction column).
- **Ground Truth Balance:** HIGH = $3,873$ ($67.95\%$), LOW = $1,827$ ($32.05\%$).
- **Active Predictions ($\tau^* = 0.40$):** HIGH = $2,983$ ($52.33\%$), LOW = $2,717$ ($47.67\%$).
- **Overall Accuracy:** $\frac{3,602}{5,700} = \mathbf{63.1930\%}$.
- **Balanced Accuracy:** $\frac{66.9403\% + 61.4253\%}{2} = \mathbf{64.1828\%}$.
- **Macro-Averaged F1:** $\mathbf{61.6141\%}$ (LOW F1: $53.83\%$, HIGH F1: $69.40\%$).
- **Weighted-Averaged F1:** $\mathbf{64.4085\%}$.

### 3.2 Exact Per-Student Recomputation Table (All 20 Students)

| Student ID | Evaluated Rows | Ground Truth HIGH | Ground Truth LOW | Model Pred HIGH | Model Pred LOW | Correct Count | Incorrect Count | Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 285 | 273 ($95.8\%$) | 12 ($4.2\%$) | 242 | 43 | 230 | 55 | **$80.70\%$** |
| **1** | 285 | 17 ($6.0\%$) | 268 ($94.0\%$) | 75 | 210 | 213 | 72 | **$74.74\%$** |
| **2** | 285 | 35 ($12.3\%$) | 250 ($87.7\%$) | 102 | 183 | 180 | 105 | **$63.16\%$** |
| **3** | 285 | 276 ($96.8\%$) | 9 ($3.2\%$) | 107 | 178 | 98 | 187 | **$34.39\%$** |
| **4** | 285 | 180 ($63.2\%$) | 105 ($36.8\%$) | 124 | 161 | 193 | 92 | **$67.72\%$** |
| **5** | 285 | 208 ($73.0\%$) | 77 ($27.0\%$) | 130 | 155 | 139 | 146 | **$48.77\%$** |
| **6** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 189 | 96 | 189 | 96 | **$66.32\%$** |
| **7** | 285 | 247 ($86.7\%$) | 38 ($13.3\%$) | 181 | 104 | 181 | 104 | **$63.51\%$** |
| **8** | 285 | 78 ($27.4\%$) | 207 ($72.6\%$) | 151 | 134 | 88 | 197 | **$30.88\%$** |
| **9** | 285 | 90 ($31.6\%$) | 195 ($68.4\%$) | 119 | 166 | 154 | 131 | **$54.04\%$** |
| **10** | 285 | 205 ($71.9\%$) | 80 ($28.1\%$) | 204 | 81 | 186 | 99 | **$65.26\%$** |
| **11** | 285 | 271 ($95.1\%$) | 14 ($4.9\%$) | 160 | 125 | 168 | 117 | **$58.95\%$** |
| **12** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 53 | 232 | 53 | 232 | **$18.60\%$** |
| **13** | 285 | 224 ($78.6\%$) | 61 ($21.4\%$) | 216 | 69 | 173 | 112 | **$60.70\%$** |
| **14** | 285 | 34 ($11.9\%$) | 251 ($88.1\%$) | 47 | 238 | 220 | 65 | **$77.19\%$** |
| **15** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 260 | 25 | 260 | 25 | **$91.23\%$** |
| **17** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 247 | 38 | 247 | 38 | **$86.67\%$** |
| **18** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 101 | 184 | 101 | 184 | **$35.44\%$** |
| **19** | 285 | 285 ($100.0\%$) | 0 ($0.0\%$) | 258 | 27 | 258 | 27 | **$90.53\%$** |
| **20** | 285 | 25 ($8.8\%$) | 260 ($91.2\%$) | 17 | 268 | 271 | 14 | **$95.09\%$** |
| **TOTAL** | **5,700** | **3,873 ($67.95\%$)** | **1,827 ($32.05\%$)** | **2,983** | **2,717** | **3,602** | **2,098** | **$63.1930\%$** |

*Verification:* Sum of correct ($3,602$) + incorrect ($2,098$) = $5,700$. Accuracy = $3,602 / 5,700 = 63.1930\%$. Matches exactly to 4 decimal places.

### 3.3 Deep Investigation of Student 15 and Student 19
We performed an audit into Student 15 and Student 19 across ground truth, offline predictions, and live inference:
1. **In Frozen Offline Evaluation (`4_manual_ground_truth_aligned.csv`):**
   - **Student 15:** Exactly 285 evaluated rows, exactly 285 GT HIGH ($100.0\%$). Model predicted 260 HIGH, 25 LOW. Accuracy is **$91.2281\%$**. Mean $P(\text{HIGH}) = 0.8747$. Mean box: $[704.4, 144.7, 756.6, 226.3]$ (Far Right, Row 2; $c_x = 730.5, c_y = 185.5$).
   - **Student 19:** Exactly 285 evaluated rows, exactly 285 GT HIGH ($100.0\%$). Model predicted 258 HIGH, 27 LOW. Accuracy is **$90.5263\%$**. Mean $P(\text{HIGH}) = 0.8862$. Mean box: $[743.4, 113.9, 811.0, 178.2]$ (Far Right, Back Row; $c_x = 777.2, c_y = 146.0$).
   - **Data Completeness:** Zero missing values, zero duplicates, zero ambiguous labels in the frozen CSV.
2. **Ambiguity Discovered During Cross-Pipeline Inspection:**
   - In **Live Runtime Inference (`runs/inference.jsonl`)**, ByteTrack dynamically assigns IDs based on detection order:
     - ByteTrack ID 15 appears at centroid $(323, 154)$ (Center-Left Desk), whereas CVAT Ground Truth ID 15 is at $(730.5, 185.5)$ (Far Right Desk).
     - ByteTrack ID 19 appears at centroid $(616, 146)$ and is lost after 40 frames due to occlusion, whereas CVAT Ground Truth ID 19 is at $(777.2, 146.0)$ for all 300 frames.
   - **Finding:** In live inference, track numbers do **not** correspond to the manual ground-truth student IDs. Evaluating live inference tracks directly against CVAT numbers by track index alone would be completely invalid. The offline evaluation harness correctly aligns by manual bounding box spatial coordinates.

---

## 4. Track Centroids and Cross-Clip Identity Scope

### Issue Flagged:
Earlier text risked overstating what the spatial centroid analysis proves regarding student identity.

### Precise Epistemic Clarification:
- **What Centroid Analysis Proves (Directly Measured Finding):**
  - Track ID 0 is located at $(53, 193)$ in `clip_004`, at $(431, 313)$ in `clip_001`, and at $(457, 240)$ in `clip_008`.
  - Track ID 2 is located at $(275, 278)$ in `clip_004`, at $(773, 345)$ in `clip_001`, and at $(317, 245)$ in `clip_008`.
  - **Proven Fact:** ByteTrack track IDs are clip-local indices. They reset across video clips and have zero spatial or semantic continuity between recordings.
- **What Centroid Analysis DOES NOT Prove (Must Not Be Claimed):**
  - It does **not** prove whether the human students in `clip_001` are the same physical individuals as in `clip_004` or `clip_008` (e.g., if a student swapped desks between sessions).
  - It does **not** provide biometric re-identification.
- **Methodological Limitation:**  
  Because the dataset lacks student biometric re-identification or cross-clip metadata, **we cannot determine whether any physical student in the validation set also appears in the test set**. Therefore, track-disjoint evaluation guarantees trajectory independence, but cannot claim proven physical-student disjointness.

---

## 5. Epistemic Separation of Claims

Every technical statement in our thesis documentation is categorized into four tiers:

### Tier 1: Directly Measured Findings (Empirical Facts)
1. `classroom_manual_binary_model_v2_clean.pt` achieves **$87.85\%$ Macro-F1**, **$88.03\%$ accuracy**, and **$87.83\%$ balanced accuracy** on the 2,280 validation sequences of held-out tracks `[2, 5, 8, 11]` from clips 1 & 8 at calibrated threshold $\tau^* = 0.40$.
2. On the 5,700 evaluated sequences of `runs/4.mp4`, `v2_clean` achieves **$61.61\%$ Macro-F1**, **$64.41\%$ weighted-F1**, **$64.18\%$ balanced accuracy**, and **$63.19\%$ raw accuracy** at $\tau^* = 0.40$.
3. Sum of all per-student correct counts ($3,602$) and incorrect counts ($2,098$) matches the aggregate confusion matrix cell-for-cell ($1,223 + 2,379 = 3,602$; $604 + 1,494 = 2,098$).
4. Student 12 ground truth is $100\%$ HIGH ($285$ frames), while model predictions are $81.4\%$ LOW ($232$ frames). Directly measured features show Student 12 has continuous downward head pitch ($+30.5^\circ$) and mean predicted $P(\text{HIGH}) = 0.185$.
5. Mean model confidence on incorrect predictions is $84.50\%$, demonstrating that errors are driven by confident feature bias rather than boundary ambiguity.

### Tier 2: Methodological Limitations (Known Constraints)
1. **Observed Benchmark Limitation:** `runs/4.mp4` was observed during V1 model selection and cannot be described as an untouched independent test set.
2. **Re-Identification Limitation:** Track IDs are clip-local. The validation split is trajectory-disjoint, not proven physical-student disjoint.
3. **Tracking Discrepancy:** Real-time ByteTrack tracking in live inference produces unconstrained tracks (24 IDs) that do not match CVAT student numbering.
4. **Binary Label Collapse:** Moderate attention was collapsed into LOW in manual ground truth; the model is evaluated on a binary distinction (Attentive vs Semi/Non-Attentive), not 3-class.

### Tier 3: Untested Hypotheses (Plausible Hypotheses Requiring Experiments)
1. **Camera Parallax Hypothesis:** Off-center students looking forward at the blackboard exhibit non-zero camera-relative head yaw, causing false inattention predictions.
2. **Desk-Pitch Note-Taking Hypothesis:** Attentive note-taking causes downward head pitch ($>25^\circ$) that the model misinterprets as sleeping or inattention due to lack of desk-relative reference.
3. **Resolution Degradation Hypothesis:** Facial landmark precision and L2CS gaze estimation degrade with distance in back rows, increasing prediction uncertainty.

### Tier 4: Proposed Future Experiments (Actionable Research)
1. **Extrinsic Camera Matrix Calibration:** Calibrate camera extrinsics and normalize yaw/pitch relative to the instructional focal point (teacher/board).
2. **Controlled Pipeline Parity Run:** Feed the exact 5,700 offline feature sequences through `AttentionPipeline.scorer.push()` with real-time tracking disabled to verify numerical identity ($|s_{\text{live}} - 100 \times p_{\text{eval}}| < 10^{-5}$).
3. **Independent Recording Split:** Evaluate on a completely new 45-minute lecture recording in a different classroom with multi-annotator consensus ($\kappa \ge 0.75$).

---

## 6. Supervisor Target Clarification

### Crucial Reporting Standard:
The target F1-score of $\ge 0.84$ **must NOT be described as universally achieved**. It must be reported transparently as follows:

| Evaluation Partition | Sequence Count | Protocol Nature | Macro-F1 | Weighted-F1 | Balanced Acc | Raw Acc | Target $\ge 0.84$ Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Split (Clips 1 & 8, Tracks `[2, 5, 8, 11]`)** | 2,280 | Held-out trajectory-disjoint split (Seed 42) | **$87.85\%$** | **$88.05\%$** | **$87.83\%$** | **$88.03\%$** | **EXCEEDED ($0.8785 \ge 0.84$)** |
| **Benchmark Test Set (`runs/4.mp4`, 20 Students)** | 5,700 | Cross-recording observed benchmark | **$61.61\%$** | **$64.41\%$** | **$64.18\%$** | **$63.19\%$** | **NOT MET ($0.6161 < 0.84$)** |

### Questions for Supervisor Approval:
1. **Primary Metric Specification:** Does the committee require Macro-Averaged F1, Weighted-Averaged F1, or High-class F1 as the primary thesis criterion?
2. **Evaluation Split Acceptance:** Does the committee accept the trajectory-disjoint validation result ($87.85\%$) as the headline validation metric, with the cross-recording benchmark ($61.61\%$) documented as an out-of-domain case study?
3. **Label Formulation:** Does the committee approve the binary task definition (Attentive vs Moderate + Non-Attentive)?

---

## 7. Audit Verification Sign-Off

- [x] Contradiction regarding `runs/4.mp4` prior observation reconciled and resolved.
- [x] Claim of "100% pipeline parity" replaced with strict distinction between model parity and pipeline disparity.
- [x] Per-student table recomputed directly from frozen aligned CSV with exact integer counts ($3,602$ correct, $2,098$ incorrect).
- [x] Student 15 and Student 19 investigated across ground truth and live inference tracking.
- [x] Track centroid evidence scope strictly clarified.
- [x] Epistemic four-tier separation formalized across all findings.
- [x] Supervisor target reported transparently without overclaiming.
- [x] Zero changes made to model checkpoints, thresholds, labels, or active inference code.
