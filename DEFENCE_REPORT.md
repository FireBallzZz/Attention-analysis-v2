# Hybrid Classroom Attention Analysis: Corrected Technical Report

*This report replaces `Student_Attention_Model_Supervisor_Report.pdf` (8 Nov 2026), whose headline numbers were affected by data leakage and metric errors (section 6).*

## 1. Summary

The system takes classroom video and runs six stages:

1. Find every student.
2. Measure head pose, gaze, behaviour and movement.
3. Score attention over 1.6-second windows.
4. Detect attention drops.
5. Suggest a teaching response to the teacher.
6. Run all of this end to end on real video, on a single laptop GPU.

All evaluation uses **human labels only** (22,878 student windows, three classrooms). It is **leave-one-classroom-out**: every model is tested on a room, camera and set of students it never saw. Every number below is the **mean over the three unseen classrooms**. Pooling all rooms into one number flatters the result, because the rooms differ in how often students are LOW.

| New classroom, no setup | Balanced acc. | ROC-AUC |
|---|---|---|
| Always HIGH (trivial baseline) | 50.0% | 0.50 |
| Neural model (BiGRU, DAiSEE-pretrained; 5 seeds) | 52.3% ± 2.3 | 0.53 |
| Best of 40 other learned variants | 58.2% | — |
| **Final model: class-relative motion + hard-evidence rules** (offline, human boxes) | **59.3%** | **0.62** |
| Motion + study protection, live pipeline, strict leave-one-classroom-out | 58.9% | 0.65 |
| **Final model: + phone detection on student crops, strict leave-one-classroom-out** (cut-off chosen on the other two rooms) | **60.2%** | **0.67** |

| Teacher's own class, after a short calibration | Balanced acc. | Accuracy | Always-HIGH acc. |
|---|---|---|---|
| First 30% of the lesson labelled, rest monitored (mean of 5 videos) | **70.8%** | **80.7%** | 70.5% |

**Main findings.**
1. Head-pose and gaze cues are measured relative to the *camera*, so they flip meaning between rooms.
2. With about 70 students whose label rarely changes within a clip, learned models (neural, image, gaze) recognise *who* a student is, not attention.
3. The one cue that keeps its meaning in every room is **how much a student moves compared with the rest of the class**. The final model is built on it.

## 2. System

| Stage | Component | Source / training |
|---|---|---|
| 1 Input | Any video, resampled to 10 fps | — |
| 2 Detect & track | YOLO11-Pose + tracker | COCO pretrained |
| 3 Modalities | Head pose; L2CS-Net gaze; SCB-YOLO behaviour (hand-raise, reading, writing…); box motion | Gaze360; SCB-dataset |
| 4 Attention | **Class-relative motion over 1.6 s**: P(HIGH) = sigmoid(a + b·z), z = log(student motion) − median log motion of the class. Hard evidence (phone, sleeping, head on desk, peer talk) may only lower the score; a phone found on the student's enlarged crop caps it at 30; confident SCB reading / writing / hand-raising keeps it ≥ 50 (study protection, section 5.4). | 2 coefficients + threshold fitted on human labels (`checkpoints/motion_rule_v1.json`) |
| 5 Drop detector | Sustained (< 40 for 3 s) or sudden (≥ 20-point fall in 2 s); 8 s cooldown | Rules |
| 6 Recommendation | Context-aware teaching suggestion per alert | Rules (LLM option implemented but off) |

Fitted rule: a = 0.051, b = −0.548 (more motion → LOW), threshold 0.514, chosen on held-out classrooms. The previous neural model (`checkpoints/final_human_v1.pt`) is kept for comparison and can be re-enabled by setting `motion_rule: ""` in `configs/pipeline.yaml`.

## 3. Data

| Dataset | Role | Result |
|---|---|---|
| COCO (YOLO11-Pose) | Student detection and keypoints | Used |
| Gaze360 (L2CS-Net) | Gaze direction | Used |
| SCB-dataset (SCB3; SCB5 bow/turn-head and discuss tested) | Classroom behaviour per student (shown in the dashboard); image features; study protection | Its cues flip between rooms; adding SCB to the score lowers unseen-room accuracy (section 5.3) |
| DAiSEE | Pre-training of the neural model; **DAiSEE-only network applied to each student individually** (engagement, boredom, confusion, frustration, shown in the dashboard) | Pre-training: AUC 0.51 → 0.55; DAiSEE clips in training: no gain; individual DAiSEE model on classroom students: at chance (section 5.3) |
| FER2013 / RAF-DB | — | Not used: faces at classroom distance are 30–60 px and emotion is not attention |
| **Own classrooms (human CVAT labels)** | Training and evaluation | See below |

| Classroom | Videos | Windows | LOW share |
|---|---|---|---|
| Classroom 01 (computer lab) | clip 007 (features saved as `classroom_clip_004`), clip 001 | 10,049 | 13.6% |
| Room B (seminar) | 11, non attentive 1 (same students) | 8,699 | 50.5% |
| Room C | Test 8 | 4,130 | 19.5% |

**Excluded:**
* Model-generated pre-annotations (previously used as labels for clips 001 and 008).
* Test 6, where every student, including one asleep for the whole clip, was left at CVAT's default "High".

## 4. Evaluation protocol

* **Splits.** Whole classrooms never cross between training and test.
* **Model choices.** Thresholds, epochs and cue choice are made on the training classrooms only. The cue choice was also checked in a nested test: in every fold, the two training rooms picked motion on their own.
* **Reporting.** Mean over unseen rooms, next to the always-HIGH baseline; 95% CIs resample whole people.
* **Seed variability.** The neural model's pooled balanced accuracy ranged from 58% to 70% across random seeds, which is why seeds are averaged and per-room means are reported.

## 5. Results

### 5.1 Final model on unseen classrooms (leave-one-classroom-out)

| Test classroom | Balanced acc. | ROC-AUC | LOW recall | Accuracy | Always-HIGH acc. |
|---|---|---|---|---|---|
| Classroom 01 | 56.9% | 0.60 | 80.9% | 39.1% | 86.9% |
| Room B | 54.5% | 0.56 | 53.9% | 54.5% | 49.2% |
| Room C | 66.4% | 0.71 | 49.3% | 77.0% | 81.3% |
| **Mean** | **59.3%** | **0.62** | **61.4%** | 56.9% | 72.5% |

The threshold favours catching LOW moments, so accuracy falls below the always-HIGH baseline in Classroom 01, where 87% of windows are HIGH.

**Live end-to-end check.** The full pipeline was run on the five labelled videos and each pipeline box was matched to the hand-drawn box it lies inside. The earlier IoU ≥ 0.3 rule discarded about 40% of correctly detected students, because the hand-drawn boxes include the desk and are 2–3× larger. Over all 6,715 matched student-frames the results were mean AUC 0.66, balanced accuracy 62.8% and accuracy 69.1%. Per video: clip 007 57%, clip 001 58%, video 11 74%, non attentive 1 48%, Test 8 78%. Non attentive 1 is a staged clip where 88% of students are labelled LOW; a class-relative score cannot flag a whole class (section 7). This confirms the live implementation behaves like the evaluation. It is not an unseen-room estimate, because the rule's two coefficients were fitted on these rooms. The same check exposed that the rubric's "active study" boost *reversed* the scores (AUC 0.28 in clip 001), so it was removed (`combine_mode: model_floor`).

**Are the CVAT students and the model's students the same people?** (`scripts/verify_matching.py`; images in `runs/final_outputs/matching_check/`. Pairs are formed in the exact video frame the model analysed: every analysed timestamp falls within 0.015 frames of a video frame that has CVAT boxes.) Each CVAT student was paired with the same model ID for 90–98% of the clip. In crowded rows, however, a model box often lies inside two overlapping hand-drawn boxes (29% of pairs in video 11, 44% in non attentive 1), and visual checks showed some of these pairs were the wrong student. On **verified pairs only** (same model ID throughout, no overlap ambiguity):

| Video | Verified pairs | Balanced acc. | Accuracy | AUC | Always-HIGH acc. |
|---|---|---|---|---|---|
| clip 007 | 89% | 59.0% | 73.9% | 0.65 | 78.5% |
| clip 001 | 89% | 61.3% | 76.3% | 0.69 | 92.2% |
| video 11 | 72% | 83.2% | 74.6% | 0.88 | 90.4% |
| non attentive 1 (staged) | 51% | 29.2% | 54.6% | 0.23 | 6.6% |
| Test 8 | 94% | 76.2% | 76.2% | 0.83 | 84.9% |
| **Mean** | | **61.8%** | **71.1%** | 0.66 | |
| **Mean without the staged clip** | | **69.9%** | **75.3%** | **0.76** | |

In the staged clip the scores are reversed: students acting inattentive sit still with phones, while the few attentive ones move. A movement-based cue cannot handle this case.

**Strict leave-one-classroom-out for the final model.** The deployed rule's 3 numbers (two coefficients and the cut-off) are fitted on all three classrooms, so the per-room figures above and in 5.3–5.5 (mean 61.8%) let the cut-off see the test room. With the cut-off chosen on the other two rooms only, the result is Classroom 01 56.1%, Room B 58.7%, Room C 62.0%: **mean 58.9% balanced accuracy, 63.7% accuracy, macro-F1 55.3%, AUC 0.648, 51.2% of LOW moments caught**. This matches the offline 59.3%  **Adding phone detection on enlarged student crops** (below) raises the strict result to Classroom 01 57.0%, Room B 60.5%, Room C 63.0%: **mean 60.2% balanced accuracy, 64.3% accuracy, macro-F1 56.3%, AUC 0.668, 54.3% of LOW moments caught. This is the number to quote for an unseen classroom.**

**Phone detection on student crops** (`scripts/phone_crop_eval.py`). Phones are only a few pixels wide in the full frame; the full-frame detector found one in 3.9% of student-frames of the phone clip. The pipeline now enlarges each student's crop to 640 px and runs the COCO phone detector (`yolo11n.pt`, class "cell phone"). Confidence ≥ 0.25 caps the score at 30. This is a fixed rule; nothing is fitted on the labels, and the result is the same for cut-offs 0.25–0.35. It improves every classroom (strict 58.9% → 60.2%, AUC 0.648 → 0.668). Leaving the staged clip out of fitting changed nothing (cut-offs 49–52, same 60.2%).

### 5.2 Why learned models do not transfer

| Inside one video | AUC |
|---|---|
| Image model, **different** students | 0.30 – 0.57 |
| Image model, **same** students later in the clip | 0.68 – 0.95 |

Single cues also flip between rooms. "Facing the camera" has AUC 0.26–0.33 in Classroom 01 but 0.73 in Room B. Class-relative motion points the same way in every room.

### 5.3 Which datasets help the attention score?

Every component was recorded from the live pipeline on the five labelled videos (6,687 matched student-frames) and normalised within its own video. Combinations were then tested leave-one-classroom-out (`scripts/collect_components.py`, `scripts/ensemble_eval.py`).

| Attention score built from | Balanced acc. | ROC-AUC |
|---|---|---|
| **Motion (final)** | **60.2%** | **0.65** |
| Motion + DAiSEE individual model | 58.8% | 0.60 |
| Motion + BiGRU | 58.7% | 0.62 |
| BiGRU alone | 58.6% | 0.57 |
| Motion + rubric | 55.0% | 0.52 |
| DAiSEE individual model alone | 53.4% | 0.48 |
| Motion + SCB behaviour | 50.3% | 0.57 |
| SCB behaviour alone | 46.7% | 0.39 |

The DAiSEE-only network, applied to each student as an individual, is unreliable on classroom students (single-room AUC 0.38–0.65; it flips between rooms). It was trained on close-up webcam faces, and classroom students are small and seen from the side. SCB behaviour cues flip between rooms (AUC 0.19–0.44). Adding either one lowers accuracy, so the score uses motion only; DAiSEE and SCB outputs are reported per student in the dashboard. In this live-component test motion reaches 60.2%, close to the offline 59.3% in 5.1.

### 5.4 Why studying students were scored LOW, and the fix

The motion model scores movement relative to the class. Writing, turning pages and looking up and down at the board all move the head and hands, so **31% of attentive students were scored LOW**. When the SCB detector *confidently* sees reading, writing or hand-raising (14.9% of moments), 78% of those students are labelled attentive (72% overall), yet motion alone scored 41% of them LOW.

**Study protection** is a fixed rule with nothing fitted on the labels. If SCB's hand-raising / reading / writing probability is ≥ 0.3, the score may not fall below 50. Hard evidence (phone, sleeping) still overrides it.

| Unseen classrooms (live components) | Balanced acc. per room | Mean | ROC-AUC |
|---|---|---|---|
| Motion only | 59 / 60 / 63% | 60.7% | 0.646 |
| **Motion + study protection (final)** | 58 / 59 / **69%** | **61.8%** | **0.648** |

The result is identical for cut-offs from 0.3 to 0.5. Room C gains 6 points, and the other two rooms lose 1–1.5 points. Unlike the rubric's old head-pose "active study" boost (which reversed the scores), this rule only trusts confident behaviour detections. Reproduce: `python scripts/study_protection_eval.py`.

**Also tried and rejected** (same unseen-classroom test):

| Idea | Result | vs current |
|---|---|---|
| Separate head and body/hand motion (`scripts/motion_parts_eval.py`) | head only 54.6%, body only 58.6%, both 53.6% | full-box motion 59.3%; body/hand motion *also* goes with LOW |
| Smoothing each student's score over 1–8 s, window chosen on training rooms (`scripts/smoothing_eval.py`) | 62.4% | 64.9% without smoothing (both measured before the matching fix) |

| Optical flow instead of pixel difference (`scripts/optical_flow_eval.py`) | 56.0% | pixel difference 59.3% (same protocol) |
| **SCB-Dataset5** ([official release](https://huggingface.co/datasets/wintonYF/SCB-Dataset)): new detectors for bow head, turn head and discussion, trained on 3,274 SCB images (val mAP50 0.43 and 0.70) and used as hard evidence or in a learned combination (`scripts/train_scb5_detectors.py`, `scripts/scb5_eval.py`) | 61.5–61.6% as hard evidence; 56.4% learned | 61.8%; the cues fire on only 0–1.5% of student-frames in these classrooms, and "turn head" flips direction between rooms |

| **Peer-context scoring** (behaviour compared with the rest of the class, after "Context Matters", CVPR-W 2026; `scripts/peer_context_eval.py`) | 57.1–59.3% (strict) | 60.2%; behaving like the class predicts attention only in Room B (AUC 0.78), at chance elsewhere (0.49–0.50) |
| **DAiSEE with unsupervised domain adaptation** (classifier trained only on 70,224 DAiSEE windows; classroom features aligned per video by standardisation or CORAL, no classroom labels; `scripts/daisee_uda_eval.py`) | AUC 0.46–0.49 (Classroom 01: 0.31–0.40, reversed) | DAiSEE engagement (webcam close-ups) does not transfer to classroom cameras even after alignment |

| **Gaze target estimation** (Gaze-LLE, frozen DINOv2 ViT-B, `gazelle_dinov2_vitb14_inout`; where each student looks and whether the target is in frame; `scripts/gaze_target_eval.py`) | gaze cues alone 52.1%, with motion 56.0% (strict) | 59.6% on the same grid; cues flip between rooms (in-frame gaze AUC 0.58 / 0.15 / 0.45). Visual checks show the predicted targets collapse onto salient objects (one notebook, other faces) at classroom distance |

So motion + study protection + phone-on-crop is the best model these features, these public datasets and this amount of labelled data support.

### 5.5 Parameter sensitivity

One parameter changed at a time, others at the deployed value. This is a sensitivity analysis, not tuning: a value is adopted only if it also wins when chosen on the training classrooms alone (`scripts/sensitivity_scores.py`, `scripts/sensitivity_motion.py`).

| Parameter | Deployed | Values tested → mean balanced accuracy | Result |
|---|---|---|---|
| HIGH/LOW cut-off | 50 | 40: 56.0% · 45: 58.6% · **50: 61.8%** · 55: 57.2% · 60: 51.2% | Deployed value is best. Lower cut-offs raise plain accuracy (71.7% at 45) but catch fewer LOW students |
| Study protection: SCB threshold | 0.3 | off: 60.7% · 0.2–0.5: **61.8%** · 0.6–0.7: 60.5% | Stable from 0.2 to 0.5 |
| Study protection: minimum score | 50 | 45: 60.7% · **50–70: 61.8%** | Stable from 50 up |
| Frame gap for motion | 0.1 s | **0.1 s: 59.7%** · 0.2 s: 59.4% · 0.3 s: 57.9% | Deployed value is best (chosen by every training fold) |
| Crop size for motion | 64 px | 32: 60.2% · **64: 59.7%** · 128: 58.5% | No meaningful difference; 64 px has the best AUC |
| Motion window | 16 frames (1.6 s) | 8: 59.4% · **16: 59.7%** · 24: 60.5% · 32: 64.2% · 48: 63.8% | Longer windows looked better on later frames (nested 63.4%), but the full-clip test (3 s: 57.8%) and live score smoothing showed no gain, and a 3.2 s window doubles the warm-up. Not adopted; needs more data to confirm |

The model is robust: no single setting moves the result by more than about ±3 points, which is within the noise of three test classrooms.

### 5.6 Calibrated setting (towards 80%)

Within each video, a model on classroom-behaviour image features is trained on the first part of the lesson and tested on the rest, with a 2-second gap. The table gives the mean over five videos.

| Calibration | Balanced acc. | Accuracy | ROC-AUC | Always-HIGH acc. |
|---|---|---|---|---|
| First 30% labelled | 70.8% | 80.7% | 0.72 | 70.5% |
| First 50% labelled (+ head-gaze features) | 71.1% | 79.3% | 0.75 | 71.7% |

This uses the same students, at new moments. It suits a teacher's own class monitored over time; it does not mean the system works out of the box in any room. It is evaluated offline and not yet built into the dashboard.

## 6. Corrections to the earlier report

| Earlier claim | Problem | Corrected |
|---|---|---|
| Clip 001: 95.13% accuracy | Always answering HIGH scores 95.22%; 0 of 51 LOW cases found | Macro-F1 48.75%, balanced accuracy 49.95% |
| Clip 001: 97.50% macro-F1 | That was the F1 of the HIGH class only | Fixed in code (`attention_pipeline/metrics.py`) |
| Clip 007: 80.65% (held-out) | Clip 007 was a training clip (its features are `classroom_clip_004`) | Kept as training data; tested only when its classroom is held out |
| 87.22% on 4,386 held-out sequences | Matrix sums to 4,602; 2/3 of labels model-made; same people under different IDs | Rebuilt (section 5) |
| Video Project 3: 90.22% vs ground truth | No human labels; agreement with the rule-based rubric | Shown only as an unlabelled demo |
| Threshold 0.45 "validated" | Tuned on the human test annotation | Thresholds chosen on training classrooms |
| SCB-5, DeepSORT, 12.6 vs 30.6 FPS, 32-D with emotion | Inconsistent or inaccurate | Emotion channels are constant (unused) |
| *(audit, 25 Nov)* Live check matched boxes by IoU ≥ 0.3 | Pipeline boxes are 2–3× smaller than the hand-drawn ones, so ~40% of correctly detected students were discarded | Matching by containment; all numbers re-measured on 6,687 instead of 4,394 student-frames |
| *(audit, 25 Nov)* Video processor at "10 fps" | Floating-point drift ran 24 fps videos at 8 fps, so motion was measured over a different gap than evaluated | Fixed 10 fps schedule in `scripts/video_processor.py` |

## 7. Limitations and future work

**Limitations:**
* Three labelled classrooms with short clips, so results vary between rooms (54–66%).
* One clip ("non attentive 1") is staged.
* Motion is an indirect cue: an attentive student who moves (writing, gesturing) can still score lower when the SCB detector is not confident about the activity.
* Scores are relative to the class: in every video roughly 40–60% of students fall below 50, so the system ranks students within a class and cannot say that a whole class is attentive. Absolute alerts come from the hard-evidence rules (phone, sleeping, head on desk).
* The system measures visible behaviour, not learning; recommendations are rule-based.

**Future work:**
* **Request DIPSER** (54 students, 51 h, per-second attention ratings in a real classroom; access by application). It provides exactly what is missing: many students whose attention changes over time.
* A board-relative gaze feature (one click per camera). A general gaze-target model (Gaze-LLE) was tested and does not work at classroom distance; this needs a gaze model trained on classroom footage.
* A calibration step in the dashboard (label the first minutes of a class).
* A second annotator, to report Cohen's κ.

## 8. Reproduce

```bash
python scripts/motion_rule.py                                          # evaluate + fit the final model
python scripts/evaluate_pipeline_vs_human.py --combine model_floor     # live end-to-end check
python scripts/daisee_ablation.py                                      # DAiSEE contribution (pre-training)
python scripts/collect_components.py && python scripts/ensemble_eval.py  # DAiSEE / SCB / motion combinations
python scripts/study_protection_eval.py                               # study protection
python scripts/phone_crop_eval.py                                     # phone detection on crops
python scripts/loro_experiments.py                                     # 32-D learned variants
python scripts/identity_check.py                                       # identity vs attention
python -m pytest                                                       # 127 tests
python scripts/build_defence_dashboard.py                              # dashboard data + videos
python scripts/make_report_pdf.py                                      # this report as PDF
```
