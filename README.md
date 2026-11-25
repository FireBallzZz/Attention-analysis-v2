# Classroom Attention Analysis

A six-stage pipeline that finds every student in a classroom video, measures pose, gaze, behaviour and movement, scores each student's attention over time, raises teacher alerts when attention drops, and suggests a teaching response. It includes a web dashboard where you can upload a video and see the result.

| Stage | Component | Source |
|---|---|---|
| 1 Input | Any video, analysed at 10 fps | — |
| 2 Detect & track | YOLO11-Pose + ByteTrack | COCO pretrained (`yolo11n-pose.pt`) |
| 3 Modalities | Head pose; L2CS-Net gaze; SCB-YOLO behaviour (hand-raise, reading, writing, phone…); box motion | Gaze360 (`checkpoints/l2cs_finetuned.pt`), SCB-dataset (`checkpoints/scb_yolo.pt`) |
| 4 Attention | **Class-relative motion** over 1.6 s (P(HIGH) = sigmoid(a + b·z), z = student's log motion minus the class median). Hard evidence (phone, sleeping, head on desk) can only lower the score; confident SCB reading/writing/hand-raising keeps it ≥ 50 ("study protection") | `checkpoints/motion_rule_v1.json`, fitted on human labels |
| 5 Drop detector | Sustained (< 40 for 3 s) or sudden (≥ 20-point fall in 2 s), 8 s cooldown | Rules |
| 6 Recommendation | Teaching suggestion per alert | Rules |

The DAiSEE-only network (`checkpoints/baseline/fusion_daisee.pt`) and the classroom BiGRU (`checkpoints/final_human_v1.pt`) also run on every student. Their outputs are reported in the scores file but are not used in the score, because they lowered accuracy on unseen classrooms.

## Accuracy (honest)

Evaluated with human CVAT labels, **leave-one-classroom-out**: the model is fitted on two classrooms and tested on a third whose room, camera and students it has never seen. Mean over the three classrooms:

| Measure | Value |
|---|---|
| **Balanced accuracy, unseen classroom** | **58.9%** (56.1 / 58.7 / 62.0%), chance = 50% |
| Accuracy | 63.7% (always answering "attentive" gives 71.7%) |
| Macro-F1 | 55.3% |
| ROC-AUC | 0.648 |
| With a short per-class calibration (teacher labels the start of the lesson) | ~71% balanced, ~81% accuracy |
| Real (non-staged) annotated videos, verified student pairs | 69.2% balanced, 75.4% accuracy |

Full method, every experiment (40+ variants, DAiSEE / SCB / SCB-Dataset5 tests, parameter sensitivity) and the corrections to an earlier, leaked evaluation are in **[DEFENCE_REPORT.md](DEFENCE_REPORT.md)** ([PDF](DEFENCE_REPORT.pdf)). `docs/Student_Attention_Model_Supervisor_Report.pdf` and the `audit/` folder are **superseded**: their headline numbers came from data leakage.

## Run it

Requires Python 3.10+ and `git`. A CUDA GPU is recommended (about 3 s per second of video on an RTX 3060); CPU works but is slow.

```bash
git clone https://github.com/FireBallzZz/Attention-analysis-v2.git
cd Attention-analysis-v2
pip install -r requirements.txt          # for GPU, first install torch/torchvision from https://pytorch.org
pip install -e .
python scripts/setup_runtime.py          # fetches the YOLOv7 code (pinned commit) that scb_yolo.pt needs
```

All model weights are in the repository (`yolo11n-pose.pt`, `yolo11n.pt`, `checkpoints/`). No dataset download is needed to run the model.

**Analyse one video:**

```bash
python analyse_video.py "path/to/classroom.mp4"
```

This writes `results/<name>_annotated.mp4` (overlay: box, score and cue per student; green = HIGH, red = LOW) and `results/<name>_scores.jsonl` (per frame and per student: score, cue, head pose, gaze, SCB behaviour, DAiSEE ratings).

**Dashboard with upload:**

```bash
python launch_defence_dashboard.py       # Windows: double-click Start_Defence_Dashboard.bat
```

The dashboard opens at http://127.0.0.1:8765. Under **Analyse a video**, upload a clip; the full pipeline runs in the background and the result opens with its timeline, alerts, student table and seat heatmap. The demo videos shown on the Live demo page are not included, because they contain real students; the uploaded-video view works fully.

**Tests:**

```bash
python -m pytest                          # 127 tests; data-dependent ones skip without the private data
```

## How to read the scores

- **Scores are relative to the class.** A student who moves more than their classmates scores lower. Read a video by *who scores lowest*; in every class roughly 40–60% of students fall below 50, even when most are attentive.
- The text under each box is the detected behaviour, not the score.
- Hard evidence (phone, head on desk) lowers a score; confident reading or writing keeps it at 50 or above.

## Repository layout

| Path | Contents |
|---|---|
| `src/attention_pipeline/` | Pipeline (detection, tracking, modalities, motion model, rules, drop detector, recommendations) |
| `configs/pipeline.yaml` | The deployed configuration |
| `checkpoints/` | Model weights used at runtime |
| `analyse_video.py`, `launch_defence_dashboard.py` | Entry points |
| `scripts/` | Evaluation and training scripts (`motion_rule.py`, `evaluate_cross_classroom.py`, `verify_matching.py`, `sensitivity_*.py`, …) |
| `reports/` | All evaluation results (JSON / Markdown) |
| `Anno/` | Human CVAT annotations (boxes and HIGH/LOW labels only) |
| `dashboard/defence/` | Dashboard page |
| `tests/` | Automated tests |

## Reproducing the evaluation

The evaluation scripts need the classroom videos and extracted features (`runs/`, `data/processed/`). Those are not published, because they show real students. With access to them:

```bash
python scripts/motion_rule.py                                        # fit + leave-one-classroom-out
python scripts/evaluate_pipeline_vs_human.py --combine model_floor   # live check vs human labels
python scripts/verify_matching.py                                    # same student, same frame
python scripts/build_defence_dashboard.py                            # dashboard data
```

## Data and licences

- COCO / YOLO11 weights: Ultralytics (AGPL-3.0).
- YOLOv7 code: WongKinYiu/yolov7 (GPL-3.0), fetched at setup.
- L2CS-Net / Gaze360: research use.
- SCB-dataset: https://github.com/Whiffe/SCB-dataset.
- DAiSEE: research use only.

The weights here are derived from these sources and are provided for **research and education only**. The classroom footage used for evaluation was recorded for this thesis and is not distributed.
