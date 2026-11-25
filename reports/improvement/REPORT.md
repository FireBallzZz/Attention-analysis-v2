# Can unseen-data accuracy reach 80%?

Scripts:
* `scripts/loro_experiments.py` (32-D features)
* `scripts/crop_embeddings.py` + `scripts/loro_crop_experiments.py` (image crops)
* `scripts/head_embeddings.py` (head crops through the L2CS gaze network)

Raw results: `features32.json`, `crops.json`, `head.json`, `calibrated.json`.

## 1. Setting A: brand-new classroom, no setup

Leave-one-classroom-out with human labels. The threshold is always chosen on held-out people from the training rooms.

40 variants were tried:

| Group | Variants |
|---|---|
| **Features** | 32-D pipeline features; classroom-behaviour image embeddings (SCB-YOLO); person embeddings (YOLO11-pose); head-crop gaze-network features (L2CS, 2048-D) and gaze direction relative to the class |
| **Models** | Logistic regression, gradient boosting, BiGRU |
| **Unsupervised per-video normalisation** | On / off |
| **Scoring unit** | 1.6 s windows or 5 s segments |

| Best variants (mean over unseen rooms) | Balanced acc. |
|---|---|
| 32-D, no camera cues, gradient boosting, 5 s segments | 58.2% |
| BiGRU, no camera cues (5 seeds) | 52.3% ± 2.3 |
| Image / head embeddings (best) | 54.4% |

Pooling all rooms into one number gave higher figures (up to 63%), but it flatters, because rooms differ in how often students are LOW. **No learned variant gets close to 80%.**

### Why: models recognise people, not attention

Same features, logistic regression, within each video:

| Split inside one video | AUC |
|---|---|
| Different students (unseen people) | 0.30 – 0.66 (chance) |
| Same students, later moments | 0.68 – 0.95 |

In 10-second clips most students keep one label throughout. LOW is concentrated in about 35 of ~70 people, so the data holds roughly 70 independent examples. A model can only learn "this person looks like this", and that does not carry over to a new person. Visual checks show the labels themselves are sensible: LOW crops really do show turned heads, phones, and looking away while resting on a hand.

## 2. Setting B: after a short per-class calibration

Within each video, a model is trained on the first part of the lesson (the human labels from that period) and tested on the rest. There is a 2-second gap between the two parts so neighbouring frames cannot leak. The threshold is chosen on the calibration part only.

Mean over the five videos (correction: an earlier version of this table pooled all videos into one number, giving a flattering 78.7% / 80.2% balanced accuracy):

| Calibration | Features | Balanced acc. | Accuracy | ROC-AUC | Always-HIGH acc. |
|---|---|---|---|---|---|
| First 30% of the lesson | Classroom-behaviour crops | **70.8%** | **80.7%** | 0.72 | 70.5% |
| First 50% of the lesson | Crops + head-gaze features | **71.1%** | 79.3% | 0.75 | 71.7% |

## 2b. Final model for the no-setup setting: class-relative motion

`scripts/motion_rule.py`: box motion measured exactly as the live pipeline does, relative to the class's median (no labels needed). Leave-one-classroom-out, mean over unseen rooms: **59.3% balanced accuracy, AUC 0.62**, compared with 52.3% for the BiGRU over 5 seeds. In a nested check, the training rooms chose motion on their own in every fold. The live pipeline check (`reports/final_model/pipeline_vs_human_floor.json`) gives mean AUC 0.71 over five videos. Those rooms were used to fit the rule's two coefficients, so this is an integration check, not an unseen-room estimate.

**What this means and what it doesn't:**
* The model has seen these same students during calibration; only the time is new.
* Clips are 10 s, so "30%" is about 3 s of labelling per class here. In a real deployment it would be the first minutes of the first lesson.
* Some optimism remains from the short clips: the same people appear in both parts, even with the 2-second gap.
* It is a realistic deployment mode for a teacher's own class across a semester. It is not "works in any classroom out of the box".

## 3. What would raise Setting A

* **Many more students whose behaviour changes during the clip**, so each person has both HIGH and LOW moments. Hundreds of people, not ~70.
* **Real rather than staged inattention.**
* **More classrooms and camera angles.**
* **A board-relative head-direction feature.**

None of these is a modelling trick. They are all data.
