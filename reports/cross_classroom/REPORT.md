# Cross-classroom evaluation with the new human labels (High / Low)

Scripts: `scripts/evaluate_cross_classroom.py` (main), `scripts/relative_gaze_features.py` (feature test).
Raw numbers: `results.json`. No existing checkpoint, dataset or report was changed.

## 1. Data

| Room | Videos | Students | Windows | Low windows | Label source |
|---|---|---|---|---|---|
| classroom_01 | clip_004 + clip_001 | 23 | 10,049 | 1,367 (13.6%) | human CVAT (earlier) |
| room_B | `11.mp4` + `non attentive 1.mov` (same room, same students) | 14–16 | 8,699 | 4,392 (50.5%) | new human CVAT |
| room_C | `Test 8.mp4` | 19 | 4,130 | 807 (19.5%) | new human CVAT |
| ~~room_D~~ | `Test 6.mp4` | 12 | 2,700 | 0 | **excluded**, see below |

New files:
* CVAT exports: `Anno/cvat_new/<video>/annotations.xml`
* CSVs: `Anno/<video>_human_ground_truth.csv`
* Features: `data/processed/human_<video>.npz`

The features were extracted with the same `extract_classroom_sequences.py` used for the old training data.

**Problems found in the new labels:**
* **Test 6 is effectively unlabelled.** All 240 frames of every student are `High`, which is CVAT's default value. That includes the student at the front right, who is asleep on the desk for the whole clip.
* **Short stray tracks.** Each video has 12–40 extra one-frame tracks on frame 0 that duplicate a real student. They are ignored automatically, but can be deleted in CVAT.
* **`non attentive 1` is 87% Low.** It looks staged ("act inattentive"). Staged behaviour may not look like real inattention.

## 2. Protocol

* **Leave one classroom out.** Train on the other rooms and test on a room whose students and camera were never seen.
* **Inner validation.** Persons held out from the training rooms choose the epoch and the threshold.
* **Repeats.** The BiGRU is the deployed recipe with 3 seeds averaged. A logistic regression on window statistics is the simple baseline.
* **Confidence intervals.** 95% CIs come from a person-level bootstrap.

## 3. Results on unseen classrooms

ROC-AUC 0.50 = coin flip; below 0.50 = the scores are partly reversed.

| Test room | Model | Accuracy | Balanced acc. | Low recall | ROC-AUC [95% CI] | Always-High acc. |
|---|---|---|---|---|---|---|
| classroom_01 | BiGRU | 19.1% | 37.8% | 63.4% | 0.32 [0.16–0.52] | 86.4% |
| | LogReg | 23.0% | 48.5% | 83.6% | 0.44 [0.29–0.63] | |
| room_B | BiGRU | 62.5% | 62.7% | 41.3% | 0.59 [0.46–0.70] | 49.5% |
| | LogReg | 61.3% | 61.3% | 61.2% | 0.58 [0.41–0.71] | |
| | deployed checkpoint | 36.9% | 36.6% | 65.6% | 0.33 | |
| room_C | BiGRU | 44.4% | 47.3% | 52.0% | 0.43 [0.26–0.64] | 80.5% |
| | LogReg | 61.7% | 48.2% | 25.9% | 0.44 [0.31–0.59] | |
| | deployed checkpoint | 34.0% | 48.1% | 71.1% | 0.46 | |

The deployed checkpoint on classroom_01 (AUC 0.84) is not shown as a result because it was trained on that room.

**Conclusion: on a classroom it has never seen, the current model is no better than a coin flip.** This holds for the deployed checkpoint too.

## 4. Why: head/gaze cues are measured relative to the camera

For each video, the table gives the AUC of one feature on its own. Above 0.5 means a high value goes with High attention; below 0.5 means it goes with Low.

| Video | "facing front" | head yaw | gaze pitch |
|---|---|---|---|
| clip_004 | 0.33 | 0.25 | |
| clip_001 | 0.26 | 0.26 | |
| 11 | **0.73** | **0.70** | **0.73** |

The same cue means opposite things in different rooms. Head pose and gaze are measured relative to the **camera**, but attention means looking at the **teacher or board**, and the board is in a different place relative to the camera in each room. The model learns where one room's board is, and that does not transfer.

Experiment, using logistic regression and mean AUC over the three unseen rooms:

| Head/gaze features | classroom_01 | room_B | room_C | Mean AUC |
|---|---|---|---|---|
| Raw (camera-relative) | 0.44 | 0.58 | 0.44 | 0.48 |
| Relative to the student's own usual direction | 0.44 | 0.55 | 0.60 | 0.53 |
| Relative to the whole class's usual direction | 0.40 | 0.53 | 0.50 | 0.48 |
| **Removed entirely** (behaviour, posture, motion and eyes only) | **0.63** | **0.63** | 0.53 | **0.60** |

In their current form, the direction cues hurt generalisation. Behaviour and posture cues transfer better, but they are still weak.

## 5. What to do next (priority order)

1. **Fix Test 6 in CVAT.** Set the sleeping student, and anyone else off-task, to Low. That adds a fourth room. Delete the one-frame duplicate tracks in all four files.
2. **Make direction cues relative to the board.** For each camera, click once on the board/teacher position. Then compute, for each student, the angle between their head direction and the direction to that point. This is the physically correct "is looking at the front" feature, and it should transfer between rooms.
3. **Label more rooms, not more clips of the same room.** Each new classroom is worth far more than another 10 s of a room already in the data. Label real (not staged) inattention where possible.
4. **Try a model that looks at the student image directly**, for example a pretrained CNN on the student crop. The 32 hand-made features throw away a lot of information.
5. **Thesis framing.** Present cross-classroom generalisation as the open problem. The camera-relative-gaze finding is itself a solid, defensible contribution. Report every number with the leave-one-classroom-out protocol and next to the always-High baseline.
