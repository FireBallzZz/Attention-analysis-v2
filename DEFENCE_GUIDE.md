# Defence day guide

## Before the defence (once, about 10 minutes)
1. Double-click **`Start_Defence_Dashboard.bat`**. It opens the dashboard in your browser.
2. Click through every page: Overview, Live demo, Analyse a video, Results, Improving accuracy, Key finding, Data & labels, Corrections, Q&A.
3. On **Live demo**, play each of the 5 videos once, so they're cached and you know which one to show.
4. On **Analyse a video**, drop in a short clip once (10 s takes about 40 s). It runs the full pipeline in the background and opens in the Live demo view, with its heatmap. Doing it once beforehand also loads the models. Uploads are kept in `runs/defence_uploads/`.
5. Bring **`DEFENCE_REPORT.pdf`**. It replaces the old supervisor PDF, which must **not** be shown.

Everything works **offline**. Always start the dashboard with the `.bat` file: opening `index.html` directly still shows the results, but uploading won't work.

## What to show (about 9 minutes)

| Time | Dashboard page | Say |
|---|---|---|
| 1 min | Overview | The problem, the six stages, the contributions. |
| 2 min | Live demo → *Computer-lab classroom* | Boxes, scores, the timeline following the video. Click an alert to jump to it. Scroll to the **heatmap** and switch to *Follow the video* while it plays. |
| 1 min | Analyse a video | Upload a short clip live and open the result. |
| 2 min | Results | "On classrooms it never saw, the final model reaches 59% balanced accuracy and AUC 0.62. The neural model only reaches 52%, close to chance. Accuracy alone would mislead; here's why." |
| 1 min | Improving accuracy | "Learned models recognise *who* a student is, not attention: with 70 students they memorise identities. The cue that transfers is movement relative to the class. With a short calibration on the teacher's own class: 71% balanced, 81% accuracy." |
| 1 min | Key finding | "Head/gaze is measured from the camera, so the same cue flips meaning between rooms." |
| 1 min | Corrections | "My earlier numbers had leakage. I found and fixed it. Here is the honest version." |

## Seven things to remember
1. Final model on a **brand-new classroom: 59.3% balanced accuracy, AUC 0.62** (chance 50% / 0.5). The neural BiGRU only reaches 52.3%.
2. After a short **per-class calibration: 70.8% balanced, 80.7% accuracy**.
3. Every number is the **mean over unseen classrooms**. Pooling flatters, because rooms differ in how often students are LOW.
4. **DAiSEE** pre-training was measured: AUC 0.51 → 0.55. Adding DAiSEE clips to training gave nothing more.
5. Learned models **recognise people, not attention**: AUC up to 0.95 on known students, about 0.5 on new ones.
6. **Study protection**: confident reading/writing keeps a student at 50 or above. **Strict unseen-classroom result: 58.9% balanced, 63.7% accuracy, AUC 0.648** (61.8% if the cut-off is also fitted on the test room; don't quote that one). Live check on 5 videos, all detected students: AUC 0.66, balanced 62.8%, accuracy 69.1%.
7. **127 automated tests** pass.

## Likely questions, short answers
- **"Your earlier report said 95%."** The test clip overlapped the training data, and one class's F1 was reported as macro-F1. I audited and corrected everything; the Corrections page shows each case.
- **"How do you know there's no leakage now?"** Labels are human-only. Test classrooms share no people, camera or video with training. Thresholds and cue choice are made on the training rooms only; a test checks that no person appears in two rooms.
- **"Why a simple motion rule and not a deep model?"** It is the only cue that keeps its meaning in every unseen classroom: the training rooms picked it on their own in every fold. Two coefficients can't memorise people; a deep model with 70 students learned identities.
- **"Why not 80% on any classroom?"** I tested over 40 learned variants (image, head-gaze, DAiSEE); on new people they stay at 52–58%. Published online-to-classroom transfer is about 71%. Getting more would take many more students whose attention changes over time, for example the DIPSER dataset.
- **"Did you use DAiSEE / FER / SCB?"** Yes, and I measured each. DAiSEE pre-trains the neural model (a small gain). The DAiSEE-only model also scores every student individually, shown in the dashboard. SCB detects each student's behaviour, also shown. But adding DAiSEE or SCB to the attention score *lowered* accuracy on new classrooms (motion alone 63%, + DAiSEE 57%, + SCB 54%): the DAiSEE model was trained on webcam close-ups, and SCB cues flip between rooms. FER: not used; faces are 30–60 px and emotion isn't attention.
- **"Isn't motion a weak cue? Someone writing moves too."** Yes. Motion alone scored 31% of attentive students LOW, mostly writers. So when the SCB behaviour detector is confident a student is reading, writing or raising a hand, the score can't drop below 50 ("study protection"). That raised unseen-room balanced accuracy from 60.7% to 61.8%, mostly in Room C (+6 points). Hard evidence (phone, sleeping) still lowers the score.
- **"Does this measure learning?"** No. It measures visible behaviour linked to attention. It's a support tool for the teacher.
- **"Privacy?"** It runs locally, stores only track numbers (no identities), and recommendations avoid singling students out.
- **"Where is the LLM?"** The recommendation stage is rule-based; the LLM option exists in the config but is switched off. Future work.

## Where things are

| What | Path |
|---|---|
| Dashboard | `dashboard/defence/index.html` (rebuild: `python scripts/build_defence_dashboard.py`) |
| Corrected report | `DEFENCE_REPORT.md` / `DEFENCE_REPORT.pdf` (rebuild: `python scripts/make_report_pdf.py`) |
| Final model | `checkpoints/motion_rule_v1.json`, configured in `configs/pipeline.yaml` (`motion_rule`, `combine_mode: model_floor`) |
| Previous neural model | `checkpoints/final_human_v1.pt` (set `motion_rule: ""` to use it) |
| Evaluation results | `reports/improvement/` (motion rule, DAiSEE, variants, calibration), `reports/final_model/` (live check) |
| Labelling tool | `..\..\Labeling_Tool\Start_Labeling.bat` |
