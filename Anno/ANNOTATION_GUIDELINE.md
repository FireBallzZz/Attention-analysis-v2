# Smart Classroom Student Attention Annotation Guideline & Protocol

**Thesis Project:** *Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms*  
**Annotation Tool:** Computer Vision Annotation Tool (CVAT) — Video Interpolation Mode  
**Task Specification:** Object Label: `Students` (Bounding Box) | Attribute: `Attention` (Radio: `High`, `Moderate`, `Low`)  
**Resolution:** $832 \times 464$ @ $29.97$ fps (300 frames per 10-second clip)

---

## 1. Objective and Scientific Scope

The goal of this annotation task is to establish genuine, defensible human ground truth for individual student attention in a physical smart classroom lecture environment. Annotations serve as the gold standard for training and evaluating multimodal computer vision models (combining head pose, eye gaze, facial expressions, and classroom body posture).

---

## 2. Attention Level Definitions and Observable Behavioral Rubric

Attention is a psychological construct inferred from observable behavioral and physical indicators. To avoid subjective annotator drift, assign labels based strictly on the following observable visual indicators:

### A. `High` Attention (Attentive)
- **Primary Visual Indicators:**
  - **Gaze / Orientation:** Head and gaze steadily directed toward the instructional focal point (teacher, blackboard, projection screen, or actively working on teacher-directed tasks).
  - **Body Posture:** Upright posture, torso oriented forward; shoulders level or leaning slightly forward toward the presentation.
  - **Task Interactions:** Active handwriting/typing in clear synchronization with the lecture; raising hand to ask or answer a question; actively holding a textbook or handout at reading distance.
  - **Gaze Tolerance:** Minor saccadic eye movements are acceptable if head pose remains stable toward instructional focus ($|\text{yaw}| \le 25^\circ$).

### B. `Moderate` Attention (Semi-Attentive / Ambient / Transition)
- **Primary Visual Indicators:**
  - **Ambient Listening:** Student is sitting upright or resting chin on hand while looking slightly away from the instructor (e.g., staring out of the window or at the ceiling for short intervals while listening).
  - **Passive Reading / Transitioning:** Looking down at a notebook or tablet without active pen movement; turning pages; adjusting writing utensils; transitioning between looking at the board and looking down.
  - **Mild Fatigue:** Slow blinking or occasional yawning while still maintaining a generally upright head posture ($|\text{yaw}| \le 45^\circ$, $|\text{pitch}| \le 35^\circ$).
  - **Brief Distraction:** Looking at a neighboring student's desk or an ambient classroom object for under 2 seconds before returning gaze to the front.

### C. `Low` Attention (Inattentive / Disengaged)
- **Primary Visual Indicators:**
  - **Extreme Head Aversion:** Head clearly turned sideways away from instruction ($|\text{yaw}| > 45^\circ$) for sustained periods, looking at the floor, back of the room, or out the window.
  - **Off-Task Posture:** Head resting completely on desk or arms; slouching heavily backward with eyes half-closed or closed (sleeping).
  - **Off-Task Behavior:** Active non-instructional cell phone usage below desk level; private whispered conversation with a peer while teacher is speaking; doodling or fidgeting with personal items.
  - **Prolonged Eye Closure:** Eyes closed continuously for more than 1 second (micro-sleep / dozing off).

---

## 3. Resolving Edge Cases & Ambiguities

| Classroom Scenario | Observed Cues | Correct Annotation | Rationale |
| :--- | :--- | :---: | :--- |
| **Writing notes vs. Doodling** | Head down, hand moving on paper | **`High`** if movement matches lecture pace; **`Low`** if accompanied by slouching, laughing, or off-task items. | In a lecture, note-taking requires looking down; penalizing downward head pose indiscriminately causes false `Low` classifications. |
| **Resting chin on hand / elbow on desk** | Head upright, eyes toward board | **`Moderate`** | Indicates partial fatigue, but student remains visually receptive to instruction. |
| **Looking at neighbor's notes** | Head turned toward adjacent peer's desk ($< 2$s) | **`Moderate`** | Ambient peer check / catching up on missed lecture points. |
| **Peer conversation** | Mutual head turning, continuous talking | **`Low`** | Off-task social interaction during a formal lecture. |
| **Turning to board behind student** | Student swivels 180° to look at back screen | **`High`** | Direction matches secondary instructional display. |

---

## 4. CVAT Annotation Protocol & Quality Control

### A. Bounding Box Framing Standards
1. **Extent:** Bounding box must enclose the student from the top of the head down to the visible lower torso / desk edge. Include raised hands.
2. **Tightness:** Avoid including empty chairs, background walls, or adjacent students in the box.
3. **Occlusions:** If a student is partially occluded by a desk or another student, draw the box around the **visible body area** and mark the `occluded` checkbox.
4. **Out of Frame:** If a student leaves the room, mark the box as `outside`.

### B. Track-Based Interpolation Workflow
1. Create a single `track` per student on frame 0.
2. Lock track ID from frame 0 through 299 (do not delete or recreate tracks midway).
3. Set keyframes only when student position or attention state changes. CVAT will automatically interpolate bounding boxes linearly between keyframes.
4. When an attention state changes (e.g. from `High` to `Low`), set a keyframe and update the `Attention` radio attribute.

### C. Dual-Annotator Consensus Procedure
To ensure scientific reliability:
1. **Primary Annotator (`Annotator A`):** Annotates 100% of the clip (all 21 students across 300 frames).
2. **Verification Annotator (`Annotator B`):** Independently annotates a random 20% sample of students (e.g., 4 students across all 300 frames = 1,200 observations).
3. **Inter-Annotator Agreement (IAA):** Compute Cohen's Kappa ($\kappa$). If $\kappa < 0.75$, both annotators convene to review and resolve discrepancies until consensus is reached.

---

## 5. Export and Format Verification
Export annotations from CVAT using the **CVAT for video 1.1 (XML)** format. The export should be saved to `Anno/<clip_id>_annotations.xml` and ingested using `scripts/ingest_cvat_annotations.py`.
