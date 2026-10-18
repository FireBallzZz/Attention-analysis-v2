# Classroom attention annotations

## `clip_004_attention.csv`

Frame-by-frame student boxes and attention labels for `video/clip_004.mp4`.

### How students are chosen
1. Count real students first (must have a visible face and confidence ≥ 0.25).
2. Lock that count so empty wall / monitor / fan boxes are dropped.
3. For each kept box, measure eye gaze, head pose, and posture, then label attention.

| Column | Meaning |
|---|---|
| `frame` | Frame index, starting at 0 |
| `timestamp_sec` | Time in the clip |
| `student_id` | Track id for that person |
| `x1,y1,x2,y2` | Bounding box in pixels |
| `width,height` | Box size |
| `attention` | `HIGH`, `MODERATE`, or `LOW` |
| `score` | Attention score 0–100 |
| `cue` | Why that band was chosen |
| `eye_gaze_yaw`, `eye_gaze_pitch` | L2CS-Net eye gaze (degrees) |
| `head_yaw`, `head_pitch`, `head_roll` | Head posture (degrees) |
| `eye_aspect`, `mouth_aspect` | Eye / mouth openness |
| `posture`, `posture_confidence` | Behavior / posture class |
| `behavior` | Same posture class name |

Colors in `clip_004_annotated.mp4`: green = HIGH, orange = MODERATE, red = LOW.
