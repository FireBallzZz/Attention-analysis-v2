import json
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE_ROOT = Path("Attention-analysis-model")
RUNS_DIR = WORKSPACE_ROOT / "runs"
REPORTS_DIR = RUNS_DIR / "reports"

ACCURACY_DATA = {
    "test_6": {"acc": 100.0, "f1": 100.0, "tn": 173, "fp": 0, "fn": 0, "tp": 555},
    "video_project_3": {"acc": 94.35, "f1": 96.63, "tn": 959, "fp": 244, "fn": 158, "tp": 5757},
    "test_5": {"acc": 92.07, "f1": 94.48, "tn": 292, "fp": 92, "fn": 4, "tp": 822},
    "clip_001": {"acc": 87.51, "f1": 93.33, "tn": 1, "fp": 88, "fn": 62, "tp": 1050},
    "clip_007": {"acc": 94.41, "f1": 97.12, "tn": 0, "fp": 0, "fn": 50, "tp": 844},
    "test_4": {"acc": 88.28, "f1": 91.02, "tn": 557, "fp": 122, "fn": 104, "tp": 1145},
    "test_7": {"acc": 92.75, "f1": 80.15, "tn": 340, "fp": 344, "fn": 33, "tp": 4481},
}

VIDEO_METADATA = {
    "test_6": {"title": "Test 6: Phone & Reading Disentanglement", "resolution": "1280x720", "fps": 24.0, "length_s": 10.0},
    "video_project_3": {"title": "Video Project 3: Dense Multi-Row Classroom (53 Students)", "resolution": "1920x1080", "fps": 30.0, "length_s": 22.5},
    "test_5": {"title": "Test 5: Seated Classroom Phone Detection", "resolution": "1280x720", "fps": 24.0, "length_s": 10.0},
    "clip_001": {"title": "Clip 001: CVAT XML Ground-Truth Benchmark", "resolution": "832x464", "fps": 30.0, "length_s": 10.0},
    "clip_007": {"title": "Clip 007: Manual Bounding Box CSV Benchmark", "resolution": "832x464", "fps": 30.0, "length_s": 10.0},
    "test_4": {"title": "Test 4: Lecture Hall Posture & Active Study", "resolution": "1920x1080", "fps": 30.0, "length_s": 10.0},
    "test_7": {"title": "Test 7: 92-Student Multi-Row Lecture Audit", "resolution": "1920x1080", "fps": 30.0, "length_s": 24.0},
}

def parse_scores_file(key, filename):
    filepath = RUNS_DIR / filename
    if not filepath.exists():
        return None

    meta = VIDEO_METADATA.get(key, {"title": key, "resolution": "1920x1080", "fps": 24.0, "length_s": 10.0})
    acc = ACCURACY_DATA.get(key, {"acc": 90.0, "f1": 90.0, "tn": 0, "fp": 0, "fn": 0, "tp": 0})

    student_data = defaultdict(lambda: {
        "x": [], "y": [], "w": [], "h": [],
        "scores": [], "cues": defaultdict(int),
        "timeline": []
    })
    
    timeline = []
    behavior_counts = defaultdict(int)
    all_scores = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            t = float(row.get("timestamp", 0.0))
            frame_scores = []
            high_count = 0
            low_count = 0
            
            for s in row.get("students", []):
                sid = s.get("student_id")
                bbox = s.get("bbox")
                score = s.get("score")
                cue = s.get("cue", "oriented")
                band = s.get("band", "UNKNOWN")
                
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    student_data[sid]["x"].append((x1 + x2) / 2.0)
                    student_data[sid]["y"].append((y1 + y2) / 2.0)
                    student_data[sid]["w"].append(x2 - x1)
                    student_data[sid]["h"].append(y2 - y1)
                
                if score is not None:
                    sc = float(score)
                    frame_scores.append(sc)
                    all_scores.append(sc)
                    student_data[sid]["scores"].append(sc)
                    student_data[sid]["timeline"].append({"t": round(t, 2), "score": round(sc, 1), "cue": cue})
                    if sc >= 50.0:
                        high_count += 1
                    else:
                        low_count += 1
                
                student_data[sid]["cues"][cue] += 1
                
                # Generalize behavior
                cue_lower = cue.lower()
                if "phone" in cue_lower:
                    behavior_counts["Using Phone"] += 1
                elif "read" in cue_lower or "writ" in cue_lower or "note" in cue_lower:
                    behavior_counts["Reading / Writing"] += 1
                elif "sleep" in cue_lower or "table" in cue_lower:
                    behavior_counts["Sleeping / Fatigue"] += 1
                elif "talk" in cue_lower or "disturb" in cue_lower:
                    behavior_counts["Peer Interaction"] += 1
                elif "hand" in cue_lower:
                    behavior_counts["Hand Raising"] += 1
                else:
                    behavior_counts["Oriented / Listening"] += 1

            if frame_scores:
                timeline.append({
                    "t": round(t, 2),
                    "mean_score": round(float(np.mean(frame_scores)), 1),
                    "high": high_count,
                    "low": low_count
                })

    # Student summary list
    students_list = []
    for sid, d in student_data.items():
        if not d["scores"]: continue
        sc_arr = np.array(d["scores"])
        top_cue = max(d["cues"].items(), key=lambda x: x[1])[0]
        att_pct = float(np.sum(sc_arr >= 50.0)) / len(sc_arr) * 100.0
        
        # Subsample timeline to max 40 points for fast rendering
        sub_timeline = d["timeline"]
        if len(sub_timeline) > 40:
            indices = np.linspace(0, len(sub_timeline) - 1, 40, dtype=int)
            sub_timeline = [sub_timeline[i] for i in indices]
            
        students_list.append({
            "id": sid,
            "x": round(float(np.median(d["x"])), 1) if d["x"] else 0,
            "y": round(float(np.median(d["y"])), 1) if d["y"] else 0,
            "w": round(float(np.median(d["w"])), 1) if d["w"] else 60,
            "h": round(float(np.median(d["h"])), 1) if d["h"] else 100,
            "mean_score": round(float(np.mean(sc_arr)), 1),
            "min_score": round(float(np.min(sc_arr)), 1),
            "max_score": round(float(np.max(sc_arr)), 1),
            "attentive_pct": round(att_pct, 1),
            "distracted_pct": round(100.0 - att_pct, 1),
            "top_cue": top_cue,
            "status": "Attentive (HIGH)" if np.mean(sc_arr) >= 50.0 else "Distracted (LOW)",
            "timeline": sub_timeline
        })

    students_list.sort(key=lambda s: s["id"])
    
    # Subsample class timeline
    if len(timeline) > 60:
        indices = np.linspace(0, len(timeline) - 1, 60, dtype=int)
        timeline = [timeline[i] for i in indices]

    # Generate realistic drop events
    drop_events = []
    for s in students_list:
        if s["mean_score"] < 50.0 or s["min_score"] < 40.0:
            top_c = s["top_cue"].lower()
            if "phone" in top_c:
                rec = "Apply non-verbal proximity control: circulate calmly down the aisle near the flagged desk without halting instructional delivery."
                sug = "Avoid public confrontation; private physical presence typically causes the student to stow the device naturally."
            elif "talk" in top_c or "disturb" in top_c:
                rec = "Pivot immediately into a structured 30-second Think-Pair-Share check to productively channel conversational energy into academic discussion."
                sug = "Re-anchor the classroom with a quick concept question rather than singling out individual conversationalists."
            elif "sleep" in top_c or "table" in top_c:
                rec = "Trigger a whole-class physical or conceptual energy reset: have all students stand for a quick stretch or conduct a rapid live poll."
                sug = "Discreetly check in with the student during the subsequent transition to screen for acute fatigue or wellness issues."
            else:
                rec = "Direct an inclusive, low-stakes retrieval question toward that sector of the room, or use a vocal inflection change to refocus eyes on the visual aid."
                sug = "Wait one brief observation cycle; isolated look-aways are normal cognitive breathers."
                
            drop_events.append({
                "student_id": s["id"],
                "timestamp": round(s["timeline"][len(s["timeline"])//2]["t"], 1) if s["timeline"] else 4.5,
                "score": s["min_score"],
                "kind": "sudden" if s["min_score"] < 35 else "sustained",
                "cue": s["top_cue"],
                "recommendation": rec,
                "suggestion": sug
            })
            
    drop_events.sort(key=lambda e: e["timestamp"])

    total_stu = len(students_list)
    high_stu = sum(1 for s in students_list if s["mean_score"] >= 50.0)
    low_stu = total_stu - high_stu

    return {
        "key": key,
        "title": meta["title"],
        "resolution": meta["resolution"],
        "fps": meta["fps"],
        "duration_s": meta["length_s"],
        "metrics": {
            "accuracy": acc["acc"],
            "macro_f1": acc["f1"],
            "class_mean_score": round(float(np.mean(all_scores)), 1) if all_scores else 0.0,
            "total_students": total_stu,
            "attentive_students": high_stu,
            "distracted_students": low_stu,
            "attentive_rate_pct": round(high_stu / max(total_stu, 1) * 100.0, 1),
            "confusion_matrix": acc
        },
        "timeline": timeline,
        "behavior_distribution": dict(behavior_counts),
        "students": students_list,
        "drop_events": drop_events
    }

def main():
    files = {
        "test_6": "test_6_scores.jsonl",
        "video_project_3": "video_project_3_scores.jsonl",
        "test_5": "test_5_scores.jsonl",
        "clip_001": "clip_001_scores.jsonl",
        "clip_007": "clip_007_scores.jsonl",
        "test_4": "test_4_scores.jsonl",
        "test_7": "test_7_scores.jsonl"
    }
    
    dataset = {}
    for key, fn in files.items():
        data = parse_scores_file(key, fn)
        if data:
            dataset[key] = data
            print(f"Parsed {key}: {data['metrics']['total_students']} students, Mean Score={data['metrics']['class_mean_score']}, Acc={data['metrics']['accuracy']}%")

    out_file = WORKSPACE_ROOT / "dashboard" / "dashboard_dataset.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved consolidated dataset: {out_file.resolve()} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
