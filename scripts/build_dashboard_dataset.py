"""Build the consolidated dashboard dataset from real run artifacts.

Every metric surfaced in the dashboard is sourced from a file on disk:
- ``runs/reports/<video>_metrics.json`` for ground-truth evaluations
- ``runs/reports/<video>*_accuracy.txt`` for heuristic-agreement evaluations

Nothing is hardcoded, so the dashboard cannot silently disagree with the
audit trail.

Source artifacts are checked for staleness before anything is written. An
accuracy report that predates the scores file it describes means the pipeline
was re-tuned after the report was written, so the report describes a model
that no longer exists. Rather than silently serving those stale numbers, the
build fails loudly and names the files to regenerate.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = WORKSPACE_ROOT / "runs"
REPORTS_DIR = RUNS_DIR / "reports"

# Ground-truth provenance per benchmark. "human" means the comparison is
# against human CVAT annotation; "heuristic" means it is an internal
# consistency check against the rubric and is NOT a supervised benchmark.
GROUND_TRUTH_KIND = {
    "clip_001": "human",
    "clip_007": "human",
    "test_4": "heuristic",
    "test_5": "heuristic",
    "test_6": "heuristic",
    "test_7": "heuristic",
    "video_project_3": "heuristic",
}

_REPORT_STEMS = {
    "clip_001": "clip_001",
    "clip_007": "clip_007",
    "test_4": "test 4",
    "test_5": "Test 5",
    "test_6": "Test 6",
    "test_7": "Test 7",
    "video_project_3": "Video Project 3",
}

VIDEO_METADATA = {
    "clip_001": {
        "title": "Clip 001",
        "subtitle": "CVAT human annotation benchmark",
        "resolution": "832x464",
        "fps": 30.0,
    },
    "clip_007": {
        "title": "Clip 007",
        "subtitle": "CVAT manual bounding-box benchmark",
        "resolution": "832x464",
        "fps": 30.0,
    },
    "test_4": {
        "title": "Test 4",
        "subtitle": "Multi-row lecture hall, posture & active study",
        "resolution": "1920x1080",
        "fps": 30.0,
    },
    "test_5": {
        "title": "Test 5",
        "subtitle": "Seated classroom phone detection",
        "resolution": "1280x720",
        "fps": 24.0,
    },
    "test_6": {
        "title": "Test 6",
        "subtitle": "Phone & reading disentanglement",
        "resolution": "1280x720",
        "fps": 24.0,
    },
    "test_7": {
        "title": "Test 7",
        "subtitle": "92-student multi-row lecture audit",
        "resolution": "1920x1080",
        "fps": 30.0,
    },
    "video_project_3": {
        "title": "Video Project 3",
        "subtitle": "Dense multi-row classroom",
        "resolution": "1920x1080",
        "fps": 30.0,
    },
}

_BEHAVIOR_ORDER = [
    "Reading / Writing",
    "Oriented / Listening",
    "Hand Raising",
    "Peer Interaction",
    "Using Phone",
    "Sleeping / Fatigue",
]

_ACC_PATTERNS = {
    "agreement": re.compile(r"VIDEO HEURISTIC AGREEMENT:\s*([\d.]+)%"),
    "students_compared": re.compile(r"Students compared\s*:\s*(\d+)"),
    "frames_scored": re.compile(r"Frames scored\s*:\s*(\d+)"),
}
_BAND_PATTERN = re.compile(r"^(HIGH|MODERATE|LOW)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)", re.M)


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Recompute a full metric block from a confusion matrix.

    True Macro-F1 is the mean of the per-class F1 scores. The pipeline's
    legacy ``macro_f1`` field was positive-class F1 only, so it is never
    reused here.
    """
    n = tn + fp + fn + tp
    if n == 0:
        return {}
    prec_low = tn / (tn + fn) if (tn + fn) else 0.0
    rec_low = tn / (tn + fp) if (tn + fp) else 0.0
    prec_high = tp / (tp + fp) if (tp + fp) else 0.0
    rec_high = tp / (tp + fn) if (tp + fn) else 0.0
    f1_low = _f1(prec_low, rec_low)
    f1_high = _f1(prec_high, rec_high)
    return {
        "n_samples": n,
        "accuracy": round((tp + tn) / n * 100.0, 2),
        "balanced_accuracy": round((rec_low + rec_high) / 2 * 100.0, 2),
        "macro_f1": round((f1_low + f1_high) / 2 * 100.0, 2),
        "f1_low": round(f1_low * 100.0, 2),
        "f1_high": round(f1_high * 100.0, 2),
        "precision_low": round(prec_low * 100.0, 2),
        "recall_low": round(rec_low * 100.0, 2),
        "precision_high": round(prec_high * 100.0, 2),
        "recall_high": round(rec_high * 100.0, 2),
        "sensitivity": round(rec_high * 100.0, 2),
        "specificity": round(rec_low * 100.0, 2),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def latest_accuracy_report_path(stem: str) -> Path | None:
    """The most recent ``*_accuracy.txt`` for a video, if any exists."""
    files = sorted(
        REPORTS_DIR.glob(f"{stem}*_accuracy.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def find_stale_sources(files: dict[str, str]) -> list[str]:
    """Report source artifacts that are older than the scores they describe.

    A scores file is rewritten whenever the pipeline is re-run. An accuracy
    report derived from it must therefore be at least as new, otherwise the
    dashboard would present metrics from a previous version of the model.
    """
    problems: list[str] = []
    for key, filename in files.items():
        scores = RUNS_DIR / filename
        if not scores.exists():
            problems.append(f"{key}: scores file missing ({filename})")
            continue

        stem = _REPORT_STEMS[key]
        report = latest_accuracy_report_path(stem)
        if report is None:
            problems.append(f"{key}: no accuracy report for '{stem}'")
            continue

        if report.stat().st_mtime < scores.stat().st_mtime:
            problems.append(
                f"{key}: accuracy report '{report.name}' is older than "
                f"'{filename}' - it describes an older pipeline run"
            )
    return problems


def read_latest_accuracy_report(stem: str) -> dict | None:
    """Parse the most recent ``*_accuracy.txt`` for a video."""
    files = sorted(
        REPORTS_DIR.glob(f"{stem}*_accuracy.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    text = files[0].read_text(encoding="utf-8", errors="replace")
    out: dict = {"source_file": files[0].name}
    for key, pattern in _ACC_PATTERNS.items():
        m = pattern.search(text)
        if m:
            out[key] = float(m.group(1)) if key == "agreement" else int(m.group(1))
    bands = {}
    for band, prec, rec, f1, support in _BAND_PATTERN.findall(text):
        bands[band] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }
    if bands:
        out["bands"] = bands
    return out


def read_metrics_json(stem: str) -> dict:
    path = REPORTS_DIR / f"{stem}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_canonical_audit_table() -> dict:
    """Read the audit's canonical metric table.

    ``audit/METRIC_RECALCULATION.csv`` is the delivered, defense-grade
    recalculation and is the authoritative source for the benchmarks it
    covers. Values are keyed by video so the dashboard shows exactly the
    numbers in the audit rather than a raw log that may predate it.
    """
    path = WORKSPACE_ROOT / "audit" / "METRIC_RECALCULATION.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    table: dict = {}
    for row in rows:
        name = row["Evaluation_Set"]
        if "Clip 001" in name:
            key = "clip_001"
        elif "Clip 007" in name:
            key = "clip_007"
        elif "Video Project 3" in name:
            key = "video_project_3"
        elif name.startswith("Test 4"):
            key = "test_4"
        elif "Held-Out Validation" in name:
            key = "__validation__"
        else:
            continue
        # The calibrated and default validation rows carry identical figures.
        if key in table:
            continue
        table[key] = row
    return table


def build_evaluation(key: str, canonical: dict) -> dict:
    """Assemble the evaluation block for one video from real artifacts."""
    kind = GROUND_TRUTH_KIND[key]
    report = read_latest_accuracy_report(_REPORT_STEMS[key])
    mj = read_metrics_json(key)

    block = {
        "kind": kind,
        "accuracy": None,
        "macro_f1": None,
        "balanced_accuracy": None,
        "n_samples": None,
        "confusion": None,
        "source": None,
    }

    # Prefer the audit's canonical row when this video is covered by it.
    row = canonical.get(key)
    if row:
        tn, fp, fn, tp = (int(row[c]) for c in ("TN", "FP", "FN", "TP"))
        computed = metrics_from_confusion(tn, fp, fn, tp)
        block.update(
            {
                "accuracy": float(row["Accuracy_Pct"]),
                "macro_f1": float(row["True_Macro_F1_Pct"]),
                "balanced_accuracy": float(row["Balanced_Acc_Pct"]),
                "n_samples": int(row["N_Samples"]),
                "confusion": computed,
                "source": "Human CVAT annotation" if kind == "human" else "Heuristic rubric agreement",
                "source_detail": row["Annotation_Source"],
                "matched_observations": int(row["N_Samples"]),
                "provenance": "audit/METRIC_RECALCULATION.csv",
            }
        )
        return block

    if kind == "human":
        gt = mj.get("ground_truth_evaluation") or {}
        cm = gt.get("confusion_matrix")
        if cm:
            computed = metrics_from_confusion(cm["TN"], cm["FP"], cm["FN"], cm["TP"])
            block.update(
                {
                    "accuracy": computed["accuracy"],
                    "macro_f1": computed["macro_f1"],
                    "balanced_accuracy": computed["balanced_accuracy"],
                    "n_samples": computed["n_samples"],
                    "confusion": computed,
                    "source": "Human CVAT annotation",
                    "source_detail": gt.get("dataset"),
                    "matched_observations": gt.get("matched_observations"),
                    "provenance": f"runs/reports/{key}_metrics.json",
                }
            )
            return block

    # Heuristic consistency check (no human ground truth exists for this clip).
    if report and "agreement" in report:
        bands = report.get("bands", {})
        high = bands.get("HIGH", {})
        low = bands.get("LOW", {})
        prec_h, rec_h = high.get("precision", 0.0), high.get("recall", 0.0)
        prec_l, rec_l = low.get("precision", 0.0), low.get("recall", 0.0)
        f1_h, f1_l = high.get("f1", 0.0), low.get("f1", 0.0)
        support = sum(b.get("support", 0) for b in bands.values())
        block.update(
            {
                "accuracy": round(report["agreement"], 2),
                "macro_f1": round((f1_h + f1_l) / 2, 2) if (f1_h or f1_l) else None,
                "balanced_accuracy": round((rec_h + rec_l) / 2, 2) if (rec_h or rec_l) else None,
                "n_samples": support or report.get("students_compared"),
                "confusion": {
                    "precision_high": prec_h,
                    "recall_high": rec_h,
                    "precision_low": prec_l,
                    "recall_low": rec_l,
                    "f1_high": f1_h,
                    "f1_low": f1_l,
                },
                "source": "Heuristic rubric agreement",
                "source_detail": "No human ground truth for this clip",
                "bands": bands,
                "frames_scored": report.get("frames_scored"),
                "source_file": report.get("source_file"),
                "provenance": f"runs/reports/{report.get('source_file', '')}",
            }
        )
    return block


def classify_cue(cue: str) -> str:
    c = (cue or "").lower()
    if "phone" in c:
        return "Using Phone"
    if "read" in c or "writ" in c or "note" in c:
        return "Reading / Writing"
    if "sleep" in c or "table" in c or "collapsed" in c:
        return "Sleeping / Fatigue"
    if "talk" in c or "disturb" in c or "peer" in c:
        return "Peer Interaction"
    if "hand" in c:
        return "Hand Raising"
    return "Oriented / Listening"


def parse_scores_file(key: str, filename: str, canonical: dict):
    filepath = RUNS_DIR / filename
    if not filepath.exists():
        return None

    meta = VIDEO_METADATA[key]
    student_data = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "scores": [],
            "cues": defaultdict(int),
            "timeline": [],
            "observations": 0,
        }
    )

    timeline = []
    behavior_counts = defaultdict(int)
    all_scores = []
    last_timestamp = 0.0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t = float(row.get("timestamp", 0.0))
            last_timestamp = max(last_timestamp, t)
            frame_scores = []
            high_count = 0
            low_count = 0

            for s in row.get("students", []):
                sid = s.get("student_id")
                bbox = s.get("bbox")
                score = s.get("score")
                cue = s.get("cue", "oriented")

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    student_data[sid]["x"].append((x1 + x2) / 2.0)
                    student_data[sid]["y"].append((y1 + y2) / 2.0)
                    student_data[sid]["w"].append(x2 - x1)
                    student_data[sid]["h"].append(y2 - y1)

                student_data[sid]["observations"] += 1

                if score is not None:
                    sc = float(score)
                    frame_scores.append(sc)
                    all_scores.append(sc)
                    student_data[sid]["scores"].append(sc)
                    student_data[sid]["timeline"].append(
                        {"t": round(t, 2), "score": round(sc, 1), "cue": cue}
                    )
                    if sc >= 50.0:
                        high_count += 1
                    else:
                        low_count += 1

                student_data[sid]["cues"][cue] += 1
                behavior_counts[classify_cue(cue)] += 1

            if frame_scores:
                timeline.append(
                    {
                        "t": round(t, 2),
                        "mean_score": round(float(np.mean(frame_scores)), 1),
                        "high": high_count,
                        "low": low_count,
                    }
                )

    students_list = []
    for sid, d in student_data.items():
        if not d["scores"]:
            continue
        sc_arr = np.array(d["scores"])
        top_cue = max(d["cues"].items(), key=lambda x: x[1])[0]
        att_pct = float(np.sum(sc_arr >= 50.0)) / len(sc_arr) * 100.0

        sub_timeline = d["timeline"]
        if len(sub_timeline) > 40:
            indices = np.linspace(0, len(sub_timeline) - 1, 40, dtype=int)
            sub_timeline = [sub_timeline[i] for i in indices]

        students_list.append(
            {
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
                "behavior": classify_cue(top_cue),
                "observations": d["observations"],
                "status": "Attentive (HIGH)" if np.mean(sc_arr) >= 50.0 else "Distracted (LOW)",
                "timeline": sub_timeline,
            }
        )

    students_list.sort(key=lambda s: s["id"])

    if len(timeline) > 60:
        indices = np.linspace(0, len(timeline) - 1, 60, dtype=int)
        timeline = [timeline[i] for i in indices]

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

            drop_events.append(
                {
                    "student_id": s["id"],
                    "timestamp": round(s["timeline"][len(s["timeline"]) // 2]["t"], 1)
                    if s["timeline"]
                    else 4.5,
                    "score": s["min_score"],
                    "kind": "sudden" if s["min_score"] < 35 else "sustained",
                    "cue": s["top_cue"],
                    "behavior": s["behavior"],
                    "recommendation": rec,
                    "suggestion": sug,
                }
            )

    drop_events.sort(key=lambda e: e["timestamp"])

    total_stu = len(students_list)
    high_stu = sum(1 for s in students_list if s["mean_score"] >= 50.0)
    low_stu = total_stu - high_stu

    ordered_behaviors = {
        k: behavior_counts.get(k, 0) for k in _BEHAVIOR_ORDER if behavior_counts.get(k, 0) > 0
    }

    return {
        "key": key,
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "resolution": meta["resolution"],
        "fps": meta["fps"],
        "duration_s": round(last_timestamp, 2),
        "metrics": {
            "class_mean_score": round(float(np.mean(all_scores)), 1) if all_scores else 0.0,
            "total_students": total_stu,
            "attentive_students": high_stu,
            "distracted_students": low_stu,
            "attentive_rate_pct": round(high_stu / max(total_stu, 1) * 100.0, 1),
        },
        "evaluation": build_evaluation(key, canonical),
        "timeline": timeline,
        "behavior_distribution": ordered_behaviors,
        "students": students_list,
        "drop_events": drop_events,
    }


def main():
    files = {
        "test_6": "test_6_scores.jsonl",
        "video_project_3": "video_project_3_scores.jsonl",
        "test_5": "test_5_scores.jsonl",
        "clip_001": "clip_001_scores.jsonl",
        "clip_007": "clip_007_scores.jsonl",
        "test_4": "test_4_scores.jsonl",
        "test_7": "test_7_scores.jsonl",
    }

    dataset = {}
    canonical = read_canonical_audit_table()

    stale = find_stale_sources(files)
    if stale:
        print("STALE SOURCE ARTIFACTS DETECTED:")
        for problem in stale:
            print(f"  - {problem}")
        print(
            "\nThe accuracy reports describe an older pipeline run than the scores "
            "files on disk.\nRegenerate them, e.g.:\n"
            "    python -c \"import sys; sys.path.insert(0, 'src'); "
            "from attention_pipeline.video_accuracy import write_video_accuracy; "
            "from pathlib import Path; "
            "[write_video_accuracy(Path('runs/<video>.mp4'), "
            "Path('runs/<video>_overlay.mp4'), Path('runs/<video>_scores.jsonl')) "
            "for v in [...]]\"\n"
            "then re-run scripts/build_dashboard_dataset.py."
        )
        raise SystemExit(1)

    for key, fn in files.items():
        data = parse_scores_file(key, fn, canonical)
        if data:
            dataset[key] = data
            ev = data["evaluation"]
            acc = ev.get("accuracy")
            acc_txt = f"{acc}%" if acc is not None else "n/a"
            print(
                f"Parsed {key}: {data['metrics']['total_students']} students, "
                f"mean={data['metrics']['class_mean_score']}, "
                f"eval={ev['kind']} acc={acc_txt} macroF1={ev.get('macro_f1')}"
            )

    out_file = WORKSPACE_ROOT / "dashboard" / "dashboard_dataset.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved consolidated dataset: {out_file.resolve()} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
