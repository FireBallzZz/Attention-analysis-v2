"""Build the offline defence dashboard: dashboard/defence/{index.html, data.js, media/}.

Collects every honest result (person CV, cross-classroom, ablation, final
model), the demo-video outputs of the final pipeline (runs/defence/*.jsonl +
overlay videos) and the test-suite status into data.js. Videos are re-encoded
to H.264 so any browser plays them. Open dashboard/defence/index.html by
double-click; no server or internet needed.

    python scripts/build_defence_dashboard.py            # full build (runs pytest)
    python scripts/build_defence_dashboard.py --skip-tests
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "dashboard" / "defence"
MEDIA = OUT / "media"
REP = ROOT / "reports"

DEMOS = [  # (key, title, room, has human labels, original video)
    ("classroom_01_clip001", "Computer-lab classroom (clip 001)", "classroom_01", True, "runs/clip_001.mp4"),
    ("room_B_11", "Seminar room B (video 11)", "room_B", True, "runs/11.mp4"),
    ("room_B_non_attentive_1", "Seminar room B (non attentive 1)", "room_B", True, "runs/non attentive 1.mov"),
    ("room_C_test_8", "Classroom C (Test 8)", "room_C", True, "runs/Test 8.mp4"),
    ("lecture_hall_video_project_3", "Lecture hall (Video Project 3)", "unlabelled", False, "runs/Video Project 3.mp4"),
]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def r4(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


# ---------------------------------------------------------------- demo videos
def demo_payload(key: str, title: str, room: str, labelled: bool, source: str):
    from dashboard_payload import build_payload, clean_frame, transcode

    jl = ROOT / "runs" / "defence" / f"{key}_scores.jsonl"
    vid = ROOT / "runs" / "defence" / f"{key}_overlay.mp4"
    if not jl.exists() or not vid.exists():
        return None
    mp4 = MEDIA / f"{key}.mp4"
    if not mp4.exists() or mp4.stat().st_mtime < vid.stat().st_mtime:
        print(f"  transcoding {key} ...")
        transcode(vid, mp4)
    size = clean_frame(ROOT / source, MEDIA / f"{key}.jpg")
    return build_payload(key=key, title=title, room=room, labelled=labelled, jsonl=jl,
                         video_url=f"media/{key}.mp4", poster_url=f"media/{key}.jpg", frame_size=size)


# ---------------------------------------------------------------- analysis
def cue_flip_and_ablation():
    """Per-video single-feature AUC (shows camera-relative cues flip) and the
    leave-one-classroom-out logistic-regression feature ablation."""
    from sklearn.metrics import roc_auc_score

    from evaluate_cross_classroom import load_all
    from honest_evaluation import best_threshold, metrics, train_logreg, window_stats
    from relative_gaze_features import relative_to_baseline

    X, y, room, person, clip, info = load_all(False)
    flips = []
    names = {21: "Facing the camera", 0: "Head yaw", 4: "Gaze pitch", 22: "Torso lean"}
    for c in ["clip_004", "clip_001", "11", "non_attentive_1", "test_8"]:
        m = clip == c
        if len(np.unique(y[m])) < 2:
            continue
        mean = X[m].mean(1)
        flips.append({"video": c, **{names[k]: r4(roc_auc_score(y[m], mean[:, k])) for k in names}})

    def loro(XX):
        F = window_stats(XX)
        out = {}
        for r in sorted(np.unique(room)):
            te = room == r
            tr = ~te
            rng = np.random.default_rng(11)
            tp = np.unique(person[tr])
            inner = rng.choice(tp, int(len(tp) * 0.25), replace=False)
            mi = np.isin(person, inner) & tr
            f = train_logreg(F[tr & ~mi], y[tr & ~mi])
            t = best_threshold(y[mi], f(F[mi]))
            out[r] = r4(metrics(y[te], f(F[te]), t)["roc_auc"])
        out["mean"] = r4(np.mean([v for v in out.values()]))
        return out

    grp = np.array([f"{c}|{p}" for c, p in zip(clip, person)])
    Xn = X.copy()
    Xn[:, :, [0, 1, 2, 3, 4, 21]] = 0
    ablation = [
        {"variant": "Raw (relative to camera)", **loro(X)},
        {"variant": "Relative to the student's usual direction", **loro(relative_to_baseline(X, grp))},
        {"variant": "Relative to the whole class's direction", **loro(relative_to_baseline(X, clip.astype(str)))},
        {"variant": "Head/gaze direction removed", **loro(Xn)},
    ]
    return flips, ablation, info


def improvement_summary():
    """Settings A (new room, no setup) vs B (per-class calibration) and the identity check."""
    imp = REP / "improvement"
    if not (imp / "calibrated.json").exists():
        return None
    variants = []
    for f in ("features32", "crops", "head"):
        if (imp / f"{f}.json").exists():
            for k, v in load(imp / f"{f}.json").items():
                variants.append({"name": k, "balacc": v["pooled"]["mean_room_balacc"],
                                 "auc": v["pooled"]["mean_room_auc"]})
    variants.sort(key=lambda v: -v["balacc"])  # mean over unseen rooms (pooling flatters)
    cal = load(imp / "calibrated.json")
    def per_video_mean(entry):
        pv = entry["per_video"]
        mean = {f: round(float(np.mean([m[f] for m in pv.values()])), 4)
                for f in ("balacc", "acc", "macro_f1", "low_recall", "always_high")}
        mean["auc"] = round(float(np.nanmean([m["auc"] for m in pv.values()])), 4)
        return mean | {"per_video": {k: m["balacc"] for k, m in pv.items()}}

    pick = {"30%": "calibrate on first 30% of each video | classroom-behaviour crops (SCB)",
            "50%": "calibrate on first 50% of each video | SCB + head gaze"}
    calib = {k: per_video_mean(cal[v]) for k, v in pick.items() if v in cal}
    ident = load(imp / "identity_check.json") if (imp / "identity_check.json").exists() else {}
    seeds = load(imp / "seed_variability.json")["mean_std"] if (imp / "seed_variability.json").exists() else None
    dz = load(imp / "daisee_ablation.json") if (imp / "daisee_ablation.json").exists() else {}
    daisee = {k: {"mean_room_balacc": round(float(np.mean([m["balanced_accuracy"] for m in v["per_room"].values()])), 4),
                  "mean_room_auc": round(float(np.mean([m["roc_auc"] for m in v["per_room"].values()])), 4)}
              for k, v in dz.items()}
    ens = load(imp / "ensemble.json") if (imp / "ensemble.json").exists() else {}
    show = [("motion", "Motion (final)"), ("motion+rubric", "Motion + rubric"), ("motion+neural", "Motion + BiGRU"),
            ("neural", "BiGRU alone"), ("motion+daisee", "Motion + DAiSEE individual"),
            ("motion+scb", "Motion + SCB behaviour"), ("motion+daisee+scb", "Motion + DAiSEE + SCB"),
            ("daisee", "DAiSEE individual alone"), ("scb", "SCB behaviour alone")]
    datasets = [{"name": label, "balacc": ens[k]["mean"]["balacc"], "auc": ens[k]["mean"]["auc"]}
                for k, label in show if k in ens]
    sp = imp / "study_protection.json"
    if sp.exists():
        v = load(sp)["variants"].get("protect if SCB study >= 0.3")
        if v:
            datasets.insert(0, {"name": "Motion + study protection (final)", "balacc": v["mean"]["balacc"],
                                "auc": v["mean"]["auc"]})
            for row in datasets[1:]:
                row["name"] = row["name"].replace("Motion (final)", "Motion only")
    return {"n_variants": len(variants), "top": variants[:5], "calibrated": calib, "identity": ident,
            "bigru_seeds": seeds, "daisee": daisee, "datasets": datasets}


def final_model_summary():
    """Final model = class-relative motion + hard-evidence floor (scripts/motion_rule.py)."""
    rule = load(ROOT / "checkpoints" / "motion_rule_v1.json")
    mr = load(REP / "improvement" / "motion_rule.json")
    win = "1.6s" if rule.get("window", 16) == 16 else "3s"
    per = mr[win]["per_room"]
    unseen = mr[win]["mean_over_rooms"]
    live_p = REP / "final_model" / "pipeline_vs_human_floor.json"
    live = load(live_p) if live_p.exists() else {}
    return {
        "name": "Class-relative motion + hard-evidence floor",
        "checkpoint": "checkpoints/motion_rule_v1.json",
        "coef": {"a": round(rule["a"], 4), "b": round(rule["b"], 4), "threshold": round(rule["threshold"], 4)},
        "unseen": {"balanced_accuracy": unseen["balacc"], "accuracy": unseen["acc"], "macro_f1": unseen["macro_f1"],
                   "roc_auc": unseen["auc"], "low_recall": unseen["low_recall"],
                   "always_attentive_accuracy": unseen["always_high"]},
        "per_room": per,
        "live": {v: {k: m[k] for k in ("auc", "balanced_accuracy", "accuracy", "always_high", "matched")}
                 for v, m in live.items()},
        "live_mean": {k: round(float(np.mean([m[k] for m in live.values()])), 4)
                      for k in ("auc", "balanced_accuracy", "accuracy")} if live else None,
    }


def run_tests():
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    return {"passed": passed, "failed": failed, "summary": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)

    honest = load(REP / "honest_eval" / "results.json")
    cross = load(REP / "cross_classroom" / "results.json")
    cross_nd = load(REP / "cross_classroom_no_direction" / "results.json")
    final = load(REP / "final_model" / "training.json")

    print("analysis ...")
    flips, ablation, rooms = cue_flip_and_ablation()

    def room_rows(res):
        rows = []
        for r, v in res["per_room"].items():
            for name, key in (("BiGRU", "bigru"), ("Logistic regression", "logreg")):
                m = v[key]
                rows.append({"room": r, "model": name, "acc": r4(m["accuracy"]),
                             "balacc": r4(m["balanced_accuracy"]), "macro_f1": r4(m["macro_f1"]),
                             "low_recall": r4(m["low_recall"]), "auc": r4(m["roc_auc"]),
                             "auc_ci": m["ci95_person_bootstrap"]["roc_auc"],
                             "always_high": r4(m["always_attentive_accuracy"])})
            d = v["deployed_checkpoint_thr_0.50"]
            rows.append({"room": r, "model": "Old deployed model", "acc": r4(d["accuracy"]),
                         "balacc": r4(d["balanced_accuracy"]), "macro_f1": r4(d["macro_f1"]),
                         "low_recall": r4(d["low_recall"]), "auc": r4(d["roc_auc"]), "auc_ci": None,
                         "always_high": r4(d["always_attentive_accuracy"]),
                         "contaminated": r == "classroom_01"})
        return rows

    cv = honest["leave_persons_out_cv"]
    person_cv = []
    for key, name in (("always_attentive", "Always HIGH"), ("bigru_seed_ensemble", "BiGRU (old recipe)"),
                      ("logreg_window_stats", "Logistic regression")):
        m = cv[key]["pooled"]
        ci = cv[key].get("ci95_person_bootstrap", {})
        person_cv.append({"model": name, "acc": r4(m["accuracy"]), "balacc": r4(m["balanced_accuracy"]),
                          "macro_f1": r4(m["macro_f1"]), "low_recall": r4(m["low_recall"]),
                          "auc": r4(m["roc_auc"]), "auc_ci": ci.get("roc_auc")})

    print("demo videos ...")
    demos = [d for d in (demo_payload(*x) for x in DEMOS) if d]
    tests = {"passed": None, "failed": None, "summary": "not run"} if args.skip_tests else run_tests()
    print("tests:", tests["summary"])

    data = {
        "built": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tests": tests,
        "rooms": rooms,
        "human_windows": int(sum(v["windows"] for v in rooms.values())),
        "human_low": int(sum(v["low"] for v in rooms.values())),
        "final_model": final_model_summary(),
        "bigru_final": {
            "checkpoint": final["checkpoint"], "sha256": final["sha256"][:16],
            "threshold": final["threshold"], "masked": final["masked_channels"],
        },
        "old_claims": [
            {"claim": "Clip 001: 95.13% accuracy", "truth": "Always answering HIGH scores 95.22%; the model found 0 of 51 LOW cases",
             "fixed": "Macro-F1 48.75%, balanced accuracy 49.95%"},
            {"claim": "Clip 001: 97.50% macro-F1", "truth": "That was the F1 of the HIGH class only",
             "fixed": "True macro-F1 48.75%"},
            {"claim": "Clip 007: 80.65% accuracy on held-out data", "truth": "Clip 007 labels are the clip_004 training labels (data leakage)",
             "fixed": "Removed as a test set"},
            {"claim": "87.22% on 4,386 held-out students' sequences", "truth": "Matrix sums to 4,602; 2/3 of labels were made by an older model; same people under different IDs",
             "fixed": "Rebuilt with human labels and person/classroom splits"},
            {"claim": "Video Project 3: 90.22% vs ground truth", "truth": "No human labels exist; it was agreement with the rule-based rubric",
             "fixed": "Shown only as an unlabelled demo"},
            {"claim": "Threshold 0.45 'validated'", "truth": "Chosen on the human test annotation",
             "fixed": "Threshold chosen on held-out persons only"},
        ],
        "person_cv": person_cv,
        "person_cv_info": {"windows": honest["data"]["total_windows"], "persons": honest["data"]["persons"],
                           "persons_low": honest["data"]["persons_with_any_disengaged"],
                           "non_overlapping": honest["data"]["approx_non_overlapping_windows"]},
        "cross_all": room_rows(cross),
        "cross_final": room_rows(cross_nd),
        "pooled": {"all_channels": {k: r4(v) for k, v in cross["pooled_held_out"]["bigru"].items() if isinstance(v, (int, float))},
                   "no_direction": {k: r4(v) for k, v in cross_nd["pooled_held_out"]["bigru"].items() if isinstance(v, (int, float))}},
        "flips": flips,
        "improvement": improvement_summary(),
        "ablation": ablation,
        "demos": demos,
    }
    (OUT / "data.js").write_text("window.DASH = " + json.dumps(data, indent=1) + ";\n", encoding="utf-8")
    print(f"wrote {OUT / 'data.js'} ({len(demos)} demo videos)")


if __name__ == "__main__":
    main()
