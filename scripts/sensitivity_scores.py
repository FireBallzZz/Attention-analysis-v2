"""Sensitivity of accuracy to the score-level parameters (live pipeline outputs).

Uses data/processed/pipeline_components.npz. One parameter is changed at a time;
everything else stays at the deployed value. Reported per classroom and as the
mean over the three classrooms. This is a SENSITIVITY table: values that look
best here are not adopted unless they also win when chosen on training rooms only.

    python scripts/sensitivity_scores.py
Writes reports/improvement/sensitivity_scores.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent


def evaluate(y, room, score, cut=50.0):
    per = {}
    for r in np.unique(room):
        k = room == r
        p = (score[k] >= cut).astype(int)
        per[str(r)] = {"balacc": balanced_accuracy_score(y[k], p), "acc": accuracy_score(y[k], p),
                       "low_recall": float(((p == 0) & (y[k] == 0)).sum() / max(1, (y[k] == 0).sum()))}
    mean = {m: round(float(np.mean([v[m] for v in per.values()])), 4) for m in ("balacc", "acc", "low_recall")}
    return {"mean": mean, "per_room": {r: round(v["balacc"], 4) for r, v in per.items()}}


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room = d["label"].astype(int), d["room"]
    motion = d["motion"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)

    def final(prot=0.3, floor=50.0):
        if prot is None:
            return motion
        return np.where(study >= prot, np.maximum(motion, floor + 1e-3), motion)

    res = {"cutoff": {}, "study_threshold": {}, "study_floor": {}}
    base = final()
    print("1) HIGH/LOW cut-off (deployed: 50)")
    for c in range(30, 75, 5):
        res["cutoff"][c] = evaluate(y, room, base, c)
        m = res["cutoff"][c]["mean"]
        print(f"   cut {c:3d}: balacc {m['balacc']:.3f}  acc {m['acc']:.3f}  LOW caught {m['low_recall']:.3f}  rooms {res['cutoff'][c]['per_room']}")
    print("2) study-protection SCB threshold (deployed: 0.3)")
    for t in (None, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7):
        res["study_threshold"][str(t)] = evaluate(y, room, final(prot=t))
        m = res["study_threshold"][str(t)]["mean"]
        print(f"   {'off' if t is None else t:>4}: balacc {m['balacc']:.3f}  acc {m['acc']:.3f}  rooms {res['study_threshold'][str(t)]['per_room']}"
              f"  (protected {0 if t is None else (study >= t).mean() * 100:.1f}% of student-frames)")
    print("3) study-protection minimum score (deployed: 50)")
    for f in (45, 50, 55, 60, 70):
        res["study_floor"][f] = evaluate(y, room, final(floor=f))
        m = res["study_floor"][f]["mean"]
        print(f"   floor {f}: balacc {m['balacc']:.3f}  acc {m['acc']:.3f}  rooms {res['study_floor'][f]['per_room']}")
    (ROOT / "reports/improvement/sensitivity_scores.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
