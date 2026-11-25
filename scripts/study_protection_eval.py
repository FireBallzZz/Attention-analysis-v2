"""Why are reading / writing / watching students sometimes scored LOW, and does
"study protection" fix it?

Uses the live-pipeline components (scripts/collect_components.py). The motion
model scores movement relative to the class, and studying moves the hands and
head, so some attentive students fall below 50. Rule tested (fixed, nothing
fitted on labels): if the SCB detector's hand-raising / reading / writing
probability is >= t, the score may not drop below 50.

    python scripts/study_protection_eval.py
Writes reports/improvement/study_protection.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, m = d["label"].astype(int), d["room"], d["motion"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)
    hi = y == 1
    conf = study >= 0.3
    diag = {
        "attentive_scored_low_pct": round(float(((m < 50) & hi).sum() / hi.sum() * 100), 1),
        "confident_study_share_pct": round(float(conf.mean() * 100), 1),
        "confident_study_labelled_high_pct": round(float(y[conf].mean() * 100), 1),
        "all_labelled_high_pct": round(float(y.mean() * 100), 1),
        "confident_study_scored_low_by_motion_pct": round(float((m[conf] < 50).mean() * 100), 1),
    }
    print(diag)
    res = {"diagnosis": diag, "variants": {}}
    for t in (None, 0.3, 0.4, 0.5):
        per = {}
        for r in np.unique(room):
            k = room == r
            sc = m[k].copy()
            if t is not None:
                sc = np.where(study[k] >= t, np.maximum(sc, 50.0 + 1e-3), sc)
            per[r] = {"balacc": round(float(balanced_accuracy_score(y[k], (sc >= 50).astype(int))), 4),
                      "auc": round(float(roc_auc_score(y[k], sc)), 4)}
        name = "motion only" if t is None else f"protect if SCB study >= {t}"
        mean = {k: round(float(np.mean([v[k] for v in per.values()])), 4) for k in ("balacc", "auc")}
        res["variants"][name] = {"per_room": per, "mean": mean}
        print(f"{name:30s} balacc {mean['balacc']:.3f}  AUC {mean['auc']:.3f}  " +
              " ".join(f"{r}:{v['balacc']:.2f}" for r, v in per.items()))
    out = ROOT / "reports/improvement/study_protection.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
