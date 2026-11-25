"""Which datasets help? Leave-one-classroom-out over live-pipeline components.

Components (from scripts/collect_components.py):
  motion  - class-relative motion            (own classroom data)
  daisee  - DAiSEE-only network per student   (DAiSEE) + its 4 ratings
  scb     - SCB behaviour probabilities       (SCB-dataset)
  neural  - classroom-fine-tuned BiGRU        (DAiSEE init + classroom labels)
  rubric  - rule-based rubric

Each component is z-scored within its own video (label-free), a logistic
regression is trained on two classrooms, the threshold is chosen on those
training classrooms, and the third classroom is the test. Reported: mean over
the three unseen rooms. The winning combination is fitted on all rooms and
saved to reports/improvement/ensemble_best.json.

    python scripts/ensemble_eval.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def zvid(F, video):
    G = F.astype(float).copy()
    for v in np.unique(video):
        m = video == v
        G[m] = (G[m] - G[m].mean(0)) / (G[m].std(0) + 1e-6)
    return G


def best_thr(y, p):
    grid = np.quantile(p, np.linspace(0.02, 0.98, 49))
    return float(grid[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def loro(F, y, room):
    per = {}
    for r in sorted(np.unique(room)):
        te, tr = room == r, room != r
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(F[tr], y[tr])
        t = best_thr(y[tr], clf.predict_proba(F[tr])[:, 1])
        p = clf.predict_proba(F[te])[:, 1]
        pr = (p >= t).astype(int)
        per[r] = {"balacc": balanced_accuracy_score(y[te], pr), "acc": accuracy_score(y[te], pr),
                  "macro_f1": f1_score(y[te], pr, average="macro"), "auc": roc_auc_score(y[te], p),
                  "low_recall": float(((pr == 0) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())),
                  "always_high": float(y[te].mean())}
    per = {r: {k: round(float(v), 4) for k, v in m.items()} for r, m in per.items()}
    mean = {k: round(float(np.mean([m[k] for m in per.values()])), 4) for k in per[next(iter(per))]}
    return per, mean


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, video = d["label"].astype(int), d["room"], d["video"]
    blocks = {
        "motion": d["motion"][:, None],
        "daisee": np.concatenate([d["daisee"][:, None], np.stack(d["daisee_aux"])], 1),
        "scb": np.stack(d["scb"]).astype(float),
        "neural": np.nan_to_num(d["neural"], nan=50.0)[:, None],
        "rubric": d["rubric"][:, None],
    }
    Z = {k: zvid(v, video) for k, v in blocks.items()}
    print(f"samples {len(y)}  LOW share {(y == 0).mean():.3f}\n")

    print("single-component AUC per classroom (>0.5: high value goes with HIGH attention)")
    for k, v in Z.items():
        print(f"  {k:7s}", "  ".join(f"{r}:{roc_auc_score(y[room == r], v[room == r, 0]):.2f}" for r in np.unique(room)))
    print()

    results = {}
    names = list(Z)
    for n in range(1, len(names) + 1):
        for combo in itertools.combinations(names, n):
            F = np.concatenate([Z[c] for c in combo], 1)
            per, mean = loro(F, y, room)
            results["+".join(combo)] = {"per_room": per, "mean": mean}
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["mean"]["balacc"])
    print(f"{'components':40s} balacc   AUC   acc  (mean over unseen rooms)")
    for k, v in ranked[:12]:
        m = v["mean"]
        print(f"{k:40s} {m['balacc']:.3f}  {m['auc']:.3f} {m['acc']:.3f}  | " +
              " ".join(f"{r}:{p['balacc']:.2f}" for r, p in v["per_room"].items()))
    for k in ("motion", "daisee", "scb", "neural", "rubric"):
        m = results[k]["mean"]
        print(f"single {k:8s} balacc {m['balacc']:.3f} AUC {m['auc']:.3f}")

    best = ranked[0][0]
    comps = best.split("+")
    F = np.concatenate([Z[c] for c in comps], 1)
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(F, y)
    thr = float(np.mean([best_thr(y[room != r], LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
                                  .fit(F[room != r], y[room != r]).predict_proba(F[room != r])[:, 1])
                         for r in np.unique(room)]))
    widths = {c: Z[c].shape[1] for c in comps}
    out = {"components": comps, "widths": widths, "coef": clf.coef_[0].round(5).tolist(),
           "intercept": float(clf.intercept_[0]), "threshold": thr,
           "expected_unseen_classroom": results[best]["mean"], "per_unseen_room": results[best]["per_room"],
           "note": "inputs are z-scored within the video (live: running class mean/std)"}
    (ROOT / "reports/improvement/ensemble_best.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "reports/improvement/ensemble.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nbest: {best} -> reports/improvement/ensemble_best.json")
    print("weights:", dict(zip([f"{c}[{i}]" for c in comps for i in range(widths[c])], out["coef"])))


if __name__ == "__main__":
    main()
