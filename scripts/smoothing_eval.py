"""Temporal smoothing of the final score (motion + study protection).

Each student's score is averaged over the last W seconds (causal - only the
past, as it would run live). W is chosen on the two training classrooms and
tested on the third (nested leave-one-classroom-out).

    python scripts/smoothing_eval.py
Writes reports/improvement/smoothing.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def smooth(score, video, track, t, w):
    if w <= 0:
        return score.copy()
    out = np.empty_like(score)
    groups = defaultdict(list)
    for i, (v, k) in enumerate(zip(video, track)):
        groups[(v, k)].append(i)
    for idx in groups.values():
        idx = np.array(idx)[np.argsort(t[idx])]
        tt, ss = t[idx], score[idx]
        cs = np.cumsum(ss)
        lo = np.searchsorted(tt, tt - w, side="left")
        n = np.arange(len(idx)) - lo + 1
        prev = np.where(lo > 0, cs[np.maximum(lo - 1, 0)], 0.0)
        out[idx] = (cs - prev) / n
    return out


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, video, track, t = d["label"].astype(int), d["room"], d["video"], d["track"], d["t"].astype(float)
    study = np.stack(d["scb"]).astype(float)[:, :3].max(1)
    base = np.where(study >= 0.3, np.maximum(d["motion"].astype(float), 50.0 + 1e-3), d["motion"].astype(float))
    S = {w: smooth(base, video, track, t, w) for w in WINDOWS}
    bal = lambda s, k: balanced_accuracy_score(y[k], (s[k] >= 50).astype(int))  # noqa: E731
    table = {w: {r: {"balacc": round(float(bal(S[w], room == r)), 4),
                     "auc": round(float(roc_auc_score(y[room == r], S[w][room == r])), 4)}
                 for r in np.unique(room)} for w in WINDOWS}
    for w in WINDOWS:
        m = np.mean([v["balacc"] for v in table[w].values()])
        print(f"W={w:>3}s  mean balacc {m:.3f}  AUC {np.mean([v['auc'] for v in table[w].values()]):.3f}  " +
              " ".join(f"{r}:{v['balacc']:.2f}" for r, v in table[w].items()))
    nested = {}
    for r in np.unique(room):
        others = [o for o in np.unique(room) if o != r]
        w_best = max(WINDOWS, key=lambda w: np.mean([table[w][o]["balacc"] for o in others]))
        nested[r] = {"chosen_window_s": w_best, **table[w_best][r]}
    mean = {k: round(float(np.mean([v[k] for v in nested.values()])), 4) for k in ("balacc", "auc")}
    print("\nnested (window chosen on the other two rooms):", nested, "\nmean", mean)
    out = {"grid": {str(w): v for w, v in table.items()}, "nested": nested, "nested_mean": mean}
    (ROOT / "reports/improvement/smoothing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
