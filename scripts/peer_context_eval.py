"""Peer-context scoring (idea from "Context Matters", CVPR-W 2026, arXiv 2601.06394).

A behaviour means different things depending on what the class is doing:
writing while classmates write is engagement; doing something different from
everybody else is a warning sign. For each student-frame, the SCB behaviour
probabilities are compared with the MEAN behaviour of the other students in
the same frame (cosine similarity). Label-free.

Variants (strict leave-one-classroom-out; anything learned is fitted on the
two training classrooms only, cut-off chosen there too):
  current final          motion + study protection + phone-on-crop
  + peer rule            if behaviour is clearly unlike the class (similarity
                         in the class's lowest 20%), score = min(score, 45)
  learned                logistic on [motion z, peer similarity, phone], then
                         the same study / phone rules

    python scripts/peer_context_eval.py
Writes reports/improvement/peer_context.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ROOMS = ["classroom_01", "room_B", "room_C"]


def peer_similarity(scb, video, t):
    sim = np.full(len(scb), np.nan)
    groups = defaultdict(list)
    for i, (v, tt) in enumerate(zip(video, t)):
        groups[(v, round(float(tt), 2))].append(i)
    for idx in groups.values():
        if len(idx) < 3:
            continue
        P = scb[idx]
        total = P.sum(0)
        for j, i in enumerate(idx):
            peer = (total - P[j]) / (len(idx) - 1)
            sim[i] = float(P[j] @ peer / (np.linalg.norm(P[j]) * np.linalg.norm(peer) + 1e-9))
    return sim


def per_video_pct(x, video):
    """Rank of each value within its own video (0..1); label-free normalisation."""
    out = np.full(len(x), 0.5)
    for v in np.unique(video):
        m = (video == v) & ~np.isnan(x)
        r = x[m].argsort().argsort()
        out[m] = r / max(1, m.sum() - 1)
    return out


def strict(y, room, score):
    grid = np.arange(25, 75.5, 0.5)
    per = {}
    for r in ROOMS:
        cut = max(grid, key=lambda c: np.mean([balanced_accuracy_score(y[room == o], (score[room == o] >= c).astype(int))
                                               for o in ROOMS if o != r]))
        k = room == r
        per[r] = {"balacc": round(float(balanced_accuracy_score(y[k], (score[k] >= cut).astype(int))), 4),
                  "auc": round(float(roc_auc_score(y[k], score[k])), 4)}
    return per, {m: round(float(np.mean([v[m] for v in per.values()])), 4) for m in ("balacc", "auc")}


def main():
    d = np.load(ROOT / "data/processed/pipeline_components.npz", allow_pickle=True)
    y, room, video, t = d["label"].astype(int), d["room"], d["video"], d["t"].astype(float)
    m = d["motion"].astype(float)
    scb = np.stack(d["scb"]).astype(float)
    study = scb[:, :3].max(1)
    phone = np.load(ROOT / "data/processed/phone_crop_conf.npy")

    def rules(s):
        s = np.where(study >= 0.3, np.maximum(s, 50.001), s)
        return np.where(phone >= 0.25, np.minimum(s, 30.0), s)

    sim = peer_similarity(scb, video, t)
    simp = per_video_pct(sim, video)
    res = {"peer_similarity_auc": {}, "variants": {}}
    print("peer-similarity AUC per classroom (> 0.5: behaving like the class goes with HIGH)")
    for r in ROOMS:
        k = (room == r) & ~np.isnan(sim)
        res["peer_similarity_auc"][r] = round(float(roc_auc_score(y[k], sim[k])), 3)
    print("  ", res["peer_similarity_auc"])

    base = rules(m)
    variants = {"current final": base}
    for q in (0.1, 0.2, 0.3):
        variants[f"+ peer rule (lowest {int(q * 100)}% -> <= 45)"] = np.where(simp <= q, np.minimum(base, 45.0), base)
    # learned combination, fitted on training rooms only, per fold
    z = np.log(np.maximum(m, 1.0))
    F = np.column_stack([z, simp, np.nan_to_num(phone)])
    learned = np.zeros(len(y))
    for r in ROOMS:
        tr, te = room != r, room == r
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(F[tr], y[tr])
        learned[te] = clf.predict_proba(F[te])[:, 1] * 100
    variants["learned: motion + peer + phone (per fold)"] = rules(learned)
    for name, s in variants.items():
        per, mean = strict(y, room, s)
        res["variants"][name] = {"per_room": per, "mean": mean}
        print(f"{name:45s} strict balacc {mean['balacc']:.3f} AUC {mean['auc']:.3f} | " +
              " ".join(f"{r}:{v['balacc']:.2f}" for r, v in per.items()))
    (ROOT / "reports/improvement/peer_context.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
