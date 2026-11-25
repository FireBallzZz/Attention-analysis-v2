"""Train the final (defence) attention classifier on human labels only.

Choices, each made by the leave-one-classroom-out test in
scripts/evaluate_cross_classroom.py, never on a reported test score:
  * data: every human-labelled room (classroom_01, room_B, room_C); Test 6 is
    excluded until it is really labelled; no model-made labels;
  * input: camera-relative head/gaze channels (0,1,2,3,4,21) masked out,
    because their meaning flips between rooms (mean unseen-room AUC
    0.44 -> 0.54 when removed);
  * epoch and decision threshold: chosen on ~25% of persons held out from
    training (inner validation), never on a test room.

The expected performance on a NEW classroom is the leave-one-classroom-out
result (reports/cross_classroom_no_direction), not anything measured here.

    python scripts/train_final_model.py
Writes checkpoints/final_human_v1.pt and reports/final_model/training.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_cross_classroom import load_all  # noqa: E402
from honest_evaluation import best_threshold, metrics, predict, train_bigru  # noqa: E402

DIRECTION_CHANNELS = [0, 1, 2, 3, 4, 21]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--inner-val-frac", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/final_human_v1.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, room, person, clip, info = load_all(include_test6=False)
    mask = np.ones(X.shape[-1], dtype=np.float32)
    mask[DIRECTION_CHANNELS] = 0.0
    Xm = X * mask

    rng = np.random.default_rng(args.seed)
    persons = np.unique(person)
    for _ in range(500):
        inner = rng.choice(persons, size=int(len(persons) * args.inner_val_frac), replace=False)
        val = np.isin(person, inner)
        # inner validation must contain every room and enough LOW windows
        if (y[val] == 0).sum() >= 100 and set(room[val]) == set(room):
            break
    fit = ~val
    print(f"fit windows {fit.sum()} ({len(np.unique(person[fit]))} persons), "
          f"inner-val windows {val.sum()} ({len(inner)} persons)")

    model = train_bigru(Xm[fit], y[fit], Xm[val], y[val], device, seed=args.seed, epochs=args.epochs)
    p_val = predict(model, Xm[val], device)
    thr = best_threshold(y[val], p_val)
    val_m = metrics(y[val], p_val, thr)
    print(f"threshold {thr:.2f} | inner-val balanced acc {val_m['balanced_accuracy']:.3f} "
          f"AUC {val_m['roc_auc']:.3f} (optimistic: same rooms as training)")

    model.input_mask.copy_(torch.from_numpy(mask).to(device))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    loro = json.loads((ROOT / "reports/cross_classroom_no_direction/results.json").read_text())
    torch.save({
        "model": model.state_dict(),
        "input_mask": mask,
        "calibrated_threshold": thr,
        "architecture": "ManualBinaryAttentionClassifier (BiGRU + Multi-Pool [384-D]), DAiSEE init",
        "training_data": {r: v for r, v in info.items()},
        "label_source": "human CVAT annotations only (High/Moderate -> 1, Low -> 0)",
        "masked_channels": DIRECTION_CHANNELS,
        "inner_val_metrics": val_m,
        "expected_unseen_classroom_performance": loro["pooled_held_out"]["bigru"],
        "seed": args.seed,
    }, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    rep = ROOT / "reports/final_model"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "training.json").write_text(json.dumps({
        "checkpoint": str(out.relative_to(ROOT)), "sha256": sha, "threshold": thr,
        "rooms": info, "inner_val_metrics": val_m, "masked_channels": DIRECTION_CHANNELS,
        "expected_unseen_classroom_performance": loro["pooled_held_out"]["bigru"],
        "per_unseen_room": {r: v["bigru"] for r, v in loro["per_room"].items()},
    }, indent=2, default=float), encoding="utf-8")
    print(f"saved {out} sha256 {sha[:16]}")


if __name__ == "__main__":
    main()
