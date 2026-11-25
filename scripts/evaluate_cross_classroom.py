"""Leave-one-classroom-out evaluation on human-labelled videos only.

Rooms (people never shared between rooms):
  classroom_01 : clip_004 (Anno/ground_truth.csv) + clip_001 (human clip-1 file)
  room_B       : 11.mp4 + non attentive 1.mov   (same room, same students)
  room_C       : Test 8.mp4
  room_D       : Test 6.mp4  -- excluded by default: every student, including
                 one asleep on the desk for the whole clip, is labelled High.

For each room: train on the other rooms, choose epoch + threshold on persons
held out from those training rooms, then score the unseen room once.
Writes reports/cross_classroom/{results.json, REPORT.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from honest_evaluation import (  # noqa: E402
    DEPLOYED,
    best_threshold,
    cluster_bootstrap,
    load_human_data,
    metrics,
    new_model,
    predict,
    train_bigru,
    train_logreg,
    window_stats,
)

PROC = ROOT / "data" / "processed"
NEW = {  # npz name -> (room, video label)
    "human_11.npz": ("room_B", "11"),
    "human_non_attentive_1.npz": ("room_B", "non_attentive_1"),
    "human_test_8.npz": ("room_C", "test_8"),
    "human_test_6.npz": ("room_D", "test_6"),
}


def load_all(include_test6: bool):
    X, y, person, clip, track, frame, stats = load_human_data(0.30, 60.0)
    rooms = np.array(["classroom_01"] * len(y))
    persons = np.array([f"classroom_01:{p}" for p in person])
    clips = clip.astype(object)
    Xs, ys, rs, ps, cs = [X], [y], [rooms], [persons], [clips]
    info = {"classroom_01": {"windows": int(len(y)), "low": int((y == 0).sum())}}
    for f, (room, label) in NEW.items():
        if room == "room_D" and not include_test6:
            continue
        p = PROC / f
        if not p.exists():
            print(f"missing {f}, skipped")
            continue
        d = np.load(p, allow_pickle=True)
        yy = (d["targets"] != 2).astype(np.int64)
        Xs.append(d["features"].astype(np.float32))
        ys.append(yy)
        rs.append(np.array([room] * len(yy)))
        # Persons are only guaranteed distinct within one video; room_B's two
        # videos share students, but the room is always held out as a whole.
        ps.append(np.array([f"{label}:{s}" for s in d["student_ids"]]))
        cs.append(np.array([label] * len(yy), dtype=object))
        info.setdefault(room, {"windows": 0, "low": 0})
        info[room]["windows"] += int(len(yy))
        info[room]["low"] += int((yy == 0).sum())
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs),
            np.concatenate(ps), np.concatenate(cs), info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--inner-val-frac", type=float, default=0.25)
    ap.add_argument("--include-test6", action="store_true")
    ap.add_argument("--out-dir", default="reports/cross_classroom")
    ap.add_argument("--drop-channels", default="",
                    help="Comma list of feature channels to zero, e.g. 0,1,2,3,4,21 (camera-relative head/gaze)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, room, person, clip, info = load_all(args.include_test6)
    print("rooms:", json.dumps(info))
    drop = [int(c) for c in args.drop_channels.split(",") if c.strip()]
    # The old deployed checkpoint always sees the full, unmasked input.
    dep = new_model(device)
    dep.load_state_dict(torch.load(DEPLOYED, map_location="cpu", weights_only=False)["model"])
    p_dep = predict(dep, X, device)

    if drop:
        X = X.copy()
        X[:, :, drop] = 0.0
        print("zeroed channels:", drop)
    feats = window_stats(X)
    pid = np.unique(person, return_inverse=True)[1]

    results = {"config": vars(args), "rooms": info, "per_room": {}}
    pooled = {"bigru": np.zeros(len(y)), "bigru_thr": np.zeros(len(y)),
              "logreg": np.zeros(len(y)), "logreg_thr": np.zeros(len(y))}
    for test_room in sorted(np.unique(room)):
        te = np.where(room == test_room)[0]
        tr = np.where(room != test_room)[0]
        rng = np.random.default_rng(11)
        tp = np.unique(person[tr])
        for _ in range(500):
            inner = rng.choice(tp, size=max(2, int(len(tp) * args.inner_val_frac)), replace=False)
            m_in = np.isin(person[tr], inner)
            if (y[tr][m_in] == 0).sum() >= 50 and (y[tr][~m_in] == 0).sum() >= 50:
                break
        fit, val = tr[~m_in], tr[m_in]

        lr = train_logreg(feats[fit], y[fit])
        t_lr = best_threshold(y[val], lr(feats[val]))
        p_lr = lr(feats[te])

        ps, ts = [], []
        for s in range(args.seeds):
            model = train_bigru(X[fit], y[fit], X[val], y[val], device, seed=77 + s, epochs=args.epochs)
            ts.append(best_threshold(y[val], predict(model, X[val], device)))
            ps.append(predict(model, X[te], device))
        p_gru, t_gru = np.mean(ps, 0), float(np.mean(ts))
        pooled["bigru"][te], pooled["bigru_thr"][te] = p_gru, t_gru
        pooled["logreg"][te], pooled["logreg_thr"][te] = p_lr, t_lr

        def pack(p, t):
            m = metrics(y[te], p, t)
            m["threshold"] = round(t, 2)
            m["ci95_person_bootstrap"] = cluster_bootstrap(y[te], p, t, pid[te])
            return m

        r = {
            "test_windows": int(len(te)),
            "test_low_windows": int((y[te] == 0).sum()),
            "test_persons": int(len(np.unique(person[te]))),
            "train_rooms": sorted(set(room[tr])),
            "bigru": pack(p_gru, t_gru),
            "logreg": pack(p_lr, t_lr),
            "deployed_checkpoint_thr_0.50": metrics(y[te], p_dep[te], 0.5),
        }
        if test_room == "classroom_01":
            r["deployed_checkpoint_note"] = "deployed model TRAINED on classroom_01; not a held-out score"
        results["per_room"][test_room] = r
        print(f"[{test_room}] n={len(te)} low={int((y[te] == 0).sum())} | "
              f"BiGRU balacc {r['bigru']['balanced_accuracy']:.3f} auc {r['bigru']['roc_auc']:.3f} | "
              f"LogReg balacc {r['logreg']['balanced_accuracy']:.3f} auc {r['logreg']['roc_auc']:.3f} | "
              f"deployed balacc {r['deployed_checkpoint_thr_0.50']['balanced_accuracy']:.3f} "
              f"auc {r['deployed_checkpoint_thr_0.50']['roc_auc']:.3f} | always-HIGH acc "
              f"{r['bigru']['always_attentive_accuracy']:.3f}")

    # Pooled over all held-out rooms (each window scored by a model that never saw its room)
    pool = {}
    for name in ("bigru", "logreg"):
        pred = (pooled[name] >= pooled[f"{name}_thr"]).astype(float)
        m = metrics(y, pred, 0.5)
        m["roc_auc_within_room_mean"] = float(np.nanmean(
            [results["per_room"][r][name]["roc_auc"] for r in results["per_room"]]))
        m["ci95_person_bootstrap"] = cluster_bootstrap(y, pred, 0.5, pid)
        pool[name] = m
    results["pooled_held_out"] = pool

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out / 'results.json'}")
    for name, m in pool.items():
        print(f"POOLED {name}: acc {m['accuracy']:.3f} balacc {m['balanced_accuracy']:.3f} "
              f"macroF1 {m['macro_f1']:.3f} lowRec {m['low_recall']:.3f} "
              f"mean-room AUC {m['roc_auc_within_room_mean']:.3f} always-HIGH {m['always_attentive_accuracy']:.3f}")


if __name__ == "__main__":
    main()
