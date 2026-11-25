"""What does DAiSEE contribute? Leave-one-classroom-out, final-model recipe
(BiGRU, camera-relative channels 0-4,21 masked), three variants:

  no_daisee      : random initialisation, classroom data only
  daisee_init    : DAiSEE-pretrained backbone (fusion_daisee.pt), classroom data only  [= final model]
  daisee_init+mix: DAiSEE init AND DAiSEE windows added to the training data
                   (all DAiSEE LOW windows + an equal number of HIGH windows)

Threshold and epoch are chosen on held-out classroom people from the training rooms.
Writes reports/improvement/daisee_ablation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.models.attention_fusion import AttentionFusionNet  # noqa: E402
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier  # noqa: E402
from evaluate_cross_classroom import load_all  # noqa: E402
from honest_evaluation import BACKBONE, best_threshold, metrics, predict  # noqa: E402

MASK = [0, 1, 2, 3, 4, 21]


def make(init: bool, device):
    base = AttentionFusionNet(feat_dim=32, hidden=64)
    if init:
        ck = torch.load(BACKBONE, map_location="cpu", weights_only=False)
        base.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=False)
    # a randomly initialised encoder must be trainable, otherwise it stays random
    return ManualBinaryAttentionClassifier(base, freeze_frame_encoder=init).to(device)


def train(Xtr, ytr, Xva, yva, device, seed, init, epochs=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make(init, device)
    counts = np.bincount(ytr, minlength=2)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([counts[1] / max(counts[0], 1) * 0.5, 1.0],
                                                  dtype=torch.float32, device=device))
    groups = [{"params": model.temporal.parameters(), "lr": 8e-5 if init else 1e-3, "weight_decay": 1e-4},
              {"params": model.head.parameters(), "lr": 1.5e-3, "weight_decay": 1e-3}]
    if not init:
        groups.append({"params": model.frame_encoder.parameters(), "lr": 1e-3, "weight_decay": 1e-4})
    opt = torch.optim.AdamW(groups)
    Xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    g = torch.Generator().manual_seed(seed)
    best, state = -1.0, None
    from sklearn.metrics import balanced_accuracy_score
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm) - 127, 128):
            b = perm[i:i + 128].to(device)
            loss = crit(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = balanced_accuracy_score(yva, (predict(model, Xva, device) >= 0.5).astype(int))
        if s > best:
            best, state = s, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def main(seeds=(77, 78)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, room, person, clip, info = load_all(False)
    X = X.copy()
    X[:, :, MASK] = 0
    dz = np.load(ROOT / "data/processed/fusion_daisee_full_v4.npz", allow_pickle=True)
    yd = dz["binary_targets"].astype(np.int64)
    rng = np.random.default_rng(0)
    lo = np.where(yd == 0)[0]
    hi = rng.choice(np.where(yd == 1)[0], size=len(lo), replace=False)
    sel = np.sort(np.concatenate([lo, hi]))
    Xd = dz["features"][sel].astype(np.float32)
    Xd[:, :, MASK] = 0
    yd = yd[sel]
    print(f"DAiSEE windows added in the mix variant: {len(yd)} ({(yd == 0).sum()} LOW)")

    res = {}
    for variant in ("no_daisee", "daisee_init", "daisee_init+mix"):
        P, T = np.zeros(len(y)), np.zeros(len(y))
        per = {}
        for r in sorted(np.unique(room)):
            te, tr = room == r, room != r
            rr = np.random.default_rng(11)
            tp = np.unique(person[tr])
            for _ in range(300):
                inner = rr.choice(tp, max(2, int(len(tp) * 0.25)), replace=False)
                mi = np.isin(person, inner) & tr
                if (y[mi] == 0).sum() >= 50 and (y[tr & ~mi] == 0).sum() >= 50:
                    break
            fit = tr & ~mi
            Xf, yf = X[fit], y[fit]
            if variant == "daisee_init+mix":
                Xf, yf = np.concatenate([Xf, Xd]), np.concatenate([yf, yd])
            ps, ts = [], []
            for s in seeds:
                m = train(Xf, yf, X[mi], y[mi], device, s, init=variant != "no_daisee")
                ts.append(best_threshold(y[mi], predict(m, X[mi], device)))
                ps.append(predict(m, X[te], device))
            P[te], T[te] = np.mean(ps, 0), float(np.mean(ts))
            per[r] = {k: round(float(v), 4) for k, v in metrics(y[te], P[te], T[te][0]).items()
                      if k in ("balanced_accuracy", "roc_auc", "accuracy", "low_recall")}
        pooled = metrics(y, (P >= T).astype(float), 0.5)
        res[variant] = {"per_room": per, "pooled": {k: round(float(v), 4) for k, v in pooled.items()
                                                     if k in ("balanced_accuracy", "accuracy", "macro_f1", "low_recall")}}
        print(f"{variant:18s} pooled balacc {pooled['balanced_accuracy']:.3f} acc {pooled['accuracy']:.3f} "
              f"macroF1 {pooled['macro_f1']:.3f} | " +
              "  ".join(f"{k}: bal {v['balanced_accuracy']:.2f} auc {v['roc_auc']:.2f}" for k, v in per.items()))
    out = ROOT / "reports/improvement/daisee_ablation.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
