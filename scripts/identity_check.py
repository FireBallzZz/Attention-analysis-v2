"""Do the image models learn attention or identity?

Within each video, using classroom-behaviour crop embeddings (logistic regression):
  * unseen students : 4-fold split by student  -> can it judge a NEW person?
  * same students   : train first half, test second half -> can it follow a KNOWN person?
Writes reports/improvement/identity_check.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from loro_crop_experiments import temporal_mean

def fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

d = np.load(ROOT / "data/processed/crop_embeddings.npz", allow_pickle=True)
y, video, track, frame = d["label"], d["video"], d["track"], d["frame"]
E = temporal_mean(d["scb"], video, track, frame)
out = {}
for v in np.unique(video):
    m = video == v; X, yy, g, f = E[m], y[m], track[m], frame[m]
    oof = np.zeros(len(yy))
    for tr, te in StratifiedGroupKFold(min(4, len(np.unique(g[yy == 0]))), shuffle=True, random_state=0).split(X, yy, g):
        oof[te] = fit_pred(X[tr], yy[tr], X[te])
    tr = f < np.median(f)
    out[v] = {"unseen_students_auc": round(float(roc_auc_score(yy, oof)), 3),
              "same_students_later_auc": round(float(roc_auc_score(yy[~tr], fit_pred(X[tr], yy[tr], X[~tr]))), 3)}
    print(v, out[v])
(ROOT / "reports/improvement/identity_check.json").write_text(json.dumps(out, indent=1))
