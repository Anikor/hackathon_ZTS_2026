import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import add_features, class_weights

warnings.filterwarnings("ignore")


def r01(a):
    return (rankdata(a) - 1) / (len(a) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--probs", default="submissions_l3s")
    ap.add_argument("--sub", default="submissions_l3s/sub/level3_submission_refine12x1.csv")
    ap.add_argument("--out", default="submissions_l3s/sub/level3_submission_eight.csv")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--w", type=float, default=0.3)
    a = ap.parse_args()
    tr = pd.read_csv(Path(a.data) / "level3_train.csv")
    te = pd.read_csv(Path(a.data) / "level3_test.csv")
    sub = pd.read_csv(a.sub)
    pdir = Path(a.probs)
    known = json.loads((pdir / "classes.json").read_text())
    Pt = np.load(pdir / "blend_test.npy") * np.load(pdir / "blend_scales.npy")
    known_pred = np.array(known)[Pt.argmax(1)]
    s = sub["unknown_score"].values
    n_unk = int((sub["attack_cat"] == "Unknown").sum())
    X, Xt = add_features(tr), add_features(te)
    y = tr["attack_cat"].map({c: i for i, c in enumerate(known)}).values
    pos = np.argsort(-s)[: int(n_unk * a.k)]
    params = dict(objective="multiclass", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=-1, random_state=1)
    pu = np.zeros(len(Xt))
    for fold, (trn, _) in enumerate(KFold(5, shuffle=True, random_state=0).split(pos)):
        Xa = pd.concat([X, Xt.iloc[pos[trn]]], ignore_index=True)
        ya = np.r_[y, np.full(len(trn), len(known))]
        m = lgb.LGBMClassifier(**params).fit(Xa, ya, sample_weight=class_weights(ya, 0.5))
        mask = np.ones(len(Xt), bool); mask[pos[trn]] = False
        pu[mask] += m.predict_proba(Xt[mask])[:, len(known)] / 4
        print("fold", fold, "done", flush=True)
    score = (1 - a.w) * r01(s) + a.w * r01(pu)
    pred = known_pred.astype(object)
    pred[np.argsort(-score)[:n_unk]] = "Unknown"
    out = pd.DataFrame({"id": te["id"], "attack_cat": pred, "unknown_score": np.round(score, 6)})
    out.to_csv(a.out, index=False)
    print("wrote", a.out, "unknown", n_unk, "spearman vs input", round(pd.Series(score).corr(pd.Series(s), method="spearman"), 4))


if __name__ == "__main__":
    main()
