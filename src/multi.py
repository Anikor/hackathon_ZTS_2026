import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import add_features, class_weights, tune_class_scales

warnings.filterwarnings("ignore")


def fit_predict(name, X, y, Xv, Xt, n_class):
    sw = class_weights(y, 0.5)
    if name == "lgb":
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.03, num_leaves=63,
                               min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv[0], Xv[1])], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "et":
        m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1, max_features=0.7, n_jobs=-1, random_state=42)
        m.fit(X, y, sample_weight=sw)
    elif name == "rf":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.4, n_jobs=-1, random_state=42)
        m.fit(X, y, sample_weight=sw)
    else:
        raise ValueError(name)
    return m.predict_proba(Xv[0]), m.predict_proba(Xt)


def train(a):
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(d / f"level{a.level}_train.csv")
    te = pd.read_csv(d / f"level{a.level}_test.csv")
    classes = sorted(tr["attack_cat"].unique())
    y = tr["attack_cat"].map({c: i for i, c in enumerate(classes)}).values
    X, Xt = add_features(tr), add_features(te)
    (out / "classes.json").write_text(json.dumps(classes))
    np.save(out / "y.npy", y)
    for name in a.models.split(","):
        t0 = time.time()
        oof, pt = np.zeros((len(X), len(classes))), np.zeros((len(Xt), len(classes)))
        for trn, val in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            pv, ptt = fit_predict(name, X.iloc[trn], y[trn], (X.iloc[val], y[val]), Xt, len(classes))
            oof[val] = pv
            pt += ptt / 5
        np.save(out / f"{name}_oof.npy", oof); np.save(out / f"{name}_test.npy", pt)
        print(f"{name:4s} OOF macro F1 {f1_score(y, oof.argmax(1), average='macro'):.5f}  {time.time() - t0:.0f} s", flush=True)


def blend(a):
    d, out = Path(a.data), Path(a.out)
    classes = json.loads((out / "classes.json").read_text())
    y = np.load(out / "y.npy")
    ids = pd.read_csv(d / f"level{a.level}_test.csv", usecols=["id"])["id"]
    names = [m for m in a.models.split(",") if (out / f"{m}_oof.npy").exists()]
    O = {m: np.load(out / f"{m}_oof.npy") for m in names}
    T = {m: np.load(out / f"{m}_test.npy") for m in names}
    score = lambda P: f1_score(y, P.argmax(1), average="macro")
    from itertools import combinations_with_replacement
    best, chosen = -1.0, None
    for r in (1, 2, 3):
        for combo in combinations_with_replacement(names, r):
            if r > 1 and len(set(combo)) == 1:
                continue
            P = np.mean([O[c] for c in combo], 0)
            f = score(P * tune_class_scales(P, y, rounds=1))
            print(f"{'+'.join(combo):16s} {f:.5f}", flush=True)
            if f > best + 1e-5:
                best, chosen = f, list(combo)
    print("chosen", chosen, round(best, 5), flush=True)
    P = np.mean([O[c] for c in chosen], 0)
    Pt = np.mean([T[c] for c in chosen], 0)
    s = tune_class_scales(P, y, rounds=3)
    tuned = score(P * s)
    best = score(P)
    np.save(out / "blend_oof.npy", P); np.save(out / "blend_test.npy", Pt); np.save(out / "blend_scales.npy", s)
    pred = np.array(classes)[(Pt * s).argmax(1)]
    pd.DataFrame({"id": ids, "attack_cat": pred}).to_csv(out / f"level{a.level}_blend.csv", index=False)
    per = f1_score(y, (P * s).argmax(1), average=None)
    info = {"weights": dict(Counter(chosen)), "oof_macro_f1_raw": best, "oof_macro_f1_tuned": tuned,
            "scales": dict(zip(classes, np.round(s, 3).tolist())), "per_class_f1": dict(zip(classes, np.round(per, 4).tolist()))}
    (out / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "blend"])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions_l2x")
    ap.add_argument("--models", default="lgb,et,rf")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else blend(a)
