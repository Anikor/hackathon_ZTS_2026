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
    name, _, seed = name.partition("@")
    seed = int(seed) if seed else 42
    if name == "lgb":
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.03, num_leaves=63,
                               min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv[0], Xv[1])], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "et":
        m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1, max_features=0.7, n_jobs=-1, random_state=seed)
        m.fit(X, y, sample_weight=sw)
    elif name == "rf":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=1, max_features=0.4, n_jobs=-1, random_state=42)
        m.fit(X, y, sample_weight=sw)
    elif name == "lgb2":
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=4000, learning_rate=0.02, num_leaves=127,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=2.0, random_state=7, n_jobs=-1, verbose=-1)
        m.fit(X, y, sample_weight=class_weights(y, 0.75), eval_set=[(Xv[0], Xv[1])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    elif name == "lgb3":
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=4000, learning_rate=0.02, num_leaves=127,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=2.0, random_state=2026, n_jobs=-1, verbose=-1)
        m.fit(X, y, sample_weight=class_weights(y, 1.0), eval_set=[(Xv[0], Xv[1])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    elif name == "et2":
        m = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1, max_features=0.85, n_jobs=-1, random_state=7)
        m.fit(X, y, sample_weight=class_weights(y, 0.75))
    elif name == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=9, subsample=0.8, colsample_bytree=0.6,
                          min_child_weight=2, tree_method="hist", objective="multi:softprob", n_jobs=-1,
                          random_state=42, early_stopping_rounds=150)
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv[0], Xv[1])], verbose=False)
    elif name == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, loss_function="MultiClass",
                               random_seed=42, verbose=0, thread_count=-1, early_stopping_rounds=150,
                               allow_writing_files=False)
        m.fit(X, y, sample_weight=sw, eval_set=(Xv[0], Xv[1]))
    elif name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
                                           random_state=42)
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
    if a.fixed:
        names = sorted(set(a.fixed.split("+")))
    names = sorted(names, key=lambda m: -f1_score(y, np.load(out / f"{m}_oof.npy").argmax(1), average="macro"))[:a.top]
    O = {m: np.load(out / f"{m}_oof.npy") for m in names}
    T = {m: np.load(out / f"{m}_test.npy") for m in names}
    score = lambda P: f1_score(y, P.argmax(1), average="macro")
    tuned = lambda P: score(P * tune_class_scales(P, y, rounds=1))
    chosen, best = ([], -1.0) if not a.fixed else (a.fixed.split("+"), 0.0)
    for _ in range(0 if a.fixed else 8):
        cands = [(tuned(np.mean([O[c] for c in chosen + [k]], 0)), k) for k in names]
        f, k = max(cands)
        print(" ".join(f"{n}:{v:.4f}" for v, n in sorted(cands, reverse=True)), flush=True)
        if f <= best + 1e-4:
            break
        chosen.append(k); best = f
        print(f"+ {k} -> {best:.5f}", flush=True)
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


def combos(a):
    d, out = Path(a.data), Path(a.out)
    classes = json.loads((out / "classes.json").read_text())
    y = np.load(out / "y.npy")
    ids = pd.read_csv(d / f"level{a.level}_test.csv", usecols=["id"])["id"]
    res = {}
    for i, combo in enumerate(a.combos.split(";")):
        parts = combo.split("+")
        P = np.mean([np.load(out / f"{m}_oof.npy") for m in parts], 0)
        Pt = np.mean([np.load(out / f"{m}_test.npy") for m in parts], 0)
        s = tune_class_scales(P, y, rounds=4)
        f = f1_score(y, (P * s).argmax(1), average="macro")
        res[combo] = f
        pd.DataFrame({"id": ids, "attack_cat": np.array(classes)[(Pt * s).argmax(1)]}).to_csv(out / f"combo_{i}.csv", index=False)
        print(f"combo_{i} {combo:32s} tuned macro F1 {f:.5f}", flush=True)
    (out / "combos.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "blend", "combos"])
    ap.add_argument("--combos", default="")
    ap.add_argument("--fixed", default="", help='blend: skip the search, e.g. "lgb+lgb+et"')
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions_l2x")
    ap.add_argument("--models", default="lgb,et,rf")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    {"train": train, "blend": blend, "combos": combos}[a.cmd](a)
