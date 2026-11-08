import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import add_features

warnings.filterwarnings("ignore")


def make(name, seed=42):
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, learning_rate=0.06, depth=8, random_seed=seed, verbose=0, thread_count=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=9, subsample=0.8, colsample_bytree=0.6,
                             min_child_weight=2, tree_method="hist", n_jobs=-1, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=0.4, n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=1500, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if name == "et3":
        return ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1, max_features=0.85, n_jobs=-1, random_state=7)
    if name == "et4":
        return ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1, max_features=0.6, criterion="entropy", n_jobs=-1, random_state=2026)
    if name == "et2":
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1, max_features=0.7, n_jobs=-1, random_state=seed)
    raise ValueError(name)


def best_threshold(y, p):
    ths = np.linspace(0.2, 0.8, 121)
    f = [f1_score(y, p >= t) for t in ths]
    k = int(np.argmax(f))
    return float(ths[k]), float(f[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions_l1x")
    ap.add_argument("--models", default="cat,xgb,rf,hgb,et2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(Path(a.data) / "level1_train.csv")
    te = pd.read_csv(Path(a.data) / "level1_test.csv")
    X, Xt = add_features(tr), add_features(te)
    y = tr["label"].astype(int).values
    for name in a.models.split(","):
        t0 = time.time()
        oof, pt = np.zeros(len(X)), np.zeros(len(Xt))
        for trn, val in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            m = make(name).fit(X.iloc[trn], y[trn])
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
            pt += m.predict_proba(Xt)[:, 1] / 5
        np.save(out / f"{name}_oof.npy", oof); np.save(out / f"{name}_test.npy", pt)
        th, f1 = best_threshold(y, oof)
        print(f"{name:6s} OOF F1 {f1:.5f}  th {th:.3f}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
