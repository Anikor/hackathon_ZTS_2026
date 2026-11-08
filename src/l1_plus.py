import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import add_features

warnings.filterwarnings("ignore")
FREQ_COLS = ["src_bytes", "dst_bytes", "flow_duration", "flow_rate", "src_load_bps", "dst_load_bps",
             "src_mean_pkt_size", "dst_mean_pkt_size", "src_tcp_base_seq", "dst_tcp_base_seq",
             "src_interpacket_time", "dst_interpacket_time", "tcp_rtt", "src_jitter", "dst_jitter"]


def freq_features(X, Xt):
    both = pd.concat([X, Xt], ignore_index=True)
    for c in [c for c in FREQ_COLS if c in X.columns]:
        vc = both[c].value_counts()
        X[f"freq_{c}"] = X[c].map(vc).values
        Xt[f"freq_{c}"] = Xt[c].map(vc).values
    return X, Xt


def params(kind, seed):
    p = dict(objective="binary", n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=10,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=seed,
             n_jobs=-1, verbose=-1)
    if kind == "big":
        p.update(learning_rate=0.02, num_leaves=127, min_child_samples=20, colsample_bytree=0.5)
    return p


def cv(X, y, Xt, kind, seeds):
    oof, pt = np.zeros(len(X)), np.zeros(len(Xt))
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(**params(kind, seed))
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / len(seeds)
            pt += m.predict_proba(Xt)[:, 1] / (5 * len(seeds))
    return oof, pt


def cv_et(X, y, Xt, seed=42):
    oof, pt = np.zeros(len(X)), np.zeros(len(Xt))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pt += m.predict_proba(Xt)[:, 1] / 5
    return oof, pt


def best_threshold(y, p):
    ths = np.linspace(0.2, 0.8, 121)
    f = [f1_score(y, p >= t) for t in ths]
    k = int(np.argmax(f))
    return float(ths[k]), float(f[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions_l1x")
    ap.add_argument("--variants", default="base,freq,big,freq_big,freq_seeds")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(Path(a.data) / "level1_train.csv")
    te = pd.read_csv(Path(a.data) / "level1_test.csv")
    y = tr["label"].astype(int).values
    results = {}
    for v in a.variants.split(","):
        t0 = time.time()
        X, Xt = add_features(tr), add_features(te)
        if "freq" in v:
            X, Xt = freq_features(X, Xt)
        kind = "big" if "big" in v else "base"
        seeds = [42, 7, 2026] if "seeds" in v else [42]
        if v == "et":
            oof, pt = cv_et(X, y, Xt)
        elif v == "big_et":
            o1, p1 = cv(X, y, Xt, "big", [42])
            o2, p2 = cv_et(X, y, Xt)
            oof, pt = (o1 + o2) / 2, (p1 + p2) / 2
        else:
            oof, pt = cv(X, y, Xt, kind, seeds)
        th, f1 = best_threshold(y, oof)
        results[v] = {"oof_f1": f1, "threshold": th, "runtime_s": round(time.time() - t0, 1)}
        np.save(out / f"{v}_oof.npy", oof); np.save(out / f"{v}_test.npy", pt)
        pd.DataFrame({"id": te.id, "label": (pt >= th).astype(int)}).to_csv(out / f"level1_{v}.csv", index=False)
        print(f"{v:12s} OOF F1 {f1:.5f}  th {th:.3f}  {results[v]['runtime_s']} s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
