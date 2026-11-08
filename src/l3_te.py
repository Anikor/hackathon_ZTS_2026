import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import add_features, class_weights, tune_class_scales

warnings.filterwarnings("ignore")
KEYS = {
    "sig6": ["src_bytes", "dst_bytes", "src_packets", "dst_packets", "src_ttl", "dst_ttl"],
    "bytes2": ["src_bytes", "dst_bytes"],
    "sb": ["src_bytes"],
}


def keyed(df, cols):
    return pd.util.hash_pandas_object(df[cols], index=False).values


def te_fit_apply(k_fit, y_fit, k_apply, n_class, prior, alpha=2.0):
    d = pd.DataFrame({"k": k_fit, "y": y_fit})
    ct = pd.crosstab(d.k, d.y).reindex(columns=range(n_class), fill_value=0)
    n = ct.sum(1)
    rates = (ct.values + alpha * prior) / (n.values[:, None] + alpha)
    idx = pd.Index(ct.index).get_indexer(k_apply)
    out = np.tile(prior, (len(k_apply), 1)).astype(float)
    cnt = np.zeros(len(k_apply))
    hit = idx >= 0
    out[hit] = rates[idx[hit]]
    cnt[hit] = n.values[idx[hit]]
    return np.c_[out, np.log1p(cnt)]


def te_features(df_fit, y_fit, df_apply, n_class, inner=False):
    prior = np.bincount(y_fit, minlength=n_class) / len(y_fit)
    blocks = []
    for name, cols in KEYS.items():
        kf, ka = keyed(df_fit, cols), keyed(df_apply, cols)
        if inner:
            res = np.zeros((len(df_fit), n_class + 1))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(df_fit, y_fit):
                res[b] = te_fit_apply(kf[a], y_fit[a], kf[b], n_class, prior)
            blocks.append(res)
        else:
            blocks.append(te_fit_apply(kf, y_fit, ka, n_class, prior))
    return np.hstack(blocks)


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tr = pd.read_csv(f"data/level{level}_train.csv")
    classes = sorted(tr.attack_cat.unique()); n_class = len(classes)
    y = tr.attack_cat.map({c: i for i, c in enumerate(classes)}).values
    X = add_features(tr)
    params = dict(objective="multiclass", n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    for use_te in (False, True):
        t0 = time.time()
        oof = np.zeros((len(X), n_class))
        for trn, val in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            Xa, Xv = X.iloc[trn].values, X.iloc[val].values
            if use_te:
                Xa = np.hstack([Xa, te_features(tr.iloc[trn], y[trn], tr.iloc[trn], n_class, inner=True)])
                Xv = np.hstack([Xv, te_features(tr.iloc[trn], y[trn], tr.iloc[val], n_class)])
            m = lgb.LGBMClassifier(**params)
            m.fit(Xa, y[trn], sample_weight=class_weights(y[trn], 0.5), eval_set=[(Xv, y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[val] = m.predict_proba(Xv)
        s = tune_class_scales(oof, y, rounds=2)
        per = f1_score(y, (oof * s).argmax(1), average=None)
        print(f"TE={use_te}: raw {f1_score(y, oof.argmax(1), average='macro'):.4f} tuned {f1_score(y, (oof * s).argmax(1), average='macro'):.4f} "
              f"{dict(zip(classes, np.round(per, 3)))}  {time.time() - t0:.0f}s", flush=True)
        np.save(f"/private/tmp/claude-501/-Users-anikor/3384f5ec-08f2-4740-9fa7-6cb94321fab6/scratchpad/l3_te_oof_{int(use_te)}.npy", oof)


if __name__ == "__main__":
    main()
