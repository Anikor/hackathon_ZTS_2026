import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve import class_weights, tune_class_scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions_l2x")
    ap.add_argument("--models", default="lgb,et,rf,lgb2,lgb3,xgb,et2")
    ap.add_argument("--tag", default="stack")
    a = ap.parse_args()
    out = Path(a.out)
    classes = json.loads((out / "classes.json").read_text())
    y = np.load(out / "y.npy")
    ids = pd.read_csv(Path(a.data) / f"level{a.level}_test.csv", usecols=["id"])["id"]
    names = [m for m in a.models.split(",") if (out / f"{m}_oof.npy").exists()]
    Z = np.hstack([np.log(np.clip(np.load(out / f"{m}_oof.npy"), 1e-6, 1)) for m in names])
    Zt = np.hstack([np.log(np.clip(np.load(out / f"{m}_test.npy"), 1e-6, 1)) for m in names])
    print("meta features", Z.shape, "from", names, flush=True)
    params = dict(objective="multiclass", n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, random_state=11,
                  n_jobs=-1, verbose=-1)
    oof, pt = np.zeros((len(Z), len(classes))), np.zeros((len(Zt), len(classes)))
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=123).split(Z, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(Z[trn], y[trn], sample_weight=class_weights(y[trn], 0.5), eval_set=[(Z[val], y[val])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = m.predict_proba(Z[val])
        pt += m.predict_proba(Zt) / 5
        print("  fold done, iters", m.best_iteration_, flush=True)
    raw = f1_score(y, oof.argmax(1), average="macro")
    s = tune_class_scales(oof, y, rounds=4)
    tuned = f1_score(y, (oof * s).argmax(1), average="macro")
    np.save(out / f"{a.tag}_oof.npy", oof); np.save(out / f"{a.tag}_test.npy", pt)
    pd.DataFrame({"id": ids, "attack_cat": np.array(classes)[(pt * s).argmax(1)]}).to_csv(out / f"level{a.level}_{a.tag}.csv", index=False)
    per = f1_score(y, (oof * s).argmax(1), average=None)
    info = {"models": names, "oof_macro_f1_raw": raw, "oof_macro_f1_tuned": tuned,
            "per_class_f1": dict(zip(classes, np.round(per, 4).tolist())), "scales": dict(zip(classes, np.round(s, 3).tolist()))}
    (out / f"{a.tag}.json").write_text(json.dumps(info, indent=2))
    print(f"STACK raw {raw:.5f} tuned {tuned:.5f}", flush=True)
    print(json.dumps(info["per_class_f1"]), flush=True)


if __name__ == "__main__":
    main()
