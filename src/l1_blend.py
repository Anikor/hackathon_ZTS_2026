import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def best_threshold(y, p):
    ths = np.linspace(0.3, 0.7, 81)
    f = [f1_score(y, p >= t) for t in ths]
    k = int(np.argmax(f))
    return float(ths[k]), float(f[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--dir", default="submissions_l1x")
    ap.add_argument("--models", default="base,big,et,cat,xgb,rf,hgb,et2")
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    d = Path(a.dir)
    y = pd.read_csv(Path(a.data) / "level1_train.csv")["label"].astype(int).values
    ids = pd.read_csv(Path(a.data) / "level1_test.csv", usecols=["id"])["id"]
    names = [m for m in a.models.split(",") if (d / f"{m}_oof.npy").exists()]
    O = {m: np.load(d / f"{m}_oof.npy") for m in names}
    T = {m: np.load(d / f"{m}_test.npy") for m in names}
    for m in names:
        print(f"{m:6s} {best_threshold(y, O[m])[1]:.5f}")

    chosen, best = [], -1.0
    for _ in range(a.steps):
        cand = []
        for m in names:
            p = np.mean([O[c] for c in chosen + [m]], axis=0)
            cand.append((best_threshold(y, p)[1], m))
        f, m = max(cand)
        if f <= best + 1e-5:
            break
        chosen.append(m); best = f
        print(f"+ {m:6s} -> {f:.5f}")

    w = Counter(chosen)
    p_oof = np.mean([O[c] for c in chosen], axis=0)
    p_test = np.mean([T[c] for c in chosen], axis=0)
    th, f1 = best_threshold(y, p_oof)
    out = d / "level1_blend.csv"
    pd.DataFrame({"id": ids, "label": (p_test >= th).astype(int)}).to_csv(out, index=False)
    (d / "blend.json").write_text(json.dumps({"weights": dict(w), "threshold": th, "oof_f1": f1}, indent=2))
    print(f"blend {dict(w)}  th {th:.3f}  OOF F1 {f1:.5f}  -> {out}")


if __name__ == "__main__":
    main()
