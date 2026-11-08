import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

out, n_unk, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
subs = [pd.read_csv(f) for f in files]
base = subs[-1]
for s in subs:
    assert (s["id"].values == base["id"].values).all()
ranks = [(rankdata(s["unknown_score"]) - 1) / (len(s) - 1) for s in subs]
score = np.mean(ranks, axis=0)
for f, r in zip(files, ranks):
    print(f"{f.split(chr(92))[-1]:45s} spearman with ensemble {pd.Series(r).corr(pd.Series(score), method='spearman'):.4f}")
known = base["attack_cat"].copy()
pred = known.where(known != "Unknown", other=None)
fallback = subs[0]["attack_cat"]
for s in subs:
    pred = pred.fillna(s["attack_cat"].where(s["attack_cat"] != "Unknown"))
pred = pred.fillna("Exploits")
top = pd.Series(score).rank(ascending=False, method="first") <= n_unk
pred[top.values] = "Unknown"
top_sets = [set(np.argsort(-r)[:n_unk]) for r in ranks]
print("rows in top-%d of every detector:" % n_unk, len(set.intersection(*top_sets)), "| of ensemble top shared with best file:",
      len(set(np.argsort(-score)[:n_unk]) & top_sets[-1]))
pd.DataFrame({"id": base["id"], "attack_cat": pred, "unknown_score": np.round(score, 6)}).to_csv(out, index=False)
print("wrote", out)
