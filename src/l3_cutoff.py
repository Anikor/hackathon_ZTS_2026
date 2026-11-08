import sys
import pandas as pd

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = pd.read_csv(src)
top = s["unknown_score"].rank(ascending=False, method="first") <= n
changed = int((top & (s["attack_cat"] != "Unknown")).sum())
s.loc[top, "attack_cat"] = "Unknown"
s.to_csv(dst, index=False)
print("Unknown", int((s["attack_cat"] == "Unknown").sum()), "newly flagged", changed)
