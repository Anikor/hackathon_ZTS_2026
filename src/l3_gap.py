import sys
import numpy as np
import pandas as pd

s = np.sort(pd.read_csv(sys.argv[1])["unknown_score"].values)[::-1]
for k in range(400, 1001, 50):
    print(f"rank {k:4d}: score {s[k - 1]:.4f}")
d = s[300:1500] - s[301:1501]
top = np.argsort(-d)[:8] + 301
print("largest drops between consecutive ranks:", [(int(k), round(float(d[k - 301]), 4)) for k in sorted(top)])
