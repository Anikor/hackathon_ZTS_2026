import sys
import pandas as pd

s = pd.read_csv(sys.argv[1])
t = pd.read_csv(sys.argv[2], usecols=["id"])
ok = {"Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Backdoor", "Unknown"}
assert list(s.columns) == ["id", "attack_cat", "unknown_score"]
assert len(s) == len(t) and (s.id.values == t.id.values).all()
assert set(s.attack_cat) <= ok and s.unknown_score.between(0, 1).all()
print("FORMAT OK", len(s), "unknown", int((s.attack_cat == "Unknown").sum()))
