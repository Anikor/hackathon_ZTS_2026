import sys
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def score(level: int, sub_path: str, truth_path: str) -> float:
    sub, truth = pd.read_csv(sub_path), pd.read_csv(truth_path)
    m = truth.merge(sub, on="id", suffixes=("_true", "_pred"))
    assert len(m) == len(truth) == len(sub), "row/id mismatch"
    if level == 1:
        return f1_score(m.label_true, m.label_pred)
    if level == 2:
        return f1_score(m.attack_cat_true, m.attack_cat_pred, average="macro")
    f1 = f1_score(m.attack_cat_true, m.attack_cat_pred, average="macro")
    auc = roc_auc_score(m.attack_cat_true == "Unknown", m.unknown_score)
    print(f"  macroF1(8)={f1:.4f}  AUROC={auc:.4f}")
    return 0.6 * f1 + 0.4 * auc


if __name__ == "__main__":
    import json
    from pathlib import Path
    lvl, sub = int(sys.argv[1]), Path(sys.argv[2])
    sc = score(lvl, str(sub), sys.argv[3])
    print(f"level {lvl} score: {sc:.4f}")
    cv = sub.with_name(sub.name.replace("_submission", "_cv")).with_suffix(".json")
    if cv.exists():
        d = json.loads(cv.read_text()); d["mock_score"] = sc
        cv.write_text(json.dumps(d, indent=2))
