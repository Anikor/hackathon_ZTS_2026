import argparse
import warnings
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
L2_CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
              "Analysis", "Backdoor", "Shellcode", "Worms"]
L3_KNOWN = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Backdoor"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in ("id", "label", "attack_cat") if c in df.columns]).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    eps = 1.0
    extra = {
        "bytes_ratio": lambda d: (d.src_bytes + eps) / (d.dst_bytes + eps),
        "pkts_ratio": lambda d: (d.src_packets + eps) / (d.dst_packets + eps),
        "src_bytes_per_pkt": lambda d: d.src_bytes / (d.src_packets + eps),
        "dst_bytes_per_pkt": lambda d: d.dst_bytes / (d.dst_packets + eps),
        "total_bytes": lambda d: d.src_bytes + d.dst_bytes,
        "ttl_pair": lambda d: d.src_ttl * 1000 + d.dst_ttl,
        "dst_silent": lambda d: (d.dst_bytes == 0).astype(int),
        "load_ratio": lambda d: (d.src_load_bps + eps) / (d.dst_load_bps + eps),
    }
    for name, f in extra.items():
        try:
            X[name] = f(X)
        except AttributeError:
            pass
    X = X.replace([np.inf, -np.inf], 0)
    return X


def lgb_params(n_class: int, fast: bool) -> dict:
    p = dict(n_estimators=400 if fast else 1200, learning_rate=0.05, num_leaves=63,
             min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
             reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    p["objective"] = "binary" if n_class == 2 else "multiclass"
    return p


def class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: (cnt.max() / n) ** power for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y])


def cv_fit_predict(X, y, X_test, n_class, fast, weight_power=0.0, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    k = 1 if n_class == 2 else n_class
    oof = np.zeros((len(X), k))
    test = np.zeros((len(X_test), k))
    for f, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**lgb_params(n_class, fast))
        sw = class_weights(y[tr], weight_power) if weight_power else None
        m.fit(X.iloc[tr], y[tr], sample_weight=sw,
              eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p_va, p_te = m.predict_proba(X.iloc[va]), m.predict_proba(X_test)
        if n_class == 2:
            p_va, p_te = p_va[:, 1:], p_te[:, 1:]
        oof[va] = p_va
        test += p_te / folds
        log(f"  fold {f + 1}/{folds} done (best_iter={m.best_iteration_})")
    return oof, test


def tune_class_scales(P: np.ndarray, y: np.ndarray, rounds: int = 3) -> np.ndarray:
    s = np.ones(P.shape[1])
    grid = np.exp(np.linspace(np.log(0.25), np.log(6), 25))
    best = f1_score(y, (P * s).argmax(1), average="macro")
    for _ in range(rounds):
        for c in range(P.shape[1]):
            for g in grid:
                t = s.copy(); t[c] = g
                sc = f1_score(y, (P * t).argmax(1), average="macro")
                if sc > best + 1e-6:
                    best, s = sc, t
    return s


def check_submission(sub: pd.DataFrame, test: pd.DataFrame, cols, allowed=None):
    assert list(sub.columns) == cols, sub.columns
    assert len(sub) == len(test) and (sub.id.values == test.id.values).all(), "ids mismatch"
    assert not sub.isna().any().any()
    if allowed is not None:
        bad = set(sub[cols[1]].unique()) - set(allowed)
        assert not bad, f"labels not allowed: {bad}"


def level1(train, test, fast):
    X, Xt = add_features(train), add_features(test)
    y = train["label"].astype(int).values
    oof, pt = cv_fit_predict(X, y, Xt, 2, fast)
    oof, pt = oof[:, 0], pt[:, 0]
    ths = np.linspace(0.2, 0.8, 61)
    f1s = [f1_score(y, oof >= t) for t in ths]
    th = float(ths[int(np.argmax(f1s))])
    cv = {"oof_f1@0.5": f1_score(y, oof >= 0.5), "oof_f1@best": max(f1s), "threshold": th}
    log("L1 CV:", cv)
    sub = pd.DataFrame({"id": test.id, "label": (pt >= th).astype(int)})
    check_submission(sub, test, ["id", "label"], [0, 1])
    return sub, cv


def level2(train, test, fast):
    classes = sorted(train.attack_cat.unique())
    enc = {c: i for i, c in enumerate(classes)}
    X, Xt = add_features(train), add_features(test)
    y = train.attack_cat.map(enc).values
    oof, pt = cv_fit_predict(X, y, Xt, len(classes), fast, weight_power=0.5)
    raw = f1_score(y, oof.argmax(1), average="macro")
    s = tune_class_scales(oof, y)
    tuned = f1_score(y, (oof * s).argmax(1), average="macro")
    per_class = f1_score(y, (oof * s).argmax(1), average=None)
    cv = {"oof_macro_f1_raw": raw, "oof_macro_f1_tuned": tuned,
          "scales": dict(zip(classes, np.round(s, 3).tolist())),
          "per_class_f1": dict(zip(classes, np.round(per_class, 4).tolist())),
          "classes": classes,
          "oof_confusion": confusion_matrix(y, (oof * s).argmax(1)).tolist(),
          "class_counts": dict(zip(classes, np.bincount(y).tolist()))}
    log("L2 CV:", json.dumps({k: v for k, v in cv.items() if k != "oof_confusion"}, indent=1))
    pred = np.array(classes)[(pt * s).argmax(1)]
    sub = pd.DataFrame({"id": test.id, "attack_cat": pred})
    check_submission(sub, test, ["id", "attack_cat"], L2_CLASSES)
    return sub, cv


def rank01(a):
    return (rankdata(a) - 1) / max(len(a) - 1, 1)


def knn_distance(X_ref, X_query, k=10, max_ref=40_000, exclude_self=False):
    rng = np.random.default_rng(SEED)
    if len(X_ref) > max_ref:
        X_ref = X_ref[rng.choice(len(X_ref), max_ref, replace=False)]
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(X_ref)
    d, _ = nn.kneighbors(X_query)
    return d[:, int(exclude_self):].mean(1)


def to_knn_space(X: pd.DataFrame, scaler=None):
    Z = np.log1p(X.clip(lower=0).values.astype(float))
    if scaler is None:
        scaler = StandardScaler().fit(Z)
    return scaler.transform(Z), scaler


def pseudo_label_score(X_tr, X_te, score, top_frac=0.04, seed=0):
    from sklearn.model_selection import KFold
    pos = np.where(score >= np.quantile(score, 1 - top_frac))[0]
    rng = np.random.default_rng(seed)
    neg = rng.choice(len(X_tr), min(len(X_tr), 20 * len(pos)), replace=False)
    p = dict(lgb_params(2, True), n_estimators=300, num_leaves=31, min_child_samples=20, random_state=seed)
    out = np.zeros(len(X_te))
    for a, _ in KFold(5, shuffle=True, random_state=seed).split(pos):
        Xa = pd.concat([X_tr.iloc[neg], X_te.iloc[pos[a]]], ignore_index=True)
        ya = np.r_[np.zeros(len(neg)), np.ones(len(a))].astype(int)
        m = lgb.LGBMClassifier(**p).fit(Xa, ya)
        mask = np.ones(len(X_te), bool); mask[pos[a]] = False
        out[mask] += m.predict_proba(X_te[mask])[:, 1] / 5
    return out


def adversarial_score(X_tr, X_te, fast):
    Xa = pd.concat([X_tr, X_te], ignore_index=True)
    ya = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)
    p = dict(lgb_params(2, True), n_estimators=300, num_leaves=31)
    out = np.zeros(len(X_te))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(Xa, ya):
        m = lgb.LGBMClassifier(**p).fit(Xa.iloc[tr], ya[tr])
        te_va = va[va >= len(X_tr)]
        out[te_va - len(X_tr)] = m.predict_proba(Xa.iloc[te_va])[:, 1]
    return out


def level3(train, test, fast, unknown_frac=None, test_labels=None, weights_override=None, probs_dir=None, refine=False, refine_frac=0.04, refine_iters=1):
    known = sorted(train.attack_cat.unique())
    enc = {c: i for i, c in enumerate(known)}
    X, Xt = add_features(train), add_features(test)
    y = train.attack_cat.map(enc).values

    if probs_dir:
        pdir = Path(probs_dir)
        assert json.loads((pdir / "classes.json").read_text()) == known
        oof, pt, s = np.load(pdir / "blend_oof.npy"), np.load(pdir / "blend_test.npy"), np.load(pdir / "blend_scales.npy")
    else:
        oof, pt = cv_fit_predict(X, y, Xt, len(known), fast, weight_power=0.5)
        s = tune_class_scales(oof, y)
    cv = {"oof_known_macro_f1": f1_score(y, (oof * s).argmax(1), average="macro")}
    log("L3 closed-set CV:", cv)

    log("L3 leave-one-class-out simulation ...")
    Ztr, scaler = to_knn_space(X)
    names = ("1-maxprob", "entropy", "knn", "adversarial")
    sim, sim_comp = {}, {}
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(skf.split(X, y))
    for c in [k for k in known if k != "Normal"]:
        ci = enc[c]
        tr = tr_idx[y[tr_idx] != ci]
        va_known = va_idx[y[va_idx] != ci]
        va_unk = va_idx[y[va_idx] == ci]
        va_unk = rng.choice(va_unk, min(len(va_unk), int(0.02 * len(va_known))), replace=False)
        va = np.r_[va_known, va_unk]
        is_unk = np.r_[np.zeros(len(va_known)), np.ones(len(va_unk))].astype(int)
        rest = [k for k in known if k != c]
        yr = np.array([rest.index(known[v]) for v in y[tr]])
        m = lgb.LGBMClassifier(**dict(lgb_params(len(rest), True), n_estimators=300))
        m.fit(X.iloc[tr], yr, sample_weight=class_weights(yr, 0.5))
        P = m.predict_proba(X.iloc[va])
        sim_comp[c] = (is_unk, {
            "1-maxprob": rank01(1 - P.max(1)),
            "entropy": rank01(-(P * np.log(P + 1e-12)).sum(1)),
            "knn": rank01(knn_distance(Ztr[tr], Ztr[va])),
            "adversarial": rank01(adversarial_score(X.iloc[tr], X.iloc[va], True)),
        })
        sim[c] = {k: roc_auc_score(is_unk, v) for k, v in sim_comp[c][1].items()}
        log(f"  hide {c:15s} AUROC", {k: round(v, 3) for k, v in sim[c].items()})
    cv["loco_auroc"] = {k: float(np.mean([sim[c][k] for c in sim])) for k in names}
    cv["loco_auroc_per_class"] = sim

    best_w, best_auc = None, -1
    grid = [0, 0.5, 1, 2]
    for w4 in np.array(np.meshgrid(grid, grid, grid, grid)).T.reshape(-1, 4):
        if w4.sum() == 0:
            continue
        auc = np.mean([roc_auc_score(u, sum(wi * comp_[k] for wi, k in zip(w4, names)))
                       for u, comp_ in sim_comp.values()])
        if auc > best_auc + 1e-6:
            best_auc, best_w = auc, w4
    cv["loco_blend_auroc"] = float(best_auc)

    Pt = pt * s
    Pt = Pt / Pt.sum(1, keepdims=True)
    comp = {
        "1-maxprob": 1 - Pt.max(1),
        "entropy": -(Pt * np.log(Pt + 1e-12)).sum(1),
        "knn": knn_distance(Ztr, to_knn_space(Xt, scaler)[0]),
        "adversarial": adversarial_score(X, Xt, fast),
    }
    w = dict(zip(names, best_w.tolist()))
    if weights_override:
        w = {k: 0.0 for k in names}
        w.update({k: float(v) for k, v in (kv.split("=") for kv in weights_override.split(","))})
    score = sum(w[k] * rank01(comp[k]) for k in names) / sum(w.values())
    if refine:
        base = score
        for it in range(refine_iters):
            comp["pseudo"] = pseudo_label_score(X, Xt, score, top_frac=refine_frac, seed=it)
            score = 0.5 * rank01(base) + 0.5 * rank01(comp["pseudo"])
        cv["refine"] = f"pseudo-label, top {refine_frac:.0%}, {refine_iters} iter, 50/50 rank blend"
    cv["blend_weights"] = {k: round(v, 3) for k, v in w.items()}

    if unknown_frac is None:
        unknown_frac = 1 - len(train) / 141_700
    th = np.quantile(score, 1 - unknown_frac)
    pred = np.array(known)[Pt.argmax(1)].astype(object)
    pred[score >= th] = "Unknown"
    cv.update(unknown_frac=float(unknown_frac), threshold=float(th))

    if test_labels is not None:
        truth = test_labels.attack_cat.values
        unk = (truth == "Unknown").astype(int)
        cv["mock_component_auroc"] = {k: roc_auc_score(unk, v) for k, v in comp.items()}
        cv["mock_blend_auroc"] = roc_auc_score(unk, score)
        sweep = {}
        for fr in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
            p2 = np.array(known)[Pt.argmax(1)].astype(object)
            p2[score >= np.quantile(score, 1 - fr)] = "Unknown"
            sweep[fr] = f1_score(truth, p2, average="macro")
        cv["mock_macro_f1_by_unknown_frac"] = sweep

    log("L3:", json.dumps(cv, indent=1, default=float))
    sub = pd.DataFrame({"id": test.id, "attack_cat": pred, "unknown_score": np.round(score, 6)})
    check_submission(sub, test, ["id", "attack_cat", "unknown_score"], L3_KNOWN + ["Unknown"])
    assert sub.unknown_score.between(0, 1).all()
    return sub, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions")
    ap.add_argument("--fast", action="store_true", help="fewer trees, for quick iteration")
    ap.add_argument("--unknown-frac", type=float, default=None)
    ap.add_argument("--l3-weights", default=None, help='override blend, e.g. "entropy=1"')
    ap.add_argument("--probs", default=None, help="level 3: closed-set probabilities from multi.py blend")
    ap.add_argument("--refine", action="store_true", help="level 3: pseudo-label refinement of unknown_score")
    ap.add_argument("--refine-frac", type=float, default=0.04)
    ap.add_argument("--refine-iters", type=int, default=1)
    ap.add_argument("--mock-labels", default=None, help="dry run: hidden test labels csv")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / f"level{a.level}_train.csv")
    test = pd.read_csv(data / f"level{a.level}_test.csv")
    log(f"level {a.level}: train {train.shape}, test {test.shape}")

    t0 = time.time()
    if a.level == 1:
        sub, cv = level1(train, test, a.fast)
    elif a.level == 2:
        sub, cv = level2(train, test, a.fast)
    else:
        labels = pd.read_csv(a.mock_labels) if a.mock_labels and a.level == 3 else None
        sub, cv = level3(train, test, a.fast, a.unknown_frac, labels, a.l3_weights, a.probs, a.refine, a.refine_frac, a.refine_iters)

    tag = "_" + a.l3_weights.replace("=", "").replace(",", "_") if a.l3_weights else ""
    tag += f"_refine{int(a.refine_frac * 100)}x{a.refine_iters}" if getattr(a, "refine", False) else ""
    path = out / f"level{a.level}_submission{tag}.csv"
    sub.to_csv(path, index=False)
    cv["runtime_s"] = round(time.time() - t0, 1)
    (out / f"level{a.level}_cv{tag}.json").write_text(json.dumps(cv, indent=2, default=float))
    log(f"wrote {path} ({len(sub)} rows) in {cv['runtime_s']} s")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
