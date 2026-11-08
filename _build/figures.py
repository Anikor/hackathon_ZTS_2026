import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
GREYS = ["#222222", "#666666", "#9a9a9a", "#cfcfcf"]


def load(d, n):
    f = Path(d) / n
    return json.loads(f.read_text()) if f.exists() else {}


def pipeline(out):
    fig, ax = plt.subplots(figsize=(7.6, 2.4)); ax.axis("off")
    W, G, Y, H = 0.176, 0.030, 0.52, 0.36
    boxes = ["level{N}_train.csv\n39 caracteristici", "+ 8 caracteristici\nderivate",
             "LightGBM, 5-fold,\nponderi de clasă", "prag / scalare clase\npe predicții OOF",
             "verificare format\n→ submission.csv"]
    for k, text in enumerate(boxes):
        x = k * (W + G)
        ax.add_patch(plt.Rectangle((x, Y), W, H, fill=False, lw=1.1))
        ax.text(x + W / 2, Y + H / 2, text, ha="center", va="center", fontsize=8.3)
        if k:
            ax.annotate("", xy=(x, Y + H / 2), xytext=(x - G, Y + H / 2), arrowprops=dict(arrowstyle="->", lw=1.0))
    x2 = 2 * (W + G)
    ax.add_patch(plt.Rectangle((x2 - 0.06, 0.02), W + 0.32, 0.3, fill=False, lw=1.0, ls="--"))
    ax.text(x2 - 0.06 + (W + 0.32) / 2, 0.17, "Nivelul 3: unknown_score = blend de ranguri\n"
            "(entropie, 1 − p_max, kNN, train-vs-test)", ha="center", va="center", fontsize=8.3)
    ax.annotate("", xy=(x2 + W / 2, 0.32), xytext=(x2 + W / 2, Y), arrowprops=dict(arrowstyle="->", lw=1.0, ls="--"))
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 0.92)
    fig.savefig(out); plt.close(fig)


def class_counts(l2, out):
    cc = l2["class_counts"]; items = sorted(cc.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar([k for k, _ in items], [v for _, v in items], color=GREYS[1])
    ax.set_yscale("log"); ax.set_ylabel("rânduri în train (log)")
    for i, (_, v) in enumerate(items):
        ax.text(i, v * 1.15, f"{v:,}".replace(",", " "), ha="center", fontsize=8.5)
    plt.xticks(rotation=25, ha="right"); fig.savefig(out); plt.close(fig)


def confusion(l2, out):
    cls = l2["classes"]; cm = np.array(l2["oof_confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(cls)), cls, rotation=40, ha="right"); ax.set_yticks(range(len(cls)), cls)
    ax.set_xlabel("clasa prezisă"); ax.set_ylabel("clasa reală")
    fig.savefig(out); plt.close(fig)


def loco(l3, out):
    per = l3["loco_auroc_per_class"]; names = list(next(iter(per.values())).keys())
    cls = list(per.keys()); x = np.arange(len(cls)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for k, n in enumerate(names):
        ax.bar(x + (k - 1.5) * w, [per[c][n] for c in cls], w, label=n, color=GREYS[k], edgecolor="black", lw=0.4)
    ax.axhline(0.5, color="black", lw=0.8, ls=":"); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x, cls, fontsize=9.5); ax.set_xlabel("clasa ascunsă din train")
    ax.set_ylabel("AUROC"); ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out); plt.close(fig)


def frac(l3, l3e, out):
    s = l3.get("mock_macro_f1_by_unknown_frac")
    if not s:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    xs = [float(k) * 100 for k in s]
    ax.plot(xs, list(s.values()), "o-", color=GREYS[0], label="blend (ponderi din simulare)")
    if l3e.get("mock_macro_f1_by_unknown_frac"):
        ax.plot(xs, list(l3e["mock_macro_f1_by_unknown_frac"].values()), "s--", color=GREYS[1], label="doar entropia")
    ax.axvline(l3["unknown_frac"] * 100, color="black", lw=0.8, ls=":")
    ax.text(l3["unknown_frac"] * 100 + 0.1, ax.get_ylim()[0] + 0.001, " pragul ales", fontsize=9)
    ax.set_xlabel("fluxuri marcate Unknown (% din test)"); ax.set_ylabel("macro F1 (8 etichete)")
    ax.legend(frameon=False, fontsize=9); fig.savefig(out); plt.close(fig)
    return True


if __name__ == "__main__":
    sub, figs = Path(sys.argv[1]), Path(sys.argv[2]); figs.mkdir(parents=True, exist_ok=True)
    l2, l3, l3e = load(sub, "level2_cv.json"), load(sub, "level3_cv.json"), load(sub, "level3_cv_entropy1.json")
    pipeline(figs / "pipeline.png")
    if l2: class_counts(l2, figs / "clase.png"); confusion(l2, figs / "confuzie.png")
    if l3: loco(l3, figs / "loco.png"); frac(l3, l3e, figs / "prag.png")
    print("figures:", sorted(p.name for p in figs.glob("*.png")))
