"""Dark-theme charts for the 10-slide presentation (navy slide surface #0D1B2A).

All values read from project artifacts at runtime:
- dataset splits        data/processed CSV row counts (pandas)
- SemRel ranking        outputs/phase5/results.json (families) + outputs/baseline/results.json
- confusion matrix      outputs/colab/afroxlmr_hau_semrel_seed42/predictions_semrel_disjoint_test.json
- radar metrics         results.json families (acc/F1/ROC-AUC/Spearman on disjoint test)
- language-level bars   results.json families (constructed test accuracy)

Palette validated against #0D1B2A (dark mode): blue/orange/aqua/yellow dark steps.
Backgrounds transparent so the slide navy shows through.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parents[1] / "presentation_assets"
ASSETS.mkdir(exist_ok=True)

NAVY = "#0D1B2A"
INK = "#E8EDF4"
INK2 = "#A8B4C4"
GRID = "#24344A"
C = {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500"}
GOOD = "#0ca30c"

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.transparent": True, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9.5, "axes.titlecolor": INK,
})

RESULTS = json.loads((ROOT / "outputs/phase5/results.json").read_text())
BASELINE = json.loads((ROOT / "outputs/baseline/results.json").read_text())


def fam(f, s, m):
    return RESULTS["families"][f][s][m]


# --------------------------------------------------------- S4 dataset splits
def s4_datasets() -> None:
    paths = [
        ("SemRel-Hausa official", "data/processed/semrel_hau_official/{s}.csv", "human"),
        ("SemRel-Hausa disjoint", "data/processed/semrel_hau_disjoint/{s}.csv", "human"),
        ("Yoruba silver (translate-train)", "data/processed/constructed/translate_train_yor.csv", "silver"),
        ("Igbo silver (translate-train)", "data/processed/constructed/translate_train_ibo.csv", "silver"),
        ("Hausa silver (translate-train)", "data/processed/constructed/translate_train_hau.csv", "silver"),
    ]
    rows = []
    for name, pat, kind in paths:
        if "{s}" in pat:
            counts = [len(pd.read_csv(ROOT / pat.format(s=k))) for k in ("train", "dev", "test")]
            pos = sum(pd.read_csv(ROOT / pat.format(s=k))["score"].ge(0.5).sum() for k in ("train", "dev", "test"))
        else:
            df = pd.read_csv(ROOT / pat)
            counts = [int((df["split"] == k).sum()) for k in ("train", "dev", "test")]
            pos = int(df["label"].eq(1).sum())
        rows.append((name, *counts, sum(counts), round(100 * pos / sum(counts), 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.6), width_ratios=[1.35, 1])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, (lab, col) in enumerate(zip(("train", "dev", "test"), (C["blue"], C["aqua"], C["yellow"]))):
        vals = np.array([r[1 + k] for r in rows], dtype=float)
        ax1.barh(y, vals, left=left, height=0.58, color=col, label=lab)
        for yi, v, l in zip(y, vals, left):
            if v > 200:
                ax1.text(l + v / 2, yi, f"{int(v):,}", ha="center", va="center",
                         fontsize=7.5, color="#0D1B2A" if col != C["yellow"] else "#0D1B2A")
        left += vals
    ax1.set_yticks(y, names, fontsize=8.5)
    ax1.set_xlim(0, 3450)
    ax1.set_xlabel("sentence pairs")
    ax1.set_title("(a) Split sizes (pairs) — disjoint re-split uses 80/10/10 folds", fontsize=9)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=8, frameon=False)
    ax1.grid(axis="x", color=GRID, linewidth=0.6)
    ax1.set_axisbelow(True)

    pos = [r[5] for r in rows]
    bars = ax2.barh(y, pos, 0.58, color=[C["blue"], C["blue"], C["orange"], C["orange"], C["orange"]])
    ax2.axvline(50, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax2.text(50.6, 4.42, "50%", fontsize=7.5, color=INK2)
    for b, v in zip(bars, pos):
        ax2.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
    ax2.set_yticks(y, names, fontsize=8.5)
    ax2.set_xlim(0, 75)
    ax2.set_xlabel("positive-class share (%)")
    ax2.set_title("(b) Class balance (binarised labels)", fontsize=9)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(ASSETS / "s4_datasets.png")
    plt.close(fig)
    print("s4 done")


# ------------------------------------------------------- S6 SemRel ranking
def s6_ranking() -> None:
    rows = [
        ("AfroXLMR-FT (human labels)", fam("afroxlmr_hau_semrel", "semrel_disjoint_test", "spearman_mean"), C["blue"]),
        ("LaBSE fine-tuned", fam("labse_finetune_hau_semrel", "semrel_disjoint_test", "spearman_mean"), C["blue"]),
        ("AfriE5 (zero-shot)", fam("afrie5_zeroshot_semrel", "semrel_disjoint_test", "spearman_mean"), C["aqua"]),
        ("LaBSE (zero-shot)", fam("labse_zeroshot_semrel", "semrel_disjoint_test", "spearman_mean"), C["aqua"]),
        ("TF-IDF + cosine", BASELINE["sets"]["semrel_hau_disjoint"]["similarity_vs_gold"]["spearman"], "#7C8CA5"),
        ("AfriBERTa-FT", fam("afriberta_hau_semrel", "semrel_disjoint_test", "spearman_mean"), C["aqua"]),
        ("AfroXLMR-FT (silver labels)", fam("afroxlmr_hau_constructed", "semrel_disjoint_test", "spearman_mean"), C["yellow"]),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    y = np.arange(len(rows))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for b, r in zip(bars, rows):
        ax.text(r[1] + 0.012, b.get_y() + b.get_height() / 2, f"{r[1]:.3f}",
                va="center", fontsize=8, color=INK)
    ax.axvline(0.69, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.695, -0.42, "published fine-tuned LaBSE (0.69)", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Spearman ρ vs SemRel-Hausa gold — leakage-free disjoint test (n = 255)")
    ax.tick_params(axis="y", labelsize=8.3)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "s6_ranking.png")
    plt.close(fig)
    print("s6 ranking done")


# ------------------------------------------------------- S6 confusion (Hausa)
def s6_confusion() -> None:
    rid = "afroxlmr_hau_semrel_seed42"
    pred = json.loads((ROOT / "outputs/colab" / rid / "predictions_semrel_disjoint_test.json").read_text())
    thr = RESULTS["runs"][rid]["threshold"]
    y = np.array(pred["labels"])
    p = (np.array(pred["scores"]) >= thr).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for g, q in zip(y, p):
        cm[g, q] += 1
    norm = cm / cm.sum(axis=1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("navy", ["#12233B", "#3987e5", "#B7D3F6"])
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for gi in range(2):
        for qi in range(2):
            frac = norm[gi, qi]
            ax.text(qi, gi, f"{labels[gi][qi]}\n{cm[gi, qi]}  ({frac:.2f})", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#0D1B2A" if frac > 0.55 else INK)
    ax.set_xticks([0, 1], ["pred: not para.", "pred: paraphrase"], fontsize=8)
    ax.set_yticks([0, 1], ["gold: not para.", "gold: paraphrase"], fontsize=8)
    ax.set_title(f"Hausa (human labels), tuned threshold τ = {thr:.2f}", fontsize=8.5)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "s6_confusion.png")
    plt.close(fig)
    print("s6 confusion done")


# ---------------------------------------------------------------- S7 radar
def s7_radar() -> None:
    systems = [
        ("AfroXLMR-FT", [fam("afroxlmr_hau_semrel", "semrel_disjoint_test", m) for m in
                         ("accuracy_mean", "f1_mean", "roc_auc_mean", "spearman_mean")], C["blue"]),
        ("LaBSE-FT", [fam("labse_finetune_hau_semrel", "semrel_disjoint_test", m) for m in
                      ("accuracy_mean", "f1_mean", "roc_auc_mean", "spearman_mean")], C["orange"]),
        ("AfriE5 zero-shot", [fam("afrie5_zeroshot_semrel", "semrel_disjoint_test", m) for m in
                              ("accuracy_mean", "f1_mean", "roc_auc_mean", "spearman_mean")], C["aqua"]),
        ("TF-IDF + cosine", [BASELINE["sets"]["semrel_hau_disjoint"]["test_metrics"][m] for m in
                             ("accuracy", "f1", "roc_auc")] +
         [BASELINE["sets"]["semrel_hau_disjoint"]["similarity_vs_gold"]["spearman"]], "#7C8CA5"),
    ]
    axes = ["Accuracy", "F1", "ROC-AUC", "Spearman ρ"]
    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.3), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for name, vals, col in systems:
        v = vals + vals[:1]
        ax.plot(ang, v, color=col, linewidth=1.6, label=name)
        ax.fill(ang, v, color=col, alpha=0.10)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes, fontsize=8.5, color=INK)
    ax.set_ylim(0.3, 0.95)
    ax.set_yticks([0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.4", "0.6", "0.8"], fontsize=7, color=INK2)
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines["polar"].set_color(GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "s7_radar.png")
    plt.close(fig)
    print("s7 radar done")


# ------------------------------------------------------- S8 language bars
def s8_language() -> None:
    sets = {"yor": "constructed_test_yor", "ibo": "constructed_test_ibo", "hau": "constructed_test_hau"}
    groups = [
        ("In-language silver FT", [fam(f"afroxlmr_{k}_constructed", s, "accuracy_mean") for k, s in sets.items()], C["orange"]),
        ("EN-PAWS transfer (zero-shot)", [fam("afroxlmr_en_paws", s, "accuracy_mean") for s in sets.values()], C["blue"]),
        ("TF-IDF + cosine", [BASELINE["sets"][f"translate_train_{k}"]["test_metrics"]["accuracy"] for k in sets], "#7C8CA5"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(3)
    w = 0.26
    sig = {0: "n.s.", 1: "p<10⁻⁴", 2: "p<10⁻⁴"}  # silver-vs-baseline McNemar, min-p across seeds
    for gi, (name, vals, col) in enumerate(groups):
        bars = ax.bar(x + (gi - 1) * w, vals, w, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center",
                    fontsize=7.3, rotation=90, color=INK)

    ax.set_xticks(x, ["Yoruba", "Igbo", "Hausa"], fontsize=9.5)
    ax.set_ylim(0.40, 1.0)
    ax.set_ylabel("accuracy (constructed test folds)")
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "s8_language.png")
    plt.close(fig)
    print("s8 done")


if __name__ == "__main__":
    s4_datasets()
    s6_ranking()
    s6_confusion()
    s7_radar()
    s8_language()
