"""Generate Figures 2-6 for the paper from the project's real artifacts.

Every number plotted is read from an artifact at runtime:
- Fig 2  training curves      <- outputs/colab/<run>/history.json (seed 42, 3 arms)
- Fig 3  confusion matrices   <- outputs/colab/<run>/predictions_*.json + tuned thresholds
- Fig 4  SemRel Spearman ranking <- outputs/phase5/results.json families + baseline results
- Fig 5  reliability diagrams <- predictions_dev.json + outputs/calibration/<run>.json
- Fig 6  leakage flip         <- results.json + baseline results.json (official vs disjoint)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from figstyle import CRIT, GOOD, INK, INK2, MUTED, SEQ, SERIES, SLOT_ORDER, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
OUT5 = ROOT / "outputs" / "phase5"
RESULTS = json.loads((OUT5 / "results.json").read_text())
BASELINE = json.loads((ROOT / "outputs/baseline/results.json").read_text())


def fam(family: str, set_name: str, metric: str) -> float:
    return RESULTS["families"][family][set_name][metric]


def run(run_id: str, set_name: str, metric: str) -> float:
    return RESULTS["runs"][run_id]["sets"][set_name][metric]


# ---------------------------------------------------------------- Figure 2
def fig2_training_curves() -> None:
    runs = [
        ("afroxlmr_hau_semrel_seed42", "(a) Hausa — SemRel (human labels)"),
        ("afroxlmr_yor_constructed_seed42", "(b) Yorùbá — constructed (silver)"),
        ("afroxlmr_ibo_constructed_seed42", "(c) Igbo — constructed (silver)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    for j, (rid, title) in enumerate(runs):
        hist = json.loads((ROOT / "outputs/colab" / rid / "history.json").read_text())
        log = hist["log_history"]
        tr = [e for e in log if "loss" in e and "eval_loss" not in e]
        ev = [e for e in log if "eval_loss" in e]
        ax = axes[0, j]
        ax.plot([t["step"] for t in tr], [t["loss"] for t in tr],
                color=SERIES["blue"], linewidth=1.4, label="train loss")
        ax.plot([t["step"] for t in ev], [t["eval_loss"] for t in ev],
                "o", markersize=3.5, color=SERIES["orange"], label="dev loss")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0.3, 0.85)
        if j == 0:
            ax.set_ylabel("cross-entropy loss")
            ax.legend(loc="lower left", fontsize=6.2)
        best_ep = int(np.argmax([e["eval_f1"] for e in ev])) + 1
        ax2 = axes[1, j]
        ax2.plot([e["epoch"] for e in ev], [e["eval_f1"] for e in ev],
                 color=SERIES["blue"], linewidth=1.4, marker="o", markersize=3.5)
        ax2.plot(ev[best_ep - 1]["epoch"], ev[best_ep - 1]["eval_f1"], "*",
                 markersize=9, color=SERIES["orange"], zorder=5)
        ax2.annotate(f"best {ev[best_ep-1]['eval_f1']:.3f}",
                     (ev[best_ep - 1]["epoch"], ev[best_ep - 1]["eval_f1"]),
                     textcoords="offset points", xytext=(8, 8), fontsize=6.2, color=INK2)
        ax2.set_xlabel("epoch")
        ax2.set_ylim(0.0, 1.0)
        if j == 0:
            ax2.set_ylabel("dev F1")
    for ax in axes[0]:
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    for ax in axes[1]:
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_xlim(0.8, 5.2)
        ax.set_xticks([1, 2, 3, 4, 5])
    fig.tight_layout(h_pad=1.2)
    fig.savefig(FIG / "fig2_training_curves.png")
    plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------- Figure 3
def fig3_confusion() -> None:
    cells = [
        ("yor", "afroxlmr_yor_constructed_seed42", "predictions_constructed_test_yor.json"),
        ("ibo", "afroxlmr_ibo_constructed_seed42", "predictions_constructed_test_ibo.json"),
        ("hau", "afroxlmr_hau_semrel_seed42", "predictions_semrel_disjoint_test.json"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, (lang, rid, pf) in zip(axes, cells):
        pred = json.loads((ROOT / "outputs/colab" / rid / pf).read_text())
        thr = RESULTS["runs"][rid]["threshold"]
        y = np.array(pred["labels"])
        p = (np.array(pred["scores"]) >= thr).astype(int)
        cm = np.zeros((2, 2), dtype=int)
        for g, q in zip(y, p):
            cm[g, q] += 1
        norm = cm / cm.sum(axis=1, keepdims=True)
        labels = ["gold 0\n(not paraphrase)", "gold 1\n(paraphrase)"]
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for gi in range(2):
            for qi in range(2):
                frac = norm[gi, qi]
                ax.text(qi, gi, f"{cm[gi, qi]}\n({frac:.2f})", ha="center", va="center",
                        fontsize=7, color="#ffffff" if frac > 0.55 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(labels, fontsize=6.4)
        ax.set_title(f"{'(a) Yorùbá' if lang == 'yor' else '(b) Igbo' if lang == 'ibo' else '(c) Hausa'}"
                     f" — thr {thr:.2f}", fontsize=8)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("row-normalised rate", fontsize=7)
    fig.savefig(FIG / "fig3_confusion.png")
    plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------- Figure 4
def fig4_spearman_ranking() -> None:
    rows = [
        ("AfroXLMR-FT (human labels)", fam("afroxlmr_hau_semrel", "semrel_disjoint_test", "spearman_mean"), SERIES["blue"]),
        ("LaBSE-FT", fam("labse_finetune_hau_semrel", "semrel_disjoint_test", "spearman_mean"), SERIES["orange"]),
        ("AfriE5 (zero-shot)", fam("afrie5_zeroshot_semrel", "semrel_disjoint_test", "spearman_mean"), SERIES["orange"]),
        ("LaBSE (zero-shot)", fam("labse_zeroshot_semrel", "semrel_disjoint_test", "spearman_mean"), SERIES["aqua"]),
        ("TF-IDF + cosine", BASELINE["sets"]["semrel_hau_disjoint"]["similarity_vs_gold"]["spearman"], "#9a9891"),
        ("AfriBERTa-FT", fam("afriberta_hau_semrel", "semrel_disjoint_test", "spearman_mean"), SERIES["aqua"]),
        ("AfroXLMR-FT (silver labels)", fam("afroxlmr_hau_constructed", "semrel_disjoint_test", "spearman_mean"), SERIES["yellow"]),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    bars = ax.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=6.8, color=INK)
    ax.axvline(0.69, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.692, 6.45, "published fine-tuned LaBSE, SemRel-Hausa (0.69)",
            fontsize=6.2, color=MUTED, va="center")
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Spearman ρ vs SemRel-Hausa gold scores — leakage-free disjoint test (n = 255)")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig4_spearman_ranking.png")
    plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------- Figure 5
def _ece_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, edges[1:-1], right=False)
    conf, acc, frac = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); frac.append(0.0)
            continue
        conf.append(probs[m].mean())
        acc.append(y[m].mean())
        frac.append(m.sum() / len(probs))
    return np.array(conf), np.array(acc), np.array(frac)


def fig5_reliability() -> None:
    cases = [
        ("afroxlmr_hau_semrel_seed42", "(a) Hausa (human labels) — T = 1.66"),
        ("afroxlmr_yor_constructed_seed42", "(b) Yorùbá (silver) — T = 3.46"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (rid, title) in zip(axes, cases):
        pred = json.loads((ROOT / "outputs/colab" / rid / "predictions_dev.json").read_text())
        p = np.clip(np.array(pred["scores"], dtype=float), 1e-6, 1 - 1e-6)
        y = np.array(pred["labels"])
        cal = json.loads((ROOT / "outputs/calibration" / f"{rid}.json").read_text())
        t = cal["temperature"]
        pc = 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / t))
        for arr, col, name in ((p, SERIES["aqua"], f"before (ECE {cal['ece_before']:.3f})"),
                               (pc, SERIES["blue"], f"after (ECE {cal['ece_after']:.3f})")):
            conf, acc, _ = _ece_bins(arr, y)
            ax.plot(conf, acc, "o-", markersize=3.5, linewidth=1.2, color=col, label=name)
        ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=MUTED, linewidth=0.9)
        ax.set_xlabel("mean predicted probability (10 bins, dev)")
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.legend(loc="upper left", fontsize=6.2)
    axes[0].set_ylabel("observed positive rate")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_reliability.png")
    plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------- Figure 6
def fig6_leakage() -> None:
    systems = [
        ("AfroXLMR-FT\n(human labels)", "afroxlmr_hau_semrel"),
        ("AfriBERTa-FT\n(human labels)", "afriberta_hau_semrel"),
        ("AfroXLMR-FT\n(silver)", "afroxlmr_hau_constructed"),
        ("TF-IDF + cosine", None),
    ]
    official, disjoint = [], []
    for _, f in systems:
        if f is None:
            official.append(BASELINE["sets"]["semrel_hau_official"]["test_metrics"]["accuracy"])
            disjoint.append(BASELINE["sets"]["semrel_hau_disjoint"]["test_metrics"]["accuracy"])
        else:
            official.append(fam(f, "semrel_official_test", "accuracy_mean"))
            disjoint.append(fam(f, "semrel_disjoint_test", "accuracy_mean"))
    # Spearman panel: systems with continuous gold scores on both splits
    sp_systems = [
        ("AfroXLMR-FT\n(human)", "afroxlmr_hau_semrel"),
        ("LaBSE-FT", "labse_finetune_hau_semrel"),
        ("AfriE5\n(zero-shot)", "afrie5_zeroshot_semrel"),
        ("TF-IDF", None),
    ]
    sp_off, sp_dis = [], []
    for _, f in sp_systems:
        if f is None:
            sp_off.append(BASELINE["sets"]["semrel_hau_official"]["similarity_vs_gold"]["spearman"])
            sp_dis.append(BASELINE["sets"]["semrel_hau_disjoint"]["similarity_vs_gold"]["spearman"])
        else:
            sp_off.append(fam(f, "semrel_official_test", "spearman_mean"))
            sp_dis.append(fam(f, "semrel_disjoint_test", "spearman_mean"))

    x = np.arange(len(systems))
    w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for off in range(2):
        vals = official if off == 0 else disjoint
        col = SERIES["blue"] if off == 0 else SERIES["orange"]
        name = "official split (leaky)" if off == 0 else "disjoint split (clean)"
        bars = ax1.bar(x + (off - 0.5) * w, vals, w, color=col, label=name)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                     ha="center", fontsize=5.8, rotation=90, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in systems], fontsize=6.2)
    ax1.set_ylim(0.4, 0.98)
    ax1.set_ylabel("accuracy (3-seed mean)")
    ax1.set_title("(a) Accuracy: official vs disjoint split", fontsize=8)
    ax1.legend(fontsize=6.2, loc="upper right")
    ax1.grid(axis="y", color="#e1e0d9", linewidth=0.6)

    x2 = np.arange(len(sp_systems))
    for off in range(2):
        vals = sp_off if off == 0 else sp_dis
        col = SERIES["blue"] if off == 0 else SERIES["orange"]
        bars = ax2.bar(x2 + (off - 0.5) * w, vals, w, color=col)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                     ha="center", fontsize=5.8, rotation=90, color=INK)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([s[0] for s in sp_systems], fontsize=6.2)
    ax2.set_ylim(0.3, 0.88)
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("(b) Ranking: official vs disjoint split", fontsize=8)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig6_leakage_flip.png")
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    apply_style()
    fig2_training_curves()
    fig3_confusion()
    fig4_spearman_ranking()
    fig5_reliability()
    fig6_leakage()
