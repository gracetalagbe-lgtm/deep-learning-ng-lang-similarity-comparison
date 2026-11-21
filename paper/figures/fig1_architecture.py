"""Figure 1 - End-to-end system architecture diagram (snake layout, 2 rows x 3 stages).

Grounded in the repository code map:
- data acquisition   src/afripara/data/sources.py + download.py (sha256 provenance)
- audit/normalise    src/afripara/data/audit.py, features/normalization.py
- split/construct    src/afripara/data/splits.py, construct.py, data/mt.py
- training           scripts/colab_train_transformers.py (29 runs, Colab T4)
- local scoring      src/afripara/evaluation/metrics.py
- calibration/demo   scripts/calibrate_models.py, inference/predictor.py, app/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, INK2, MUTED, SEQ, apply_style  # noqa: E402

STAGES = [
    ("1", "Data acquisition",
     "SemRel2024-Hausa, Menyo-20k,\nIGBONLP, AfriXNLI, PAWS-Wiki (en)\nsha256 provenance in data/raw/"),
    ("2", "Audit & normalisation",
     "clean_text: NFC, strip U+200B/\n200E/200F/FEFF, collapse spaces;\ndedup + split-leakage audit\n(644 shared train-test sentences)"),
    ("3", "Split & construction",
     "connected-component splitter\n(0 leakage; 2041/255/255);\nNLLB-600M translate-train;\nlabel-provenance columns"),
    ("4", "Training (Colab T4)",
     "29 runs: AfroXLMR ×12,\nAfriBERTa ×5, LaBSE ×7,\nAfriE5 ×4, EN-PAWS ×1;\nLR 2e-5, eff. batch 32, fp16, ≤5 ep"),
    ("5", "Local scoring",
     "dev-tuned thresholds (99-pt grid);\nSpearman/Pearson; McNemar exact;\npaired bootstrap (10k resamples);\n3-seed mean ± std"),
    ("6", "Calibration & interface",
     "temperature scaling fit on dev;\npredictor: calibrated prob +\ndev threshold; Streamlit demo\nserving all 3 languages"),
]


def draw() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    box_w, box_h = 29.0, 20.0
    xs = [3.0, 35.5, 68.0]
    ys = {"top": 31.0, "bot": 5.0}

    def box(x, y, num, title, body):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.4,rounding_size=1.0",
                linewidth=1.0, edgecolor=SEQ[4], facecolor="#f3f7fd",
            )
        )
        ax.add_patch(Circle((x + 3.2, y + box_h - 3.4), 1.9, facecolor=SEQ[4], edgecolor="none"))
        ax.text(x + 3.2, y + box_h - 3.45, num, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff")
        ax.text(x + 6.4, y + box_h - 3.4, title, ha="left", va="center",
                fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + 1.8, y + box_h - 7.2, body, ha="left", va="top",
                fontsize=6.1, color=INK2, linespacing=1.5)

    # Top row: 1 -> 2 -> 3 ; bottom row: 6 <- 5 <- 4 (snake flow)
    for i in range(3):
        num, title, body = STAGES[i]
        box(xs[i], ys["top"], num, title, body)
    for i in range(3):
        num, title, body = STAGES[3 + i]
        box(xs[2 - i], ys["bot"], num, title, body)

    def harrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.2, color=MUTED))

    top_y = ys["top"] + box_h / 2
    bot_y = ys["bot"] + box_h / 2
    harrow(xs[0] + box_w + 0.8, xs[1] - 0.8, top_y)
    harrow(xs[1] + box_w + 0.8, xs[2] - 0.8, top_y)
    # vertical drop from stage 3 to stage 4 (same column)
    ax.add_patch(FancyArrowPatch((xs[2] + box_w / 2, ys["top"] - 0.8),
                                 (xs[2] + box_w / 2, ys["bot"] + box_h + 0.8),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color=MUTED))
    harrow(xs[2] - 0.8, xs[1] + box_w + 0.8, bot_y)
    harrow(xs[1] - 0.8, xs[0] + box_w + 0.8, bot_y)

    ax.text(50, 54.0, "Colab T4 GPU (stage 4)  →  local CPU (stages 1–3, 5–6)",
            ha="center", fontsize=7.0, color=MUTED, style="italic")

    out = Path(__file__).resolve().parent / "fig1_architecture.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()
