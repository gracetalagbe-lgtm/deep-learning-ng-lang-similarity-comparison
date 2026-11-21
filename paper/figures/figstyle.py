"""Shared publication-figure style for the paper.

Palette: validated categorical order (blue, orange, aqua, yellow) from the
project dataviz method; sequential blue ramp for magnitude/heatmaps; recessive
ink and hairline grids. All figures are print-friendly: hue choices are
colour-blind safe and every mark carries a direct label or linestyle secondary
encoding so identity survives grayscale printing.
"""
from __future__ import annotations

import matplotlib as mpl

# Categorical slots (fixed order, never cycled) - validated light-mode palette.
SERIES = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
}
SLOT_ORDER = ["blue", "orange", "aqua", "yellow"]

# Sequential blue ramp (light -> dark) for heatmaps / ordered stages.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#898781"        # axis labels / muted annotations
GRID = "#e1e0d9"         # hairline gridline
AXIS = "#c3c2b7"         # baseline / axis line
SURFACE = "#ffffff"      # paper figures render on white (journal print)

GOOD = "#006300"         # delta-good text (green, >= 3:1 on white)
CRIT = "#d03b3b"         # negative-result text


def apply_style() -> None:
    """Apply the shared rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "text.color": INK,
        }
    )
