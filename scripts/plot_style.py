"""Shared publication style for Ascend-FFT benchmark figures."""

from pathlib import Path

import matplotlib.pyplot as plt
COLORS = {
    "ours": "#167D9A",
    "native": "#D1495B",
    "e2e": "#2A9D8F",
    "application": "#E9A23B",
    "model": "#6C5CE7",
    "neutral": "#667085",
    "grid": "#D0D5DD",
}


def apply_publication_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "axes.edgecolor": "#344054",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_on_canvas(fig, path, dpi=160, padding=0.08):
    """Save at the figure's native aspect ratio for readable web embedding."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=padding, facecolor="white")
    plt.close(fig)
