"""Shared publication style for Ascend-FFT benchmark figures."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

CANVAS = (1600, 900)
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


def save_on_canvas(fig, path, canvas=CANVAS, padding=36):
    """Save a plot on a fixed canvas without stretching its contents."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    image.thumbnail((canvas[0] - 2 * padding, canvas[1] - 2 * padding), Image.Resampling.LANCZOS)
    output = Image.new("RGBA", canvas, "white")
    output.alpha_composite(image, ((canvas[0] - image.width) // 2,
                                   (canvas[1] - image.height) // 2))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(target, optimize=True)
