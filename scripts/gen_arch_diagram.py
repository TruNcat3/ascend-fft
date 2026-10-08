#!/usr/bin/env python3
"""Generate the 16:9 method overview used by README and the design guide."""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["svg.fonttype"] = "path"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "architecture.svg")
OUT_PNG = os.path.join(ROOT, "docs", "figures", "architecture.png")

INK = "#172B35"
MUTED = "#526671"
BLUE = "#167D9A"
GREEN = "#2A9D8F"
AMBER = "#E9A23B"
RED = "#D1495B"
PURPLE = "#6C5CE7"
PAPER = "#F8FAFB"


def card(ax, x, y, w, h, title, lines, color, title_size=13, body_size=10.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
        linewidth=1.4, edgecolor=color, facecolor="white", zorder=2))
    ax.add_patch(plt.Rectangle((x, y + h - 1.7), w, 1.7, color=color, zorder=3))
    ax.text(x + 2.2, y + h - 4.1, title, fontsize=title_size, fontweight="bold",
            color=INK, va="center", zorder=4)
    top = y + h - 8.0
    for index, line in enumerate(lines):
        ax.text(x + 2.2, top - index * 3.7, line, fontsize=body_size,
                color=MUTED, va="center", zorder=4)


def arrow(ax, start, end, color=INK, width=1.8, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16, linewidth=width,
        linestyle=(0, (4, 3)) if dashed else "solid", color=color, zorder=6,
        shrinkA=2, shrinkB=2))


def pill(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10.5,
            fontweight="bold", bbox=dict(boxstyle="round,pad=.45", fc=color, ec="none"),
            zorder=8)


def main():
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    ax.set_facecolor(PAPER)

    ax.text(6, 85, "Ascend-FFT: from butterfly constraints to hardware mapping", fontsize=22,
            fontweight="bold", color=INK, va="center")
    ax.text(6, 80.8, "The paradigm stays fixed; profiling and measurement choose mapping parameters and compute cores.",
            fontsize=11.5, color=MUTED, va="center")

    card(ax, 6, 58, 42, 18, "1  Data-only expansion", [
        "Cross-stage dependencies limit ready tiles",
        "Short dependency chains leave pipeline bubbles",
        "More batches do not remove per-tile dependencies",
    ], RED)
    card(ax, 59, 58, 42, 18, "2  Stage-only expansion", [
        "Intermediate state consumes on-chip UB",
        "Reordering, barriers, and coefficients grow",
        "Capacity overflow materializes boundaries in GM",
    ], AMBER)
    card(ax, 112, 58, 42, 18, "3  Roofline is not enough", [
        "Compute or bandwidth ceilings are not throughput",
        "Pipeline bubbles and launch cost remain hidden",
        "Dependencies, capacity, and scheduling interact",
    ], PURPLE)

    for x in (27, 80, 133):
        arrow(ax, (x, 57.2), (x, 51.8), color=MUTED)

    card(ax, 30, 34, 100, 17, "Parameterized hybrid dataflow", [
        "Stage dimension: Us (space) x Ts (time)     Data dimension: Ud (space) x Td (time)",
        "Layout, fusion, residence, and processing unit remain orthogonal choices",
        "Homogeneous subgraphs can stream; tile size, partition count, and core are not constants",
    ], BLUE, title_size=15, body_size=11.3)
    pill(ax, 18, 42.5, "dependency wall", RED)
    pill(ax, 142, 42.5, "capacity wall", AMBER)
    arrow(ax, (26, 42.5), (29.3, 42.5), color=RED)
    arrow(ax, (134, 42.5), (130.7, 42.5), color=AMBER)

    arrow(ax, (80, 33.2), (80, 28.7), color=BLUE, width=2.2)

    card(ax, 6, 9, 31, 18, "Hardware facts H", [
        "AIV / UB / MTE / GM",
        "Alignment, masks, and launch costs",
        "Measured probes; never extrapolated blindly",
    ], GREEN)
    card(ax, 47, 9, 31, 18, "Feasibility and cost Q", [
        "Enumeration keeps rejection reasons",
        "Structural counts estimate eta",
        "Hardware profile ranks candidates",
    ], PURPLE)
    card(ax, 88, 9, 31, 18, "Ascend lowering", [
        "AIV spatial work and batch folding",
        "UB residence, Gather, and MTE overlap",
        "Radix core is separate from architecture",
    ], AMBER)
    card(ax, 129, 9, 25, 18, "Plan", [
        "prepare",
        "measure",
        "run / R2C / C2R",
    ], BLUE)

    arrow(ax, (37.8, 18), (46.2, 18))
    arrow(ax, (78.8, 18), (87.2, 18))
    arrow(ax, (119.8, 18), (128.2, 18))
    arrow(ax, (141.5, 8.2), (62.5, 8.2), color=GREEN, dashed=True)
    ax.text(102, 4.8, "Measurement feedback: measured candidates outrank estimates", fontsize=10.5,
            color=GREEN, ha="center")

    ax.text(6, 2.4,
            "cuButterfly architecture paradigm -> Ascend AIV/UB/MTE lowering (no CUDA kernel reuse)",
            fontsize=9.5, color=MUTED)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT_PNG, format="png", dpi=100, facecolor=PAPER,
                metadata={"Software": "scripts/gen_arch_diagram.py"})
    fig.savefig(OUT, format="svg", facecolor=PAPER,
                metadata={"Date": None, "Creator": "scripts/gen_arch_diagram.py"})
    plt.close(fig)

    with open(OUT, encoding="utf-8") as handle:
        svg = handle.read()
    ids = sorted(set(re.findall(r'\bid="(p[0-9a-f]{8,})"', svg)))
    for index, old in enumerate(ids):
        new = f"clip{index}"
        svg = svg.replace(old, new).replace(f"url(#{old})", f"url(#{new})")
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(svg)
    print(f"-> {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")
    print(f"-> {os.path.relpath(OUT_PNG, ROOT)} ({os.path.getsize(OUT_PNG) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
