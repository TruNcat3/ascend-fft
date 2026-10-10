#!/usr/bin/env python3
"""Render editable documentation SVGs to PNG previews and PDF exports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
DIAGRAMS = (
    "method_layers",
    "fft_space_time_example",
    "flow_core_space",
    "ascend_data_path_detail",
    "kplane_layout_example",
    "profile_to_plan",
)


def svg_dimensions(source: str) -> tuple[int, int]:
    match = re.search(r'<svg[^>]+width="(\d+)"[^>]+height="(\d+)"', source)
    if not match:
        raise ValueError("SVG must declare integer width and height")
    return int(match.group(1)), int(match.group(2))


def render(name: str) -> None:
    source_path = FIGURES / f"{name}.svg"
    source = source_path.read_text(encoding="utf-8")
    width, height = svg_dimensions(source)
    if "<title" not in source or "<desc" not in source:
        raise ValueError(f"{source_path} must contain accessible title and desc")

    cairosvg.svg2png(
        bytestring=source.encode("utf-8"),
        write_to=str(FIGURES / f"{name}.png"),
        output_width=width,
        output_height=height,
    )
    cairosvg.svg2pdf(
        bytestring=source.encode("utf-8"),
        write_to=str(FIGURES / f"{name}.pdf"),
        output_width=width,
        output_height=height,
    )
    print(f"rendered {name}: {width}x{height} SVG/PNG/PDF")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", choices=DIAGRAMS)
    args = parser.parse_args()
    for name in args.names or DIAGRAMS:
        render(name)


if __name__ == "__main__":
    main()
