#!/usr/bin/env python3
"""Generate the Week 7 distribution diagram using Matplotlib only."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


WIDTH_PX = 1388
HEIGHT_PX = 476
DPI = 200

X_MAX = 100
Y_MAX = 34

COLORS = {
    "background": "#e8e8e8",
    "model_fill": "#cbc3da",
    "model_edge": "#7a7485",
    "sample_fill": "#d5e1cf",
    "sample_edge": "#8f9588",
    "arrow_fill": "#f0e3b3",
    "arrow_edge": "#8e866f",
    "truth_fill": "#c6d7f2",
    "truth_edge": "#6f7f9a",
    "text": "#111111",
    "arc": "#5979a0",
}


def add_text(ax, x, y, text, fontsize=16, rotation=0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        family="DejaVu Sans",
        rotation=rotation,
    )


def add_arrow(
    ax,
    start,
    end,
    label,
    *,
    tail_width=3.8,
    head_width=6.5,
    head_length=4.2,
    label_shift=0.0,
    fontsize=13,
):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    neck_x = x2 - ux * head_length
    neck_y = y2 - uy * head_length

    points = np.array(
        [
            [x1 + px * tail_width / 2, y1 + py * tail_width / 2],
            [neck_x + px * tail_width / 2, neck_y + py * tail_width / 2],
            [neck_x + px * head_width / 2, neck_y + py * head_width / 2],
            [x2, y2],
            [neck_x - px * head_width / 2, neck_y - py * head_width / 2],
            [neck_x - px * tail_width / 2, neck_y - py * tail_width / 2],
            [x1 - px * tail_width / 2, y1 - py * tail_width / 2],
        ]
    )

    arrow = Polygon(
        points,
        closed=True,
        facecolor=COLORS["arrow_fill"],
        edgecolor=COLORS["arrow_edge"],
        linewidth=1.0,
        joinstyle="miter",
    )
    ax.add_patch(arrow)

    label_x = x1 + dx * 0.48 + px * label_shift
    label_y = y1 + dy * 0.48 + py * label_shift
    rotation = math.degrees(math.atan2(dy, dx))
    if rotation > 90 or rotation < -90:
        rotation += 180
    add_text(ax, label_x, label_y, label, fontsize=fontsize, rotation=rotation)


def add_cloud(ax, center, width, height):
    cx, cy = center
    rx = width / 2
    ry = height / 2

    t = np.linspace(0, 2 * np.pi, 320, endpoint=False)
    scallop = 1.0 + 0.08 * np.sin(6 * t) + 0.03 * np.sin(12 * t + 0.7)
    x = cx + rx * np.cos(t) * scallop
    y = cy + ry * np.sin(t) * scallop

    vertices = np.column_stack([x, y])
    vertices = np.vstack([vertices, vertices[0]])

    codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY

    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=COLORS["truth_fill"],
        edgecolor=COLORS["truth_edge"],
        linewidth=1.0,
    )
    ax.add_patch(patch)


def generate_diagram(output_path: Path) -> Path:
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(COLORS["background"])

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])

    # Small clipped arc at the top, matching the provided reference.
    ax.add_patch(
        Ellipse(
            (65.8, 34.8),
            width=2.4,
            height=1.8,
            fill=False,
            edgecolor=COLORS["arc"],
            linewidth=1.6,
        )
    )

    ax.add_patch(
        Ellipse(
            (16.0, 16.2),
            width=20.0,
            height=9.6,
            facecolor=COLORS["model_fill"],
            edgecolor=COLORS["model_edge"],
            linewidth=1.0,
        )
    )
    add_text(ax, 16.0, 16.2, "model", fontsize=18)

    ax.add_patch(
        Rectangle(
            (35.5, 18.3),
            width=24.0,
            height=11.2,
            facecolor=COLORS["sample_fill"],
            edgecolor=COLORS["sample_edge"],
            linewidth=1.0,
        )
    )
    add_text(ax, 47.5, 23.9, "Empirical Data\nSample", fontsize=18)

    ax.add_patch(
        Rectangle(
            (35.3, 1.8),
            width=24.3,
            height=11.1,
            facecolor=COLORS["sample_fill"],
            edgecolor=COLORS["sample_edge"],
            linewidth=1.0,
        )
    )
    add_text(ax, 47.45, 7.35, "New\nSample", fontsize=18)

    add_cloud(ax, center=(82.0, 22.0), width=19.8, height=9.8)
    add_text(ax, 82.0, 22.0, "Hidden\nTruth", fontsize=18)

    add_arrow(ax, (34.0, 25.7), (24.8, 23.1), "learn", label_shift=0.35, fontsize=14)
    add_arrow(ax, (24.0, 10.5), (34.9, 7.7), "predict", label_shift=-0.35, fontsize=14)
    add_arrow(ax, (75.8, 24.8), (60.0, 24.2), "i.i.d.", label_shift=0.25, fontsize=15)
    add_arrow(ax, (75.6, 9.4), (60.0, 8.2), "i.i.d.", label_shift=0.25, fontsize=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("distribution.png")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output PNG path (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_diagram(args.output.resolve())
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
