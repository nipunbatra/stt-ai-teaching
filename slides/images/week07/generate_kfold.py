#!/usr/bin/env python3
"""Generate the K-Fold Cross-Validation diagram using Matplotlib."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


DPI = 200
K = 5

TRAIN_COLOR = "#4a7fb5"
VAL_COLOR = "#c0504d"
BG_COLOR = "#ffffff"
TEXT_COLOR = "#222222"


def generate_kfold_diagram(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    box_w = 1.0
    box_h = 0.6
    gap_x = 0.15
    gap_y = 0.35
    left_margin = 1.8
    top = K * (box_h + gap_y)

    for fold in range(K):
        y = top - (fold + 1) * (box_h + gap_y)
        # Label
        ax.text(
            left_margin - 0.3, y + box_h / 2,
            f"Fold {fold + 1}",
            ha="right", va="center",
            fontsize=13, fontweight="bold", color=TEXT_COLOR,
        )
        for j in range(K):
            x = left_margin + j * (box_w + gap_x)
            is_val = (j == fold)
            color = VAL_COLOR if is_val else TRAIN_COLOR
            label = "Val" if is_val else "Train"
            rect = mpatches.FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                x + box_w / 2, y + box_h / 2,
                label,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
            )
        # Score label on the right
        score_x = left_margin + K * (box_w + gap_x) + 0.1
        ax.text(
            score_x, y + box_h / 2,
            f"  score_{fold + 1}",
            ha="left", va="center",
            fontsize=12, color=TEXT_COLOR, family="monospace",
        )

    # Title
    ax.text(
        left_margin + K * (box_w + gap_x) / 2 - gap_x / 2,
        top + 0.35,
        "K-Fold Cross-Validation  (K = 5)",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=TEXT_COLOR,
    )

    # Bottom formula
    formula_y = -0.35
    formula_x = left_margin + K * (box_w + gap_x) / 2 - gap_x / 2
    box_text = r"CV Score $= \frac{1}{K}\sum_{k=1}^{K}$ score$_k$"
    ax.text(
        formula_x, formula_y, box_text,
        ha="center", va="center",
        fontsize=13, color=TEXT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#f0f0f0", edgecolor="#aaaaaa", linewidth=1.2,
        ),
    )

    # Legend
    legend_y = formula_y
    legend_x = left_margin + K * (box_w + gap_x) + 0.6
    for color, label in [(TRAIN_COLOR, "Train"), (VAL_COLOR, "Validation")]:
        rect = mpatches.FancyBboxPatch(
            (legend_x, legend_y - 0.15), 0.3, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            legend_x + 0.45, legend_y,
            label, ha="left", va="center",
            fontsize=10, color=TEXT_COLOR,
        )
        legend_y -= 0.45

    x_max = left_margin + K * (box_w + gap_x) + 2.5
    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.0, top + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("kfold_diagram.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    out = generate_kfold_diagram(args.output.resolve())
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
