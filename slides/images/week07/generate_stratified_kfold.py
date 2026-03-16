#!/usr/bin/env python3
"""Generate the Stratified K-Fold From Scratch diagram using Matplotlib.

Recreates the 4-step visual:
  1. Separate by class
  2. Shuffle within each class
  3. Deal round-robin into K=5 folds
  4. Result — each fold preserves the class ratio
"""

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

# Colors
CLASS_A_COLOR = "#2171b5"   # blue
CLASS_B_COLOR = "#e6851f"   # orange
CLASS_A_LIGHT = "#c6dbef"
CLASS_B_LIGHT = "#fdd0a2"
BG_COLOR = "#ffffff"
TEXT_COLOR = "#222222"
HEADER_COLOR = "#333333"

# Triangle for class B
def draw_triangle(ax, cx, cy, size, color, text, text_color="white", fontsize=8):
    """Draw an upward-pointing triangle centered at (cx, cy)."""
    half = size / 2
    tri = plt.Polygon(
        [(cx, cy + half), (cx - half, cy - half * 0.6), (cx + half, cy - half * 0.6)],
        closed=True, facecolor=color, edgecolor="white", linewidth=1.2,
    )
    ax.add_patch(tri)
    ax.text(cx, cy - 0.02, str(text), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def draw_square(ax, cx, cy, size, color, text, text_color="white", fontsize=8):
    """Draw a rounded square centered at (cx, cy)."""
    half = size / 2
    rect = mpatches.FancyBboxPatch(
        (cx - half, cy - half), size, size,
        boxstyle="round,pad=0.03",
        facecolor=color, edgecolor="white", linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, str(text), ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def draw_sample(ax, cx, cy, size, cls, idx, fontsize=8):
    """Draw a sample marker (square for A, triangle for B)."""
    if cls == "A":
        draw_square(ax, cx, cy, size, CLASS_A_COLOR, idx, fontsize=fontsize)
    else:
        draw_triangle(ax, cx, cy, size, CLASS_B_COLOR, idx, fontsize=fontsize)


def section_header(ax, x, y, text, fontsize=10):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=HEADER_COLOR)


def generate_diagram(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    S = 0.48  # sample marker size
    SP = 0.60  # spacing between samples

    # ─── INITIAL DATASET ───
    section_header(ax, 0.2, 8.6, "INITIAL DATASET (15 Samples)")
    ax.text(0.2, 8.25, "70% Class A, 30% Class B Ratio", fontsize=7, color="#666666")

    # Row 1: 7 Class A
    init_row1_y = 7.7
    for i in range(7):
        draw_square(ax, 0.5 + i * SP, init_row1_y, S, CLASS_A_COLOR, i, fontsize=7)
    # Row 2: 3 Class A + 5 Class B
    init_row2_y = 7.1
    for i in range(3):
        draw_square(ax, 0.5 + i * SP, init_row2_y, S, CLASS_A_COLOR, 7 + i, fontsize=7)
    for i in range(5):
        draw_triangle(ax, 2.5 + i * SP, init_row2_y, S, CLASS_B_COLOR, 10 + i, fontsize=7)

    # Legend
    draw_square(ax, 5.5, 8.6, 0.3, CLASS_A_COLOR, "", fontsize=5)
    ax.text(5.75, 8.6, "Class A (10, 70%)", fontsize=7, va="center", color=TEXT_COLOR)
    draw_triangle(ax, 5.5, 8.25, 0.3, CLASS_B_COLOR, "", fontsize=5)
    ax.text(5.75, 8.25, "Class B (5, 30%)", fontsize=7, va="center", color=TEXT_COLOR)

    # ─── STEP 1: SEPARATE BY CLASS ───
    s1_x = 0.2
    s1_y = 6.3
    section_header(ax, s1_x, s1_y, "STEP 1: SEPARATE BY CLASS")

    ax.text(s1_x, s1_y - 0.4, "Class A (10 samples)", fontsize=7, color="#666666")
    for i in range(10):
        draw_square(ax, 0.5 + i * SP, s1_y - 0.85, S, CLASS_A_COLOR, i, fontsize=7)

    ax.text(s1_x, s1_y - 1.25, "Class B (5 samples)", fontsize=7, color="#666666")
    for i in range(5):
        draw_triangle(ax, 0.5 + i * SP, s1_y - 1.7, S, CLASS_B_COLOR, 10 + i, fontsize=7)

    # ─── STEP 2: SHUFFLE WITHIN EACH CLASS ───
    s2_x = 7.2
    s2_y = 8.6
    section_header(ax, s2_x, s2_y, "STEP 2: SHUFFLE WITHIN EACH CLASS")

    # Shuffled Class A
    shuffled_a = [8, 2, 3, 0, 9, 4, 7, 5, 1, 6]
    ax.text(s2_x, s2_y - 0.4, "Class A (shuffled)", fontsize=7, color="#666666")
    for i, idx in enumerate(shuffled_a):
        draw_square(ax, 7.5 + i * SP, s2_y - 0.85, S, CLASS_A_COLOR, idx, fontsize=7)

    # Shuffled Class B
    shuffled_b = [13, 10, 14, 12, 11]
    ax.text(s2_x, s2_y - 1.25, "Class B (shuffled)", fontsize=7, color="#666666")
    for i, idx in enumerate(shuffled_b):
        draw_triangle(ax, 7.5 + i * SP, s2_y - 1.7, S, CLASS_B_COLOR, idx, fontsize=7)

    # ─── STEP 3: DEAL ROUND-ROBIN INTO K=5 FOLDS ───
    s3_x = 7.2
    s3_y = 6.3
    section_header(ax, s3_x, s3_y, "STEP 3: DEAL ROUND-ROBIN INTO K=5 FOLDS")

    # Class A dealt round-robin into 5 folds (2 per fold)
    ax.text(s3_x, s3_y - 0.35, "Class A → 2 per fold:", fontsize=7, color="#666666")
    fold_a = [[8, 2], [3, 0], [9, 4], [7, 5], [1, 6]]
    for f in range(5):
        fx = 7.5 + f * 1.3
        for j, idx in enumerate(fold_a[f]):
            draw_square(ax, fx + j * SP, s3_y - 0.8, S * 0.9, CLASS_A_COLOR, idx, fontsize=6)

    # Class B dealt round-robin (1 per fold)
    ax.text(s3_x, s3_y - 1.2, "Class B → 1 per fold:", fontsize=7, color="#666666")
    fold_b = [[13], [10], [14], [12], [11]]
    for f in range(5):
        fx = 7.5 + f * 1.3
        for j, idx in enumerate(fold_b[f]):
            draw_triangle(ax, fx + j * SP * 0.5 + 0.15, s3_y - 1.65, S * 0.9, CLASS_B_COLOR, idx, fontsize=6)

    # Fold labels
    for f in range(5):
        fx = 7.5 + f * 1.3 + 0.15
        ax.text(fx, s3_y - 2.05, f"Fold {f+1}", fontsize=6, ha="center",
                fontweight="bold", color="#555555")

    # ─── STEP 4: RESULT ───
    s4_x = 7.2
    s4_y = 3.8
    section_header(ax, s4_x, s4_y, "STEP 4: RESULT")

    fold_labels = ["FOLD 1", "FOLD 2", "FOLD 3", "FOLD 4", "FOLD 5"]
    for f in range(5):
        fx = 7.5 + f * 1.3
        # Fold header
        ax.text(fx + 0.15, s4_y - 0.35, fold_labels[f], fontsize=7, ha="center",
                fontweight="bold", color=TEXT_COLOR)

        # Draw box around fold
        rect = mpatches.FancyBboxPatch(
            (fx - 0.3, s4_y - 1.85), 1.1, 1.3,
            boxstyle="round,pad=0.06",
            facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=1,
        )
        ax.add_patch(rect)

        # Class A samples
        for j, idx in enumerate(fold_a[f]):
            draw_square(ax, fx + j * SP * 0.85, s4_y - 0.8, S * 0.85,
                       CLASS_A_COLOR, idx, fontsize=6)

        # Class B sample
        for j, idx in enumerate(fold_b[f]):
            draw_triangle(ax, fx + 0.15, s4_y - 1.35, S * 0.85,
                         CLASS_B_COLOR, idx, fontsize=6)

        # Count label
        ax.text(fx + 0.15, s4_y - 1.95, "2A + 1B =\n3 Samples",
                fontsize=5.5, ha="center", va="top", color="#666666")

    # Preserved ratio note
    ax.text(
        7.2 + 5 * 1.3 / 2 - 0.5, s4_y - 2.6,
        "Preserved Ratio: 67% Class A / 33% Class B",
        fontsize=8, ha="center", va="center",
        fontweight="bold", color="#2e7d32",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#81c784"),
    )

    # ─── BOTTOM LEFT: WHY STRATIFY? ───
    why_x = 0.3
    why_y = 3.8
    section_header(ax, why_x, why_y, "WHY STRATIFY?")

    reasons = [
        "Without stratification, a fold could get",
        "0 minority samples → meaningless score.",
        "",
        "Stratified K-Fold guarantees each fold",
        "has ≈ same class ratio as the full dataset.",
        "",
        "Essential for imbalanced datasets.",
    ]
    for i, line in enumerate(reasons):
        weight = "bold" if i in (3, 4, 6) else "normal"
        ax.text(why_x + 0.1, why_y - 0.5 - i * 0.35, line,
                fontsize=7.5, color=TEXT_COLOR, fontweight=weight)

    # Connecting arrow from Step 1 area down to Why box
    ax.annotate(
        "", xy=(3.5, 4.2), xytext=(3.5, 4.6),
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("stratified_kfold_scratch.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    out = generate_diagram(args.output.resolve())
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
