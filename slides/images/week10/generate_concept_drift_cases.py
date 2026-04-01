#!/usr/bin/env python3
"""Generate concept drift cases figure: same input, different label over time."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def card(ax, y, title, x_text, old_label, new_label, why):
    ax.text(0.02, y + 0.17, title, fontsize=13, fontweight="bold", color="#1f2d3d")

    input_box = patches.FancyBboxPatch(
        (0.02, y + 0.01), 0.28, 0.12,
        boxstyle="round,pad=0.02",
        facecolor="#eef4fb", edgecolor="#4A90D9", linewidth=1.8
    )
    ax.add_patch(input_box)
    ax.text(0.16, y + 0.07, x_text, ha="center", va="center", fontsize=11)

    old_box = patches.FancyBboxPatch(
        (0.38, y + 0.01), 0.18, 0.12,
        boxstyle="round,pad=0.02",
        facecolor="#ecf8f1", edgecolor="#2D7D46", linewidth=1.8
    )
    ax.add_patch(old_box)
    ax.text(0.47, y + 0.07, old_label, ha="center", va="center", fontsize=11, fontweight="bold", color="#2D7D46")
    ax.text(0.47, y - 0.005, "Before", ha="center", va="top", fontsize=9, color="#2D7D46")

    new_box = patches.FancyBboxPatch(
        (0.70, y + 0.01), 0.18, 0.12,
        boxstyle="round,pad=0.02",
        facecolor="#fff2f0", edgecolor="#c44536", linewidth=1.8
    )
    ax.add_patch(new_box)
    ax.text(0.79, y + 0.07, new_label, ha="center", va="center", fontsize=11, fontweight="bold", color="#c44536")
    ax.text(0.79, y - 0.005, "Later", ha="center", va="top", fontsize=9, color="#c44536")

    ax.annotate("", xy=(0.38, y + 0.07), xytext=(0.30, y + 0.07),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#666"))
    ax.annotate("", xy=(0.70, y + 0.07), xytext=(0.56, y + 0.07),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#666"))

    ax.text(0.90, y + 0.07, why, ha="left", va="center", fontsize=10, color="#444")


def generate(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card(
        ax, 0.68,
        "Customer scoring",
        "orders/week = 4\nspend = Rs 500",
        "Premium",
        "Regular",
        "Free-delivery subscription\nmade frequent orders normal"
    )

    card(
        ax, 0.40,
        "Review helpfulness",
        "\"delivery in 2 days\"",
        "Ordinary",
        "Helpful",
        "Delivery delays changed what\nusers cared about"
    )

    card(
        ax, 0.12,
        "Medical triage",
        "same scan +\nsame score",
        "Observe",
        "Admit",
        "Hospital guideline or threshold\nchanged"
    )

    ax.text(0.02, 0.95, "Concept Drift: Same Input, Different Correct Label",
            fontsize=16, fontweight="bold", color="#1f2d3d")
    ax.text(0.02, 0.90,
            "The world context changes, so the old mapping from X to Y is no longer correct.",
            fontsize=11, color="#555")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("concept_drift_cases.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
