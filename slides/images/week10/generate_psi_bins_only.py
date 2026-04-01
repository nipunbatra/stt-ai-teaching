#!/usr/bin/env python3
"""Generate a simple PSI intuition figure: compare percentages bin by bin."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate(output_path: Path) -> Path:
    bins = ["18-25", "25-32", "32-40", "40-50", "50+"]
    train = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    prod = np.array([0.15, 0.20, 0.25, 0.25, 0.15])

    x = np.arange(len(bins))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150)

    bars1 = ax.bar(x - w / 2, train * 100, w, color="steelblue", alpha=0.85, label="Training")
    bars2 = ax.bar(x + w / 2, prod * 100, w, color="coral", alpha=0.85, label="Production")

    for bar, val in zip(bars1, train):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.0%}", ha="center", fontsize=10, color="steelblue", fontweight="bold")
    for bar, val in zip(bars2, prod):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.0%}", ha="center", fontsize=10, color="coral", fontweight="bold")

    # Call out one bin explicitly
    ax.annotate(
        "This bin changed a lot:\n30% -> 15%",
        xy=(x[0], 31),
        xytext=(0.6, 35),
        textcoords="data",
        fontsize=10,
        color="#c44536",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c44536", lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=11)
    ax.set_ylabel("Percentage of records", fontsize=11)
    ax.set_title("PSI Step 1: Compare What Percentage Falls in Each Bin", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("psi_bins_only.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
