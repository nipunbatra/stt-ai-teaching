#!/usr/bin/env python3
"""Generate exact permutation-gap plots for the toy p-value example."""

import argparse
import itertools
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def all_mean_gaps(values):
    n = len(values)
    group_size = n // 2
    gaps = []
    for idx in itertools.combinations(range(n), group_size):
        idx = set(idx)
        group_a = [values[i] for i in range(n) if i in idx]
        group_b = [values[i] for i in range(n) if i not in idx]
        gap = abs(np.mean(group_a) - np.mean(group_b))
        gaps.append(round(float(gap), 10))
    return gaps


def plot_case(ax, gaps, observed_gap, title, color, p_text):
    counts = Counter(gaps)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    bar_colors = [color if x >= observed_gap - 1e-9 else "#cfd8e3" for x in xs]
    ax.bar(xs, ys, width=0.32, color=bar_colors, edgecolor="white")
    ax.axvline(observed_gap, color="#c44536", linestyle="--", linewidth=2)
    ax.text(
        observed_gap + 0.08,
        max(ys) * 0.88,
        f"Observed gap = {observed_gap:.1f}",
        color="#c44536",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="top",
    )
    ax.text(
        0.03,
        0.95,
        title,
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#22313f",
    )
    ax.text(
        0.03,
        0.84,
        p_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#555",
    )
    ax.set_xlabel("Mean gap after re-splitting", fontsize=10)
    ax.set_ylabel("Number of re-splits", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate(output_path: Path) -> Path:
    a = [24, 25, 25, 26, 24]
    b_similar = [24, 25, 26, 25, 24]
    b_different = [31, 32, 33, 31, 32]

    gaps_similar = all_mean_gaps(a + b_similar)
    gaps_different = all_mean_gaps(a + b_different)

    obs_similar = abs(np.mean(a) - np.mean(b_similar))
    obs_different = abs(np.mean(a) - np.mean(b_different))

    p_similar = sum(g >= obs_similar - 1e-9 for g in gaps_similar) / len(gaps_similar)
    p_different = sum(g >= obs_different - 1e-9 for g in gaps_different) / len(gaps_different)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    plot_case(
        axes[0],
        gaps_similar,
        obs_similar,
        "Case A: Similar samples",
        "#7fb3d5",
        f"All 252 re-splits checked\np = {int(p_similar * 252)}/252 = {p_similar:.2f}",
    )
    plot_case(
        axes[1],
        gaps_different,
        obs_different,
        "Case B: Very different samples",
        "#f5a089",
        f"All 252 re-splits checked\np = {int(round(p_different * 252))}/252 = {p_different:.3f}",
    )

    fig.suptitle(
        "Exact p-value idea: look at all possible re-splits under the 'no real difference' story",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Colored bars are re-splits at least as extreme as the observed mean gap.",
        ha="center",
        fontsize=10,
        color="#555",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("p_value_intuition.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
