#!/usr/bin/env python3
"""Generate exact toy p-value figure from all 252 re-splits."""

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


def exact_gap_distribution(sample_a, sample_b):
    values = sample_a + sample_b
    n = len(values)
    gaps = []

    for idx in itertools.combinations(range(n), len(sample_a)):
        chosen = set(idx)
        group_a = [values[i] for i in range(n) if i in chosen]
        group_b = [values[i] for i in range(n) if i not in chosen]
        gap = abs(sum(group_a) / len(group_a) - sum(group_b) / len(group_b))
        gaps.append(round(gap, 10))

    observed = abs(sum(sample_a) / len(sample_a) - sum(sample_b) / len(sample_b))
    observed = round(observed, 10)
    counts = Counter(gaps)
    p_value = sum(g >= observed for g in gaps) / len(gaps)
    return counts, observed, p_value, len(gaps)


def plot_case(ax, counts, observed, p_value, total, title, color):
    xs = np.array(sorted(counts))
    ys = np.array([counts[x] for x in xs])
    mask = xs >= observed - 1e-12
    pos = np.arange(len(xs))

    ax.bar(
        pos[~mask],
        ys[~mask],
        width=0.78,
        color="#cfd8e3",
        edgecolor="white",
        linewidth=1.0,
    )
    ax.bar(
        pos[mask],
        ys[mask],
        width=0.78,
        color=color,
        edgecolor="white",
        linewidth=1.0,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean gap after re-splitting", fontsize=10)
    ax.set_ylabel("Number of re-splits", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{x:.1f}" for x in xs], fontsize=9)

    ymax = max(ys) * 1.22
    ax.set_ylim(0, ymax)
    obs_idx = int(np.where(np.isclose(xs, observed))[0][0])
    obs_hits = int(sum(ys[mask]))
    ax.text(
        0.03,
        0.96,
        f"p = {obs_hits}/{total} = {p_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#22313f",
        va="top",
    )
    ax.annotate(
        "Observed gap",
        xy=(obs_idx, ys[obs_idx]),
        xytext=(obs_idx, ymax * 0.90),
        ha="center",
        color="#c44536",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#c44536", lw=1.8),
    )
    ax.text(0.03, 0.86, "Colored bars count toward the p-value", transform=ax.transAxes, fontsize=9, color="#666", va="top")


def generate(output_path: Path) -> Path:
    sample_a = [24, 25, 25, 26, 24]
    sample_b_similar = [24, 25, 26, 25, 24]
    sample_b_different = [31, 32, 33, 31, 32]

    similar_counts, similar_obs, similar_p, total = exact_gap_distribution(sample_a, sample_b_similar)
    diff_counts, diff_obs, diff_p, _ = exact_gap_distribution(sample_a, sample_b_different)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=150)

    plot_case(
        axes[0],
        similar_counts,
        similar_obs,
        similar_p,
        total,
        "Case A: Similar samples",
        color="steelblue",
    )
    plot_case(
        axes[1],
        diff_counts,
        diff_obs,
        diff_p,
        total,
        "Case B: Very different samples",
        color="coral",
    )

    fig.suptitle(
        "Toy p-value example: exact mean-gap distribution from all 252 possible re-splits",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.25,
        -0.01,
        "Left: every bar counts, so p = 252/252 = 1.000",
        ha="center",
        fontsize=10,
        color="#555",
    )
    fig.text(
        0.75,
        -0.01,
        "Right: only the far-right bar counts, so p = 2/252 = 0.008",
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
