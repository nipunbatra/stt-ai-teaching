#!/usr/bin/env python3
"""Generate KS comparison figure with no-drift and drift cases."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_ecdf_step(values, x_min, x_max):
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    x_step = np.concatenate([[x_min], values, [x_max]])
    y_step = np.concatenate([[0.0], y, [1.0]])
    return x_step, y_step


def plot_case(ax_hist, ax_cdf, train, prod, title):
    train_sorted = np.sort(train)
    prod_sorted = np.sort(prod)

    x_min = min(train.min(), prod.min()) - 1.5
    x_max = max(train.max(), prod.max()) + 1.5

    train_x, train_y = make_ecdf_step(train, x_min, x_max)
    prod_x, prod_y = make_ecdf_step(prod, x_min, x_max)

    all_values = np.sort(np.concatenate([train, prod]))
    cdf_train_interp = np.searchsorted(train_sorted, all_values, side="right") / len(train_sorted)
    cdf_prod_interp = np.searchsorted(prod_sorted, all_values, side="right") / len(prod_sorted)
    gaps = np.abs(cdf_train_interp - cdf_prod_interp)
    max_idx = np.argmax(gaps)
    max_x = all_values[max_idx]
    max_train = cdf_train_interp[max_idx]
    max_prod = cdf_prod_interp[max_idx]
    d_stat = gaps[max_idx]

    bins = np.linspace(x_min, x_max, 26)
    ax_hist.hist(train, bins=bins, alpha=0.6, color="steelblue", label="Training", density=True)
    ax_hist.hist(prod, bins=bins, alpha=0.6, color="coral", label="Production", density=True)
    ax_hist.set_title(title, fontsize=13, fontweight="bold")
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax_cdf.step(train_x, train_y, where="post", color="steelblue", linewidth=2.3)
    ax_cdf.step(prod_x, prod_y, where="post", color="coral", linewidth=2.3)
    ax_cdf.annotate(
        "",
        xy=(max_x, max_train),
        xytext=(max_x, max_prod),
        arrowprops=dict(arrowstyle="<->", color="#c44536", lw=2),
    )
    label_x = min(max_x + 0.6, x_max - 2.2)
    ax_cdf.text(label_x, (max_train + max_prod) / 2, f"D = {d_stat:.2f}",
                fontsize=10, color="#c44536", fontweight="bold", va="center")
    ax_cdf.set_xlabel("Feature value", fontsize=10)
    ax_cdf.set_ylabel("CDF", fontsize=10)
    ax_cdf.set_xlim(x_min, x_max)
    ax_cdf.set_ylim(-0.02, 1.05)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.spines["right"].set_visible(False)


def generate(output_path: Path) -> Path:
    np.random.seed(42)

    fig = plt.figure(figsize=(12.4, 5.8), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    train1 = np.random.normal(30, 5, 400)
    prod1 = np.random.normal(30.4, 5.2, 400)

    train2 = np.random.normal(30, 5, 400)
    prod2 = np.random.normal(38, 6.5, 400)

    plot_case(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), train1, prod1, "Case A: No Real Drift")
    plot_case(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), train2, prod2, "Case B: Clear Drift")

    fig.text(0.25, 0.01, "Left: similar distributions, small CDF gap", ha="center", fontsize=9, color="#555")
    fig.text(0.75, 0.01, "Right: shifted distributions, large CDF gap", ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(top=0.94, bottom=0.11, left=0.08, right=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("ks_two_cases.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
