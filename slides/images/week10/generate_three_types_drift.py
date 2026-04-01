#!/usr/bin/env python3
"""Generate three-panel diagram showing data drift, concept drift, and label drift."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    # --- Panel 1: Data Drift — histograms shift ---
    ax = axes[0]
    train_ages = np.random.normal(28, 6, 500)
    prod_ages = np.random.normal(38, 8, 500)
    ax.hist(train_ages, bins=25, alpha=0.6, color='steelblue', label='Training', density=True)
    ax.hist(prod_ages, bins=25, alpha=0.6, color='coral', label='Production', density=True)
    ax.set_title('Data Drift', fontsize=14, fontweight='bold', color='steelblue')
    ax.set_xlabel('User Age', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.5, -0.22, 'P(X) changed, P(Y|X) same',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic', color='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Concept Drift — same data, boundary moves ---
    ax = axes[1]
    n = 100
    X1 = np.random.randn(n, 2) * 0.8 + np.array([2, 2])
    X2 = np.random.randn(n, 2) * 0.8 + np.array([4, 4])

    # January labels (class boundary at x+y=6)
    jan_labels_1 = (X1[:, 0] + X1[:, 1] < 6).astype(int)
    jan_labels_2 = (X2[:, 0] + X2[:, 1] < 6).astype(int)

    # Plot with "January" colors but "June" relabeling:
    # After concept drift, the same point near boundary means something different
    ax.scatter(X1[:, 0], X1[:, 1], c='steelblue', alpha=0.5, s=20, label='Class A')
    ax.scatter(X2[:, 0], X2[:, 1], c='coral', alpha=0.5, s=20, label='Class B')

    # Old boundary (January)
    x_line = np.linspace(0, 6, 50)
    ax.plot(x_line, 6 - x_line, 'k--', linewidth=2, label='Old boundary')
    # New boundary (June) — shifted
    ax.plot(x_line, 7.5 - x_line, color='#c44536', linewidth=2, linestyle='-', label='New boundary')

    ax.set_title('Concept Drift', fontsize=14, fontweight='bold', color='#c44536')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 7)
    ax.text(0.5, -0.22, 'P(Y|X) changed, P(X) same',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic', color='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Label Drift — class proportions change ---
    ax = axes[2]
    categories = ['Normal', 'Fraud']
    train_counts = [99, 1]
    prod_counts = [95, 5]

    x = np.arange(len(categories))
    w = 0.35
    bars1 = ax.bar(x - w/2, train_counts, w, color='steelblue', alpha=0.8, label='Training')
    bars2 = ax.bar(x + w/2, prod_counts, w, color='coral', alpha=0.8, label='Production')

    # Labels on bars
    for bar, val in zip(bars1, train_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10, color='steelblue')
    for bar, val in zip(bars2, prod_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10, color='coral')

    ax.set_title('Label Drift', fontsize=14, fontweight='bold', color='#8B6914')
    ax.set_ylabel('Percentage', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    ax.text(0.5, -0.22, 'P(Y) changed',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic', color='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("three_types_of_drift.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
