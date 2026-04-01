#!/usr/bin/env python3
"""Generate concept drift decision boundary diagram — side-by-side January vs June."""

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Generate two clusters
    n = 80
    cluster_a = np.random.randn(n, 2) * 0.7 + np.array([2, 3])
    cluster_b = np.random.randn(n, 2) * 0.7 + np.array([4, 1.5])

    # Decision boundary: y = -x + 5
    x_line = np.linspace(0, 6, 100)
    boundary = -x_line + 5

    # --- January: boundary works well ---
    ax1.scatter(cluster_a[:, 0], cluster_a[:, 1], c='steelblue', s=25, alpha=0.7,
                label='Deliver (Y=1)', zorder=3)
    ax1.scatter(cluster_b[:, 0], cluster_b[:, 1], c='coral', s=25, alpha=0.7,
                label='No Deliver (Y=0)', zorder=3)
    ax1.plot(x_line, boundary, 'k-', linewidth=2.5, label='Decision boundary', zorder=4)
    ax1.fill_between(x_line, boundary, 6, alpha=0.08, color='steelblue')
    ax1.fill_between(x_line, -1, boundary, alpha=0.08, color='coral')
    ax1.set_title('January (Training)', fontsize=13, fontweight='bold', color='#2d7d46')
    ax1.set_xlabel('Order Frequency', fontsize=11)
    ax1.set_ylabel('Time of Day', fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(-0.5, 5.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Count misclassified
    jan_correct_a = np.sum(cluster_a[:, 1] > -cluster_a[:, 0] + 5)
    jan_correct_b = np.sum(cluster_b[:, 1] < -cluster_b[:, 0] + 5)
    jan_acc = (jan_correct_a + jan_correct_b) / (2 * n)
    ax1.text(0.5, -0.15, f'Accuracy: {jan_acc:.0%}',
             transform=ax1.transAxes, ha='center', fontsize=12,
             fontweight='bold', color='#2d7d46')

    # --- June: cluster B shifted up, old boundary fails ---
    cluster_b_shifted = cluster_b.copy()
    cluster_b_shifted[:, 1] += 2.0  # shift up by 2

    ax2.scatter(cluster_a[:, 0], cluster_a[:, 1], c='steelblue', s=25, alpha=0.7,
                label='Deliver (Y=1)', zorder=3)
    ax2.scatter(cluster_b_shifted[:, 0], cluster_b_shifted[:, 1], c='coral', s=25, alpha=0.7,
                label='No Deliver (Y=0)', zorder=3)

    # Same old boundary
    ax2.plot(x_line, boundary, 'k--', linewidth=2.5, label='Old boundary', zorder=4, alpha=0.6)
    ax2.fill_between(x_line, boundary, 6, alpha=0.05, color='steelblue')
    ax2.fill_between(x_line, -1, boundary, alpha=0.05, color='coral')

    # Mark misclassified points
    misclassified = cluster_b_shifted[cluster_b_shifted[:, 1] > -cluster_b_shifted[:, 0] + 5]
    ax2.scatter(misclassified[:, 0], misclassified[:, 1], marker='x', c='red',
                s=50, linewidths=2, zorder=5, label='Misclassified')

    ax2.set_title('June (Production)', fontsize=13, fontweight='bold', color='#c44536')
    ax2.set_xlabel('Order Frequency', fontsize=11)
    ax2.set_ylabel('Time of Day', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-0.5, 5.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    jun_correct_a = np.sum(cluster_a[:, 1] > -cluster_a[:, 0] + 5)
    jun_correct_b = np.sum(cluster_b_shifted[:, 1] < -cluster_b_shifted[:, 0] + 5)
    jun_acc = (jun_correct_a + jun_correct_b) / (2 * n)
    ax2.text(0.5, -0.15, f'Accuracy: {jun_acc:.0%}',
             transform=ax2.transAxes, ha='center', fontsize=12,
             fontweight='bold', color='#c44536')

    fig.suptitle('Same Boundary, Different World → Concept Drift',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("concept_drift_boundary.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
