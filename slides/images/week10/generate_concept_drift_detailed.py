#!/usr/bin/env python3
"""Generate concept drift visualization: same X, different Y — before and after."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Shared data points (same X in both panels)
    n = 150
    orders_per_week = np.random.uniform(1, 6, n)
    avg_spend = np.random.uniform(100, 800, n)

    # --- Before Zomato Gold: premium if orders > 3 AND spend > 400 ---
    labels_before = ((orders_per_week > 3) & (avg_spend > 400)).astype(int)

    premium_b = labels_before == 1
    regular_b = labels_before == 0

    ax1.scatter(orders_per_week[regular_b], avg_spend[regular_b],
                c='steelblue', alpha=0.6, s=30, label='Regular', zorder=3)
    ax1.scatter(orders_per_week[premium_b], avg_spend[premium_b],
                c='coral', alpha=0.6, s=30, label='Premium', zorder=3)

    # Decision boundary
    ax1.axvline(3, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax1.axhline(400, color='black', linestyle='-', linewidth=2, alpha=0.7, xmin=0.4)
    ax1.fill_between([3, 6.2], 400, 850, alpha=0.08, color='coral')
    ax1.fill_between([0.8, 3], 50, 850, alpha=0.05, color='steelblue')
    ax1.fill_between([3, 6.2], 50, 400, alpha=0.05, color='steelblue')

    ax1.set_xlabel('Orders per Week', fontsize=11)
    ax1.set_ylabel('Avg Spend (₹)', fontsize=11)
    ax1.set_title('Before Zomato Gold\n(Jan 2024)', fontsize=13, fontweight='bold', color='#2d7d46')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(0.8, 6.2)
    ax1.set_ylim(50, 850)
    ax1.text(4.5, 600, 'Premium\nzone', fontsize=11, color='coral', alpha=0.6,
             ha='center', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- After Zomato Gold: premium if orders > 5 AND spend > 600 ---
    # (Everyone orders more now, so threshold moved up)
    labels_after = ((orders_per_week > 5) & (avg_spend > 600)).astype(int)

    premium_a = labels_after == 1
    regular_a = labels_after == 0

    ax2.scatter(orders_per_week[regular_a], avg_spend[regular_a],
                c='steelblue', alpha=0.6, s=30, label='Regular', zorder=3)
    ax2.scatter(orders_per_week[premium_a], avg_spend[premium_a],
                c='coral', alpha=0.6, s=30, label='Premium', zorder=3)

    # Old boundary (wrong now)
    ax2.axvline(3, color='black', linestyle='--', linewidth=2, alpha=0.3, label='Old boundary')
    ax2.axhline(400, color='black', linestyle='--', linewidth=2, alpha=0.3, xmin=0.4)

    # New true boundary
    ax2.axvline(5, color='#c44536', linestyle='-', linewidth=2.5, alpha=0.8, label='New boundary')
    ax2.axhline(600, color='#c44536', linestyle='-', linewidth=2.5, alpha=0.8, xmin=0.82)
    ax2.fill_between([5, 6.2], 600, 850, alpha=0.1, color='coral')

    # Highlight misclassified region
    ax2.fill_between([3, 5], 400, 600, alpha=0.15, color='#FFD700',
                     label='Was "premium"\nnow "regular"')

    ax2.set_xlabel('Orders per Week', fontsize=11)
    ax2.set_ylabel('Avg Spend (₹)', fontsize=11)
    ax2.set_title('After Zomato Gold\n(Jun 2024)', fontsize=13, fontweight='bold', color='#c44536')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(0.8, 6.2)
    ax2.set_ylim(50, 850)
    ax2.text(4, 500, 'Model says\n"premium"\nbut wrong!',
             fontsize=10, color='#c44536', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3f3', edgecolor='#c44536', alpha=0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Concept Drift: Same Customers, Different Labels',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("concept_drift_detailed.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
