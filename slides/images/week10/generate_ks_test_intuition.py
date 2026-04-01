#!/usr/bin/env python3
"""Generate KS test intuition diagram: two CDFs with maximum gap highlighted."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    # Simulated food delivery data: user age
    train = np.random.normal(28, 6, 500)
    prod = np.random.normal(35, 8, 500)

    # Compute CDFs
    train_sorted = np.sort(train)
    prod_sorted = np.sort(prod)
    cdf_train = np.arange(1, len(train) + 1) / len(train)
    cdf_prod = np.arange(1, len(prod) + 1) / len(prod)

    # Get KS statistic
    ks_stat, _ = ks_2samp(train, prod)

    # Find the point of maximum gap
    all_values = np.sort(np.concatenate([train, prod]))
    cdf_train_interp = np.searchsorted(train_sorted, all_values, side='right') / len(train)
    cdf_prod_interp = np.searchsorted(prod_sorted, all_values, side='right') / len(prod)
    gaps = np.abs(cdf_train_interp - cdf_prod_interp)
    max_idx = np.argmax(gaps)
    max_x = all_values[max_idx]
    max_cdf_train = cdf_train_interp[max_idx]
    max_cdf_prod = cdf_prod_interp[max_idx]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(train_sorted, cdf_train, color='steelblue', linewidth=2.5,
            label='Training (Jan)', zorder=3)
    ax.plot(prod_sorted, cdf_prod, color='coral', linewidth=2.5,
            label='Production (Jun)', zorder=3)

    # Shade the gap region at max point
    ax.fill_betweenx([min(max_cdf_train, max_cdf_prod), max(max_cdf_train, max_cdf_prod)],
                     max_x - 0.5, max_x + 0.5, alpha=0.3, color='#FFD700', zorder=2)

    # Draw the max gap arrow
    ax.annotate('', xy=(max_x, max_cdf_train), xytext=(max_x, max_cdf_prod),
                arrowprops=dict(arrowstyle='<->', color='#c44536', lw=2.5))

    ax.text(max_x + 1.5, (max_cdf_train + max_cdf_prod) / 2,
            f'D = {ks_stat:.3f}\n(max gap)',
            fontsize=13, fontweight='bold', color='#c44536', va='center')

    ax.set_xlabel('User Age', fontsize=12)
    ax.set_ylabel('Cumulative Fraction (CDF)', fontsize=12)
    ax.set_title('KS Test Step 2: Measure the Biggest Vertical Gap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("ks_test_intuition.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
