#!/usr/bin/env python3
"""Generate CDF intuition diagram: histogram on left, CDF step function on right."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    # Exact data from the slide table: 50 people
    ages = ([20] * 10 + [25] * 10 + [30] * 15 + [35] * 10 + [40] * 5)
    ages = np.array(ages, dtype=float)
    # Add small jitter for histogram visibility
    np.random.seed(42)
    ages_jittered = ages + np.random.uniform(-2, 2, len(ages))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # --- Left: Histogram ---
    ax1.hist(ages_jittered, bins=np.arange(16, 45, 3), color='steelblue', alpha=0.7,
             edgecolor='white', linewidth=0.8)
    ax1.set_title('Histogram: How Many at Each Age?', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Annotate the peak
    ax1.annotate('15 people\naged ~30', xy=(30, 13), xytext=(36, 14),
                 fontsize=10, color='steelblue', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='steelblue'))

    # --- Right: Empirical CDF (step function) ---
    sorted_ages = np.sort(ages)
    cdf = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages)

    ax2.step(sorted_ages, cdf, where='post', color='steelblue', linewidth=2.5)
    ax2.scatter(sorted_ages[::1], cdf[::1], color='steelblue', s=8, zorder=4)

    # Annotate CDF(25) = 0.40
    idx_25 = np.searchsorted(sorted_ages, 25, side='right') - 1
    cdf_at_25 = cdf[idx_25]
    ax2.plot([25, 25], [0, cdf_at_25], 'k--', alpha=0.4, linewidth=1)
    ax2.plot([15, 25], [cdf_at_25, cdf_at_25], 'k--', alpha=0.4, linewidth=1)
    ax2.scatter([25], [cdf_at_25], color='#c44536', s=80, zorder=5)
    ax2.annotate(f'CDF(25) = {cdf_at_25:.2f}\n"40% of people are\n age 25 or younger"',
                 xy=(25, cdf_at_25), xytext=(31, 0.25),
                 fontsize=10, color='#c44536', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#c44536', lw=1.5))

    # Annotate CDF(30) = 0.70
    idx_30 = np.searchsorted(sorted_ages, 30, side='right') - 1
    cdf_at_30 = cdf[idx_30]
    ax2.plot([30, 30], [0, cdf_at_30], 'k--', alpha=0.3, linewidth=1)
    ax2.scatter([30], [cdf_at_30], color='#2d7d46', s=60, zorder=5)
    ax2.annotate(f'CDF(30) = {cdf_at_30:.2f}', xy=(30, cdf_at_30),
                 xytext=(34, 0.78), fontsize=10, color='#2d7d46',
                 arrowprops=dict(arrowstyle='->', color='#2d7d46', lw=1.2))

    ax2.set_title('CDF: What % Are Below x?', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Cumulative Fraction', fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(17, 43)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Same Data, Two Views', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("cdf_intuition.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
