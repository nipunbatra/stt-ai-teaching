#!/usr/bin/env python3
"""Generate PSI step-by-step visual: grouped bar chart with per-bin contributions."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    # Exact data from worked example in slides
    bins = ['18-25', '25-32', '32-40', '40-50', '50+']
    Q = np.array([0.30, 0.25, 0.20, 0.15, 0.10])  # Training
    P = np.array([0.15, 0.20, 0.25, 0.25, 0.15])   # Production

    # PSI per bin
    psi_per_bin = (P - Q) * np.log(P / Q)
    psi_total = np.sum(psi_per_bin)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=150,
                                    gridspec_kw={'height_ratios': [3, 2]})

    # --- Top: Grouped bar chart ---
    x = np.arange(len(bins))
    w = 0.35
    bars1 = ax1.bar(x - w/2, Q * 100, w, color='steelblue', alpha=0.85, label='Training')
    bars2 = ax1.bar(x + w/2, P * 100, w, color='coral', alpha=0.85, label='Production')

    # Label each bar
    for bar, val in zip(bars1, Q):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{val:.0%}', ha='center', fontsize=10, fontweight='bold', color='steelblue')
    for bar, val in zip(bars2, P):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{val:.0%}', ha='center', fontsize=10, fontweight='bold', color='coral')

    # Arrows showing direction of change
    for i in range(len(bins)):
        if P[i] > Q[i]:
            ax1.annotate('', xy=(x[i] + w/2, Q[i]*100 + 3),
                        xytext=(x[i] - w/2, Q[i]*100 + 3),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
        elif P[i] < Q[i]:
            ax1.annotate('', xy=(x[i] + w/2, P[i]*100 + 3),
                        xytext=(x[i] - w/2, P[i]*100 + 3),
                        arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))

    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, fontsize=11)
    ax1.set_ylabel('Percentage', fontsize=11)
    ax1.set_title('Step 1: Compare Bins — Training vs Production', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 38)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Bottom: PSI contribution per bin ---
    colors = ['#c44536' if v > 0.05 else '#FFB347' if v > 0.02 else '#2d7d46' for v in psi_per_bin]
    bars3 = ax2.bar(x, psi_per_bin, 0.6, color=colors, alpha=0.85, edgecolor='white')

    for bar, val in zip(bars3, psi_per_bin):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, fontsize=11)
    ax2.set_ylabel('PSI contribution', fontsize=11)
    ax2.set_title(f'Step 2: Sum Contributions → PSI = {psi_total:.3f}  (Yellow Zone: Moderate Drift)',
                  fontsize=13, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Traffic light legend
    ax2.text(0.98, 0.85, '< 0.1 = Green  |  0.1–0.25 = Yellow  |  > 0.25 = Red',
             transform=ax2.transAxes, ha='right', fontsize=9, color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ccc'))

    fig.tight_layout(h_pad=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("psi_step_by_step.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
