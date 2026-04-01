#!/usr/bin/env python3
"""Generate retrain recovery chart: accuracy with periodic retraining vs without monitoring."""

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

    months = np.arange(1, 25)
    noise = np.random.normal(0, 0.005, len(months))

    # --- With retraining ---
    acc_retrain = np.zeros(len(months))
    # Phase 1: decay from 0.95 to ~0.76
    for i in range(8):
        acc_retrain[i] = 0.95 - 0.025 * i
    # Retrain v2 at month 9
    acc_retrain[8] = 0.93
    for i in range(9, 16):
        acc_retrain[i] = 0.93 - 0.02 * (i - 8)
    # Retrain v3 at month 17
    acc_retrain[16] = 0.94
    for i in range(17, 24):
        acc_retrain[i] = 0.94 - 0.018 * (i - 16)

    acc_retrain += noise

    # --- Without monitoring (continuous decay) ---
    acc_no_monitor = 0.95 - 0.018 * months + noise * 0.5
    acc_no_monitor = np.clip(acc_no_monitor, 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    # No monitoring line
    ax.plot(months, acc_no_monitor, color='#999', linewidth=2, linestyle='--',
            alpha=0.7, label='Without monitoring', zorder=2)
    ax.fill_between(months, acc_no_monitor, 0.5, alpha=0.05, color='#999')

    # With retraining line
    ax.plot(months, acc_retrain, color='steelblue', linewidth=2.5,
            label='With monitoring + retraining', zorder=3)

    # Mark retrain events
    retrain_months = [9, 17]
    retrain_accs = [acc_retrain[8], acc_retrain[16]]
    ax.scatter(retrain_months, retrain_accs, color='#2d7d46', s=120,
               marker='*', zorder=5, label='Retrain event')

    for m, a in zip(retrain_months, retrain_accs):
        ax.axvline(m, color='#2d7d46', linestyle=':', alpha=0.4, linewidth=1.5)
        ax.annotate(f'Retrain\nv{retrain_months.index(m)+2}',
                    xy=(m, a), xytext=(m + 1.2, a + 0.03),
                    fontsize=10, color='#2d7d46', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2d7d46', lw=1.2))

    # Danger zone
    ax.axhspan(0.5, 0.70, alpha=0.08, color='#c44536')
    ax.text(23.5, 0.62, 'Danger\nzone', ha='right', fontsize=9, color='#c44536', alpha=0.7)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Monitoring + Retraining Keeps Your Model Alive', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(1, 24)
    ax.set_xticks(range(1, 25, 2))
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
    default = Path(__file__).with_name("retrain_recovery.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
