#!/usr/bin/env python3
"""Generate model accuracy decay chart for Week 10 Data Drift lecture."""

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

    months = np.arange(1, 13)
    # Logistic-style decay from 0.95 to ~0.72 with small noise
    decay = 0.95 - 0.25 * (1 / (1 + np.exp(-0.6 * (months - 7))))
    noise = np.random.normal(0, 0.008, len(months))
    accuracy = np.clip(decay + noise, 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    # Color gradient: green to red
    for i in range(len(months) - 1):
        frac = i / (len(months) - 2)
        color = (0.2 + 0.6 * frac, 0.7 - 0.5 * frac, 0.3 - 0.15 * frac)
        ax.plot(months[i:i+2], accuracy[i:i+2], color=color, linewidth=3, zorder=3)

    ax.scatter(months, accuracy, color='white', edgecolor='#333', s=60, zorder=4, linewidth=1.5)

    # Annotations
    ax.annotate('Deploy!\n95% accuracy', xy=(1, accuracy[0]),
                xytext=(2.5, 0.98), fontsize=11, color='#2d7d46', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2d7d46', lw=1.5))

    ax.annotate('"Why am I getting\nso much spam?"', xy=(12, accuracy[-1]),
                xytext=(9.5, 0.68), fontsize=11, color='#c44536', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c44536', lw=1.5))

    # Reference lines
    ax.axhline(0.95, color='#2d7d46', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(accuracy[-1], color='#c44536', linestyle='--', alpha=0.4, linewidth=1)

    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Model Accuracy Over Time (Same Code, Same Server)', fontsize=14, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=10)
    ax.set_ylim(0.62, 1.02)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("model_accuracy_decay.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
