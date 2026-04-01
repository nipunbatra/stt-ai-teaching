#!/usr/bin/env python3
"""Generate label drift visualization: class proportions shifting over time."""

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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # --- Panel 1: Fraud rate shift (tabular) ---
    ax = axes[0]
    months = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov']
    fraud_rate = [1.0, 1.2, 1.8, 2.5, 3.5, 5.0]
    colors = ['#2d7d46' if r < 2 else '#E8A838' if r < 3.5 else '#c44536' for r in fraud_rate]
    ax.bar(months, fraud_rate, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='steelblue', linestyle='--', alpha=0.5, label='Training rate (1%)')
    for i, r in enumerate(fraud_rate):
        ax.text(i, r + 0.15, f'{r}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Fraud Rate Over Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraud %', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 6.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Image classification — class balance shift ---
    ax = axes[1]
    classes = ['Cat', 'Dog', 'Bird']
    train_pct = [40, 40, 20]
    prod_pct = [10, 75, 15]
    x = np.arange(len(classes))
    w = 0.35
    bars1 = ax.bar(x - w/2, train_pct, w, color='steelblue', alpha=0.85, label='Training')
    bars2 = ax.bar(x + w/2, prod_pct, w, color='coral', alpha=0.85, label='Production')
    for b, v in zip(bars1, train_pct):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v}%', ha='center', fontsize=10, color='steelblue')
    for b, v in zip(bars2, prod_pct):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v}%', ha='center', fontsize=10, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_title('Pet Classifier Class Mix', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of images', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Sentiment distribution shift (text) ---
    ax = axes[2]
    sentiments = ['Positive', 'Neutral', 'Negative']
    train_sent = [30, 45, 25]
    prod_sent = [65, 25, 10]
    bars1 = ax.bar(x - w/2, train_sent, w, color='steelblue', alpha=0.85, label='Training')
    bars2 = ax.bar(x + w/2, prod_sent, w, color='coral', alpha=0.85, label='Production')
    for b, v in zip(bars1, train_sent):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v}%', ha='center', fontsize=10, color='steelblue')
    for b, v in zip(bars2, prod_sent):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v}%', ha='center', fontsize=10, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiments, fontsize=11)
    ax.set_title('Review Sentiment Mix', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of reviews', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 80)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Label Drift: The Outcome Proportions Shifted',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("label_drift_example.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
