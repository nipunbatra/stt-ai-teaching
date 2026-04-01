#!/usr/bin/env python3
"""Generate text drift visualization: training vs production text characteristics."""

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

    # --- Panel 1: Message length distribution shift ---
    ax = axes[0]
    train_lengths = np.random.lognormal(3.5, 0.6, 500).clip(5, 300)  # avg ~33 words
    prod_lengths = np.random.lognormal(2.0, 0.8, 500).clip(1, 300)   # avg ~7 words (SMS/WhatsApp)
    ax.hist(train_lengths, bins=30, alpha=0.6, color='steelblue', label='Training\n(emails)', density=True)
    ax.hist(prod_lengths, bins=30, alpha=0.6, color='coral', label='Production\n(WhatsApp)', density=True)
    ax.set_title('Message Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Words per message', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Vocabulary shift (top words changed) ---
    ax = axes[1]
    categories = ['Dear/Sir', 'Meeting', 'Invoice', 'lol/haha', 'pic', 'brb']
    train_freq = [0.15, 0.12, 0.10, 0.01, 0.005, 0.002]
    prod_freq =  [0.01, 0.03, 0.02, 0.18, 0.12,  0.08]

    x = np.arange(len(categories))
    w = 0.35
    ax.barh(x - w/2, train_freq, w, color='steelblue', alpha=0.8, label='Training')
    ax.barh(x + w/2, prod_freq, w, color='coral', alpha=0.8, label='Production')
    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_title('Top Words Changed', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Emoji / special char usage ---
    ax = axes[2]
    labels = ['No emoji', 'Has emoji', 'Has URL', 'Has ₹/$/€']
    train_pct = [92, 3, 40, 15]
    prod_pct = [30, 55, 10, 5]

    x = np.arange(len(labels))
    ax.bar(x - w/2, train_pct, w, color='steelblue', alpha=0.8, label='Training')
    ax.bar(x + w/2, prod_pct, w, color='coral', alpha=0.8, label='Production')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_title('Message Characteristics', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of messages', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Text Data Drift: Email Spam Filter Deployed on WhatsApp Messages',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("text_drift_example.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
