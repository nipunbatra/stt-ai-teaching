#!/usr/bin/env python3
"""Generate drift response pipeline diagram: Monitor → Detect → Diagnose → Retrain."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def generate(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    steps = [
        {'x': 0.5, 'label': 'Monitor', 'sub': 'Track feature\ndistributions weekly',
         'color': '#4A90D9', 'icon': '📊'},
        {'x': 3.3, 'label': 'Detect', 'sub': 'KS test / PSI\nthreshold crossed → alert',
         'color': '#E8A838', 'icon': '🔍'},
        {'x': 6.1, 'label': 'Diagnose', 'sub': 'Bug? Seasonal?\nNew users?',
         'color': '#D4652F', 'icon': '🔬'},
        {'x': 8.9, 'label': 'Retrain', 'sub': 'Add new data\nDeploy v2',
         'color': '#2D7D46', 'icon': '🔄'},
    ]

    box_w, box_h = 2.4, 2.4

    for step in steps:
        x = step['x']
        color = step['color']

        # Rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x, 0.5), box_w, box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.15),
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.15
        )
        ax.add_patch(rect)
        rect_border = mpatches.FancyBboxPatch(
            (x, 0.5), box_w, box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.15),
            facecolor='none', edgecolor=color, linewidth=2.5
        )
        ax.add_patch(rect_border)

        # Label
        ax.text(x + box_w/2, 2.35, step['label'],
                ha='center', va='center', fontsize=15, fontweight='bold', color=color)
        # Subtitle
        ax.text(x + box_w/2, 1.35, step['sub'],
                ha='center', va='center', fontsize=10, color='#333', linespacing=1.4)

    # Arrows between boxes
    arrow_style = mpatches.ArrowStyle('->', head_length=0.3, head_width=0.2)
    for i in range(len(steps) - 1):
        x_start = steps[i]['x'] + box_w + 0.05
        x_end = steps[i+1]['x'] - 0.05
        ax.annotate('', xy=(x_end, 1.7), xytext=(x_start, 1.7),
                    arrowprops=dict(arrowstyle=arrow_style, color='#666',
                                   lw=2.5, connectionstyle='arc3,rad=0'))

    # Feedback loop arrow (from Retrain back to Monitor)
    ax.annotate('', xy=(steps[0]['x'] + box_w/2, 0.35),
                xytext=(steps[-1]['x'] + box_w/2, 0.35),
                arrowprops=dict(arrowstyle=arrow_style, color='#999',
                                lw=1.5, linestyle='--',
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(5.7, 0.05, 'continuous loop', ha='center', fontsize=9,
            color='#999', style='italic')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("drift_response_pipeline.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
