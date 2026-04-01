#!/usr/bin/env python3
"""Generate quantization visualization diagrams."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_number_line(output_path: Path) -> Path:
    """FP32 dense number line vs INT8 sparse number line."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), dpi=150)

    # FP32: dense ticks
    fp32_values = np.linspace(-1, 1, 200)
    ax1.plot([-1.05, 1.05], [0, 0], 'k-', linewidth=2)
    for v in fp32_values:
        ax1.plot([v, v], [-0.15, 0.15], 'steelblue', linewidth=0.5, alpha=0.6)
    ax1.set_title('FP32: Many possible values (high precision)', fontsize=12, fontweight='bold',
                  color='steelblue')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-0.8, 0.8)
    ax1.axis('off')
    ax1.text(-1.05, -0.5, '-1.0', fontsize=10, ha='center')
    ax1.text(0, -0.5, '0.0', fontsize=10, ha='center')
    ax1.text(1.05, -0.5, '1.0', fontsize=10, ha='center')

    # INT8: sparse ticks
    int8_values = np.linspace(-1, 1, 16)  # only 16 levels shown
    ax2.plot([-1.05, 1.05], [0, 0], 'k-', linewidth=2)
    for v in int8_values:
        ax2.plot([v, v], [-0.2, 0.2], 'coral', linewidth=2.5)

    # Show a few FP32 values snapping to nearest INT8
    snap_examples = [(-0.82, -0.87), (-0.33, -0.27), (0.45, 0.47), (0.71, 0.73)]
    for fp32_v, int8_v in snap_examples:
        nearest = int8_values[np.argmin(np.abs(int8_values - fp32_v))]
        ax2.annotate('', xy=(nearest, 0), xytext=(fp32_v, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1))
        ax2.plot(fp32_v, 0.5, 'o', color='steelblue', markersize=4)
        ax2.text(fp32_v, 0.65, f'{fp32_v}', fontsize=7, ha='center', color='steelblue')

    ax2.set_title('INT8: Few possible values (nearby values snap to same point)',
                  fontsize=12, fontweight='bold', color='coral')
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-0.8, 1.0)
    ax2.axis('off')
    ax2.text(-1.05, -0.5, '-1.0', fontsize=10, ha='center')
    ax2.text(0, -0.5, '0.0', fontsize=10, ha='center')
    ax2.text(1.05, -0.5, '1.0', fontsize=10, ha='center')

    fig.suptitle('Quantization: Reduce the Number of Possible Values',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def generate_weight_distribution(output_path: Path) -> Path:
    """Weight distribution before (FP32) and after (INT8) quantization."""
    np.random.seed(42)

    # Simulate neural network weights (bell-shaped, centered near 0)
    weights = np.concatenate([
        np.random.normal(0, 0.3, 5000),
        np.random.normal(0.1, 0.1, 2000),
        np.random.normal(-0.1, 0.1, 2000),
    ])

    # Quantize: map to 256 levels in the range [min, max]
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 255
    quantized = np.round((weights - w_min) / scale) * scale + w_min

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

    ax1.hist(weights, bins=80, color='steelblue', alpha=0.75, edgecolor='white', linewidth=0.5)
    ax1.set_title('FP32 Weights (Original)', fontsize=13, fontweight='bold', color='steelblue')
    ax1.set_xlabel('Weight value', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax1.text(0.02, 0.95, f'{len(np.unique(weights)):,} unique values',
             transform=ax1.transAxes, fontsize=10, va='top', color='steelblue')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.hist(quantized, bins=80, color='coral', alpha=0.75, edgecolor='white', linewidth=0.5)
    ax2.set_title('INT8 Weights (Quantized)', fontsize=13, fontweight='bold', color='coral')
    ax2.set_xlabel('Weight value', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax2.text(0.02, 0.95, f'{len(np.unique(quantized)):,} unique values\n(max 256)',
             transform=ax2.transAxes, fontsize=10, va='top', color='coral')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Most Weights Cluster Near Zero → Quantization Error Is Small',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def generate_cprofile_bars(output_path: Path) -> Path:
    """Visual cProfile output: time spent per function."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    functions = ['pd.read_csv()', 'joblib.load()', 'model.predict()', 'Other']
    times = [1.8, 0.6, 0.001, 0.02]
    colors = ['#c44536', '#E8A838', '#2D7D46', '#999']

    bars = ax.barh(functions, times, color=colors, edgecolor='white', linewidth=1.5, height=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{t:.3f}s' if t < 0.01 else f'{t:.1f}s',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('cProfile Output: Where Is the Time Going?', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation
    ax.annotate('99.9% of time is NOT in predict()!\nThe bottleneck is loading data every request.',
                xy=(1.8, 3), xytext=(1.2, 1.5),
                fontsize=10, color='#c44536', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c44536', lw=1.5))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def generate_pareto(output_path: Path) -> Path:
    """Accuracy vs model size Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    points = [
        ('FP32\n(baseline)', 400, 99.5, 'steelblue'),
        ('FP16', 200, 99.3, '#4A90D9'),
        ('INT8\n(static)', 100, 99.0, '#2D7D46'),
        ('INT8\n(dynamic)', 105, 98.5, '#2a9d8f'),
        ('Pruned\n+ INT8', 60, 98.0, '#E8A838'),
        ('INT4', 50, 95.5, '#c44536'),
        ('Distilled\n+ INT8', 35, 97.0, '#8B6914'),
    ]

    for name, size, acc, color in points:
        ax.scatter(size, acc, s=120, c=color, zorder=5, edgecolor='white', linewidth=1.5)
        ax.text(size + 8, acc, name, fontsize=9, va='center', color=color, fontweight='bold')

    # Pareto frontier curve
    pareto_x = [400, 200, 105, 60, 35]
    pareto_y = [99.5, 99.3, 98.5, 98.0, 97.0]
    ax.plot(pareto_x, pareto_y, '--', color='#888', linewidth=1.5, alpha=0.5, label='Pareto frontier')

    # Zones
    ax.axhspan(98, 100, alpha=0.05, color='#2D7D46')
    ax.text(380, 98.2, 'Sweet\nspot', fontsize=9, color='#2D7D46', alpha=0.5)

    ax.set_xlabel('Model Size (MB)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('The Tradeoff: Smaller Model vs Lower Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(94, 100.5)
    ax.set_xlim(0, 450)
    ax.invert_xaxis()
    ax.grid(alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    base = Path(__file__).parent
    print(f"Generated: {generate_number_line(base / 'quantization_number_line.png')}")
    print(f"Generated: {generate_weight_distribution(base / 'weight_distribution.png')}")
    print(f"Generated: {generate_cprofile_bars(base / 'cprofile_bars.png')}")
    print(f"Generated: {generate_pareto(base / 'pareto_frontier.png')}")
