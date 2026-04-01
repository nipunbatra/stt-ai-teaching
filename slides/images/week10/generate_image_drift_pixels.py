#!/usr/bin/env python3
"""Generate image drift pixel statistics: show how to compare high-dim X."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    # Simulate pixel statistics for clean vs noisy images
    n = 500

    # --- Panel 1: Mean brightness ---
    ax = axes[0]
    clean_brightness = np.random.normal(0.45, 0.08, n)
    noisy_brightness = np.random.normal(0.55, 0.15, n)
    ax.hist(clean_brightness, bins=30, alpha=0.6, color='steelblue',
            label='Training (clean)', density=True)
    ax.hist(noisy_brightness, bins=30, alpha=0.6, color='coral',
            label='Production (noisy)', density=True)
    ax.set_title('Mean Pixel Brightness', fontsize=12, fontweight='bold')
    ax.set_xlabel('Avg brightness (0–1)', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Contrast (std of pixels) ---
    ax = axes[1]
    clean_contrast = np.random.normal(0.25, 0.04, n)
    noisy_contrast = np.random.normal(0.18, 0.06, n)
    ax.hist(clean_contrast, bins=30, alpha=0.6, color='steelblue',
            label='Training', density=True)
    ax.hist(noisy_contrast, bins=30, alpha=0.6, color='coral',
            label='Production', density=True)
    ax.set_title('Pixel Contrast (Std Dev)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pixel std dev', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Edge density (proxy for blur) ---
    ax = axes[2]
    clean_edges = np.random.normal(0.35, 0.06, n)
    noisy_edges = np.random.normal(0.20, 0.08, n)  # blurry = fewer edges
    ax.hist(clean_edges, bins=30, alpha=0.6, color='steelblue',
            label='Training', density=True)
    ax.hist(noisy_edges, bins=30, alpha=0.6, color='coral',
            label='Production', density=True)
    ax.set_title('Edge Density (Sharpness)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Edge pixels fraction', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Images Are High-Dimensional → Compare Summary Statistics',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("image_drift_pixels.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
