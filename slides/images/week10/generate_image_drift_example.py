#!/usr/bin/env python3
"""Generate image drift examples: clean vs corrupted MNIST-style digits."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def make_digit_8(size=28):
    """Create a simple '8' digit image."""
    img = np.zeros((size, size))
    # Two circles stacked
    y, x = np.mgrid[:size, :size]
    top = ((x - 14)**2 + (y - 9)**2) < 36
    bottom = ((x - 14)**2 + (y - 19)**2) < 36
    ring_top = ((x - 14)**2 + (y - 9)**2) > 16
    ring_bottom = ((x - 14)**2 + (y - 19)**2) > 16
    img[(top & ring_top) | (bottom & ring_bottom)] = 1.0
    return img

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    digit = make_digit_8()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)

    # Row 1: Training data (clean)
    titles_train = ['Clean (Training)', 'Clean (Training)', 'Clean (Training)', 'Clean (Training)']
    for i, ax in enumerate(axes[0]):
        img = digit + np.random.normal(0, 0.05, digit.shape)
        # Small random shifts
        img = np.roll(img, np.random.randint(-1, 2), axis=0)
        img = np.roll(img, np.random.randint(-1, 2), axis=1)
        ax.imshow(img.clip(0, 1), cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(titles_train[i], fontsize=11, color='steelblue', fontweight='bold')
        ax.axis('off')

    # Row 2: Production data (drifted — different corruptions)
    corruptions = [
        ('Gaussian Noise', lambda img: img + np.random.normal(0, 0.4, img.shape)),
        ('Low Contrast', lambda img: img * 0.3 + 0.1),
        ('Brightness Shift', lambda img: np.clip(img + 0.5, 0, 1)),
        ('Blur + Rotate', lambda img: np.roll(np.roll(
            np.convolve(img.flatten(), np.ones(5)/5, mode='same').reshape(img.shape),
            3, axis=0), 3, axis=1)),
    ]

    for i, (name, corrupt_fn) in enumerate(corruptions):
        ax = axes[1][i]
        img = corrupt_fn(digit.copy())
        ax.imshow(img.clip(0, 1), cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, color='#c44536', fontweight='bold')
        ax.axis('off')

    axes[0][0].set_ylabel('Training\n(clean)', fontsize=13, fontweight='bold',
                           color='steelblue', rotation=0, labelpad=70, va='center')
    axes[1][0].set_ylabel('Production\n(drifted)', fontsize=13, fontweight='bold',
                           color='#c44536', rotation=0, labelpad=70, va='center')

    fig.suptitle('Image Data Drift: Same Digit, Different Conditions',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0.08, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("image_drift_example.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
