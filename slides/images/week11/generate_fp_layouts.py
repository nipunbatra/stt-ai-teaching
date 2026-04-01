#!/usr/bin/env python3
"""Generate floating point bit layout diagrams: FP32, FP16, INT8 comparison."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_bit_bar(ax, y, sections, total_bits, label, height=0.6):
    """Draw a segmented bit layout bar."""
    x = 0
    for name, nbits, color in sections:
        width = nbits / total_bits * 10  # scale to 10 units wide
        rect = mpatches.FancyBboxPatch(
            (x, y - height/2), width, height,
            boxstyle=mpatches.BoxStyle.Round(pad=0.02),
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + width/2, y, f'{name}\n({nbits} bits)',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        x += width

    ax.text(-0.5, y, label, ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(x + 0.2, y, f'{total_bits} bits = {total_bits//8} bytes',
            ha='left', va='center', fontsize=10, color='#555')

def generate_comparison(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.set_xlim(-3, 14)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    # Colors
    SIGN = '#c44536'
    EXP = '#4A90D9'
    MANT = '#2D7D46'
    INT = '#E8A838'

    # FP32: 1 sign + 8 exponent + 23 mantissa = 32 bits
    draw_bit_bar(ax, 4, [('Sign', 1, SIGN), ('Exponent', 8, EXP), ('Mantissa', 23, MANT)], 32, 'FP32')

    # FP16: 1 sign + 5 exponent + 10 mantissa = 16 bits
    draw_bit_bar(ax, 2.5, [('Sign', 1, SIGN), ('Exp', 5, EXP), ('Mantissa', 10, MANT)], 32, 'FP16')
    # Show actual width = 16/32 of FP32
    ax.text(5.2, 2.5, '← half the bits', fontsize=10, color='#888', va='center')

    # INT8: 8 bits, no sign/exponent/mantissa — just a whole number
    draw_bit_bar(ax, 1, [('Integer value', 8, INT)], 32, 'INT8')
    ax.text(2.7, 1, '← 1/4 the bits\nno decimal point', fontsize=10, color='#888', va='center')

    ax.set_title('How Numbers Are Stored: FP32 vs FP16 vs INT8',
                 fontsize=14, fontweight='bold', y=1.02)

    # Legend
    legend_items = [
        mpatches.Patch(color=SIGN, label='Sign (+ or -)'),
        mpatches.Patch(color=EXP, label='Exponent (how big/small)'),
        mpatches.Patch(color=MANT, label='Mantissa (precision)'),
        mpatches.Patch(color=INT, label='Integer (whole number only)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def generate_fp32_detail(output_path: Path) -> Path:
    """Detailed FP32 layout with worked example for 6.5."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.set_xlim(-0.5, 33)
    ax.set_ylim(-2, 4)
    ax.axis('off')

    SIGN = '#c44536'
    EXP = '#4A90D9'
    MANT = '#2D7D46'

    # Draw 32 individual bit boxes
    bits_6_5 = '0 10000001 10100000000000000000000'
    bits_clean = bits_6_5.replace(' ', '')
    colors_per_bit = [SIGN] + [EXP]*8 + [MANT]*23

    for i, (bit, color) in enumerate(zip(bits_clean, colors_per_bit)):
        rect = mpatches.Rectangle((i, 2), 0.9, 0.9, facecolor=color, edgecolor='white',
                                   linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(i + 0.45, 2.45, bit, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Labels
    ax.text(0.45, 3.2, 'Sign', ha='center', fontsize=9, fontweight='bold', color=SIGN)
    ax.text(4.95, 3.2, 'Exponent (8 bits)', ha='center', fontsize=9, fontweight='bold', color=EXP)
    ax.text(20, 3.2, 'Mantissa (23 bits)', ha='center', fontsize=9, fontweight='bold', color=MANT)

    # Worked example
    ax.text(16, 0.8, 'Example: storing 6.5', fontsize=12, fontweight='bold', ha='center')
    ax.text(16, 0.1, '6.5 = 1.101 × 2²   →   Sign=0, Exponent=129 (binary: 10000001), Mantissa=101000...0',
            fontsize=10, ha='center', color='#333')
    ax.text(16, -0.7, 'Formula: (-1)⁰ × 1.101₂ × 2^(129-127) = 1 × 1.625 × 4 = 6.5  ✓',
            fontsize=10, ha='center', color='#2D7D46', fontweight='bold')

    ax.set_title('FP32: 32 Bits to Store One Decimal Number', fontsize=14, fontweight='bold')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    base = Path(__file__).parent
    print(f"Generated: {generate_comparison(base / 'fp_comparison.png')}")
    print(f"Generated: {generate_fp32_detail(base / 'fp32_detail.png')}")
