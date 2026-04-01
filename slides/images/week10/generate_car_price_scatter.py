#!/usr/bin/env python3
"""Generate car price vs km scatter: training, drifted, and model predictions."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    km_train = np.random.normal(25000, 10000, 300).clip(5000, 60000)
    price_train = 8 * np.exp(-km_train / 50000) + np.random.normal(0, 0.3, 300)

    km_drift = np.random.normal(80000, 20000, 150).clip(30000, 150000)
    price_drift = 8 * np.exp(-km_drift / 50000) + np.random.normal(0, 0.3, 150)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(km_train.reshape(-1,1), price_train)

    x_line = np.linspace(3000, 160000, 300)
    y_pred = model.predict(x_line.reshape(-1,1))
    y_true = 8 * np.exp(-x_line / 50000)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

    ax.scatter(km_train/1000, price_train, c='steelblue', alpha=0.35, s=20,
               label='Training (2022: low-km cars)', zorder=3)
    ax.scatter(km_drift/1000, price_drift, c='coral', alpha=0.35, s=20,
               label='Production (2026: high-km cars)', zorder=3)

    ax.plot(x_line/1000, y_pred, 'k-', linewidth=2.5,
            label='Linear model prediction', zorder=4)
    ax.plot(x_line/1000, y_true, '--', color='#2d7d46', linewidth=2,
            label='True relationship (exponential)', zorder=4, alpha=0.7)

    ax.axvspan(3, 60, alpha=0.06, color='steelblue')
    ax.text(30, 7.5, 'Training\nrange', ha='center', fontsize=10,
            color='steelblue', alpha=0.6)

    # Annotate the gap at 100k km
    idx = np.argmin(np.abs(x_line - 100000))
    ax.annotate(f'Model: ₹{y_pred[idx]:.1f}L\nTrue: ₹{y_true[idx]:.1f}L',
                xy=(100, y_pred[idx]), xytext=(115, 4),
                fontsize=11, color='#c44536', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c44536', lw=1.5))

    # Show the negative prediction region
    neg_idx = x_line[y_pred < 0]
    if len(neg_idx) > 0:
        ax.axvspan(neg_idx[0]/1000, 160, alpha=0.08, color='#c44536')
        ax.text(neg_idx[0]/1000 + 5, 0.3, 'Model predicts\nnegative price!',
                fontsize=10, color='#c44536', fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Kilometers Driven (thousands)', fontsize=12)
    ax.set_ylabel('Price (₹ Lakhs)', fontsize=12)
    ax.set_title('Linear Model on Exponential Truth → Extrapolation Disaster',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, 160)
    ax.grid(alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("car_price_scatter.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
