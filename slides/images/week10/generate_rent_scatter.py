#!/usr/bin/env python3
"""Generate rent prediction scatter: training, drifted, and prediction line."""

import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate(output_path: Path) -> Path:
    np.random.seed(42)

    sqft_train = np.random.normal(700, 120, 300).clip(300, 1200)
    rent_train = 3 + 0.008*sqft_train + 0.000008*sqft_train**2 + np.random.normal(0, 1.2, 300)

    sqft_drift = np.random.normal(1800, 300, 150).clip(800, 3000)
    rent_drift = 3 + 0.008*sqft_drift + 0.000008*sqft_drift**2 + np.random.normal(0, 1.2, 150)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(sqft_train.reshape(-1,1), rent_train)

    x_line = np.linspace(200, 3000, 200)
    y_pred_line = model.predict(x_line.reshape(-1,1))
    y_true_line = 3 + 0.008*x_line + 0.000008*x_line**2

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

    ax.scatter(sqft_train, rent_train, c='steelblue', alpha=0.4, s=20,
               label='Training (Ahmedabad)', zorder=3)
    ax.scatter(sqft_drift, rent_drift, c='coral', alpha=0.4, s=20,
               label='Production (Mumbai)', zorder=3)

    ax.plot(x_line, y_pred_line, 'k-', linewidth=2.5, label='Model prediction (linear)', zorder=4)
    ax.plot(x_line, y_true_line, '--', color='#2d7d46', linewidth=2,
            label='True relationship (curved)', zorder=4, alpha=0.7)

    # Shade the training range
    ax.axvspan(300, 1200, alpha=0.06, color='steelblue', label='Training range')

    # Annotate the gap
    idx = 180  # around sqft=2200
    ax.annotate('Model predicts ₹20k\nTrue rent ₹55k!',
                xy=(x_line[idx], y_pred_line[idx]),
                xytext=(2200, 15), fontsize=11, color='#c44536', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c44536', lw=1.5))

    ax.annotate('', xy=(x_line[idx], y_true_line[idx]),
                xytext=(x_line[idx], y_pred_line[idx]),
                arrowprops=dict(arrowstyle='<->', color='#c44536', lw=2))

    ax.set_xlabel('Square Feet', fontsize=12)
    ax.set_ylabel('Rent (₹k/month)', fontsize=12)
    ax.set_title('Linear Model Trained on Small Apartments → Fails on Large Ones',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(200, 3100)
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
    default = Path(__file__).with_name("rent_scatter.png")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    print(f"Generated: {generate(args.output.resolve())}")
