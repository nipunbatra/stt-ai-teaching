"""
Demo 2: What Is My Model Getting Wrong? (Image Logging)
========================================================
Story: "My model says 97% accuracy — but WHICH digits does it confuse?"

Train a digit classifier, create matplotlib figure grids of correct vs
misclassified predictions, log as single images.

TrackIO features: trackio.Image() from file path, Media & Tables tab
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import trackio, tempfile, os

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = float((preds == y_test).mean())

correct = np.where(preds == y_test)[0]
wrong = np.where(preds != y_test)[0]
print(f"Test accuracy: {acc:.4f}  ({len(correct)} correct, {len(wrong)} wrong)")

tmpdir = tempfile.mkdtemp()


def plot_digit_grid(indices, X, y_true, y_pred, title, cols=8, max_show=32):
    """Create a compact matplotlib grid of digit predictions."""
    indices = indices[:max_show]
    n = len(indices)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 1.0))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=11, fontweight='bold')

    for pos in range(rows * cols):
        r, c = divmod(pos, cols)
        ax = axes[r, c]
        ax.axis('off')
        if pos >= n:
            continue
        idx = indices[pos]
        ax.imshow(X[idx].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16)
        is_wrong = y_true[idx] != y_pred[idx]
        color = '#e74c3c' if is_wrong else '#27ae60'
        label = f"{y_true[idx]}→{y_pred[idx]}" if is_wrong else str(y_true[idx])
        ax.set_title(label, fontsize=8, color=color, pad=1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(tmpdir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


# ── Log ──
trackio.init(project="cs203-week08-demo", name="prediction-analysis",
             config={"model": "RandomForest", "experiment": "predictions",
                     "test_accuracy": round(acc, 4)})

# Correct predictions grid
path = plot_digit_grid(correct, X_test, y_test, preds,
                       f"Correct Predictions ({len(correct)} total, showing 32)")
trackio.log({"correct_grid": trackio.Image(path, caption=f"{len(correct)} correct")})

# ALL misclassified grid
path = plot_digit_grid(wrong, X_test, y_test, preds,
                       f"Misclassified ({len(wrong)} total)", cols=6, max_show=30)
trackio.log({"misclassified_grid": trackio.Image(
    path, caption=f"{len(wrong)} errors — label shows true→predicted")})

trackio.log({"test_accuracy": round(acc, 4), "n_misclassified": len(wrong)})
trackio.finish()

print(f"\n→ Dashboard → 'Media & Tables' tab")
print(f"  Two compact grids: correct vs misclassified with labels")
