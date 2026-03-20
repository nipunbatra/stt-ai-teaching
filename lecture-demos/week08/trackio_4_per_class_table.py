"""
Demo 4: Per-Class Breakdown Table + Confusion Pairs (Tables)
=============================================================
Story: "97% accuracy sounds great, but digit 8 is only 89%.
        Which digits get confused with each other?"

Compute per-class precision/recall, log as a rich table.
Also find the most-confused digit pairs and show example images.

TrackIO features: trackio.Table() with mixed text + images
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import trackio

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)


def digit_to_rgb(flat_pixels, scale=2):
    """8x8 → small RGB image. scale=2 gives 16x16 (compact in tables)."""
    img = (flat_pixels.reshape(8, 8) / 16.0 * 255).astype(np.uint8)
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return np.stack([img] * 3, axis=-1)


# ── Per-class accuracy table ──
trackio.init(project="cs203-week08-demo", name="per-class-breakdown",
             space_id="nipunbatra/cs203-week08-demo",
             config={"model": "RandomForest", "experiment": "analysis"})

report = classification_report(y_test, preds, output_dict=True)
table_data = []
for digit in range(10):
    d = report[str(digit)]
    table_data.append([
        str(digit),
        f"{d['precision']:.3f}",
        f"{d['recall']:.3f}",
        f"{d['f1-score']:.3f}",
        str(int(d['support'])),
    ])

trackio.log({"per_class_metrics": trackio.Table(
    dataframe=pd.DataFrame(table_data,
                           columns=["Digit", "Precision", "Recall", "F1", "Support"]),
)})
print("Per-class metrics:")
for row in table_data:
    print(f"  Digit {row[0]}: P={row[1]} R={row[2]} F1={row[3]}")

# ── Most-confused pairs table (with example images) ──
cm = confusion_matrix(y_test, preds)
np.fill_diagonal(cm, 0)  # zero out correct predictions

# Find top confused pairs
confused_pairs = []
for _ in range(5):
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    if cm[i, j] == 0:
        break
    confused_pairs.append((int(i), int(j), int(cm[i, j])))
    cm[i, j] = 0

confusion_rows = []
for true_digit, pred_digit, count in confused_pairs:
    # Find an example of this confusion
    mask = (y_test == true_digit) & (preds == pred_digit)
    example_indices = np.where(mask)[0]
    if len(example_indices) > 0:
        img = digit_to_rgb(X_test[example_indices[0]])
        confusion_rows.append([
            trackio.Image(img, caption=f"True={true_digit}"),
            f"{true_digit} → {pred_digit}",
            str(count),
        ])

trackio.log({"top_confusions": trackio.Table(
    dataframe=pd.DataFrame(confusion_rows,
                           columns=["Example", "True → Predicted", "Count"]),
)})

trackio.finish()
print(f"\nTop confused pairs: {confused_pairs}")
print("\n→ Dashboard → 'Media & Tables' tab → see per-class breakdown")
print("  and which digit pairs get confused most often (with examples!)")
