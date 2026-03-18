"""
Demo 5: Catching Overfitting with Alerts
=========================================
Story: "My model trains overnight. I want to be notified if it
        starts overfitting so I can stop early and save compute."

Deliberately increase model complexity until train-test gap explodes.
Fire a trackio.alert() when the gap exceeds a threshold.

TrackIO features: trackio.alert(), AlertLevel, overfitting detection
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import trackio

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

trackio.init(project="cs203-week08-demo", name="overfitting-detector",
             config={"model": "DecisionTree", "experiment": "overfitting"})

OVERFIT_THRESHOLD = 0.08  # alert if train-test gap > 8%
alerted = False

for depth in range(1, 30):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_acc = float(dt.score(X_train, y_train))
    test_acc = float(dt.score(X_test, y_test))
    gap = round(train_acc - test_acc, 4)

    trackio.log({
        "max_depth": depth,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "overfit_gap": gap,
    })

    status = ""
    if gap > OVERFIT_THRESHOLD and not alerted:
        trackio.alert(
            title="Overfitting detected!",
            text=f"Train-test gap = {gap:.1%} at max_depth={depth}. "
                 f"Train={train_acc:.3f}, Test={test_acc:.3f}. "
                 f"Consider stopping here or reducing complexity.",
            level=trackio.AlertLevel.ERROR)
        alerted = True
        status = "  ← ALERT FIRED"

    print(f"  depth={depth:2d}  train={train_acc:.3f}  test={test_acc:.3f}  gap={gap:.3f}{status}")

trackio.finish()
print("\n→ Dashboard shows train/test curves diverging")
print("  Check the alert icon — it fired when overfitting was detected!")
print("  In real use, alerts can trigger Slack/Discord webhooks.")
