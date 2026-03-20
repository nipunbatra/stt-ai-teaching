"""
Demo 1: Watch a Model Learn (Training Curves)
==============================================
Story: "Is my model still improving, or has it plateaued?"

Train GradientBoosting incrementally (10→300 trees).
Log train/test accuracy at each step → dashboard shows live-style curves.

TrackIO features: scalar logging, multi-step runs
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import trackio

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

trackio.init(project="cs203-week08-demo", name="gb-training",
             space_id="nipunbatra/cs203-week08-demo",
             config={"model": "GradientBoosting", "lr": 0.1, "max_depth": 3})

for n_est in range(10, 310, 10):
    gb = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)

    train_acc = float(gb.score(X_train, y_train))
    test_acc = float(gb.score(X_test, y_test))

    trackio.log({
        "n_estimators": n_est,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "overfit_gap": round(train_acc - test_acc, 4),
    })
    print(f"  n={n_est:3d}  train={train_acc:.3f}  test={test_acc:.3f}")

trackio.finish()
print("\n→ Open dashboard, look at Metrics tab: train vs test accuracy curves")
print("  Notice where test accuracy plateaus while train keeps climbing!")
