"""
Demo 3: Which Learning Rate Wins? (Overlaid Runs)
==================================================
Story: "I want to try lr=0.01, 0.1, 0.5 — which converges faster?
        Which overfits? I don't want to juggle 3 terminal windows."

Train same model with 3 learning rates. Each is a separate run,
but same project → dashboard overlays all curves automatically.

TrackIO features: multiple runs overlaid, groups, confidence histograms
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import trackio

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

for lr in [0.01, 0.1, 0.5]:
    trackio.init(project="cs203-week08-demo", name=f"lr-{lr}",
                 group="lr-sweep",
                 config={"model": "GradientBoosting", "learning_rate": lr})

    for n_est in range(10, 210, 10):
        gb = GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)

        trackio.log({
            "n_estimators": n_est,
            "train_accuracy": round(float(gb.score(X_train, y_train)), 4),
            "test_accuracy": round(float(gb.score(X_test, y_test)), 4),
        })

    # Log confidence histogram at the end
    proba = gb.predict_proba(X_test)
    max_conf = proba.max(axis=1).astype(float)
    trackio.log({"prediction_confidence": trackio.Histogram(max_conf)})

    trackio.finish()
    print(f"  lr={lr:<6}  final_test={gb.score(X_test, y_test):.4f}")

print("\n→ Dashboard shows 3 overlaid curves — compare convergence speed")
print("  lr=0.5 converges fast but overfits; lr=0.01 is slow but stable")
