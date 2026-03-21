"""
Demo 4: Train a model and save it.

WITHOUT a volume: the saved model disappears when the container stops.
WITH a volume (-v): the model persists on your laptop.

    # Without volume — model vanishes!
    docker run train-save
    ls outputs/   # nothing there

    # With volume — model persists!
    docker run -v $(pwd)/outputs:/app/outputs train-save
    ls outputs/   # model.pkl is there!
"""
import os
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("=" * 50)
print("Demo 4: Training + Saving (Volumes)")
print("=" * 50)

# Train
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save
os.makedirs("outputs", exist_ok=True)
save_path = "outputs/model.pkl"
joblib.dump(model, save_path)
size_kb = os.path.getsize(save_path) / 1024
print(f"Saved to: {save_path} ({size_kb:.0f} KB)")

# Write a log
log_path = "outputs/training_log.txt"
with open(log_path, "w") as f:
    f.write(f"accuracy={accuracy:.4f}\n")
    f.write(f"model_size_kb={size_kb:.0f}\n")
    f.write(f"n_estimators=100\n")
print(f"Log saved to: {log_path}")

print()
print("If you ran WITHOUT -v, these files exist only")
print("inside the container. Stop it → they're GONE.")
print()
print("If you ran WITH -v $(pwd)/outputs:/app/outputs,")
print("check your laptop's outputs/ folder — they're there!")
