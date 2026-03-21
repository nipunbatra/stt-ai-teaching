"""
Train a model inside Docker — proves that dependencies work.
Without Docker, students on different machines get different sklearn versions
and potentially different results.
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn
import platform
import sys

print("=" * 50)
print("Training a model INSIDE Docker")
print("=" * 50)
print(f"Python:       {sys.version.split()[0]}")
print(f"sklearn:      {sklearn.__version__}")
print(f"OS:           {platform.system()} {platform.release()}")
print()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print()
print("Everyone running this Docker image gets EXACTLY")
print(f"this accuracy: {accuracy:.4f}")
print("Same Python, same sklearn, same OS, same result.")
