"""
Demo: Profiling a "slow" prediction endpoint
=============================================
Shows how cProfile reveals the real bottleneck.

    python profiling_demo.py
"""
import cProfile
import time
import joblib
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Setup: train and save a model + data ──
print("Setting up...")
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "/tmp/demo_model.pkl")
pd.DataFrame(X_train).to_csv("/tmp/demo_data.csv", index=False)

print(f"Model saved. Test accuracy: {model.score(X_test, y_test):.1%}\n")


# ── BAD version: loads everything per request ──
def slow_predict(sample):
    """Simulates a slow API endpoint that reloads on every call."""
    data = pd.read_csv("/tmp/demo_data.csv")
    m = joblib.load("/tmp/demo_model.pkl")
    return m.predict([sample])[0]


# ── GOOD version: loads once ──
_model = joblib.load("/tmp/demo_model.pkl")

def fast_predict(sample):
    """Model loaded once at module level."""
    return _model.predict([sample])[0]


# ── Profile the BAD version ──
sample = X_test[0]

print("=" * 60)
print("PROFILING THE SLOW VERSION")
print("=" * 60)
cProfile.run('slow_predict(sample)', sort='cumulative')

print("\n" + "=" * 60)
print("TIMING COMPARISON")
print("=" * 60)

# Time the slow version
start = time.time()
for _ in range(10):
    slow_predict(sample)
slow_time = (time.time() - start) / 10

# Time the fast version
start = time.time()
for _ in range(100):
    fast_predict(sample)
fast_time = (time.time() - start) / 100

print(f"  Slow (reload every call): {slow_time*1000:.1f} ms")
print(f"  Fast (load once):         {fast_time*1000:.2f} ms")
print(f"  Speedup:                  {slow_time/fast_time:.0f}x")
print(f"\n→ Moving two lines gave a {slow_time/fast_time:.0f}x speedup!")
print("  That's what profiling finds — the bottleneck is never where you think.")
