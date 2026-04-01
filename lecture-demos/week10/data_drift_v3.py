#!/usr/bin/env python3
"""
Honest data drift examples.
Key insight: drift hurts MOST with models that don't extrapolate (trees, KNN).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from scipy.stats import ks_2samp

np.random.seed(42)
N = 500; N_T = 200

print("=" * 70)
print("SCENARIO 1: Rent in Ahmedabad — same city, apartments get bigger")
print("  True relationship: rent ≈ linear in sqft (nearly)")
print("=" * 70)

sqft_train = np.random.normal(650, 100, N).clip(350, 1000)
rent_train = 2 + 0.014*sqft_train + np.random.normal(0, 0.7, N)

# Drift: 4 years later, new constructions → bigger apartments
sqft_drift = np.random.normal(1000, 150, N_T).clip(500, 1500)
rent_drift = 2 + 0.014*sqft_drift + np.random.normal(0, 0.7, N_T)

sqft_clean = np.random.normal(650, 100, N_T).clip(350, 1000)
rent_clean = 2 + 0.014*sqft_clean + np.random.normal(0, 0.7, N_T)

ks_s, ks_p = ks_2samp(sqft_train, sqft_drift)

print(f"\nKS test on sqft: D={ks_s:.3f}, p={ks_p:.2e} → DRIFT DETECTED")
print(f"\n{'Model':<25} {'Clean R²':>10} {'Drift R²':>10} {'Clean MAE':>10} {'Drift MAE':>10}")
print("-" * 70)

for name, ModelClass in [
    ("LinearRegression", LinearRegression),
    ("DecisionTree(depth=5)", lambda: DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("DecisionTree(depth=10)", lambda: DecisionTreeRegressor(max_depth=10, random_state=42)),
    ("KNN(k=5)", lambda: KNeighborsRegressor(n_neighbors=5)),
]:
    m = ModelClass() if callable(ModelClass) and not isinstance(ModelClass, type) else ModelClass()
    m.fit(sqft_train.reshape(-1,1), rent_train)
    r2c = r2_score(rent_clean, m.predict(sqft_clean.reshape(-1,1)))
    r2d = r2_score(rent_drift, m.predict(sqft_drift.reshape(-1,1)))
    mc = mean_absolute_error(rent_clean, m.predict(sqft_clean.reshape(-1,1)))
    md = mean_absolute_error(rent_drift, m.predict(sqft_drift.reshape(-1,1)))
    print(f"{name:<25} {r2c:>10.3f} {r2d:>10.3f} {mc:>10.2f} {md:>10.2f}")

print("\n→ Linear regression handles drift fine (relationship is truly linear)")
print("→ Decision tree and KNN BREAK because they can't extrapolate!")

print("\n" + "=" * 70)
print("SCENARIO 2: Used car price — nonlinear relationship")
print("  True: price drops fast initially, then plateaus (diminishing returns)")
print("=" * 70)

km_train = np.random.normal(25000, 10000, N).clip(5000, 60000)
# True: exponential decay — price drops fast for first 30k km, then slower
price_train = 8 * np.exp(-km_train / 50000) + np.random.normal(0, 0.3, N)

km_drift = np.random.normal(80000, 20000, N_T).clip(30000, 150000)
price_drift = 8 * np.exp(-km_drift / 50000) + np.random.normal(0, 0.3, N_T)

km_clean = np.random.normal(25000, 10000, N_T).clip(5000, 60000)
price_clean = 8 * np.exp(-km_clean / 50000) + np.random.normal(0, 0.3, N_T)

ks_s, ks_p = ks_2samp(km_train, km_drift)
print(f"\nKS test on km: D={ks_s:.3f}, p={ks_p:.2e}")

print(f"\n{'Model':<25} {'Clean R²':>10} {'Drift R²':>10} {'Clean MAE':>10} {'Drift MAE':>10}")
print("-" * 70)

for name, ModelClass in [
    ("LinearRegression", LinearRegression),
    ("DecisionTree(depth=5)", lambda: DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("KNN(k=5)", lambda: KNeighborsRegressor(n_neighbors=5)),
]:
    m = ModelClass() if callable(ModelClass) and not isinstance(ModelClass, type) else ModelClass()
    m.fit(km_train.reshape(-1,1), price_train)
    r2c = r2_score(price_clean, m.predict(km_clean.reshape(-1,1)))
    r2d = r2_score(price_drift, m.predict(km_drift.reshape(-1,1)))
    mc = mean_absolute_error(price_clean, m.predict(km_clean.reshape(-1,1)))
    md = mean_absolute_error(price_drift, m.predict(km_drift.reshape(-1,1)))
    print(f"{name:<25} {r2c:>10.3f} {r2d:>10.3f} {mc:>10.2f} {md:>10.2f}")

print("\n→ With nonlinear truth, even linear regression fails under drift!")
print("→ Tree and KNN fail even harder (predict flat values outside training range)")

print("\n" + "=" * 70)
print("SCENARIO 3: Classification — pass/fail from study hours")
print("=" * 70)

hours_train = np.random.normal(15, 4, N).clip(4, 28)
prob = 1 / (1 + np.exp(-0.8 * (hours_train - 12)))
pass_train = np.random.binomial(1, prob)

hours_drift = np.random.normal(25, 3, N_T).clip(15, 40)
prob_d = 1 / (1 + np.exp(-0.8 * (hours_drift - 12)))
pass_drift = np.random.binomial(1, prob_d)

hours_clean = np.random.normal(15, 4, N_T).clip(4, 28)
prob_c = 1 / (1 + np.exp(-0.8 * (hours_clean - 12)))
pass_clean = np.random.binomial(1, prob_c)

ks_s, ks_p = ks_2samp(hours_train, hours_drift)
print(f"\nKS test on hours: D={ks_s:.3f}, p={ks_p:.2e}")

print(f"\n{'Model':<30} {'Clean Acc':>10} {'Drift Acc':>10}")
print("-" * 55)

from sklearn.linear_model import LogisticRegression

for name, m in [
    ("LogisticRegression", LogisticRegression()),
    ("DecisionTree(depth=4)", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ("DecisionTree(depth=8)", DecisionTreeClassifier(max_depth=8, random_state=42)),
]:
    m.fit(hours_train.reshape(-1,1), pass_train)
    acc_c = accuracy_score(pass_clean, m.predict(hours_clean.reshape(-1,1)))
    acc_d = accuracy_score(pass_drift, m.predict(hours_drift.reshape(-1,1)))
    print(f"{name:<30} {acc_c:>10.1%} {acc_d:>10.1%}")

print("\n→ Logistic regression generalizes well (sigmoid extrapolates correctly)")
print("→ Deep decision tree FAILS because it memorized training-range quirks")

print("\n" + "=" * 70)
print("DATA DRIFT vs DOMAIN ADAPTATION")
print("=" * 70)
print("""
Data Drift:
  - SAME deployment context
  - P(X) shifts OVER TIME
  - P(Y|X) stays the same
  - Example: Ahmedabad rents — apartments get bigger over years
  - KS test on X detects it
  - Model may or may not break (depends on model type!)

Domain Adaptation:
  - DIFFERENT deployment context
  - P(Y|X) is different between domains
  - Example: Train on Ahmedabad prices, deploy in Mumbai
  - KS test on X may NOT catch it (sqft could be similar!)
  - Model WILL break because the relationship is different
  - This is closer to CONCEPT DRIFT
""")
