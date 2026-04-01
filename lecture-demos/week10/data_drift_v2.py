#!/usr/bin/env python3
"""
Better data drift examples with honest numbers.
Data drift = same city/context, distribution shifts OVER TIME.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ks_2samp

np.random.seed(42)

print("=" * 70)
print("SCENARIO 1A: Rent prediction — SAME CITY, distribution shifts over time")
print("  Ahmedabad 2022 → Ahmedabad 2026 (new constructions are bigger)")
print("=" * 70)

# 2022: mostly old 1-2BHK apartments
sqft_2022 = np.random.normal(650, 100, 500).clip(300, 1000)
# True: rent = 2 + 0.015*sqft (roughly linear, slight curve)
rent_2022 = 2 + 0.015 * sqft_2022 + 0.0000015 * sqft_2022**2 + np.random.normal(0, 0.8, 500)

# 2026: new constructions → larger apartments
sqft_2026 = np.random.normal(950, 180, 200).clip(400, 1600)
rent_2026 = 2 + 0.015 * sqft_2026 + 0.0000015 * sqft_2026**2 + np.random.normal(0, 0.8, 200)

# Clean test (also 2022-style)
sqft_clean = np.random.normal(650, 100, 200).clip(300, 1000)
rent_clean = 2 + 0.015 * sqft_clean + 0.0000015 * sqft_clean**2 + np.random.normal(0, 0.8, 200)

model = LinearRegression()
model.fit(sqft_2022.reshape(-1,1), rent_2022)

r2_clean = r2_score(rent_clean, model.predict(sqft_clean.reshape(-1,1)))
r2_drift = r2_score(rent_2026, model.predict(sqft_2026.reshape(-1,1)))
mae_clean = mean_absolute_error(rent_clean, model.predict(sqft_clean.reshape(-1,1)))
mae_drift = mean_absolute_error(rent_2026, model.predict(sqft_2026.reshape(-1,1)))

ks_stat, ks_p = ks_2samp(sqft_2022, sqft_2026)

print(f"Training range: {sqft_2022.min():.0f} - {sqft_2022.max():.0f} sqft")
print(f"Clean test range: {sqft_clean.min():.0f} - {sqft_clean.max():.0f} sqft")
print(f"Drifted range: {sqft_2026.min():.0f} - {sqft_2026.max():.0f} sqft")
print(f"\nClean:   R²={r2_clean:.3f}, MAE=₹{mae_clean:.2f}k")
print(f"Drifted: R²={r2_drift:.3f}, MAE=₹{mae_drift:.2f}k")
print(f"KS test: D={ks_stat:.3f}, p={ks_p:.2e}")

print("\n" + "=" * 70)
print("SCENARIO 1B: Stronger drift — 2022 → 2028 (even larger shift)")
print("=" * 70)

sqft_2028 = np.random.normal(1200, 250, 200).clip(500, 2000)
rent_2028 = 2 + 0.015 * sqft_2028 + 0.0000015 * sqft_2028**2 + np.random.normal(0, 0.8, 200)

r2_drift2 = r2_score(rent_2028, model.predict(sqft_2028.reshape(-1,1)))
mae_drift2 = mean_absolute_error(rent_2028, model.predict(sqft_2028.reshape(-1,1)))
ks_stat2, ks_p2 = ks_2samp(sqft_2022, sqft_2028)

print(f"Drifted range: {sqft_2028.min():.0f} - {sqft_2028.max():.0f} sqft")
print(f"Drifted: R²={r2_drift2:.3f}, MAE=₹{mae_drift2:.2f}k")
print(f"KS test: D={ks_stat2:.3f}, p={ks_p2:.2e}")

print("\n" + "=" * 70)
print("SCENARIO 1C: Domain adaptation (different city entirely)")
print("  Train Ahmedabad → Deploy Mumbai")
print("=" * 70)

sqft_mum = np.random.normal(500, 120, 200).clip(200, 900)
# Mumbai has DIFFERENT pricing (much more per sqft)
rent_mum = 8 + 0.035 * sqft_mum + np.random.normal(0, 1.5, 200)

r2_domain = r2_score(rent_mum, model.predict(sqft_mum.reshape(-1,1)))
mae_domain = mean_absolute_error(rent_mum, model.predict(sqft_mum.reshape(-1,1)))
ks_stat3, ks_p3 = ks_2samp(sqft_2022, sqft_mum)

print(f"Mumbai range: {sqft_mum.min():.0f} - {sqft_mum.max():.0f} sqft (similar sizes!)")
print(f"Mumbai: R²={r2_domain:.3f}, MAE=₹{mae_domain:.2f}k")
print(f"KS test on sqft: D={ks_stat3:.3f}, p={ks_p3:.2e}")
print("Note: sqft distribution may be SIMILAR but pricing is totally different!")
print("This is DOMAIN ADAPTATION, not data drift.")
print("A KS test on X would NOT catch this — you need to check Y!")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("Data drift:        Same city, P(X) shifts over time → KS test catches it")
print("Domain adaptation: Different city, P(Y|X) is different → KS on X may miss it!")
print("                   This is closer to concept drift / distribution shift")
