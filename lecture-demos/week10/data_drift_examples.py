#!/usr/bin/env python3
"""
Data Drift Examples — 4 scenarios showing sklearn models breaking under drift.
Run this to get exact numbers for slides.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

np.random.seed(42)
N_TRAIN, N_TEST = 500, 200

def ks_table(df_train, df_test, cols):
    rows = []
    for c in cols:
        stat, p = ks_2samp(df_train[c], df_test[c])
        rows.append({"feature": c, "KS": round(stat, 3), "p": round(p, 4),
                      "drift": "DRIFT" if p < 0.05 else "ok"})
    return pd.DataFrame(rows)

# ============================================================
print("=" * 60)
print("SCENARIO 1: Apartment Rent from Sqft")
print("  X: scalar (sqft), Y: scalar (rent in ₹k/month)")
print("  Model: LinearRegression")
print("=" * 60)

# True relationship is quadratic — linear model works in training range
# but extrapolates badly to larger apartments
sqft_train = np.random.normal(700, 120, N_TRAIN).clip(300, 1200)
rent_train = 3 + 0.008 * sqft_train + 0.000008 * sqft_train**2 + np.random.normal(0, 1.2, N_TRAIN)

sqft_clean = np.random.normal(700, 120, N_TEST).clip(300, 1200)
rent_clean = 3 + 0.008 * sqft_clean + 0.000008 * sqft_clean**2 + np.random.normal(0, 1.2, N_TEST)

# Drift: production data from Mumbai — much larger apartments
sqft_drift = np.random.normal(1800, 300, N_TEST).clip(800, 3000)
rent_drift = 3 + 0.008 * sqft_drift + 0.000008 * sqft_drift**2 + np.random.normal(0, 1.2, N_TEST)

model1 = LinearRegression()
model1.fit(sqft_train.reshape(-1, 1), rent_train)

r2_1c = r2_score(rent_clean, model1.predict(sqft_clean.reshape(-1, 1)))
r2_1d = r2_score(rent_drift, model1.predict(sqft_drift.reshape(-1, 1)))
mae_1c = mean_absolute_error(rent_clean, model1.predict(sqft_clean.reshape(-1, 1)))
mae_1d = mean_absolute_error(rent_drift, model1.predict(sqft_drift.reshape(-1, 1)))

print(f"Clean:   R²={r2_1c:.2f}, MAE=₹{mae_1c:.1f}k")
print(f"Drifted: R²={r2_1d:.2f}, MAE=₹{mae_1d:.1f}k")
ks_s, ks_p = ks_2samp(sqft_train, sqft_drift)
print(f"KS test: D={ks_s:.3f}, p={ks_p:.2e}")

# ============================================================
print("\n" + "=" * 60)
print("SCENARIO 2: Exam Pass/Fail from Study Hours")
print("  X: scalar (hours/week), Y: binary (pass/fail)")
print("  Model: DecisionTreeClassifier (depth=4, overfits training range)")
print("=" * 60)

# Decision tree learns a specific pattern for hours 8-22
# Drift: students from a crash course — study way more hours (25-40)
# but the tree never saw that range and makes random predictions there
hours_train = np.random.normal(15, 4, N_TRAIN).clip(4, 28)
# Nonlinear relationship: pass rate peaks around 18 hours, drops with too many
prob_train = 1 / (1 + np.exp(-1.2 * (hours_train - 10))) * (1 / (1 + np.exp(0.3 * (hours_train - 25))))
prob_train = prob_train / prob_train.max() * 0.92  # scale
pass_train = np.random.binomial(1, prob_train.clip(0.05, 0.95))

hours_clean = np.random.normal(15, 4, N_TEST).clip(4, 28)
prob_clean = 1 / (1 + np.exp(-1.2 * (hours_clean - 10))) * (1 / (1 + np.exp(0.3 * (hours_clean - 25))))
prob_clean = prob_clean / prob_clean.max() * 0.92
pass_clean = np.random.binomial(1, prob_clean.clip(0.05, 0.95))

hours_drift = np.random.normal(30, 5, N_TEST).clip(15, 50)
prob_drift = 1 / (1 + np.exp(-1.2 * (hours_drift - 10))) * (1 / (1 + np.exp(0.3 * (hours_drift - 25))))
prob_drift = prob_drift / prob_drift.max() * 0.92
pass_drift = np.random.binomial(1, prob_drift.clip(0.05, 0.95))

model2 = DecisionTreeClassifier(max_depth=4, random_state=42)
model2.fit(hours_train.reshape(-1, 1), pass_train)

acc_2c = accuracy_score(pass_clean, model2.predict(hours_clean.reshape(-1, 1)))
acc_2d = accuracy_score(pass_drift, model2.predict(hours_drift.reshape(-1, 1)))

print(f"Clean:   Accuracy={acc_2c:.1%}")
print(f"Drifted: Accuracy={acc_2d:.1%}")
ks_s, ks_p = ks_2samp(hours_train, hours_drift)
print(f"KS test: D={ks_s:.3f}, p={ks_p:.2e}")

# ============================================================
print("\n" + "=" * 60)
print("SCENARIO 3: Used Car Price from 3 Features")
print("  X: [km_driven, age_years, engine_cc], Y: price (₹ lakhs)")
print("  Model: LinearRegression")
print("=" * 60)

km_train = np.random.normal(30000, 12000, N_TRAIN).clip(5000, 80000)
age_train = np.random.normal(3, 1.2, N_TRAIN).clip(0.5, 10)
cc_train = np.random.normal(1200, 250, N_TRAIN).clip(600, 2500)
# True relationship: price decays as 1/(1+km/40000) — strongly nonlinear
# Linear model will extrapolate badly to high-km cars
def true_price(km, age, cc, n):
    return (12.0 / (1 + km / 40000) - 0.5 * age + 0.0015 * cc
            + np.random.normal(0, 0.5, n))

price_train = true_price(km_train, age_train, cc_train, N_TRAIN)
df_t3 = pd.DataFrame({"km": km_train, "age": age_train, "cc": cc_train})

# Clean test
km_c = np.random.normal(30000, 12000, N_TEST).clip(5000, 80000)
age_c = np.random.normal(3, 1.2, N_TEST).clip(0.5, 10)
cc_c = np.random.normal(1200, 250, N_TEST).clip(600, 2500)
price_c = true_price(km_c, age_c, cc_c, N_TEST)
df_c3 = pd.DataFrame({"km": km_c, "age": age_c, "cc": cc_c})

# Drift A: only km shifts (much higher mileage cars)
km_dA = np.random.normal(120000, 30000, N_TEST).clip(60000, 250000)
age_dA = np.random.normal(3, 1.2, N_TEST).clip(0.5, 10)
cc_dA = np.random.normal(1200, 250, N_TEST).clip(600, 2500)
price_dA = true_price(km_dA, age_dA, cc_dA, N_TEST)
df_dA3 = pd.DataFrame({"km": km_dA, "age": age_dA, "cc": cc_dA})

# Drift B: all 3 shift (old high-mileage small cars)
km_dB = np.random.normal(120000, 30000, N_TEST).clip(60000, 250000)
age_dB = np.random.normal(8, 2, N_TEST).clip(2, 15)
cc_dB = np.random.normal(800, 150, N_TEST).clip(600, 2500)
price_dB = true_price(km_dB, age_dB, cc_dB, N_TEST)
df_dB3 = pd.DataFrame({"km": km_dB, "age": age_dB, "cc": cc_dB})

model3 = LinearRegression()
model3.fit(df_t3, price_train)

r2_3c = r2_score(price_c, model3.predict(df_c3))
r2_3dA = r2_score(price_dA, model3.predict(df_dA3))
r2_3dB = r2_score(price_dB, model3.predict(df_dB3))
mae_3c = mean_absolute_error(price_c, model3.predict(df_c3))
mae_3dA = mean_absolute_error(price_dA, model3.predict(df_dA3))
mae_3dB = mean_absolute_error(price_dB, model3.predict(df_dB3))

print(f"Clean:           R²={r2_3c:.2f}, MAE=₹{mae_3c:.1f}L")
print(f"1 feature drift: R²={r2_3dA:.2f}, MAE=₹{mae_3dA:.1f}L")
print(f"3 feature drift: R²={r2_3dB:.2f}, MAE=₹{mae_3dB:.1f}L")
print("\nKS tests (Drift A — only km):")
print(ks_table(df_t3, df_dA3, ["km", "age", "cc"]).to_string(index=False))
print("\nKS tests (Drift B — all features):")
print(ks_table(df_t3, df_dB3, ["km", "age", "cc"]).to_string(index=False))

# ============================================================
print("\n" + "=" * 60)
print("SCENARIO 4: Loan Approval from 3 Features")
print("  X: [income, credit_score, loan_amount], Y: approved (0/1)")
print("  Model: DecisionTreeClassifier (overfits training range)")
print("=" * 60)

inc_train = np.random.normal(8, 2.5, N_TRAIN).clip(2, 20)
cs_train = np.random.normal(720, 40, N_TRAIN).clip(500, 850)
la_train = np.random.normal(5, 1.5, N_TRAIN).clip(1, 15)
# Approval depends on income-to-loan ratio (nonlinear!)
# and credit score threshold (if cs > 680, much easier to approve)
def true_approve(inc, cs, la, n):
    ratio = inc / (la + 0.1)
    logit = (1.5 * ratio + 0.03 * (cs - 680) - 1.5
             + 0.5 * (cs > 700).astype(float) * ratio)  # interaction
    prob = 1 / (1 + np.exp(-logit))
    return np.random.binomial(1, prob.clip(0.05, 0.95))

approve_train = true_approve(inc_train, cs_train, la_train, N_TRAIN)
df_t4 = pd.DataFrame({"income": inc_train, "credit": cs_train, "loan_amt": la_train})

# Clean
inc_c = np.random.normal(8, 2.5, N_TEST).clip(2, 20)
cs_c = np.random.normal(720, 40, N_TEST).clip(500, 850)
la_c = np.random.normal(5, 1.5, N_TEST).clip(1, 15)
approve_c = true_approve(inc_c, cs_c, la_c, N_TEST)
df_c4 = pd.DataFrame({"income": inc_c, "credit": cs_c, "loan_amt": la_c})

# Drift A: income drops (recession) — ratio changes
inc_dA = np.random.normal(4, 1.5, N_TEST).clip(1, 15)
cs_dA = np.random.normal(720, 40, N_TEST).clip(500, 850)
la_dA = np.random.normal(5, 1.5, N_TEST).clip(1, 15)
approve_dA = true_approve(inc_dA, cs_dA, la_dA, N_TEST)
df_dA4 = pd.DataFrame({"income": inc_dA, "credit": cs_dA, "loan_amt": la_dA})

# Drift B: all shift (recession + risky applicants + bigger loans)
inc_dB = np.random.normal(4, 1.5, N_TEST).clip(1, 15)
cs_dB = np.random.normal(640, 50, N_TEST).clip(400, 850)
la_dB = np.random.normal(10, 2.5, N_TEST).clip(2, 25)
approve_dB = true_approve(inc_dB, cs_dB, la_dB, N_TEST)
df_dB4 = pd.DataFrame({"income": inc_dB, "credit": cs_dB, "loan_amt": la_dB})

model4 = DecisionTreeClassifier(max_depth=6, random_state=42)
model4.fit(df_t4, approve_train)

acc_4c = accuracy_score(approve_c, model4.predict(df_c4))
acc_4dA = accuracy_score(approve_dA, model4.predict(df_dA4))
acc_4dB = accuracy_score(approve_dB, model4.predict(df_dB4))

print(f"Clean:           Accuracy={acc_4c:.1%}")
print(f"1 feature drift: Accuracy={acc_4dA:.1%}")
print(f"3 feature drift: Accuracy={acc_4dB:.1%}")
print("\nKS tests (Drift A — only income):")
print(ks_table(df_t4, df_dA4, ["income", "credit", "loan_amt"]).to_string(index=False))
print("\nKS tests (Drift B — all features):")
print(ks_table(df_t4, df_dB4, ["income", "credit", "loan_amt"]).to_string(index=False))

# ============================================================
print("\n" + "=" * 60)
print("GRAND SUMMARY")
print("=" * 60)
print(f"{'Scenario':<30} {'Clean':>10} {'1-feat':>10} {'All-feat':>10}")
print("-" * 62)
print(f"{'1. Rent (R²)':<30} {r2_1c:>10.2f} {r2_1d:>10.2f} {'—':>10}")
print(f"{'2. Pass/Fail (Acc)':<30} {acc_2c:>10.1%} {acc_2d:>10.1%} {'—':>10}")
print(f"{'3. Car Price (R²)':<30} {r2_3c:>10.2f} {r2_3dA:>10.2f} {r2_3dB:>10.2f}")
print(f"{'3. Car Price (MAE ₹L)':<30} {mae_3c:>10.1f} {mae_3dA:>10.1f} {mae_3dB:>10.1f}")
print(f"{'4. Loan (Acc)':<30} {acc_4c:>10.1%} {acc_4dA:>10.1%} {acc_4dB:>10.1%}")
