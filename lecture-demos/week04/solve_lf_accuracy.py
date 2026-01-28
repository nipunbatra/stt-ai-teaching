"""
Solving for Labeling Function Accuracies
=========================================
This demo shows how Snorkel estimates LF accuracy from agreement patterns.

Given observed agreement rates between LF pairs, we solve for individual accuracies.

Run: python solve_lf_accuracy.py
"""

import numpy as np

print("=" * 60)
print("SOLVING FOR LF ACCURACIES FROM AGREEMENT PATTERNS")
print("=" * 60)

# =============================================================================
# The Setup
# =============================================================================

print("\n📊 OBSERVED AGREEMENT RATES:")
print("-" * 40)

# These are the observed agreement rates between LF pairs
# Agreement = P(both correct) + P(both wrong)
#           = α_i * α_j + (1-α_i) * (1-α_j)

agree_12 = 0.85  # LF1 and LF2 agree 85% of the time
agree_13 = 0.80  # LF1 and LF3 agree 80% of the time
agree_23 = 0.90  # LF2 and LF3 agree 90% of the time

print(f"  LF₁ & LF₂ agree: {agree_12:.0%}")
print(f"  LF₁ & LF₃ agree: {agree_13:.0%}")
print(f"  LF₂ & LF₃ agree: {agree_23:.0%}")

print("\n📐 THE EQUATIONS:")
print("-" * 40)
print("  α₁α₂ + (1-α₁)(1-α₂) = 0.85")
print("  α₁α₃ + (1-α₁)(1-α₃) = 0.80")
print("  α₂α₃ + (1-α₂)(1-α₃) = 0.90")
print("\n  3 equations, 3 unknowns → solvable!")

# =============================================================================
# Method 1: Iterative Solution
# =============================================================================

print("\n" + "=" * 60)
print("METHOD 1: ITERATIVE SOLUTION")
print("=" * 60)

def agreement(a_i, a_j):
    """Expected agreement rate given two LF accuracies."""
    return a_i * a_j + (1 - a_i) * (1 - a_j)

def solve_for_alpha(agreement_rate, other_alpha):
    """Solve for α_i given agreement rate and α_j."""
    # From: α_i * α_j + (1-α_i)(1-α_j) = agreement
    # Rearranging: α_i = (agreement - 1 + α_j) / (2*α_j - 1)
    # But this can give values outside [0.5, 1], so we use optimization

    from scipy.optimize import brentq

    def error(a_i):
        return agreement(a_i, other_alpha) - agreement_rate

    try:
        return brentq(error, 0.5, 0.99)
    except:
        return other_alpha  # fallback

# Initialize with random guesses
np.random.seed(42)
α1, α2, α3 = 0.6, 0.6, 0.6

print(f"\n  Initial guess: α₁={α1:.3f}, α₂={α2:.3f}, α₃={α3:.3f}")
print("\n  Iterating...")
print("-" * 40)

for i in range(10):
    # Update each α based on agreements with others
    α1_new = solve_for_alpha(agree_12, α2)
    α2_new = solve_for_alpha(agree_23, α3)
    α3_new = solve_for_alpha(agree_13, α1)

    # Damped update for stability
    α1 = 0.5 * α1 + 0.5 * α1_new
    α2 = 0.5 * α2 + 0.5 * α2_new
    α3 = 0.5 * α3 + 0.5 * α3_new

    # Check convergence
    pred_12 = agreement(α1, α2)
    pred_13 = agreement(α1, α3)
    pred_23 = agreement(α2, α3)
    error = abs(pred_12 - agree_12) + abs(pred_13 - agree_13) + abs(pred_23 - agree_23)

    print(f"  Iter {i+1:2d}: α₁={α1:.4f}, α₂={α2:.4f}, α₃={α3:.4f}  (error={error:.6f})")

    if error < 0.0001:
        print(f"\n  ✓ Converged at iteration {i+1}!")
        break

# =============================================================================
# Verify Solution
# =============================================================================

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print(f"\n  Final solution:")
print(f"    α₁ = {α1:.3f}  (LF₁ is correct {α1:.0%} of the time)")
print(f"    α₂ = {α2:.3f}  (LF₂ is correct {α2:.0%} of the time)")
print(f"    α₃ = {α3:.3f}  (LF₃ is correct {α3:.0%} of the time)")

print(f"\n  Predicted agreement rates:")
print(f"    LF₁ & LF₂: {agreement(α1, α2):.3f} (observed: {agree_12})")
print(f"    LF₁ & LF₃: {agreement(α1, α3):.3f} (observed: {agree_13})")
print(f"    LF₂ & LF₃: {agreement(α2, α3):.3f} (observed: {agree_23})")

# =============================================================================
# Method 2: Direct Optimization
# =============================================================================

print("\n" + "=" * 60)
print("METHOD 2: SCIPY OPTIMIZATION")
print("=" * 60)

from scipy.optimize import minimize

def objective(alphas):
    """Sum of squared errors between predicted and observed agreements."""
    a1, a2, a3 = alphas
    pred_12 = agreement(a1, a2)
    pred_13 = agreement(a1, a3)
    pred_23 = agreement(a2, a3)
    return (pred_12 - agree_12)**2 + (pred_13 - agree_13)**2 + (pred_23 - agree_23)**2

result = minimize(
    objective,
    x0=[0.7, 0.7, 0.7],  # initial guess
    bounds=[(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]  # accuracies between 50% and 100%
)

print(f"\n  Optimization result:")
print(f"    α₁ = {result.x[0]:.4f}")
print(f"    α₂ = {result.x[1]:.4f}")
print(f"    α₃ = {result.x[2]:.4f}")
print(f"    Final error: {result.fun:.10f}")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
  Snorkel estimates LF accuracies WITHOUT knowing true labels!

  It only needs:
    - The label matrix (which LFs voted what on each example)
    - Counts of agreements/disagreements between LF pairs

  This is the magic of weak supervision!
""")
