"""Generate polynomial CV scores plot for Week 8 slides."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

np.random.seed(42)
n = 40
X = np.sort(np.random.uniform(0, 6, n)).reshape(-1, 1)
y = np.sin(X.ravel()) * X.ravel() + np.random.normal(0, 0.5, n)

degrees = range(1, 16)
cv_means, cv_stds = [], []
for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), LinearRegression())
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_means.append(-scores.mean())
    cv_stds.append(scores.std())

cv_means = np.array(cv_means)
cv_stds = np.array(cv_stds)

# Plot 1: CV Error
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(degrees), cv_means, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='CV Error (lower = better)')
ax.fill_between(list(degrees), cv_means - cv_stds, cv_means + cv_stds, alpha=0.15, color='#e74c3c')
best_idx = np.argmin(cv_means)
ax.plot(best_idx + 1, cv_means[best_idx], '*', color='#2ecc71', markersize=20, zorder=5, label=f'Best: degree {best_idx+1}')
ax.axvspan(0.5, 2.5, alpha=0.08, color='blue', label='Underfitting')
ax.axvspan(8.5, 15.5, alpha=0.08, color='red', label='Overfitting')
ax.set_xlabel('Polynomial Degree', fontsize=14)
ax.set_ylabel('CV Error (MSE)', fontsize=14)
ax.set_title('Which Polynomial Degree Is Best?', fontsize=16, fontweight='bold')
ax.set_xticks(list(degrees))
ax.legend(fontsize=11, loc='upper left')
ax.tick_params(labelsize=12)
ax.set_xlim(0.5, 15.5)
plt.tight_layout()
plt.savefig('polynomial_cv_scores.png', dpi=150, bbox_inches='tight', facecolor='white')

# Plot 2: Reframed as optimization
fig2, ax2 = plt.subplots(figsize=(10, 5))
neg_cv = -cv_means
ax2.plot(list(degrees), neg_cv, 'o-', color='#3498db', linewidth=2, markersize=8)
ax2.fill_between(list(degrees), neg_cv - cv_stds, neg_cv + cv_stds, alpha=0.15, color='#3498db')
ax2.plot(best_idx + 1, neg_cv[best_idx], '*', color='#2ecc71', markersize=20, zorder=5, label=f'Maximum at degree {best_idx+1}')
ax2.set_xlabel('Hyperparameter (polynomial degree)', fontsize=14)
ax2.set_ylabel('Score (higher = better)', fontsize=14)
ax2.set_title('Hyperparameter Tuning = Finding the Maximum', fontsize=16, fontweight='bold')
ax2.set_xticks(list(degrees))
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=12)
ax2.annotate('Where is the\npeak?', xy=(8, neg_cv[7]), fontsize=13, fontweight='bold',
            color='#e74c3c', ha='center',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
            xytext=(11, neg_cv[2]))
plt.tight_layout()
plt.savefig('tuning_as_optimization.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Done: polynomial_cv_scores.png, tuning_as_optimization.png")
