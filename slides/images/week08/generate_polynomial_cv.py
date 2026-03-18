"""Generate polynomial CV scores plot for Week 8 slides.
Fixed: uses log scale on y-axis so high-degree explosions are visible but interpretable."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

np.random.seed(42)
n = 60
X = np.sort(np.random.uniform(0, 6, n)).reshape(-1, 1)
y = np.sin(X.ravel()) * X.ravel() + np.random.normal(0, 0.5, n)

degrees = range(1, 16)
cv_means_mse, cv_stds_mse = [], []
train_means_mse = [], []
for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), LinearRegression())
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_means_mse.append(-scores.mean())
    cv_stds_mse.append(scores.std())

cv_means = np.array(cv_means_mse)
cv_stds = np.array(cv_stds_mse)

# ---- Plot 1: CV Error with log scale ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(list(degrees), cv_means, 'o-', color='#e74c3c', linewidth=2.5, markersize=8,
            label='CV Error (MSE, lower = better)')
best_idx = np.argmin(cv_means)
ax.semilogy(best_idx + 1, cv_means[best_idx], '*', color='#2ecc71', markersize=22,
            zorder=5, label=f'Best: degree {best_idx + 1}')

# Shaded regions
ax.axvspan(0.5, 2.5, alpha=0.08, color='blue', label='Underfitting zone')
ax.axvspan(9.5, 15.5, alpha=0.08, color='red', label='Overfitting zone')

ax.set_xlabel('Polynomial Degree (hyperparameter)', fontsize=14)
ax.set_ylabel('CV Error (MSE, log scale)', fontsize=14)
ax.set_title('Which Polynomial Degree Is Best?', fontsize=16, fontweight='bold')
ax.set_xticks(list(degrees))
ax.legend(fontsize=11, loc='upper left')
ax.tick_params(labelsize=12)
ax.set_xlim(0.5, 15.5)

# Add annotation for the explosion
ax.annotate(f'Error explodes!\n({cv_means[-1]:.0e})',
            xy=(15, cv_means[-1]), fontsize=11, fontweight='bold',
            color='#c0392b', ha='center',
            xytext=(12.5, cv_means[-1] * 0.3),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

plt.tight_layout()
plt.savefig('/Users/nipun/git/stt-ai-teaching/slides/images/week08/polynomial_cv_scores.png',
            dpi=150, bbox_inches='tight', facecolor='white')

# ---- Plot 2: Reframed as optimization (score = -MSE, higher = better) ----
fig2, ax2 = plt.subplots(figsize=(10, 5))
# Clip to show meaningful range — use R² instead for better intuition
from sklearn.model_selection import cross_val_score as cvs

r2_means = []
for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), LinearRegression())
    scores = cvs(pipe, X, y, cv=5, scoring='r2')
    r2_means.append(scores.mean())

r2_means = np.array(r2_means)
ax2.plot(list(degrees), r2_means, 'o-', color='#3498db', linewidth=2.5, markersize=8)
best_r2_idx = np.argmax(r2_means)
ax2.plot(best_r2_idx + 1, r2_means[best_r2_idx], '*', color='#2ecc71', markersize=22,
         zorder=5, label=f'Peak at degree {best_r2_idx + 1}')

ax2.set_xlabel('Hyperparameter (polynomial degree)', fontsize=14)
ax2.set_ylabel('CV Score (R², higher = better)', fontsize=14)
ax2.set_title('Hyperparameter Tuning = Finding the Peak', fontsize=16, fontweight='bold')
ax2.set_xticks(list(degrees))
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=12)
ax2.set_xlim(0.5, 15.5)

# Annotate
ax2.annotate('Where is the\npeak?', xy=(8, r2_means[7]), fontsize=13, fontweight='bold',
             color='#e74c3c', ha='center',
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
             xytext=(11, r2_means[2]))

plt.tight_layout()
plt.savefig('/Users/nipun/git/stt-ai-teaching/slides/images/week08/tuning_as_optimization.png',
            dpi=150, bbox_inches='tight', facecolor='white')

print("Done: polynomial_cv_scores.png, tuning_as_optimization.png")
