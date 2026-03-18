"""Generate a clear 1D Grid vs Random comparison.
Uses a continuous x-axis (learning rate on log scale) with a narrow peak
that grid's fixed spacing misses but random has a chance of finding."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(7)

# Continuous hyperparameter: learning rate (log scale, 1e-4 to 1e-1)
x = np.linspace(-4, -1, 500)  # log10(lr)

# True CV score: sharp peak near lr=0.003 (log10 = -2.5), drops off fast
true_score = (
    0.60
    + 0.28 * np.exp(-0.5 * ((x - (-2.52)) / 0.18)**2)   # sharp peak
    + 0.08 * np.exp(-0.5 * ((x - (-1.5)) / 0.5)**2)      # small bump
)

# Grid: 6 evenly spaced on log scale
grid_x = np.linspace(-4, -1, 6)
grid_y = np.interp(grid_x, x, true_score)

# Random: 6 random points (continuous, not snapped to grid)
rand_x = np.sort(np.random.uniform(-4, -1, 6))
rand_y = np.interp(rand_x, x, true_score)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

true_peak_x = x[np.argmax(true_score)]
true_peak_y = np.max(true_score)

for ax, pts, scores, title, color in [
    (ax1, grid_x, grid_y, 'Grid Search (6 evaluations)', '#009688'),
    (ax2, rand_x, rand_y, 'Random Search (6 evaluations)', '#5C6BC0'),
]:
    # True function as faded background
    ax.fill_between(x, 0.55, true_score, alpha=0.08, color='gray')
    ax.plot(x, true_score, 'k-', alpha=0.25, linewidth=6)
    ax.plot(x, true_score, 'k--', alpha=0.4, linewidth=1, label='True CV score (hidden)')

    # Sampled points
    ax.scatter(pts, scores, s=160, color=color, zorder=5, edgecolors='white',
               linewidths=2, label='Evaluated points')

    # Best found
    best_idx = np.argmax(scores)
    best_lr = 10**pts[best_idx]
    ax.annotate(f'Best: {scores[best_idx]:.1%}\n(lr={best_lr:.1e})',
                xy=(pts[best_idx], scores[best_idx]),
                xytext=(pts[best_idx] + 0.4, scores[best_idx] + 0.025),
                fontsize=11, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

    # True peak marker
    ax.axvline(true_peak_x, color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.scatter([true_peak_x], [true_peak_y], color='#e74c3c', s=100, marker='*',
               zorder=4, alpha=0.6)
    ax.text(true_peak_x - 0.05, true_peak_y + 0.015, f'True peak ({true_peak_y:.1%})',
            fontsize=9, color='#e74c3c', alpha=0.7, ha='right')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('log₁₀(learning_rate)', fontsize=12)
    ax.set_xlim(-4.1, -0.9)
    ax.set_ylim(0.55, 0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper left')

    # Add lr tick labels
    ax.set_xticks([-4, -3, -2, -1])
    ax.set_xticklabels(['1e-4', '1e-3', '1e-2', '1e-1'])

ax1.set_ylabel('CV Accuracy', fontsize=12)

# Explanation
gap_grid = true_peak_y - np.max(grid_y)
gap_rand = true_peak_y - np.max(rand_y)
fig.text(0.5, 0.01,
         f'Grid misses the narrow peak (gap: {gap_grid:.1%}). '
         f'Random lands closer (gap: {gap_rand:.1%}).',
         ha='center', fontsize=11, fontstyle='italic', color='#555')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/Users/nipun/git/stt-ai-teaching/slides/images/week08/grid_vs_random_1d.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Done: grid_vs_random_1d.png")
