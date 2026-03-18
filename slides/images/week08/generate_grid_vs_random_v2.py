"""Generate improved Grid vs Random Search visualizations.
Shows WHY random is better: 1D coverage + 2D projection clearly."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Figure 1: 1D demonstration ---
# True function with a narrow peak
x = np.linspace(0, 1, 500)
f = np.sin(5 * np.pi * x) * np.exp(-2 * (x - 0.7)**2) + 0.5

# Grid: 5 evenly spaced points
grid_pts = np.linspace(0.1, 0.9, 5)
grid_vals = np.interp(grid_pts, x, f)

# Random: 5 random points
rand_pts = np.sort(np.random.uniform(0.05, 0.95, 5))
rand_vals = np.interp(rand_pts, x, f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

for ax, pts, vals, title, color in [
    (ax1, grid_pts, grid_vals, 'Grid Search (5 evaluations)', '#009688'),
    (ax2, rand_pts, rand_vals, 'Random Search (5 evaluations)', '#5C6BC0')
]:
    ax.plot(x, f, 'k-', alpha=0.3, linewidth=1.5, label='True function (hidden)')
    ax.scatter(pts, vals, s=120, color=color, zorder=5, edgecolors='white', linewidths=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Hyperparameter value', fontsize=12)
    ax.set_ylabel('CV Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.3)
    best_idx = np.argmax(vals)
    ax.annotate(f'Best: {vals[best_idx]:.2f}', xy=(pts[best_idx], vals[best_idx]),
                xytext=(pts[best_idx], vals[best_idx] + 0.15),
                fontsize=11, fontweight='bold', color=color, ha='center',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    # Mark where true peak is
    peak_x = x[np.argmax(f)]
    ax.axvline(peak_x, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(peak_x + 0.02, 1.2, 'True peak', fontsize=9, color='red', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/nipun/git/stt-ai-teaching/slides/images/week08/grid_vs_random_1d.png',
            dpi=150, bbox_inches='tight', facecolor='white')

# --- Figure 2: 2D with clearer projection ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Grid
grid_vals_x = np.linspace(0.1, 0.9, 5)
grid_vals_y = np.linspace(0.1, 0.9, 5)
gx, gy = np.meshgrid(grid_vals_x, grid_vals_y)
gx_flat, gy_flat = gx.ravel(), gy.ravel()

# Random
rx = np.random.uniform(0.05, 0.95, 25)
ry = np.random.uniform(0.05, 0.95, 25)

# Background: show that the function mainly varies along x (important param)
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
zz = np.sin(3 * np.pi * xx) * np.exp(-2 * (xx - 0.65)**2) + 0.3  # only depends on x!

for ax, px, py, title, color, proj_color in [
    (ax1, gx_flat, gy_flat, 'Grid Search (25 evals)', '#009688', '#00796B'),
    (ax2, rx, ry, 'Random Search (25 evals)', '#5C6BC0', '#3949AB')
]:
    ax.contourf(xx, yy, zz, levels=20, alpha=0.25, cmap='YlOrRd')
    ax.scatter(px, py, s=60, color=color, zorder=5, edgecolors='white', linewidths=1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Important parameter', fontsize=12)
    ax.set_ylabel('Unimportant parameter', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Project onto x-axis (the important one)
    unique_x = np.unique(np.round(px, 3))
    for xi in unique_x:
        ax.plot([xi, xi], [-0.02, 0.03], color=proj_color, linewidth=2.5,
                solid_capstyle='round', clip_on=False)

    n_unique = len(unique_x)
    ax.text(0.5, -0.10, f'{n_unique} unique values of important param',
            ha='center', va='top', fontsize=11, color=proj_color,
            fontweight='bold', transform=ax.transData)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.text(0.5, 0.01, 'The function only varies along x — grid wastes 20 of 25 evals repeating the same 5 x-values!',
          ha='center', fontsize=11, fontstyle='italic', color='#555')

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('/Users/nipun/git/stt-ai-teaching/slides/images/week08/grid_vs_random_2d.png',
            dpi=150, bbox_inches='tight', facecolor='white')

print("Done: grid_vs_random_1d.png, grid_vs_random_2d.png")
