#!/usr/bin/env python3
"""Generate a publication-quality Grid Search vs Random Search diagram."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# --- Colors ---
grid_color = '#009688'       # teal
random_color = '#5C6BC0'     # indigo/purple-blue
proj_color_grid = '#00796B'
proj_color_rand = '#3949AB'

# --- Grid Search (left panel) ---
grid_vals = np.linspace(0.1, 0.9, 5)
gx, gy = np.meshgrid(grid_vals, grid_vals)
gx, gy = gx.ravel(), gy.ravel()

ax1.scatter(gx, gy, s=60, color=grid_color, zorder=5,
            edgecolors='white', linewidths=0.5)
ax1.set_title('Grid Search', fontsize=16, fontweight='bold', pad=12)
ax1.set_xlabel('Important parameter', fontsize=13)
ax1.set_ylabel('Unimportant parameter', fontsize=13)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks([])
ax1.set_yticks([])

# Projection tick marks onto x-axis
for x in grid_vals:
    ax1.plot([x, x], [0, 0.04], color=proj_color_grid,
             linewidth=2.5, solid_capstyle='round')

# Double-arrow + label underneath
ax1.annotate('', xy=(0.1, -0.08), xytext=(0.9, -0.08),
             arrowprops=dict(arrowstyle='<->', color=proj_color_grid, lw=1.5),
             annotation_clip=False)
ax1.text(0.5, -0.13, '5 unique values explored',
         ha='center', va='top', fontsize=11,
         color=proj_color_grid, fontweight='bold')

# --- Random Search (right panel) ---
rx = np.random.uniform(0.05, 0.95, 25)
ry = np.random.uniform(0.05, 0.95, 25)

ax2.scatter(rx, ry, s=60, color=random_color, zorder=5,
            edgecolors='white', linewidths=0.5)
ax2.set_title('Random Search', fontsize=16, fontweight='bold', pad=12)
ax2.set_xlabel('Important parameter', fontsize=13)
ax2.set_ylabel('Unimportant parameter', fontsize=13)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticks([])
ax2.set_yticks([])

# Projection tick marks onto x-axis
for x in sorted(rx):
    ax2.plot([x, x], [0, 0.04], color=proj_color_rand,
             linewidth=2.5, solid_capstyle='round')

ax2.annotate('', xy=(min(rx), -0.08), xytext=(max(rx), -0.08),
             arrowprops=dict(arrowstyle='<->', color=proj_color_rand, lw=1.5),
             annotation_clip=False)
ax2.text(0.5, -0.13, '25 unique values explored',
         ha='center', va='top', fontsize=11,
         color=proj_color_rand, fontweight='bold')

# --- Style both axes ---
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(length=0)

# Footnote
fig.text(0.99, 0.01, 'Based on Bergstra & Bengio, 2012',
         ha='right', va='bottom', fontsize=9,
         fontstyle='italic', color='#777777')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    '/Users/nipun/git/stt-ai-teaching/slides/images/week08/grid_vs_random_search.png',
    bbox_inches='tight', facecolor='white', dpi=150
)
print('Saved successfully.')
