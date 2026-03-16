"""Generate gold mining 1D and 2D diagrams for Week 8 slides."""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# === 1D Gold Mining ===
x = np.linspace(0, 10, 300)
gold = 2 * np.exp(-0.5 * ((x - 3) / 0.8)**2) + 1.5 * np.exp(-0.5 * ((x - 7) / 1.2)**2) + 0.3 * np.sin(x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.fill_between(x, 0, gold, alpha=0.3, color='#f39c12')
ax.plot(x, gold, color='#f39c12', linewidth=2)
ax.set_title("Hidden Gold Deposits", fontsize=13, fontweight='bold')
ax.set_xlabel("Location along the field", fontsize=11)
ax.set_ylabel("Gold amount", fontsize=11)
ax.text(3, 2.3, "Rich\nvein!", fontsize=10, ha='center', color='#d35400', fontweight='bold')
ax.set_ylim(-0.2, 3)

ax = axes[1]
ax.fill_between(x, 0, gold, alpha=0.1, color='gray')
drill_x = [1.5, 5.0, 8.5]
drill_y = [np.interp(dx, x, gold) for dx in drill_x]
ax.vlines(drill_x, 0, drill_y, colors='#2c3e50', linewidth=2, linestyles='--')
ax.scatter(drill_x, drill_y, s=100, color='#e74c3c', zorder=5, marker='v')
for dx, dy in zip(drill_x, drill_y):
    ax.annotate(f'{dy:.1f}', (dx, dy + 0.15), ha='center', fontsize=10, fontweight='bold')
ax.set_title("3 Drills Done", fontsize=13, fontweight='bold')
ax.set_xlabel("Location along the field", fontsize=11)
ax.set_ylabel("Gold found", fontsize=11)
ax.text(5, 2.5, "Where next?", fontsize=14, ha='center', color='#e74c3c', fontweight='bold')
ax.set_ylim(-0.2, 3)

ax = axes[2]
ax.fill_between(x, 0, gold, alpha=0.1, color='gray')
ax.vlines(drill_x, 0, drill_y, colors='#2c3e50', linewidth=2, linestyles='--', alpha=0.5)
ax.scatter(drill_x, drill_y, s=80, color='#95a5a6', zorder=5, marker='v')
smart_x = 3.0
smart_y = np.interp(smart_x, x, gold)
ax.vlines([smart_x], 0, smart_y, colors='#2ecc71', linewidth=3)
ax.scatter([smart_x], [smart_y], s=150, color='#2ecc71', zorder=5, marker='v')
ax.annotate(f'Jackpot!\n{smart_y:.1f}', (smart_x, smart_y + 0.15), ha='center', fontsize=10, fontweight='bold', color='#27ae60')
ax.set_title("Smart: Drill Near the Best", fontsize=13, fontweight='bold')
ax.set_xlabel("Location along the field", fontsize=11)
ax.set_ylabel("Gold found", fontsize=11)
ax.set_ylim(-0.2, 3)

plt.tight_layout()
plt.savefig('gold_mining_1d.png', dpi=150, bbox_inches='tight', facecolor='white')

# === 2D Gold Mining ===
x2 = np.linspace(0, 10, 100)
y2 = np.linspace(0, 10, 100)
X2, Y2 = np.meshgrid(x2, y2)
Z = (2 * np.exp(-0.5 * (((X2-3)/1.2)**2 + ((Y2-7)/1.0)**2)) +
     1.2 * np.exp(-0.5 * (((X2-7)/1.5)**2 + ((Y2-3)/1.5)**2)) +
     0.3 * np.sin(X2) * np.cos(Y2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.contourf(X2, Y2, Z, levels=20, cmap='YlOrRd', alpha=0.7)
gx = np.linspace(1, 9, 5); gy = np.linspace(1, 9, 5)
GX, GY = np.meshgrid(gx, gy)
ax.scatter(GX.ravel(), GY.ravel(), s=40, c='#2c3e50', marker='x', linewidths=2, zorder=5)
ax.set_title("Grid Search\n25 drills, regular spacing", fontsize=12, fontweight='bold')
ax.set_xlabel("Hyperparameter 1", fontsize=11)
ax.set_ylabel("Hyperparameter 2", fontsize=11)

ax = axes[1]
ax.contourf(X2, Y2, Z, levels=20, cmap='YlOrRd', alpha=0.7)
np.random.seed(42)
rx = np.random.uniform(0.5, 9.5, 25); ry = np.random.uniform(0.5, 9.5, 25)
ax.scatter(rx, ry, s=40, c='#2c3e50', marker='x', linewidths=2, zorder=5)
ax.set_title("Random Search\n25 drills, random locations", fontsize=12, fontweight='bold')
ax.set_xlabel("Hyperparameter 1", fontsize=11)
ax.set_ylabel("Hyperparameter 2", fontsize=11)

ax = axes[2]
ax.contourf(X2, Y2, Z, levels=20, cmap='YlOrRd', alpha=0.7)
np.random.seed(7)
bx_init = np.random.uniform(1, 9, 5); by_init = np.random.uniform(1, 9, 5)
ax.scatter(bx_init, by_init, s=40, c='#95a5a6', marker='x', linewidths=2, zorder=5, label='Initial (random)')
bx_focus = np.array([2.5,3.5,3.0,2.8,3.2,3.1,2.9,3.3,7.2,6.8,7.5,6.5,7.0,7.3,6.9,7.1,6.7,7.4,7.2,3.0])
by_focus = np.array([6.5,7.5,7.0,7.2,6.8,7.1,6.9,7.3,3.2,2.8,3.5,3.0,3.1,2.9,3.3,2.7,3.4,3.0,2.5,7.5])
ax.scatter(bx_focus, by_focus, s=40, c='#2ecc71', marker='x', linewidths=2, zorder=5, label='Focused (smart)')
ax.scatter([3.0], [7.0], s=200, c='#e74c3c', marker='*', zorder=6, edgecolors='white', linewidths=1)
ax.set_title("Bayesian Optimization\n25 drills, learns where to look", fontsize=12, fontweight='bold')
ax.set_xlabel("Hyperparameter 1", fontsize=11)
ax.set_ylabel("Hyperparameter 2", fontsize=11)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('gold_mining_2d.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Done: gold_mining_1d.png, gold_mining_2d.png")
