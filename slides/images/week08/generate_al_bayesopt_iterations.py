"""Generate ALL iterations (0-9) for Active Learning and BayesOpt,
mimicking the style from the Distill BayesOpt article."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- GP implementation (simple RBF kernel) ---
def rbf_kernel(X1, X2, length_scale=0.3, sigma_f=1.0):
    dist_sq = np.subtract.outer(X1, X2) ** 2
    return sigma_f**2 * np.exp(-0.5 * dist_sq / length_scale**2)

def gp_predict(X_train, y_train, X_test, length_scale=0.3, sigma_f=1.0, sigma_n=0.05):
    K = rbf_kernel(X_train, X_train, length_scale, sigma_f) + sigma_n**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale, sigma_f)
    K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f)
    K_inv = np.linalg.solve(K, np.eye(len(K)))
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    return mu, std

# --- True function (gold concentration) ---
def true_function(x):
    return np.sin(3 * x) * x + 0.5 * np.cos(5 * x) + 0.3 * x

# --- Acquisition functions ---
def expected_improvement(mu, std, best_y, xi=0.01):
    z = (mu - best_y - xi) / (std + 1e-10)
    ei = (mu - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-10] = 0.0
    return ei

def max_uncertainty(std):
    return std

# --- Plot one iteration ---
def plot_iteration(X_test, mu, std, X_obs, y_obs, acq_values, acq_name,
                   next_x, true_y, iteration, strategy, outpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.08})

    # Top: GP posterior + observations
    ax1.plot(X_test, true_y, 'k--', alpha=0.3, linewidth=1, label='True function')
    ax1.plot(X_test, mu, '#2196F3', linewidth=2, label='GP mean')
    ax1.fill_between(X_test, mu - 2*std, mu + 2*std, alpha=0.2, color='#2196F3', label='95% confidence')
    ax1.scatter(X_obs, y_obs, c='#E91E63', s=80, zorder=5, edgecolors='white', linewidths=1.5, label='Observations')
    if next_x is not None:
        ax1.axvline(next_x, color='#FF9800', linewidth=2, linestyle='--', alpha=0.7, label='Next sample')
    ax1.set_ylabel('f(x)', fontsize=13)
    ax1.set_title(f'{strategy} — Iteration {iteration}', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left', ncol=2)
    ax1.set_xlim(X_test[0], X_test[-1])

    # Bottom: acquisition function
    color = '#4CAF50' if strategy == 'Bayesian Optimization' else '#9C27B0'
    ax2.fill_between(X_test, 0, acq_values, alpha=0.3, color=color)
    ax2.plot(X_test, acq_values, color=color, linewidth=1.5)
    if next_x is not None:
        ax2.axvline(next_x, color='#FF9800', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel(acq_name, fontsize=11)
    ax2.set_xlim(X_test[0], X_test[-1])

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# --- Run simulations ---
np.random.seed(42)
X_test = np.linspace(0, 2, 300)
true_y = true_function(X_test)

# Initial observations (2 random points)
X_init = np.array([0.3, 1.5])
y_init = true_function(X_init) + np.random.normal(0, 0.05, len(X_init))

for strategy in ['active_learning', 'bayesopt']:
    X_obs = X_init.copy()
    y_obs = y_init.copy()

    for iteration in range(10):
        mu, std = gp_predict(X_obs, y_obs, X_test)

        if strategy == 'active_learning':
            acq = max_uncertainty(std)
            acq_name = 'Uncertainty'
        else:
            best_y = np.max(y_obs)
            acq = expected_improvement(mu, std, best_y)
            acq_name = 'Expected Improvement'

        # Pick next point
        next_idx = np.argmax(acq)
        next_x = X_test[next_idx]

        prefix = 'al' if strategy == 'active_learning' else 'ei'
        label = 'Active Learning' if strategy == 'active_learning' else 'Bayesian Optimization'
        outpath = f'/Users/nipun/git/stt-ai-teaching/slides/images/week08/distill_{prefix}_{iteration}.png'

        plot_iteration(X_test, mu, std, X_obs, y_obs, acq, acq_name,
                      next_x if iteration < 9 else None, true_y, iteration, label, outpath)

        # Evaluate and add
        next_y = true_function(next_x) + np.random.normal(0, 0.05)
        X_obs = np.append(X_obs, next_x)
        y_obs = np.append(y_obs, next_y)

    # Final state (iteration 9 shows the final model, no next point)

print("Generated all AL and BayesOpt iteration images (0-9 each)")
