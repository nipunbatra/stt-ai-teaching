#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Experiment Tracking & Reproducibility — Follow-Along Guide             ║
# ║  Week 10 · CS 203 · Software Tools and Techniques for AI               ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   Your code is versioned (Git), your environment is pinned (venv/Docker).
#   But you're tuning hyperparameters and tracking results in a spreadsheet.
#   Today you'll learn proper experiment tracking: grid/random/Bayesian
#   search, PyTorch reproducibility, and W&B for logging everything.
#
# HOW TO USE:
#   1. Open this file in your editor (VS Code, etc.)
#   2. Open a terminal side-by-side
#   3. Copy-paste each command into your terminal, one at a time
#   4. Compare your output with the expected output shown here
#   5. DO NOT run this file as a script — read it and type along
#
# LEGEND:
#   Lines without # prefix     →  commands to type
#   # >> ...                   →  expected output
#   # ...                      →  explanation / narration
#
# ═══════════════════════════════════════════════════════════════════════════



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 2-3: The Spreadsheet Problem                    ║
# ║     Show the messy spreadsheet. "Which run was best? What were the      ║
# ║     params? Let's fix this."                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: Setup — Our ML Project                                ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘

mkdir -p ~/experiments-demo && cd ~/experiments-demo

python -m venv .venv
source .venv/bin/activate

pip install scikit-learn numpy optuna wandb matplotlib

# Create our base training script:

cat > train.py << 'PYEOF'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_data(seed=42):
    """Generate synthetic movie dataset."""
    np.random.seed(seed)
    X = np.random.rand(500, 8)
    y = (X[:, 0] * 2 + X[:, 2] - X[:, 5] + np.random.randn(500) * 0.3 > 1).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def train_and_evaluate(n_estimators=100, max_depth=None, min_samples_split=2, seed=42):
    """Train a model and return accuracy."""
    set_seed(seed)
    X_train, X_test, y_train, y_test = load_data(seed)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"Accuracy: {acc:.3f}")
PYEOF

python train.py

# >> Accuracy: 0.xxx



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 5-7: Bias-Variance, Grid vs Random Search       ║
# ║     Walk through theory. Then back to terminal for hands-on.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: Grid Search — Try Every Combination                  ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > grid_search.py << 'PYEOF'
"""Grid search: exhaustive but expensive."""
from train import train_and_evaluate
from itertools import product

# Define grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
}

# Try every combination
keys = param_grid.keys()
values = param_grid.values()
combinations = list(product(*values))
print(f"Grid search: {len(combinations)} combinations\n")

best_acc, best_params = 0, {}
for combo in combinations:
    params = dict(zip(keys, combo))
    acc = train_and_evaluate(**params)
    print(f"  {params} → {acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_params = params

print(f"\nBest: {best_params} → {best_acc:.3f}")
PYEOF

python grid_search.py

# >> Grid search: 27 combinations
# >> ...
# >> Best: {...} → 0.xxx



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: Random Search — Explore More Efficiently              ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > random_search.py << 'PYEOF'
"""Random search: same budget, better coverage."""
import numpy as np
from train import train_and_evaluate

np.random.seed(42)
n_trials = 27  # same budget as grid search

best_acc, best_params = 0, {}
for i in range(n_trials):
    params = {
        "n_estimators": int(np.random.randint(10, 500)),
        "max_depth": int(np.random.choice([3, 5, 10, 20, 50])),
        "min_samples_split": int(np.random.randint(2, 20)),
    }
    acc = train_and_evaluate(**params)
    print(f"  Trial {i+1:2d}: {params} → {acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_params = params

print(f"\nBest: {best_params} → {best_acc:.3f}")
PYEOF

python random_search.py

# >> Trial  1: {...} → 0.xxx
# >> ...
# >> Best: {...} → 0.xxx
#
# Compare: random search explored more unique values per parameter!



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 8: Bayesian Optimization (Optuna)                ║
# ║     Show how it uses previous results to pick next trial.               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: Bayesian Search with Optuna                           ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > optuna_search.py << 'PYEOF'
"""Bayesian optimization: learns from past trials."""
import optuna
from train import train_and_evaluate

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    return train_and_evaluate(**params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=27)  # same budget

print(f"Best accuracy: {study.best_value:.3f}")
print(f"Best params:   {study.best_params}")

# Show how Optuna focused its search
print(f"\nTrials near best region:")
for trial in sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]:
    print(f"  {trial.params} → {trial.value:.3f}")
PYEOF

python optuna_search.py

# >> Best accuracy: 0.xxx
# >> Best params:   {...}
# Optuna typically finds a better result with the same budget!



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 10-13: PyTorch Reproducibility                  ║
# ║     Show the seed + deterministic settings. Then demo.                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: PyTorch Reproducibility                               ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > torch_repro.py << 'PYEOF'
"""Demonstrate PyTorch reproducibility challenges."""
import torch
import torch.nn as nn
import numpy as np
import random
import os

def set_seed(seed=42):
    """Complete seed function for PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

# ── Without seeds: different every time ──
print("=== Without seeds ===")
for i in range(3):
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = model(x)
    print(f"  Run {i+1}: first output = {y[0].item():.6f}")

# ── With seeds: identical every time ──
print("\n=== With seeds ===")
for i in range(3):
    set_seed(42)
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = model(x)
    print(f"  Run {i+1}: first output = {y[0].item():.6f}")

# ── DataLoader reproducibility ──
print("\n=== DataLoader with worker seeding ===")
from torch.utils.data import DataLoader, TensorDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

for i in range(2):
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        worker_init_fn=seed_worker, generator=g)
    first_batch = next(iter(loader))
    print(f"  Run {i+1}: first batch label sum = {first_batch[1].sum().item()}")
PYEOF

pip install torch --quiet 2>/dev/null
python torch_repro.py

# >> === Without seeds ===
# >>   Run 1: first output = 0.234567   (different each run)
# >>   Run 2: first output = -0.891234
# >>   Run 3: first output = 0.567890
# >>
# >> === With seeds ===
# >>   Run 1: first output = 0.123456   (SAME each run)
# >>   Run 2: first output = 0.123456
# >>   Run 3: first output = 0.123456



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 15-18: W&B Introduction                         ║
# ║     Show the W&B dashboard screenshot. Then integrate live.             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: W&B — Track Everything                               ~15 min   │
# └──────────────────────────────────────────────────────────────────────────┘

# First, login to W&B (you'll need a free account at wandb.ai):

wandb login

# >> Enter API key from https://wandb.ai/authorize
# >> Successfully logged in!

cat > train_wandb.py << 'PYEOF'
"""Training with W&B experiment tracking."""
import wandb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# Initialize W&B run
wandb.init(project="movie-predictor", config={
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "seed": 42,
})
config = wandb.config

# Setup
set_seed(config.seed)
np.random.seed(config.seed)
X = np.random.rand(500, 8)
y = (X[:, 0] * 2 + X[:, 2] - X[:, 5] + np.random.randn(500) * 0.3 > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.seed)

# Train
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    min_samples_split=config.min_samples_split,
    random_state=config.seed,
)
model.fit(X_train, y_train)

# Log metrics
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
wandb.log({
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "n_features": X.shape[1],
    "n_samples": X.shape[0],
})

# Log feature importances
for i, imp in enumerate(model.feature_importances_):
    wandb.log({f"feature_{i}_importance": imp})

print(f"Train: {train_acc:.3f}, Test: {test_acc:.3f}")
print(f"View run at: {wandb.run.url}")

wandb.finish()
PYEOF

python train_wandb.py

# >> Train: 0.xxx, Test: 0.xxx
# >> View run at: https://wandb.ai/...
#
# Open the URL — see all params, metrics, system info logged automatically!

# Run a few more with different params:

python -c "
import wandb, numpy as np, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

for n_est in [50, 200, 500]:
    wandb.init(project='movie-predictor', config={'n_estimators': n_est, 'max_depth': 10, 'seed': 42}, reinit=True)
    random.seed(42); np.random.seed(42)
    X = np.random.rand(500, 8)
    y = (X[:, 0] * 2 + X[:, 2] - X[:, 5] + np.random.randn(500) * 0.3 > 1).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=n_est, max_depth=10, random_state=42)
    m.fit(X_tr, y_tr)
    wandb.log({'test_accuracy': m.score(X_te, y_te), 'train_accuracy': m.score(X_tr, y_tr)})
    print(f'n_estimators={n_est}: test_acc={m.score(X_te, y_te):.3f}')
    wandb.finish()
"

# >> n_estimators=50:  test_acc=0.xxx
# >> n_estimators=200: test_acc=0.xxx
# >> n_estimators=500: test_acc=0.xxx
#
# Now check W&B dashboard — compare all runs side-by-side!



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 7: W&B Sweeps — Automated Hyperparameter Search          ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > sweep.yaml << 'EOF'
program: train_wandb.py
method: bayes
metric:
  name: test_accuracy
  goal: maximize
parameters:
  n_estimators:
    min: 10
    max: 500
  max_depth:
    values: [5, 10, 20, 50]
  min_samples_split:
    min: 2
    max: 20
  seed:
    value: 42
EOF

cat sweep.yaml

# Create the sweep:

wandb sweep sweep.yaml

# >> wandb: Created sweep with ID: xxxxxxxx
# >> wandb: View sweep at: https://wandb.ai/...
# >> wandb: Run sweep agent with: wandb agent your-entity/movie-predictor/xxxxxxxx

# Run the agent (it will run multiple trials):
# wandb agent <your-entity>/movie-predictor/<sweep-id> --count 10

# Check the W&B sweep dashboard — parallel coordinates plot,
# importance analysis, best runs.



# ═══════════════════════════════════════════════════════════════════════════
# WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════
#
# What we covered today:
#
#   Act 1: Setup — base ML project
#   Act 2: Grid search — exhaustive, expensive
#   Act 3: Random search — better coverage, same budget
#   Act 4: Bayesian search (Optuna) — learns from past trials
#   Act 5: PyTorch reproducibility — seeds + deterministic mode
#   Act 6: W&B — track params, metrics, artifacts
#   Act 7: W&B Sweeps — automated hyperparameter search
#
# The progression:
#   Spreadsheet → manual scripts → Optuna → W&B sweeps
#
# Next week: CI/CD — automate testing, linting, deployment
# ═══════════════════════════════════════════════════════════════════════════

cd ~
deactivate 2>/dev/null
# rm -rf ~/experiments-demo   # uncomment to clean up
