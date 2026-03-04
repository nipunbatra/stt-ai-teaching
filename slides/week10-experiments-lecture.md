---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->

# Experiment Tracking & Reproducibility

## Week 10 · CS 203: Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are

```
Week 8:  Version your CODE       (Git)
Week 9:  Version your ENVIRONMENT (venv, Docker)
Week 10: Version your EXPERIMENTS  ← you are here
Week 11: Automate everything      (CI/CD)
```

Your code is versioned. Your environment is pinned.
But which hyperparameters gave the best accuracy?

**"I think it was learning_rate=0.01... or was it 0.001?"**

---

# The Spreadsheet Problem

You're tuning your Netflix predictor:

| Run | lr | n_estimators | accuracy | notes |
|-----|-----|-------------|----------|-------|
| 1 | 0.01 | 100 | 0.82 | first try |
| 2 | 0.001 | 100 | 0.79 | worse |
| 3 | 0.01 | 200 | 0.85 | better! |
| 4 | ??? | ??? | 0.87 | forgot to write down |

**Problems:**
- Forgot to log run 4's parameters
- Can't reproduce run 3 (which code version?)
- Spreadsheet doesn't link to code, data, or model files

---

# What We Really Need

A system that **automatically** records:

| Dimension | What to Track |
|-----------|--------------|
| **Code** | Git commit hash, diff |
| **Data** | Dataset version, preprocessing steps |
| **Config** | All hyperparameters |
| **Metrics** | Train/val/test scores over time |
| **Artifacts** | Model files, plots, predictions |
| **Environment** | Python version, package versions, hardware |

**Manual tracking fails.** Automated tracking scales.

---

<!-- _class: lead -->

# Part 1: Theory — Why Tuning Matters

---

# The Bias-Variance Tradeoff

```
Error = Bias² + Variance + Irreducible Noise
```

| | High Bias | High Variance |
|--|-----------|---------------|
| **Model** | Too simple | Too complex |
| **Training error** | High | Low |
| **Test error** | High | High |
| **Fix** | More features, complex model | More data, regularization |

**Hyperparameter tuning** = finding the sweet spot between underfitting and overfitting.

---

# Bias-Variance: Visually

```
    Error
      │
      │ \                         ╱
      │  \    Test Error         ╱
      │   \                    ╱
      │    \     ___         ╱
      │     \  ╱    ╲      ╱
      │      ╲╱      ╲   ╱
      │                ╲╱  ← Sweet spot
      │     ──────────────  Training Error
      │
      └──────────────────────────────
        Simple ←  Complexity  → Complex
        (High Bias)        (High Variance)
```

**Tuning hyperparameters** moves you along this curve.

Too few trees? Underfitting. Too many with no regularization? Overfitting.

---

# What Are Hyperparameters?

**Parameters** — learned from data (weights, biases):
```python
model.fit(X, y)  # parameters are learned here
```

**Hyperparameters** — set by YOU before training:
```python
RandomForestClassifier(
    n_estimators=100,      # how many trees?
    max_depth=10,          # how deep?
    min_samples_split=5    # when to stop splitting?
)
```

**The model can't learn its own hyperparameters.** You must search for good ones.

---

# Examples of Hyperparameters

| Model | Key Hyperparameters |
|-------|-------------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split |
| **SVM** | C (regularization), kernel, gamma |
| **Neural Network** | learning_rate, batch_size, n_layers, hidden_size |
| **k-NN** | k (n_neighbors), distance metric |
| **Gradient Boosting** | learning_rate, n_estimators, max_depth, subsample |

**Some matter more than others.** Learning rate is almost always critical. Number of layers often matters less than you'd think.

---

# The Tuning Process

```
1. Define search space
   lr: [0.0001 ... 0.1]
   n_estimators: [50 ... 500]

2. Pick a search strategy
   Grid? Random? Bayesian?

3. Evaluate each configuration
   Use cross-validation (NOT test set!)

4. Select best configuration
   Retrain on full training set

5. Final evaluation on TEST SET
   Report this number (only once!)
```

**Common mistake:** Tuning on the test set → overly optimistic estimate.

---

<!-- _class: lead -->

# Part 1b: Search Strategies

---

# Grid Search

**Try every combination.**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,           # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,      # use all CPU cores
)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

**3 x 3 x 3 = 27 combinations x 5 folds = 135 fits!**

---

# Grid Search: The Problem

```
             n_estimators
          50    100    200
         ┌──────────────────┐
    5    │ 0.78  0.80  0.81 │
max  10  │ 0.82  0.84  0.85 │
depth    │ 0.83  0.85  0.86 │
   None  │ 0.83  0.85  0.86 │
         └──────────────────┘
```

**Problem:** If `max_depth` doesn't matter much (10 vs None are similar), we wasted runs exploring both. Those runs could have tried more `n_estimators` values.

**Cost grows exponentially:** 3 params x 10 values each = 1,000 combinations!

---

# Random Search

**Sample randomly from distributions.**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(3, 50),
    "min_samples_split": randint(2, 20),
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=27,       # same budget as grid
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)
```

---

# Grid vs Random: Visually

```
Grid Search:              Random Search:
┌─────────────────┐      ┌─────────────────┐
│ •   •   •       │      │    •     •       │
│                 │      │  •           •   │
│ •   •   •       │      │        •        │
│                 │      │ •          •     │
│ •   •   •       │      │      •       •  │
└─────────────────┘      └─────────────────┘
  Only 3 unique lr         9 unique lr values
  values explored!         explored!
```

**If `lr` matters more than `n_estimators`,** random search finds good `lr` values faster.

**With the same budget, random search explores more of the important dimensions.**

*(Bergstra & Bengio, 2012)*

<!-- ⌨ NOTEBOOK → Grid vs random search comparison -->

---

# When Grid Search IS Better

Grid search wins when:
- **Few hyperparameters** (1-2): exhaustive search is cheap
- **Discrete choices**: kernel=["linear", "rbf", "poly"]
- **Known good ranges**: you've narrowed down the space already

```python
# Good use of grid search: just 2 params, small space
param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10, 100],
}
# Only 8 combinations — grid is fine!
```

**Rule of thumb:** > 3 hyperparameters or continuous ranges → use random or Bayesian.

---

# Bayesian Optimization (Optuna)

**Idea:** Use results from previous runs to decide what to try next.

```
Run 1: lr=0.05  → accuracy 0.80
Run 2: lr=0.01  → accuracy 0.85   ← better!
Run 3: lr=0.008 → accuracy 0.86   ← promising region!
Run 4: lr=0.007 → accuracy 0.84   ← Optuna narrows in
Run 5: lr=0.009 → accuracy 0.87   ← best so far!
```

**Smarter than random** — builds a surrogate model of "what works" and balances:
- **Exploitation:** Try near the best so far
- **Exploration:** Try unexplored regions

---

# Optuna in Practice

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.1, log=True),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best: {study.best_value:.3f}")
print(f"Params: {study.best_params}")
```

<!-- ⌨ NOTEBOOK → Optuna demo with visualization -->

---

# Optuna: Built-in Visualizations

```python
# Which params matter most?
optuna.visualization.plot_param_importances(study)

# How did the search progress?
optuna.visualization.plot_optimization_history(study)

# Parameter relationships
optuna.visualization.plot_contour(study, params=["lr", "n_estimators"])

# Parallel coordinates
optuna.visualization.plot_parallel_coordinate(study)
```

Optuna can also **prune** bad trials early (stop training if it's going badly).

---

# Search Strategy Summary

| Strategy | Pros | Cons | Use When |
|----------|------|------|----------|
| **Grid** | Exhaustive, reproducible | Exponential cost | Few params, small space |
| **Random** | Good coverage, simple | No learning | Moderate space |
| **Bayesian** | Efficient, learns | More complex | Large space, expensive training |

**Practical advice:**
1. Start with random search to understand the landscape
2. Narrow the range based on results
3. Use Bayesian (Optuna) for fine-tuning in the narrow range

---

# Cross-Validation for Tuning

**Problem:** Tuning on a single train/test split is noisy.

**K-Fold Cross-Validation:**

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

Average score across folds → more reliable estimate.

**Warning:** Don't tune hyperparameters on the test set! Use a separate validation set or cross-validation on the training set only.

---

# CV Variants

| Variant | Use Case |
|---------|----------|
| **K-Fold** | Default choice (k=5 or 10) |
| **Stratified K-Fold** | Imbalanced classes (preserves class ratio) |
| **Leave-One-Out** | Very small datasets (k = n) |
| **Time Series Split** | Temporal data (train on past, test on future) |
| **Group K-Fold** | Grouped data (all samples from one patient in same fold) |

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

# The Data Split Hierarchy

```
Full Dataset
├── Training Set (80%)
│   ├── CV Train Fold (used for fitting)
│   └── CV Val Fold (used for hyperparameter selection)
│
└── Test Set (20%) — TOUCH ONLY ONCE AT THE END
```

**Three-level separation:**
1. **CV folds** — compare hyperparameter configurations
2. **Validation** — select the best configuration
3. **Test** — final unbiased performance estimate

**If you tune on the test set, you're cheating.** Your reported accuracy will be optimistic.

---

<!-- _class: lead -->

# Part 2: PyTorch Reproducibility

---

# PyTorch: Seeds Aren't Enough

In sklearn, `random_state=42` is sufficient. PyTorch is harder:

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**But this still isn't fully deterministic!** Some CUDA operations use non-deterministic algorithms for speed.

---

# Why Is PyTorch Non-Deterministic by Default?

Some CUDA algorithms have multiple implementations:

```
Operation: torch.nn.Conv2d forward pass

Algorithm A: Deterministic
  - Always same result
  - 10ms per batch

Algorithm B: Non-deterministic (uses atomicAdd)
  - Tiny floating-point order differences
  - 6ms per batch ← cuDNN picks this by default
```

**cuDNN benchmarks** different algorithms and picks the fastest one — which may vary between runs.

**Speed vs reproducibility** is a real engineering tradeoff.

---

# Full PyTorch Determinism

```python
import torch
import os

# 1. Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 2. Force deterministic algorithms
torch.use_deterministic_algorithms(True)

# 3. Disable cuDNN benchmark (trades speed for reproducibility)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 4. Set CUBLAS workspace config (for some CUDA ops)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

**Tradeoff:** Full determinism can be 10-20% slower on GPU.

---

# DataLoader Reproducibility

```python
# Workers use separate random states!
# Fix with a worker seed function:

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g
)
```

**Without `worker_init_fn`:** Each worker gets different random state across runs.

<!-- ⌨ NOTEBOOK → PyTorch reproducibility demo -->

---

# PyTorch Reproducibility Checklist

| Layer | What to Set |
|-------|------------|
| Python | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` |
| cuDNN | `torch.backends.cudnn.deterministic = True` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| Deterministic mode | `torch.use_deterministic_algorithms(True)` |
| CUBLAS | `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` |
| DataLoader | `worker_init_fn` + `generator` |

**Miss any one of these → non-reproducible results.**

---

# When NOT to Be Fully Deterministic

Full determinism is **not always necessary:**

- **Exploration phase:** You're trying many ideas → speed matters more
- **Large-scale training:** 10-20% slowdown over days is expensive
- **Reporting results:** Run 3-5 seeds, report mean ± std

```python
# Instead of one deterministic run, report variance:
results = []
for seed in [42, 123, 456, 789, 1024]:
    set_seed(seed)
    acc = train_and_evaluate()
    results.append(acc)

print(f"Accuracy: {np.mean(results):.3f} ± {np.std(results):.3f}")
```

**This is more informative than a single deterministic result.**

---

<!-- _class: lead -->

# Part 3: Experiment Tracking with W&B

*No more spreadsheets*

---

# What W&B Tracks — Automatically

| What | How | Manual Effort |
|------|-----|--------------|
| **Hyperparameters** | `wandb.config` | You set config dict |
| **Metrics** | `wandb.log()` | You call log() |
| **Code version** | Auto-captures git hash | Zero |
| **Environment** | requirements.txt snapshot | Zero |
| **System metrics** | CPU, GPU, memory usage | Zero |
| **Model files** | `wandb.save()` | You call save() |
| **Plots** | `wandb.log({"chart": fig})` | You log figures |

**One dashboard** — compare all runs, filter, sort, reproduce.

---

# W&B: Setup

```bash
# Install
pip install wandb

# Login (one-time)
wandb login
# → paste API key from https://wandb.ai/authorize
```

Free tier includes:
- Unlimited experiments
- 100 GB storage
- Team dashboards
- Sweep orchestration

---

# W&B Basic Usage

```python
import wandb

# Start a run
wandb.init(project="netflix-predictor", config={
    "learning_rate": 0.01,
    "n_estimators": 100,
    "seed": 42,
})

# Train your model (use wandb.config for params)
model = train(wandb.config)

# Log metrics
wandb.log({"accuracy": accuracy, "f1": f1_score})

# Log at each epoch (for training curves)
for epoch in range(100):
    loss = train_one_epoch(model)
    wandb.log({"epoch": epoch, "loss": loss})

# Save artifacts
wandb.save("model.pkl")
wandb.finish()
```

<!-- ⌨ NOTEBOOK → W&B integration demo -->

---

# W&B: Logging Rich Data

```python
# Log images
wandb.log({"confusion_matrix": wandb.Image(fig)})

# Log tables
table = wandb.Table(columns=["input", "prediction", "actual"])
for x, pred, actual in results:
    table.add_data(x, pred, actual)
wandb.log({"predictions": table})

# Log matplotlib/plotly figures
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(train_losses)
wandb.log({"loss_curve": wandb.Image(fig)})

# Log a summary metric (final value)
wandb.run.summary["best_accuracy"] = best_acc
```

---

# W&B Dashboard: What You See

**Runs Table:**
- Every experiment as a row
- Sort/filter by any metric or config value
- Group by hyperparameter

**Charts (auto-generated):**
- Training curves (loss, accuracy over time)
- Hyperparameter vs metric scatter plots
- System resource usage (GPU util, memory)

**Compare runs:**
- Side-by-side metric curves
- Diff configs between runs
- Identify which changes improved performance

---

# W&B Sweeps (Hyperparameter Search)

```yaml
# sweep.yaml
program: train_wandb.py
method: bayes          # or grid, random
metric:
  name: test_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform_values
  n_estimators:
    values: [50, 100, 200, 500]
  max_depth:
    min: 3
    max: 50
```

```bash
wandb sweep sweep.yaml          # creates sweep
wandb agent <sweep-id>          # runs trials
```

W&B handles the search strategy, logging, and visualization.

---

# W&B Sweeps: Parallel Coordinates

```
  lr        n_est    max_depth   accuracy
  │          │          │          │
0.1 ──┐     500 ──┐    50 ──┐    0.90 ──  best run (thick line)
      │          │         │
0.01 ─┤     200 ─┤    20 ─┤    0.85 ──
      │          │         │
0.001─┘     50 ──┘     5 ──┘    0.75 ──  worst run
```

The sweep dashboard shows:
- **Parallel coordinates**: trace each run's params → metric
- **Parameter importance**: which params affect the metric most
- **Contour plots**: 2D landscape of param interactions

---

# MLflow: A Self-Hosted Alternative

```python
import mlflow

mlflow.set_experiment("netflix-predictor")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.01)

    model = train(...)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
```

```bash
mlflow ui    # → http://localhost:5000
```

| | W&B | MLflow |
|--|-----|--------|
| Hosting | Cloud (free tier) | Self-hosted or cloud |
| Setup | `pip install wandb` | `pip install mlflow` |
| UI | Web dashboard | Local UI |
| Best for | Teams, sweeps, rich viz | Self-hosted, enterprise |

---

# Other Tracking Tools

| Tool | Strengths |
|------|-----------|
| **TensorBoard** | Built into TF/PyTorch, free, lightweight |
| **Neptune.ai** | Good for teams, nice UI |
| **Comet ML** | Similar to W&B, different pricing |
| **DVC** | Data versioning + experiment tracking |
| **Sacred** | Lightweight, Python-native |

**Our recommendation:** W&B for class projects (free, easy, great UI).

MLflow if you need self-hosted.

---

# Experiment Tracking Best Practices

1. **Log everything** — storage is cheap, hindsight is expensive
2. **Use meaningful run names** — `lr0.01_depth10` not `run_42`
3. **Tag experiments** — `baseline`, `augmented`, `final`
4. **Save the model file** — not just the metrics
5. **Record the git hash** — know exactly which code produced results
6. **Use config files** — don't hardcode in the training script
7. **Compare against baselines** — always have a reference point

---

# Key Takeaways

1. **Hyperparameter tuning** finds the bias-variance sweet spot
   - Random search > grid search in most cases
   - Bayesian (Optuna) is smarter still
   - Always use cross-validation, never tune on test set

2. **PyTorch reproducibility** needs more than just seeds
   - `torch.use_deterministic_algorithms(True)` + cuDNN settings
   - Seed DataLoader workers separately
   - Consider reporting mean ± std over multiple seeds

3. **Experiment tracking** (W&B/MLflow) replaces spreadsheets
   - Logs params, metrics, code, artifacts automatically
   - Compare runs in a dashboard
   - Sweeps automate hyperparameter search

**Next week:** CI/CD — automate testing and deployment

---

<!-- _class: lead -->

# Questions?

**Exam-relevant concepts:**
- Bias-variance tradeoff and how tuning navigates it
- Parameters vs hyperparameters
- Grid vs random search — why random wins in high dimensions
- K-fold cross-validation and its variants
- Train/validation/test split hierarchy
- PyTorch determinism requirements (all 8 settings)
- When full determinism is vs isn't necessary
