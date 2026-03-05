---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Tuning & Experiment Tracking

## Week 8: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are

```
Week 7:  Evaluate models properly   (CV, bias-variance, AutoML)    ✓
Week 8:  Tune & track experiments   ← you are here
Week 9:  Version your CODE          (Git)
Week 10: Version your ENVIRONMENT   (venv, Docker)
Week 11: Automate everything        (CI/CD)
Week 12: Ship it                    (APIs, demos)
Week 13: Make it fast and small     (profiling, quantization)
```

---

# Recap from Week 7

Last week we learned to **evaluate** models properly:

| Concept | Key Idea |
|---------|----------|
| Cross-validation | Never trust a single split |
| Stratified CV | Preserve class ratios |
| Data leakage | Preprocess *inside* CV with Pipelines |
| Learning curves | Do I need more data? |
| AutoML | Let the computer try everything |

**This week**: How to **tune** hyperparameters systematically, and how to **track** every experiment so you never lose a result.

---

<!-- _class: lead -->

# Part 1: Hyperparameter Tuning

*Finding the best knobs to turn*

---

# Parameters vs Hyperparameters

<img src="images/week07/params_vs_hyperparams.png" width="700" style="display: block; margin: 0 auto;">

**Parameters**: The model figures these out from data (weights, thresholds).
**Hyperparameters**: You decide these *before* training.

---

# Common Hyperparameters

| Model | Hyperparameter | What It Controls | Typical Range |
|-------|----------------|------------------|---------------|
| **Logistic Reg** | `C` | Regularization strength | 0.001 - 1000 |
| **Decision Tree** | `max_depth` | Tree complexity | 1 - 50 |
| **Random Forest** | `n_estimators` | Number of trees | 50 - 500 |
| **Random Forest** | `min_samples_leaf` | Minimum leaf size | 1 - 20 |
| **XGBoost** | `learning_rate` | Step size | 0.01 - 0.3 |
| **XGBoost** | `max_depth` | Tree complexity | 3 - 10 |

**How do you find the best combination?**

---

# Approach 0: "Grad Student Descent"

```python
# Monday
model = RandomForestClassifier(n_estimators=100, max_depth=10)
# Accuracy: 83.2%

# Tuesday
model = RandomForestClassifier(n_estimators=200, max_depth=15)
# Accuracy: 84.1%

# Wednesday
model = RandomForestClassifier(n_estimators=200, max_depth=20)
# Accuracy: 82.8%  ... wait, was Tuesday's result with max_depth=15 or 20?

# Thursday: give up, use the Tuesday one. Probably.
```

**Problems**: No record of what you tried. No systematic coverage. Easy to miss the best combination.

---

# Approach 1: Grid Search

**Try every combination on a predefined grid.**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(), param_grid, cv=5, scoring='accuracy'
)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score:  {grid.best_score_:.3f}")
```

---

# Grid Search: The Explosion Problem

```
Parameters:
  n_estimators:    [50, 100, 200]         → 3 values
  max_depth:       [5, 10, 15, None]      → 4 values
  min_samples_leaf: [1, 2, 5]             → 3 values

Total combinations: 3 × 4 × 3 = 36
Cross-validation:   36 × 5 folds = 180 model fits
```

**Now add two more parameters with 5 values each:**
$$3 \times 4 \times 3 \times 5 \times 5 = 900 \text{ combinations} \times 5 \text{ folds} = 4{,}500 \text{ fits}$$

**Grid search doesn't scale.** Each new parameter multiplies the cost.

---

# Approach 2: Random Search

**Sample random combinations instead of trying all of them.**

<img src="images/week07/grid_vs_random_search.png" width="750" style="display: block; margin: 0 auto;">

---

# Why Random Search Works Better

**Bergstra & Bengio (2012)**: A landmark result.

**Key insight**: Not all hyperparameters matter equally.

- Maybe `max_depth` matters a lot, but `min_samples_leaf` barely affects performance.
- Grid search wastes many evaluations varying the unimportant parameter.
- Random search spreads evaluations more evenly across *all* dimensions.
- With the same budget, random search explores more unique values of the important parameters.

**In practice**: 60 random trials often beats a full grid search.

---

# Random Search in Code

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),          # Sample integers 50-500
    'max_depth': randint(3, 30),               # Sample integers 3-30
    'min_samples_leaf': randint(1, 20),        # Sample integers 1-20
    'max_features': uniform(0.1, 0.9),         # Sample floats 0.1-1.0
}

search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=60,                                 # Only 60 random trials
    cv=5,
    random_state=42
)
search.fit(X, y)
print(f"Best: {search.best_score_:.3f} with {search.best_params_}")
```

---

# Grid vs Random: When to Use Which

| | Grid Search | Random Search |
|---|-------------|---------------|
| **Combinations** | All | Sampled |
| **Budget** | Grows exponentially | You control it (`n_iter`) |
| **Coverage** | Even but sparse per dimension | Better for important params |
| **Best for** | 1-2 hyperparameters | 3+ hyperparameters |
| **Guarantees** | Finds best in grid | May miss the best |

**Rule of thumb**: Use grid for quick searches (few params, few values). Use random for everything else.

---

# Approach 3: Bayesian Optimization

**Idea**: Use results so far to decide what to try next.

Grid and random search are *blind* -- they don't learn from previous trials.
Bayesian optimization builds a model of "hyperparameter → score" and picks the next point intelligently.

```
Trial 1: max_depth=5, lr=0.1   → 82%
Trial 2: max_depth=10, lr=0.01 → 85%
Trial 3: max_depth=8, lr=0.05  → ??? (model predicts ~86%, tries this region)
```

**Explores** uncertain regions + **exploits** promising regions.

---

# Optuna: Bayesian Optimization Made Easy

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best score: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

---

# Optuna: Built-In Visualizations

```python
# Which hyperparameters matter most?
optuna.visualization.plot_param_importances(study)

# How did optimization progress over trials?
optuna.visualization.plot_optimization_history(study)

# How do parameters interact?
optuna.visualization.plot_contour(study)
```

**Optuna also supports**:
- Pruning bad trials early (stop wasting time on hopeless configs)
- Multi-objective optimization (accuracy AND speed)
- Distributed search across machines

---

# How Search Strategies Explore Parameter Space

<img src="images/week08-tuning/search_strategies_comparison.png" width="800" style="display: block; margin: 0 auto;">

---

# Comparison: All Three Approaches

| | Grid | Random | Bayesian (Optuna) |
|---|------|--------|-------------------|
| **Intelligence** | None | None | Learns from trials |
| **Efficiency** | Low | Medium | High |
| **Setup** | Easy | Easy | Moderate |
| **Best for** | Few params | Many params | Expensive models |
| **Parallelizable** | Yes | Yes | Partially |

**Practical advice**:
1. Start with `RandomizedSearchCV` (simple, effective)
2. Switch to Optuna when model training is expensive (minutes per fit)

---

# The Tuning Trap: Evaluating Tuned Models

**A subtle but critical mistake:**

```python
# WRONG: Tune and evaluate on the SAME cross-validation
grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)
print(f"Best score: {grid.best_score_:.3f}")  # Optimistic!
```

**Why this is wrong**: You searched over many configurations and picked the best one. By definition, it's the luckiest. This is *selection bias*.

**The score from `GridSearchCV.best_score_` is always optimistic.**

---

# Nested Cross-Validation

**Solution**: Separate the tuning loop from the evaluation loop.

<img src="images/week07/nested_cross_validation.png" width="750" style="display: block; margin: 0 auto;">

- **Inner loop**: Tunes hyperparameters (finds best config)
- **Outer loop**: Evaluates the *tuned model* on truly held-out data

---

# Nested CV in Code

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner loop: tune hyperparameters
inner_cv = GridSearchCV(
    RandomForestClassifier(),
    param_grid={'max_depth': [5, 10, 15], 'n_estimators': [100, 200]},
    cv=3                  # 3-fold inner CV for tuning
)

# Outer loop: evaluate the tuned model
outer_scores = cross_val_score(inner_cv, X, y, cv=5)  # 5-fold outer CV

print(f"Nested CV score: {outer_scores.mean():.3f} +/- {outer_scores.std():.3f}")
```

**This is the gold standard** for reporting tuned model performance.

---

# Hyperparameter Tuning: Best Practices

<div class="insight">

1. **Always use CV for tuning** -- never tune on a single split
2. **Random search before grid** -- grid only if you have 1-2 params
3. **Set a compute budget** -- diminishing returns after ~100 trials
4. **Use nested CV for final reporting** -- `GridSearchCV.best_score_` is optimistic
5. **Log everything** -- you will want to revisit old experiments

</div>

---

# Common Tuning Mistakes

<div class="warning">

1. **Tuning on test set**: "I'll just try a few values on test..." -- now test is contaminated
2. **Too fine a grid**: `learning_rate: [0.001, 0.0011, 0.0012, ...]` -- waste of compute
3. **Ignoring interactions**: `max_depth` and `n_estimators` interact -- tune them together
4. **Not setting random seeds**: Can't reproduce the best result
5. **Reporting `best_score_` as final performance**: Always use nested CV

</div>

---

<!-- _class: lead -->

# Part 2: PyTorch Reproducibility

*Making deep learning experiments repeatable*

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

---

# PyTorch Reproducibility Checklist

<img src="images/week08-tuning/pytorch_reproducibility_layers.png" width="700" style="display: block; margin: 0 auto;">

---

# PyTorch Reproducibility Checklist (Detail)

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

<img src="images/week08-tuning/experiment_tracking_workflow.png" width="800" style="display: block; margin: 0 auto;">

---

# What an Experiment Tracker Records

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

<!-- _class: lead -->

# Part 4: MLflow & Other Tools

*Self-hosted alternatives*

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

<!-- _class: lead -->

# Key Takeaways & Exam Prep

---

# Key Takeaways

| Concept | One-Liner |
|---------|-----------|
| **Random > Grid** | Better coverage of important dimensions |
| **Bayesian (Optuna)** | Learns from past trials to search smarter |
| **Nested CV** | Tune inside, evaluate outside — gold standard |
| **PyTorch determinism** | Seeds + cuDNN settings + DataLoader seeding |
| **Multi-seed reporting** | More informative than one deterministic run |
| **W&B / MLflow** | Automated tracking replaces spreadsheets |
| **Tuning + tracking** | Complementary — tune systematically, record everything |

---

# Exam Questions

**Q1**: Why does random search often beat grid search?
> Important hyperparameters get more unique values tested (Bergstra & Bengio 2012).

**Q2**: What is nested cross-validation and when do you need it?
> Inner loop tunes hyperparameters, outer loop evaluates. Needed whenever you report performance of a *tuned* model, because `best_score_` from GridSearchCV is optimistically biased.

**Q3**: Name three things you must set for full PyTorch determinism beyond `torch.manual_seed()`.
> `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark = False`, and `worker_init_fn` for DataLoader.

**Q4**: W&B vs MLflow — when would you choose each?
> W&B for teams and rich dashboards (cloud). MLflow for self-hosted / privacy-sensitive environments.

---

# Lab Preview

| Task | Time | What You'll Do |
|------|------|----------------|
| **1. Grid vs Random** | 20 min | Same budget, compare coverage and best score |
| **2. Optuna** | 20 min | Bayesian optimization with visualizations |
| **3. Nested CV** | 10 min | Compare `best_score_` vs nested CV score |
| **4. PyTorch Seeds** | 15 min | Reproducibility with and without full seeding |
| **5. W&B Integration** | 20 min | Log experiments to a dashboard |

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

**This week's message:**

> Tune systematically (random/Bayesian search).
> Report honestly (nested CV). Track everything (W&B/MLflow).
> Reproducibility is not optional — it's engineering discipline.

**Next week**: Git — Version Your Code
