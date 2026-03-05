---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Tuning, AutoML & Experiment Tracking

## Week 8: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are

```
Week 7:  Evaluate models properly    (CV, complexity, bias-variance)   ✓
Week 8:  Tune, AutoML & track        ← you are here
Week 9:  Version your CODE           (Git)
Week 10: Version your ENVIRONMENT    (venv, Docker)
Week 11: Automate everything         (CI/CD)
Week 12: Ship it                     (APIs, demos)
Week 13: Make it fast and small      (profiling, quantization)
```

---

# Recap from Week 7

| Concept | Key Idea |
|---------|----------|
| Train/test split | Never evaluate on training data |
| Model complexity | Degree, depth → underfitting/overfitting |
| Bias-variance | Sweet spot minimizes total error |
| K-fold CV | Average over multiple splits |
| CV variants | Stratified, TimeSeries, GroupKFold |

**This week**: Systematically search for best hyperparameters, let AutoML do it, and track everything.

---

<!-- _class: lead -->

# Part 1: Hyperparameter Tuning

*Finding the best knobs to turn*

---

# The Problem: Too Many Knobs

A Random Forest has *many* hyperparameters:

| Hyperparameter | Controls | Range |
|----------------|----------|-------|
| `n_estimators` | Number of trees | 50-500 |
| `max_depth` | Tree complexity | 3-30 |
| `min_samples_leaf` | Minimum leaf size | 1-20 |
| `max_features` | Features per split | 0.1-1.0 |

And a neural network has even more: learning rate, batch size, hidden layers, dropout rate, optimizer, weight decay...

---

# Approach 1: Grid Search (The For-Loop Way)

What's *really* happening inside `GridSearchCV`?

```python
best_score, best_params = 0, {}

for n_est in [50, 100, 200]:
    for depth in [5, 10, 15]:
        for leaf in [1, 2, 5]:
            model = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth,
                min_samples_leaf=leaf)
            score = cross_val_score(model, X, y, cv=5).mean()
            if score > best_score:
                best_score = score
                best_params = {'n_est': n_est, 'depth': depth,
                               'leaf': leaf}

# That's 3 × 3 × 3 = 27 combos × 5 folds = 135 fits!
```

---

# Grid Search: The Sklearn Way

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(), param_grid,
    cv=5, scoring='accuracy')
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score:  {grid.best_score_:.3f}")
```

Same nested for-loops, but handles CV, scoring, and results tracking.

---

# Grid Search: The Explosion Problem

```
n_estimators:     3 values
max_depth:        4 values
min_samples_leaf: 3 values

Total: 3 × 4 × 3 = 36 combos × 5 folds = 180 fits
```

**Add two more parameters (5 values each):**

$$36 \times 5 \times 5 = 900 \text{ combos} \times 5 \text{ folds} = 4{,}500 \text{ fits}$$

Now imagine a neural network with 6+ hyperparameters. Grid search is hopeless.

---

# Approach 2: Random Search

**Sample random combinations instead of trying all.**

<img src="images/week07/grid_vs_random_search.png" width="700" style="display: block; margin: 0 auto;">

---

# Why Random Search Works Better

**Bergstra & Bengio (2012)**:

> *"Random search is more efficient than grid search because not all hyperparameters are equally important."*

J. Bergstra and Y. Bengio. "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13(Feb):281-305, 2012.

Grid wastes evaluations varying unimportant parameters. Random search covers each dimension more uniformly.

---

# Random Search in Code

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

search = RandomizedSearchCV(
    RandomForestClassifier(),
    {'n_estimators': randint(50, 500),
     'max_depth': randint(3, 30),
     'min_samples_leaf': randint(1, 20),
     'max_features': uniform(0.1, 0.9)},
    n_iter=60, cv=5, random_state=42)
search.fit(X, y)
```

**60 random trials often beats a full grid search.**

---

<!-- _class: lead -->

# Approach 3: Bayesian Optimization

*Use past results to decide what to try next*

---

# The Key Idea: Learn a Surrogate

Grid and Random are **blind** — they don't learn from previous evaluations.

**Bayesian Optimization**:
1. Evaluate a few random points
2. Fit a **surrogate model** (GP or tree) to results so far
3. Use an **acquisition function** to pick the most promising next point
4. Evaluate, update surrogate, repeat

The surrogate predicts both the **mean** (what score to expect) and **uncertainty** (how confident we are).

---

# 1D Example: Bayesian Optimization in Action

```python
from bayes_opt import BayesianOptimization

def black_box(x):
    return -((x - 2)**2) + 1  # unknown to optimizer

optimizer = BayesianOptimization(
    f=black_box,
    pbounds={'x': (-5, 5)},
    random_state=42)
optimizer.maximize(init_points=3, n_iter=7)
```

After 3 random points, the GP surrogate "sees" the landscape and focuses on the peak. See notebook for step-by-step visualization.

---

# Two Flavors of Bayesian Optimization

| | Gaussian Process (GP) | Tree Parzen Estimator (TPE) |
|--|----------------------|----------------------------|
| Surrogate | GP regression | Density estimators |
| Library | `bayesian-optimization`, `skopt` | Optuna |
| Strengths | Exact uncertainty, smooth functions | Scales to many params, handles categorical |
| Best for | < 20 params, continuous | Any size, mixed types |

**GP-based** models the function directly.
**TPE** models the distribution of good vs bad configs.

---

# Optuna (TPE) in Code

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best: {study.best_value:.3f}")
```

Optuna also supports **pruning**: stopping unpromising trials early.

---

# GP-Based Bayesian Optimization in Code

```python
from bayes_opt import BayesianOptimization

def rf_objective(n_estimators, max_depth, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf))
    return cross_val_score(model, X, y, cv=5).mean()

optimizer = BayesianOptimization(
    f=rf_objective,
    pbounds={'n_estimators': (50, 500),
             'max_depth': (3, 30),
             'min_samples_leaf': (1, 20)})
optimizer.maximize(init_points=10, n_iter=50)
```

---

# Optuna for Neural Networks

```python
def nn_objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden = trial.suggest_int('hidden_size', 32, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 4)

    model = build_network(hidden, n_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_loss = train_and_evaluate(model, optimizer)
    return val_loss
```

Optuna prunes unpromising trials early (e.g., if loss diverges), saving compute.

---

# Comparison: All Approaches

| | Grid | Random | GP-BayesOpt | Optuna (TPE) |
|---|------|--------|-------------|--------------|
| Intelligence | None | None | High | High |
| Efficiency | Low | Medium | High | High |
| Scales to many params | No | Yes | No (< 20) | Yes |
| Handles categorical | Yes | Yes | No | Yes |

**Practical rule**: Random for quick exploration, Optuna for serious tuning, GP for small expensive problems.

---

# The Tuning Trap: Selection Bias

```python
# WRONG: Tune and evaluate on SAME cross-validation
grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)
print(f"Best score: {grid.best_score_:.3f}")  # Optimistic!
```

**Why?** You tried many configs and picked the best. By definition, it's the luckiest.

**`GridSearchCV.best_score_` is always optimistically biased.**

---

# Nested Cross-Validation

**Separate the tuning loop from the evaluation loop.**

<img src="images/week07/nested_cross_validation.png" width="700" style="display: block; margin: 0 auto;">

- **Inner loop**: Tunes hyperparameters
- **Outer loop**: Evaluates the tuned model on truly held-out data

---

# Nested CV in Code

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner loop: tune hyperparameters
inner_cv = GridSearchCV(
    RandomForestClassifier(),
    param_grid={'max_depth': [5, 10, 15],
                'n_estimators': [100, 200]},
    cv=3)

# Outer loop: evaluate the tuned model
outer_scores = cross_val_score(inner_cv, X, y, cv=5)
print(f"Nested CV: {outer_scores.mean():.3f} ± {outer_scores.std():.3f}")
```

**This is the gold standard** for reporting tuned model performance.

---

<!-- _class: lead -->

# Part 2: AutoML

*What if the computer did all of this for you?*

---

# FLAML: Fast and Lightweight AutoML

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train,
           task="classification",
           time_budget=120)  # 2 minutes

print(f"Best model: {automl.best_estimator}")
print(f"Best score: {automl.best_config}")
predictions = automl.predict(X_test)
```

**FLAML** (Microsoft): lightweight, fast, no heavy dependencies.

---

# What FLAML Does

```
FLAML: Starting fit...
  Trying LightGBM...        val_acc=0.851  (3s)
  Trying XGBoost...          val_acc=0.848  (5s)
  Trying RandomForest...     val_acc=0.832  (2s)
  Trying ExtraTrees...       val_acc=0.828  (2s)
  Trying LRL1...             val_acc=0.789  (1s)
  Retrying LightGBM (tuned)  val_acc=0.862  (8s)
  ...

Best model: LGBMClassifier (val_acc=0.862)
Total time: 120s
```

FLAML uses **cost-frugal** search — spends more time on promising models.

---

# When to Use AutoML

| Good for | Be careful when |
|----------|-----------------|
| Tabular data (CSVs) | Model must be interpretable |
| Quick baselines | Latency matters |
| Lack time or ML expertise | Model must fit on edge device |
| Kaggle competitions | Non-tabular data (images, text) |

**Use AutoML to find the ceiling, then manually build an interpretable model that gets close.**

---

# The Complete Evaluation Workflow

```python
# Step 1: Know your floor
dummy = cross_val_score(DummyClassifier(), X, y, cv=5).mean()

# Step 2: Simple interpretable model
lr = cross_val_score(LogisticRegression(), X, y, cv=5).mean()

# Step 3: Strong model with tuning (nested CV)
search = RandomizedSearchCV(RandomForestClassifier(),
                            params, n_iter=60, cv=5)
outer = cross_val_score(search, X, y, cv=5)

# Step 4: AutoML ceiling
automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=120)
```

**If LR is close to AutoML → deploy LR (interpretable, fast).**

---

<!-- _class: lead -->

# Part 3: Reproducibility Best Practices

*Making experiments repeatable*

---

# Sklearn: Easy Reproducibility

```python
# One parameter is enough
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

Every run with `random_state=42` gives the **exact same result**.

---

# PyTorch: Seeds Aren't Enough

PyTorch has many sources of randomness:

```python
import torch, random, numpy as np, os

def set_seed(seed=42):
    random.seed(seed)                          # Python
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # cuDNN
    torch.backends.cudnn.benchmark = False     # cuDNN
    torch.use_deterministic_algorithms(True)   # All ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

**Miss any one of these → non-reproducible results.**

---

# Multi-Seed Reporting

Full determinism is not always necessary. Report variance instead:

```python
results = []
for seed in [42, 123, 456, 789, 1024]:
    set_seed(seed)
    acc = train_and_evaluate()
    results.append(acc)

print(f"Accuracy: {np.mean(results):.3f} ± {np.std(results):.3f}")
```

**More informative than a single deterministic result.**

Papers increasingly require multi-seed results (NeurIPS, ICML checklist).

---

<!-- _class: lead -->

# Part 4: Experiment Tracking

*No more spreadsheets*

---

# The Problem: "Which Run Was That?"

```python
# Monday:  lr=0.01, depth=10  → 83.2%
# Tuesday: lr=0.001, depth=15 → 84.1%
# Wednesday: ... was Tuesday depth=15 or 20?
# Thursday: "I think the best was Tuesday's run. Probably."
```

You need a system that automatically records every experiment.

---

# Trackio: Local-First Experiment Tracking

```python
import trackio

trackio.init(project="netflix-predictor", config={
    "learning_rate": 0.01,
    "n_estimators": 100,
    "seed": 42})

model = train(trackio.config)
trackio.log({"accuracy": accuracy, "f1": f1_score})

for epoch in range(100):
    loss = train_one_epoch(model)
    trackio.log({"epoch": epoch, "loss": loss})

trackio.finish()
```

**Trackio** (Hugging Face): free, local-first, W&B-compatible API.

---

# Trackio Features

| Feature | Details |
|---------|---------|
| **Local storage** | SQLite in `~/.cache/huggingface/trackio/` |
| **Dashboard** | Gradio-based, runs locally |
| **W&B-compatible API** | `init`, `log`, `finish` — same as wandb |
| **Free forever** | No cloud account needed |
| **Share** | Optionally sync to Hugging Face Spaces |

```bash
pip install trackio
trackio              # launches local dashboard
```

---

# Trackio for Training Loops

```python
import trackio

trackio.init(project="mnist-cnn", config={
    "lr": 1e-3, "epochs": 20, "batch_size": 64})

for epoch in range(20):
    for batch_x, batch_y in train_loader:
        loss = train_step(model, batch_x, batch_y)
        trackio.log({"train_loss": loss})

    val_acc = evaluate(model, val_loader)
    trackio.log({"epoch": epoch, "val_acc": val_acc})

trackio.finish()
```

Trackio auto-generates loss curves and accuracy plots in its local dashboard.

---

# Comparing Runs in Trackio

```python
# Run 1: baseline
trackio.init(project="nlp", config={"model": "lstm", "lr": 1e-3})
# ... train ...
trackio.finish()

# Run 2: improved
trackio.init(project="nlp", config={"model": "transformer", "lr": 5e-4})
# ... train ...
trackio.finish()
```

Open the local dashboard to see both runs side-by-side with their configs, metrics, and curves.

---

# Experiment Tracking Best Practices

1. **Log everything** — storage is cheap, hindsight is expensive
2. **Use meaningful run names** — `lr0.01_depth10` not `run_42`
3. **Tag experiments** — `baseline`, `augmented`, `final`
4. **Save the model file** — not just the metrics
5. **Record the git hash** — know which code produced results

---

# Other Tracking Tools

| Tool | Hosting | Best For |
|------|---------|----------|
| **Trackio** | Local | Free, simple, local-first |
| **MLflow** | Self-hosted | Enterprise, model registry |
| **W&B** | Cloud | Teams, sweeps, rich viz |
| **TensorBoard** | Local | TensorFlow/PyTorch training |

Pick based on your needs: Trackio for course projects, MLflow for production.

---

<!-- _class: lead -->

# Key Takeaways

---

# Summary

| Concept | Key Idea |
|---------|----------|
| Grid search | Exhaustive but scales poorly |
| Random search | Better coverage (Bergstra & Bengio, 2012) |
| Bayesian opt (GP) | Models the function, uses uncertainty |
| Optuna (TPE) | Scales well, handles categorical, prunes |
| Nested CV | Tune inside, evaluate outside — unbiased |
| FLAML | Lightweight AutoML, cost-frugal search |
| Reproducibility | 8 settings for PyTorch, multi-seed reporting |
| Trackio | Local-first, free experiment tracking |

---

# Exam Questions (1/2)

**Q1**: Why does random search often beat grid search?

> Important hyperparameters get more unique values tested. Bergstra, J. and Bengio, Y. "Random Search for Hyper-Parameter Optimization." JMLR 13:281-305, 2012.

**Q2**: What is the difference between GP-based and TPE-based Bayesian optimization?

> GP models the objective function directly with uncertainty. TPE models the distribution of good vs bad configurations. GP works better for small, continuous spaces; TPE scales to more parameters and handles categorical.

---

# Exam Questions (2/2)

**Q3**: What is nested CV and when do you need it?

> Inner loop tunes, outer loop evaluates. Needed because `best_score_` is optimistically biased.

**Q4**: You train a neural net with `torch.manual_seed(42)` but get different results each run. Why?

> Need to also set `np.random.seed`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

> Tune systematically (random/Bayesian).
> Report honestly (nested CV).
> Let AutoML find the ceiling. Track everything locally.

**Next week**: Git — Version Your Code
