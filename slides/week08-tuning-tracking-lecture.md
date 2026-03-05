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
Week 7:  Evaluate models properly    (CV, complexity, leakage)     ✓
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
| Data leakage | Preprocess inside CV with Pipelines |

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

# Approach 0: Manual Tuning

```python
# Monday
model = RandomForestClassifier(n_estimators=100, max_depth=10)
# Accuracy: 83.2%

# Tuesday
model = RandomForestClassifier(n_estimators=200, max_depth=15)
# Accuracy: 84.1%

# Wednesday ... was Tuesday max_depth=15 or 20?
```

**Problems**: No record. No systematic coverage. Easy to miss the best combo.

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

**Reference**: J. Bergstra and Y. Bengio. "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13(Feb):281-305, 2012.

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

# Approach 3: Bayesian Optimization (Optuna)

**Use results so far to decide what to try next.**

<img src="images/week08-tuning/search_strategies_comparison.png" width="750" style="display: block; margin: 0 auto;">

---

# Optuna in Code

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
study.optimize(objective, n_trials=50)
print(f"Best: {study.best_value:.3f}")
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

# Comparison: Grid vs Random vs Bayesian

| | Grid | Random | Bayesian (Optuna) |
|---|------|--------|-------------------|
| Intelligence | None | None | Learns from trials |
| Efficiency | Low | Medium | High |
| Setup | Easy | Easy | Moderate |
| Best for | ≤ 2 params | 3-5 params | Expensive models, NNs |

**Practical rule**: Start with `RandomizedSearchCV`, switch to Optuna for neural nets or expensive models.

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

# AutoGluon: 3 Lines of Code

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='success')
predictor.fit(train_data, time_limit=300)  # 5 minutes
predictions = predictor.predict(test_data)
```

**No model selection. No hyperparameter tuning. No ensembling.**
AutoGluon does all of it.

---

# What Happens Inside

```
AutoGluon: Starting fit...
Fitting 11 models...
  LightGBM           ✓ (32s)   val_acc=0.851
  CatBoost           ✓ (45s)   val_acc=0.856
  XGBoost            ✓ (38s)   val_acc=0.848
  RandomForest       ✓ (25s)   val_acc=0.832
  NeuralNetTorch     ✓ (65s)   val_acc=0.819
  ...

Ensembling top models...  ✓
Best: WeightedEnsemble_L2 (val_acc=0.873)
```

The ensemble beats every individual model. That's **stacking**.

---

# AutoGluon Leaderboard

```python
predictor.leaderboard(test_data)
```

```
                   model  score_val  fit_time
0    WeightedEnsemble_L2     0.873      180s
1              CatBoost     0.856       60s
2              LightGBM     0.851       40s
3               XGBoost     0.848       55s
4          RandomForest     0.832       30s
5    LogisticRegression     0.789       10s
```

---

# When to Use AutoML

| Good for | Be careful when |
|----------|-----------------|
| Tabular data (CSVs) | Model must be interpretable |
| Quick baselines | Latency matters (ensembles are slow) |
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
predictor = TabularPredictor(label='target').fit(train, time_limit=300)
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

# PyTorch Reproducibility Layers

<img src="images/week08-tuning/pytorch_reproducibility_layers.png" width="650" style="display: block; margin: 0 auto;">

**Tradeoff:** Full determinism can be 10-20% slower on GPU.

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

# The Spreadsheet Problem

<img src="images/week08-tuning/experiment_tracking_workflow.png" width="750" style="display: block; margin: 0 auto;">

---

# What W&B Tracks Automatically

| What | How |
|------|-----|
| Hyperparameters | `wandb.config` |
| Metrics | `wandb.log()` |
| Code version | Auto-captures git hash |
| Environment | requirements.txt snapshot |
| System metrics | CPU, GPU, memory usage |

**One dashboard** — compare all runs, filter, sort, reproduce.

---

# W&B Basic Usage

```python
import wandb

wandb.init(project="netflix-predictor", config={
    "learning_rate": 0.01,
    "n_estimators": 100,
    "seed": 42})

model = train(wandb.config)
wandb.log({"accuracy": accuracy, "f1": f1_score})

for epoch in range(100):
    loss = train_one_epoch(model)
    wandb.log({"epoch": epoch, "loss": loss})

wandb.finish()
```

---

# W&B for PyTorch Training

```python
wandb.init(project="mnist-cnn", config={
    "lr": 1e-3, "epochs": 20, "batch_size": 64})

for epoch in range(config.epochs):
    for batch_x, batch_y in train_loader:
        loss = train_step(model, batch_x, batch_y)
        wandb.log({"train_loss": loss})

    val_acc = evaluate(model, val_loader)
    wandb.log({"epoch": epoch, "val_acc": val_acc})
```

W&B auto-generates loss curves, accuracy plots, and system utilization graphs.

---

# MLflow: A Self-Hosted Alternative

```python
import mlflow

mlflow.set_experiment("netflix-predictor")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
```

| | W&B | MLflow |
|--|-----|--------|
| Hosting | Cloud (free tier) | Self-hosted |
| Best for | Teams, sweeps, rich viz | Privacy, enterprise |

---

# Experiment Tracking Best Practices

1. **Log everything** — storage is cheap, hindsight is expensive
2. **Use meaningful run names** — `lr0.01_depth10` not `run_42`
3. **Tag experiments** — `baseline`, `augmented`, `final`
4. **Save the model file** — not just the metrics
5. **Record the git hash** — know which code produced results

---

<!-- _class: lead -->

# Key Takeaways

---

# Summary

| Concept | Key Idea |
|---------|----------|
| Grid search | Exhaustive but scales poorly |
| Random search | Better coverage (Bergstra & Bengio, 2012) |
| Bayesian (Optuna) | Learns from past trials, prunes bad ones |
| Nested CV | Tune inside, evaluate outside — unbiased |
| AutoML | Automates selection + tuning + ensembling |
| PyTorch seeds | 8 settings for full reproducibility |
| W&B / MLflow | Automated tracking replaces spreadsheets |

---

# Exam Questions (1/2)

**Q1**: Why does random search often beat grid search?

> Important hyperparameters get more unique values tested. Bergstra, J. and Bengio, Y. "Random Search for Hyper-Parameter Optimization." JMLR 13:281-305, 2012.

**Q2**: What is nested CV and when do you need it?

> Inner loop tunes, outer loop evaluates. Needed because `best_score_` is optimistically biased.

---

# Exam Questions (2/2)

**Q3**: AutoML gets 88%, logistic regression gets 85%. Which deploy?

> Depends on context: interpretability, latency, model size. 3% may not justify complexity.

**Q4**: You train a neural net with `torch.manual_seed(42)` but get different results each run. Why?

> Need to also set `np.random.seed`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

> Tune systematically (random/Bayesian).
> Report honestly (nested CV).
> Let AutoML find the ceiling. Track everything.

**Next week**: Git — Version Your Code
