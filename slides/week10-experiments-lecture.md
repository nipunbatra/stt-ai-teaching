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

# Recap: Tuning from Week 7

You already know *how* to tune (Grid, Random, Bayesian/Optuna).

This week: how to **track** and **reproduce** those experiments.

| Week 7 | Week 10 |
|--------|---------|
| *How* to find good hyperparameters | *How* to record and reproduce runs |
| GridSearchCV, Optuna | W&B, MLflow |
| Cross-validation | PyTorch determinism |
| "Which params are best?" | "Which run was best, and can I reproduce it?" |

---

<!-- _class: lead -->

# Part 1: PyTorch Reproducibility

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

1. **PyTorch reproducibility** needs more than just seeds
   - `torch.use_deterministic_algorithms(True)` + cuDNN settings
   - Seed DataLoader workers separately
   - Report mean ± std over multiple seeds

2. **Experiment tracking** (W&B/MLflow) replaces spreadsheets
   - Logs params, metrics, code, artifacts automatically
   - Compare runs in a dashboard
   - Sweeps automate hyperparameter search

3. **Tracking + tuning are complementary**
   - Week 7 taught *how* to tune → Week 10 teaches *how* to track
   - W&B Sweeps = tuning + tracking in one tool

**Next week:** CI/CD — automate testing and deployment

---

<!-- _class: lead -->

# Questions?

**Exam-relevant concepts:**
- PyTorch determinism: all 8 settings and why each is needed
- DataLoader reproducibility (worker_init_fn + generator)
- When full determinism is vs isn't necessary
- W&B: what it tracks automatically vs manually
- MLflow vs W&B: when to use which
- Multi-seed reporting: why it's better than single deterministic run
