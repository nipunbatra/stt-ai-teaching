---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Reproducibility in Practice

## Week 9: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The Friday Night Nightmare

**11:30 PM — You (on WhatsApp):**
> *"Done! Model hit 94% accuracy. Just pushed `model_final_v2.py`. We are going to ace this."*

**11:45 PM — Your partner:**
> *"I pulled it. I'm only getting 81%."*

**11:47 PM — You:**
> *"Wait, let me run it again... I'm getting 86% now?! What hyperparameters did I use an hour ago?"*

**11:55 PM — Your partner:**
> *"I tried to fix it. Now I get `ModuleNotFoundError: No module named 'xgboost'`. I'm on a Mac, are you on Windows?"*

**12:30 AM:** 17 Stack Overflow tabs. Still broken. **0 marks.**

---

# Three Levels of Reproducibility

Every reproducibility failure comes from losing control of one of three things:

| Level | Problem | Solution |
|:------|:--------|:---------|
| **1. The Math** | "I get different results every run" | Seeds & determinism |
| **2. The Memory** | "Which of my 50 runs was the good one?" | Experiment tracking (TrackIO) |
| **3. The Machine** | "It works on my laptop but not yours" | Docker |

Today we fix all three — **in order**.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Level 1: Control the Math

*Seeds & Determinism*

---

# "Yesterday 92%, Today 85%, Same Code?!"

Run your model training twice — exact same code:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

**Run 1:** 0.92 &nbsp;&nbsp; **Run 2:** 0.85 &nbsp;&nbsp; **Run 3:** 0.88

**Which result do you report?** The problem: **uncontrolled randomness.**

Sources of randomness in ML:
- **Train/test split** — which samples go where
- **Model initialization** — starting weights (neural nets)
- **Data shuffling** — order during training
- **Dropout** — which neurons to deactivate

---

# The Minecraft Seed

In Minecraft, the world is randomly generated. But enter the **same seed** → get the **exact same world** every time.

| Minecraft | Machine Learning |
|:--|:--|
| Seed number | `random_state=42` |
| Same seed → same world | Same seed → same split, same model |
| Share seed with friend → same map | Share seed with TA → same results |

**A random seed locks the universe into one specific timeline.**

---

# How to Lock the Universe

**In sklearn:** set `random_state` everywhere

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Global seed** (for larger projects):

```python
import random, numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)  # call once at the top of your script
```

---

# Seeds: The Limits

Setting seeds makes YOUR code deterministic. But results can still differ across machines because of:

- **Different OS** → different BLAS/LAPACK math libraries under the hood
- **Different library versions** → sklearn 1.2 vs 1.4 may split data differently
- **GPU non-determinism** → CUDA operations are not always reproducible

Seeds solve **Level 1** (the math). We still need to solve the memory and the machine.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Level 2: Control the Memory

*Experiment Tracking with TrackIO*

---

# The Spreadsheet Graveyard

You've been tuning hyperparameters all week:

```
run_lr001_depth5_est100.py     → 87.2%
run_lr01_depth10_est200.py     → 91.4%  ← wait, was this the one?
run_lr01_depth10_est200_v2.py  → 89.1%  ← or this?
run_FINAL.py                   → ???
run_FINAL_FINAL.py             → ???
run_FINAL_USE_THIS.py          → definitely not this
```

You had **the** best model last Tuesday. Now you can't find it.

**Seeds made results reproducible. But you didn't record what you did.**

---

# TrackIO: Three Calls Is All You Need

**TrackIO** (by Hugging Face) — free, local-first experiment tracking.

```python
import trackio

# 1. Start a run
trackio.init(project="cs203-demo", name="RandomForest",
             config={"model": "RF", "n_estimators": 100, "max_depth": 10})

# 2. Log metrics
trackio.log({"train_accuracy": 0.95, "test_accuracy": 0.845})

# 3. Finish
trackio.finish()
```

Everything stored locally in SQLite — no account, no cloud, no cost.

> **Demo**: `python trackio_1_training_curves.py`

---

# TrackIO: Training Curves

Log metrics at each step → dashboard shows curves:

```python
trackio.init(project="cs203-demo", name="gb-training",
             config={"model": "GradientBoosting", "lr": 0.1})

for n_est in range(10, 310, 10):
    gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1)
    gb.fit(X_train, y_train)
    trackio.log({
        "n_estimators": n_est,
        "train_accuracy": float(round(gb.score(X_train, y_train), 4)),
        "test_accuracy": float(round(gb.score(X_test, y_test), 4)),
    })

trackio.finish()
```

Watch accuracy climb and plateau — just like TensorBoard, but simpler.

---

# TrackIO: Comparing Runs

Run the same model with 3 learning rates → dashboard **overlays** them:

```python
for lr in [0.01, 0.1, 0.5]:
    trackio.init(project="cs203-demo", name=f"lr-{lr}",
                 config={"learning_rate": lr})

    for n_est in range(10, 210, 10):
        gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr)
        gb.fit(X_train, y_train)
        trackio.log({
            "n_estimators": n_est,
            "train_accuracy": float(round(gb.score(X_train, y_train), 4)),
            "test_accuracy": float(round(gb.score(X_test, y_test), 4)),
        })

    trackio.finish()
```

> **Demo**: `python trackio_3_compare_hyperparams.py`

---

# TrackIO: What Did My Model Get Wrong?

Log **images** to see predictions, not just numbers:

```python
trackio.init(project="cs203-demo", name="prediction-analysis",
             config={"model": "RandomForest"})

# Create matplotlib grid of misclassified digits
fig = plot_misclassified_grid(X_test, y_test, preds)
fig.savefig("errors.png")

trackio.log({
    "misclassified": trackio.Image("errors.png",
        caption="All misclassified digits — red = wrong"),
    "test_accuracy": 0.97,
})
trackio.finish()
```

> **Demo**: `python trackio_2_misclassified.py` → check "Media & Tables" tab

---

# TrackIO: Per-Class Breakdown Table

Log **tables** for structured analysis:

```python
trackio.log({"per_class_metrics": trackio.Table(
    dataframe=pd.DataFrame(
        table_data,
        columns=["Digit", "Precision", "Recall", "F1", "Support"]),
)})
```

| Digit | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 | 1.000 | 0.970 | 0.985 | 33 |
| 5 | 0.938 | 0.957 | 0.947 | 23 |
| 9 | 0.950 | 0.950 | 0.950 | 20 |

> **Demo**: `python trackio_4_per_class_table.py`

---

# TrackIO: Alerts for Overfitting

Get notified when something goes wrong:

```python
trackio.init(project="cs203-demo", name="overfitting-detector",
             config={"model": "DecisionTree"})

for depth in range(1, 30):
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    gap = dt.score(X_train, y_train) - dt.score(X_test, y_test)

    trackio.log({"max_depth": depth, "overfit_gap": gap})

    if gap > 0.08:
        trackio.alert(title="Overfitting detected!",
                      text=f"Gap={gap:.1%} at depth={depth}",
                      level=trackio.AlertLevel.ERROR)
```

> **Demo**: `python trackio_5_overfitting_alert.py`

---

# TrackIO: The Dashboard

```bash
pip install trackio
trackio show --project cs203-demo
```

| Tab | What You See |
|-----|-------------|
| **Metrics** | Training curves, overlaid runs |
| **Media & Tables** | Images, per-class tables |
| **Runs** | All configs and final metrics |
| **System Metrics** | GPU/CPU usage (auto on Apple Silicon) |

Seeds control the math. TrackIO records the memory.
**But your partner still can't run your code...**

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Level 3: Control the Machine

*Docker*

---

# "Works on My Machine" — The Most Famous Lie in CS

You shared your code AND `requirements.txt`. Your friend installs everything.

```
$ python train.py
OSError: libgomp.so.1: cannot open shared object file
```

**What happened?** Your code depends on a system library on your Mac that doesn't exist on their Windows laptop.

Virtual environments isolate **Python packages**. They **don't** isolate:
- Operating system (Mac vs Windows vs Linux)
- System libraries (C compilers, CUDA)
- Python version itself

**We need to ship the entire computer, not just the package list.**

---

# Why Docker? Four Scenarios

**1. The Assignment Defense**
You submit your ML assignment. TA runs it on Windows → crash. With Docker: TA runs `docker run your-project` → guaranteed identical results.

**2. The Dependency Minefield**
You want to try a cool GitHub repo requiring Python 3.7 + old PyTorch. Installing it will break your other projects. Docker: isolated bubble, no damage.

**3. The Group Project Peacemaker**
4 teammates: Windows, Mac Intel, Mac Silicon, Linux. Getting a shared pipeline to work across all four? Docker gives everyone the exact same environment.

**4. The Portfolio Builder**
You built a Gradio demo and want to deploy it on HuggingFace Spaces or AWS. Cloud providers want a Docker image, not your laptop.

---

# Docker: The Key Idea

Before shipping containers, moving cargo was chaos — different ports, different cranes, different systems.

The shipping container standardized everything: **pack once, ship anywhere.**

| Without Docker | With Docker |
|:--|:--|
| "Install Python 3.10, then pip install..." | `docker run my-app` |
| "Which OS are you on?" | Doesn't matter |
| "It works on my machine" | It works on **every** machine |
| Ship the recipe | Ship the **whole kitchen** |

---

# Six Concepts — That's All

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Image** | Blueprint — OS + libraries + code (read-only) | Recipe card |
| **Container** | Running instance of an image | Dish being cooked |
| **Dockerfile** | Instructions to build an image | The recipe |
| **Docker Hub** | Registry of pre-built images | GitHub for images |
| **Volumes** | Shared folder between container and your laptop | USB drive |
| **Ports** | How you access the container's web apps | Window into the kitchen |

If Image is a **class**, Container is an **object** (instance).

---

# The Dockerfile: A 5-Line Recipe

```dockerfile
FROM python:3.10-slim          # start with Python + Linux
WORKDIR /app                   # set working directory
COPY requirements.txt .        # copy dependency list
RUN pip install -r requirements.txt  # install packages
COPY . .                       # copy your code
```

Build and run:

```bash
docker build -t netflix-predictor .               # build the image
docker run netflix-predictor python train.py       # run your code
```

**That's it.** Your code now runs identically on any machine with Docker.

---

# The Cheat Sheet (6 Commands)

```bash
# Build an image from a Dockerfile
docker build -t my-ml-app .

# Run a container (with port mapping for web apps)
docker run -p 7860:7860 my-ml-app

# List running containers
docker ps

# Stop a container
docker stop <container_id>

# Open a shell inside a running container
docker exec -it <container_id> bash

# Clean up stopped containers
docker rm <container_id>
```

---

# Think About This...

**Question 1:**
*"If we already set seeds, why might results still differ on my Mac vs the TA's Windows?"*

→ Seeds control algorithmic randomness. But different OS have different math libraries (BLAS/LAPACK), and different sklearn versions may split data differently. Docker locks the OS + versions.

**Question 2:**
*"You trained a model inside Docker for 3 hours. You stop the container. Where are your trained model weights?"*

→ **Gone.** Containers are temporary. Unless you used a **volume** to save files back to your laptop, everything inside the container disappears.

---

# Question 3: Why Map Ports?

*"Why do we run `docker run -p 7860:7860` instead of just opening a browser inside Docker?"*

→ Containers are **headless** — no screen, no browser, no GUI. They run the backend process. You use YOUR laptop's browser to look through the "port window" into the container.

```
Your laptop                    Docker container
┌────────────┐                ┌────────────┐
│            │    port 7860    │            │
│  Browser ──┼───────────────►│  Gradio    │
│            │                │  server     │
└────────────┘                └────────────┘
```

---

# The Reproducibility Stack

| Level | Tool | What It Controls |
|:------|:-----|:----------------|
| **The Math** | `random_state=42` | Algorithmic randomness |
| **The Memory** | TrackIO | What you tried & what worked |
| **The Packages** | `venv` + `requirements.txt` | Library versions |
| **The Machine** | Docker | OS + system libs + everything |

```
Seeds + TrackIO + Venv + Docker = Time Capsule
```

**You don't always need all layers.** For course projects: seeds + TrackIO + venv is enough. Add Docker when shipping to production or sharing across OS.

---

# Key Takeaways

1. **Set seeds everywhere** — `random_state=42` in every sklearn call
2. **Track experiments** — `trackio.init()` / `log()` / `finish()` → never lose a good run
3. **Pin dependencies** — `pip freeze > requirements.txt` with exact versions
4. **Dockerize for sharing** — 5-line Dockerfile guarantees it runs anywhere
5. **Reproducibility is a gift to your future self** — if no one else can run it, it might as well not exist
