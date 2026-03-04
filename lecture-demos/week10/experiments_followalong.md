---
title: "Experiment Tracking & Reproducibility — Follow-Along Guide"
subtitle: "Week 10 · CS 203 · Software Tools and Techniques for AI"
author: "Prof. Nipun Batra · IIT Gandhinagar"
date: "Spring 2026"
geometry: margin=2cm
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{CS 203 — Experiment Tracking}
  - \fancyhead[R]{Follow-Along Guide}
  - \usepackage{tcolorbox}
  - \tcbuselibrary{skins,breakable}
  - \newtcolorbox{tipbox}{colback=green!5,colframe=green!50!black,title=Tip,fonttitle=\bfseries,breakable}
  - \newtcolorbox{warningbox}{colback=red!5,colframe=red!50!black,title=Warning,fonttitle=\bfseries,breakable}
  - \newtcolorbox{slidebox}{colback=blue!5,colframe=blue!60!black,breakable}
  - \newtcolorbox{actbox}{colback=gray!8,colframe=gray!60!black,breakable,top=2mm,bottom=2mm}
---

\vspace{-0.5cm}

# How to Use This Guide

\begin{itemize}
\item Open \texttt{experiments\_followalong.sh} in your editor (left half of screen)
\item Open a terminal (right half of screen)
\item Copy-paste each command, one at a time
\item \textbf{Type it yourself} --- that's how you learn
\end{itemize}

**Legend:**  `$` = command to type. `>>` = expected output. Blue boxes = projector slide.

---

\begin{slidebox}\textbf{Projector: Slides 2--3 --- The Spreadsheet Problem}
Your experiment spreadsheet has missing entries. Which run was best? What code version produced it?
\end{slidebox}

\begin{actbox}\textbf{\large Act 1: Setup --- Our ML Project \hfill $\sim$5 min}
\end{actbox}

```bash
$ mkdir -p ~/experiments-demo && cd ~/experiments-demo
$ python -m venv .venv && source .venv/bin/activate
$ pip install scikit-learn numpy optuna wandb matplotlib
```

Create a base training script (`train.py`):

```python
def train_and_evaluate(n_estimators=100, max_depth=None,
                       min_samples_split=2, seed=42):
    # ... returns accuracy
```

```bash
$ python train.py
>> Accuracy: 0.xxx
```

\newpage

\begin{slidebox}\textbf{Projector: Slides 5--7 --- Bias-Variance, Grid vs Random Search}
Theory: hyperparameters control model complexity. Grid search is exhaustive; random search explores more unique values per parameter.
\end{slidebox}

\begin{actbox}\textbf{\large Act 2: Grid Search --- Try Every Combination \hfill $\sim$10 min}
\end{actbox}

```bash
$ python grid_search.py
>> Grid search: 27 combinations
>>   {'n_estimators': 50, 'max_depth': 5, ...} → 0.xxx
>>   ...
>> Best: {...} → 0.xxx
```

Grid search tries all 3 $\times$ 3 $\times$ 3 = 27 combinations. Exhaustive but wasteful --- many runs only vary an unimportant parameter.

\begin{actbox}\textbf{\large Act 3: Random Search --- Better Coverage \hfill $\sim$10 min}
\end{actbox}

```bash
$ python random_search.py
>> Trial  1: {...} → 0.xxx
>> ...
>> Best: {...} → 0.xxx
```

Same 27-trial budget, but each trial explores a unique combination. Random search typically finds a better result because it samples more distinct values per parameter.

\begin{tipbox}
\textbf{Rule of thumb:} Random search with 60 trials covers 95\% of important parameter space (Bergstra \& Bengio, 2012).
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slide 8 --- Bayesian Optimization (Optuna)}
Optuna builds a model of what works and focuses on promising regions.
\end{slidebox}

\begin{actbox}\textbf{\large Act 4: Bayesian Search with Optuna \hfill $\sim$10 min}
\end{actbox}

```bash
$ python optuna_search.py
>> Best accuracy: 0.xxx
>> Best params:   {...}
>> Trials near best region:
>>   {...} → 0.xxx
>>   ...
```

Optuna uses past results to pick better trials. Same 27-trial budget, usually the best result of the three methods.

\begin{actbox}\textbf{\large Act 5: PyTorch Reproducibility \hfill $\sim$10 min}
\end{actbox}

\begin{slidebox}\textbf{Projector: Slides 10--13 --- PyTorch Determinism}
Seeds alone aren't enough. You also need \texttt{torch.use\_deterministic\_algorithms(True)} and cuDNN settings.
\end{slidebox}

```bash
$ python torch_repro.py
>> === Without seeds ===
>>   Run 1: 0.234567   (different each time)
>>   Run 2: -0.891234
>> === With seeds ===
>>   Run 1: 0.123456   (SAME each time)
>>   Run 2: 0.123456
```

\begin{warningbox}
Full PyTorch determinism requires: (1) \texttt{torch.manual\_seed}, (2) \texttt{torch.use\_deterministic\_algorithms(True)}, (3) \texttt{cudnn.benchmark = False}, (4) \texttt{CUBLAS\_WORKSPACE\_CONFIG}. This can be 10--20\% slower on GPU.
\end{warningbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 15--18 --- W\&B Introduction}
Show the W\&B dashboard. One place for params, metrics, code version, artifacts.
\end{slidebox}

\begin{actbox}\textbf{\large Act 6: W\&B --- Track Everything \hfill $\sim$15 min}
\end{actbox}

Login to W\&B:

```bash
$ wandb login
>> Enter API key from https://wandb.ai/authorize
```

Run training with W\&B logging:

```bash
$ python train_wandb.py
>> Train: 0.xxx, Test: 0.xxx
>> View run at: https://wandb.ai/...
```

Key W\&B calls:

```python
wandb.init(project="movie-predictor", config={...})
wandb.log({"accuracy": 0.85, "loss": 0.32})
wandb.finish()
```

Run multiple experiments and compare in the dashboard:

```bash
$ python -c "... run with n_est=50, 200, 500 ..."
```

\begin{tipbox}
W\&B automatically logs: git commit hash, Python version, OS, GPU info, and system metrics (CPU, memory). No extra code needed.
\end{tipbox}

\begin{actbox}\textbf{\large Act 7: W\&B Sweeps --- Automated Search \hfill $\sim$10 min}
\end{actbox}

```bash
$ cat sweep.yaml        # define search space
$ wandb sweep sweep.yaml
>> Created sweep with ID: xxxxxxxx
$ wandb agent <entity>/movie-predictor/<sweep-id> --count 10
```

Check the sweep dashboard: parallel coordinates plot, parameter importance, best runs.

\newpage

# Quick Reference

| Tool | What It Does |
|------|-------------|
| `sklearn.model_selection.GridSearchCV` | Exhaustive grid search |
| `sklearn.model_selection.RandomizedSearchCV` | Random search |
| `optuna.create_study()` | Bayesian hyperparameter optimization |
| `wandb.init()` | Start experiment tracking |
| `wandb.log()` | Log metrics |
| `wandb.config` | Access hyperparameters |
| `wandb sweep` | Create automated sweep |
| `wandb agent` | Run sweep trials |

\vspace{0.5cm}

**PyTorch Reproducibility Checklist:**

| Setting | Code |
|---------|------|
| Python seed | `random.seed(42)` |
| NumPy seed | `np.random.seed(42)` |
| PyTorch seed | `torch.manual_seed(42)` |
| CUDA seed | `torch.cuda.manual_seed_all(42)` |
| Deterministic ops | `torch.use_deterministic_algorithms(True)` |
| Disable cuDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| CUBLAS config | `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` |
| DataLoader workers | `worker_init_fn=seed_worker, generator=g` |

\vspace{0.5cm}

**Exam-relevant concepts:**

- Parameters (learned) vs hyperparameters (set by you)
- Grid search: exhaustive, $O(\prod |values|)$ combinations
- Random search: better in high dimensions (explores more unique values)
- K-fold cross-validation: average over $k$ splits for reliable estimates
- Bias-variance tradeoff: tuning controls model complexity
