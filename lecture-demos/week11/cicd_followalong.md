---
title: "CI/CD & Automation — Follow-Along Guide"
subtitle: "Week 11 · CS 203 · Software Tools and Techniques for AI"
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
  - \fancyhead[L]{CS 203 — CI/CD \& Automation}
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
\item Open \texttt{cicd\_followalong.sh} in your editor (left half of screen)
\item Open a terminal (right half of screen)
\item Copy-paste each command, one at a time
\item \textbf{Type it yourself} --- that's how you learn
\end{itemize}

**Legend:**  `$` = command to type. `>>` = expected output. Blue boxes = projector slide.

---

\begin{slidebox}\textbf{Projector: Slides 2--3 --- The Manual Quality Control Problem}
Nobody manually checks tests before merging. Humans forget. Machines don't.
\end{slidebox}

\begin{actbox}\textbf{\large Act 1: Setup --- A Testable ML Project \hfill $\sim$8 min}
\end{actbox}

```bash
$ mkdir -p ~/cicd-demo && cd ~/cicd-demo
$ git init
$ python -m venv .venv && source .venv/bin/activate
$ pip install scikit-learn numpy pytest ruff pre-commit
$ mkdir -p src tests .github/workflows
```

Create `src/train.py` with `load_data()` and `train_model()` functions:

```bash
$ python src/train.py
>> Accuracy: 0.xxx
```

\newpage

\begin{slidebox}\textbf{Projector: Slides 4--6 --- Testing Pyramid + What to Test in ML}
Many unit tests (fast, cheap), fewer integration, few E2E. In ML: test shapes, types, pipeline completion --- NOT accuracy.
\end{slidebox}

\begin{actbox}\textbf{\large Act 2: Writing Tests with pytest \hfill $\sim$10 min}
\end{actbox}

Create tests in `tests/test_data.py` and `tests/test_model.py`:

```python
def test_data_shapes():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 8
    assert len(y_test) == 100

def test_predictions_valid():
    model = train_model()
    preds = model.predict(np.random.rand(10, 8))
    assert all(p in [0, 1] for p in preds)
```

```bash
$ pytest tests/ -v
>> tests/test_data.py::test_data_shapes PASSED
>> tests/test_data.py::test_data_types PASSED
>> tests/test_data.py::test_data_reproducibility PASSED
>> tests/test_model.py::test_model_trains PASSED
>> tests/test_model.py::test_predictions_valid PASSED
>> tests/test_model.py::test_model_reproducibility PASSED
>> tests/test_model.py::test_predict_proba_range PASSED
>> 7 passed
```

\begin{tipbox}
\textbf{What to test in ML code:} data shapes, data types, no NaN after preprocessing, model produces output, predictions in valid range, pipeline runs end-to-end. \textbf{What NOT to test in CI:} accuracy thresholds (that's experiment tracking).
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 8--10 --- Pre-commit Hooks + Ruff}
Hooks run before \texttt{git commit}. If they fail, the commit is rejected. Ruff is 100x faster than flake8.
\end{slidebox}

\begin{actbox}\textbf{\large Act 3: Pre-commit Hooks \hfill $\sim$10 min}
\end{actbox}

First, see Ruff catch issues:

```bash
$ ruff check src/messy.py
>> src/messy.py:1:8: F401 `os` imported but unused
>> src/messy.py:3:8: F401 `json` imported but unused
$ ruff check --fix src/messy.py && ruff format src/messy.py
```

Set up pre-commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

```bash
$ pre-commit install
$ git add -A && git commit -m "Add bad code"
>> ruff.......Failed   (auto-fixes the file)
>> ruff-format.......Failed
$ git add -A && git commit -m "Add utility function"
>> All hooks passed ✓
```

\begin{warningbox}
When a hook auto-fixes files, the commit is rejected but the fixes are applied. You need to \texttt{git add} again and commit a second time.
\end{warningbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 12--15 --- GitHub Actions}
Workflow YAML in \texttt{.github/workflows/}. Triggers: push, pull\_request. Jobs run on GitHub-hosted machines.
\end{slidebox}

\begin{actbox}\textbf{\large Act 4: GitHub Actions --- CI on Every Push \hfill $\sim$12 min}
\end{actbox}

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install ruff
      - run: ruff check .
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -r requirements.txt && pip install pytest
      - run: pytest tests/ -v
```

Key concepts: `on` = trigger, `jobs` = parallel work, `needs` = dependencies, `steps` = commands.

\begin{actbox}\textbf{\large Act 5: Push and Watch CI Run \hfill $\sim$10 min}
\end{actbox}

```bash
$ gh repo create cicd-demo --public --source=. --push
$ gh run watch
>> ✓ lint   in 15s
>> ✓ test   in 30s
>> ✓ train  in 20s
```

\newpage

\begin{slidebox}\textbf{Projector: Slide 16 --- Branch Protection}
Require CI to pass before merging. Require PR reviews. Nobody pushes directly to main.
\end{slidebox}

\begin{actbox}\textbf{\large Act 6: Branch Protection + PR Workflow \hfill $\sim$10 min}
\end{actbox}

```bash
$ git checkout -b feature/add-evaluation
# ... add src/evaluate.py + tests/test_evaluate.py ...
$ pytest tests/ -v
>> 11 passed
$ git add -A && git commit -m "Add evaluation module with tests"
$ git push -u origin feature/add-evaluation
$ gh pr create --title "Add evaluation module" --body "..."
$ gh pr checks
>> ✓ lint  ✓ test  ✓ train
$ gh pr merge --merge
```

\begin{tipbox}
The workflow: branch $\to$ code $\to$ test locally $\to$ push $\to$ CI runs $\to$ PR review $\to$ merge. This is how professional teams work.
\end{tipbox}

\begin{actbox}\textbf{\large Act 7: Makefile --- Run CI Locally \hfill $\sim$5 min}
\end{actbox}

```makefile
ci: lint test
	@echo "All checks passed!"
lint:
	ruff check .
test:
	pytest tests/ -v
```

```bash
$ make ci
>> All checks passed — safe to push!
```

---

# Quick Reference

| I want to... | Command |
|-------------|---------|
| Run tests | `pytest tests/ -v` |
| Lint code | `ruff check .` |
| Auto-fix lint | `ruff check --fix .` |
| Format code | `ruff format .` |
| Install hooks | `pre-commit install` |
| Run hooks manually | `pre-commit run --all-files` |
| Create GitHub repo | `gh repo create name --public --source=. --push` |
| Watch CI run | `gh run watch` |
| Check PR status | `gh pr checks` |
| Run CI locally | `make ci` |

\vspace{0.5cm}

**Exam-relevant concepts:**

- Testing pyramid: unit (many, fast) $>$ integration $>$ E2E (few, slow)
- CI = automated checks on every push; CD = automated build/deploy after CI
- What to test in ML: data shapes, pipeline completion, prediction validity
- What NOT to test in CI: accuracy thresholds (use experiment tracking)
- Pre-commit hooks catch issues at the earliest point --- before bad code is committed
