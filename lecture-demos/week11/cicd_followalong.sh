#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CI/CD & Automation — Follow-Along Guide                                ║
# ║  Week 11 · CS 203 · Software Tools and Techniques for AI               ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   Your code is versioned, environment is pinned, experiments are tracked.
#   But who checks that tests pass before merging? Who ensures consistent
#   style? Today you automate all of that: pre-commit hooks, pytest,
#   GitHub Actions, and branch protection.
#
# HOW TO USE:
#   1. Open this file in your editor (VS Code, etc.)
#   2. Open a terminal side-by-side
#   3. Copy-paste each command, one at a time
#   4. DO NOT run this file as a script — read it and type along
#
# LEGEND:
#   Lines without # prefix     →  commands to type
#   # >> ...                   →  expected output
#   # ...                      →  explanation / narration
#
# ═══════════════════════════════════════════════════════════════════════════



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 2-3: The Manual Quality Control Problem          ║
# ║     "Who checks that tests pass? Nobody. Until today."                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: Setup — A Testable ML Project                         ~8 min   │
# └──────────────────────────────────────────────────────────────────────────┘

mkdir -p ~/cicd-demo && cd ~/cicd-demo

git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

python -m venv .venv
source .venv/bin/activate

pip install scikit-learn numpy pytest ruff pre-commit

# Create project structure:

mkdir -p src tests .github/workflows

# Training module:

cat > src/__init__.py << 'EOF'
EOF

cat > src/train.py << 'PYEOF'
"""Movie predictor training pipeline."""
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_data(seed=42):
    """Generate synthetic movie dataset."""
    np.random.seed(seed)
    X = np.random.rand(500, 8)
    y = (X[:, 0] * 2 + X[:, 2] - X[:, 5] + np.random.randn(500) * 0.3 > 1).astype(
        int
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def train_model(n_estimators=100, seed=42):
    """Train and return a model."""
    set_seed(seed)
    X_train, X_test, y_train, y_test = load_data(seed)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = train_model()
    X_train, X_test, y_train, y_test = load_data()
    print(f"Accuracy: {model.score(X_test, y_test):.3f}")
PYEOF

python src/train.py

# >> Accuracy: 0.xxx



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 4-6: Testing Pyramid + What to Test in ML       ║
# ║     Show the pyramid. Unit > Integration > E2E. Then write tests.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: Writing Tests with pytest                             ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > tests/__init__.py << 'EOF'
EOF

cat > tests/test_data.py << 'PYEOF'
"""Unit tests for data loading."""
import numpy as np
from src.train import load_data


def test_data_shapes():
    """Data should have correct dimensions."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 8, "Expected 8 features"
    assert X_test.shape[1] == 8, "Test features should match train"
    assert len(y_train) == 400, "80% of 500 = 400 training samples"
    assert len(y_test) == 100, "20% of 500 = 100 test samples"


def test_data_types():
    """Labels should be binary integers."""
    _, _, _, y_test = load_data()
    assert set(np.unique(y_test)).issubset({0, 1}), "Labels must be 0 or 1"


def test_data_reproducibility():
    """Same seed should give same data."""
    X1, _, y1, _ = load_data(seed=42)
    X2, _, y2, _ = load_data(seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
PYEOF

cat > tests/test_model.py << 'PYEOF'
"""Unit tests for model training."""
import numpy as np
from src.train import train_model, load_data


def test_model_trains():
    """Model should train without error."""
    model = train_model()
    assert model is not None


def test_predictions_valid():
    """Predictions should be binary."""
    model = train_model()
    X_test = np.random.rand(10, 8)
    preds = model.predict(X_test)
    assert all(p in [0, 1] for p in preds)
    assert len(preds) == 10


def test_model_reproducibility():
    """Same seed should give same model."""
    model1 = train_model(seed=42)
    model2 = train_model(seed=42)
    X = np.random.rand(20, 8)
    np.testing.assert_array_equal(
        model1.predict(X), model2.predict(X)
    )


def test_predict_proba_range():
    """Probabilities should be between 0 and 1."""
    model = train_model()
    X_test = np.random.rand(10, 8)
    proba = model.predict_proba(X_test)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
PYEOF

# Run the tests:

pytest tests/ -v

# >> tests/test_data.py::test_data_shapes PASSED
# >> tests/test_data.py::test_data_types PASSED
# >> tests/test_data.py::test_data_reproducibility PASSED
# >> tests/test_model.py::test_model_trains PASSED
# >> tests/test_model.py::test_predictions_valid PASSED
# >> tests/test_model.py::test_model_reproducibility PASSED
# >> tests/test_model.py::test_predict_proba_range PASSED
# >> 7 passed

# Let's also see what happens when a test FAILS:

cat > tests/test_fail_demo.py << 'PYEOF'
"""This test should fail — demo only."""
def test_intentional_failure():
    assert 1 + 1 == 3, "Math is broken!"
PYEOF

pytest tests/test_fail_demo.py -v

# >> FAILED tests/test_fail_demo.py::test_intentional_failure
# >> AssertionError: Math is broken!

rm tests/test_fail_demo.py

# KEY: Tests verify your code RUNS correctly, not that accuracy is high.
# Accuracy testing belongs in experiment tracking (Week 10).



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 8-10: Pre-commit Hooks + Ruff                   ║
# ║     "Catch problems before they're committed." Show the framework.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: Pre-commit Hooks                                     ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

# First, let's see Ruff in action:

cat > src/messy.py << 'PYEOF'
import os
import sys
import json      # unused import!

def bad_function(   x,y,z   ):
    """poorly formatted function."""
    result=x+y
    return    result

class myClass:    # should be MyClass (PEP 8)
    def __init__(self):
        self.x = 1
        unused_var = 42    # unused variable!
PYEOF

# Check for issues:

ruff check src/messy.py

# >> src/messy.py:1:8: F401 `os` imported but unused
# >> src/messy.py:2:8: F401 `sys` imported but unused
# >> src/messy.py:3:8: F401 `json` imported but unused
# >> ...

# Auto-fix what Ruff can:

ruff check --fix src/messy.py
ruff format src/messy.py

cat src/messy.py

# >> Much cleaner! Unused imports removed, formatting fixed.

rm src/messy.py

# Now set up pre-commit hooks:

cat > .pre-commit-config.yaml << 'EOF'
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
        args: ['--maxkb=500']
EOF

# Install the hooks:

pre-commit install

# >> pre-commit installed at .git/hooks/pre-commit

# Now hooks run on every commit. Let's test:

cat > src/bad_code.py << 'PYEOF'
import os
import json
def f(x,y):
    return x+y
PYEOF

git add -A
git commit -m "Add bad code"

# >> ruff....................................................Failed
# >> - Fixing src/bad_code.py
# >> ruff-format.............................................Failed
# >> - Reformatted src/bad_code.py
# >>
# >> The commit was REJECTED. But the files were auto-fixed!

cat src/bad_code.py

# >> def f(x, y):
# >>     return x + y
#
# Unused imports removed, formatting fixed. Now commit again:

git add -A
git commit -m "Add utility function"

# >> ruff....................................................Passed
# >> ruff-format.............................................Passed
# >> [main abc1234] Add utility function

# The hook auto-fixed the code, then the second commit succeeded.

rm src/bad_code.py



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 12-15: GitHub Actions concepts + workflow        ║
# ║     Show workflow YAML structure. Then create one live.                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: GitHub Actions — CI on Every Push                    ~12 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > .github/workflows/ci.yml << 'EOF'
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
        with:
          python-version: "3.10"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    needs: lint          # only test if lint passes
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest tests/ -v

  train:
    runs-on: ubuntu-latest
    needs: test          # only train if tests pass
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python src/train.py
EOF

cat .github/workflows/ci.yml

# Key points:
#   - "on: push/pull_request" = trigger
#   - "jobs" = what to run (lint, test, train)
#   - "needs" = job dependencies (test waits for lint)
#   - "runs-on: ubuntu-latest" = GitHub-provided machine
#   - "steps" = individual commands

# Save requirements:

pip freeze > requirements.txt

# Commit everything:

git add -A
git commit -m "Add CI workflow, tests, and pre-commit config"

# >> [main ...] Add CI workflow, tests, and pre-commit config



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: Push to GitHub and Watch CI Run                      ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

# Create a repo on GitHub (using gh CLI):

gh repo create cicd-demo --public --source=. --push

# >> ✓ Created repository your-name/cicd-demo on GitHub
# >> ✓ Pushed commits to ...

# Check the Actions tab:

gh run list

# >> STATUS  TITLE                          WORKFLOW  BRANCH
# >> *       Add CI workflow, tests, ...    CI        main

# Watch a specific run:

gh run watch

# >> ✓ lint     in 15s
# >> ✓ test     in 30s
# >> ✓ train    in 20s

# All green! Now let's break something:

cat > src/broken.py << 'PYEOF'
import os   # unused import — ruff will catch this
def broken():
    pass
PYEOF

git add -A
git commit -m "Add broken code"

# >> pre-commit hooks will catch this and fix it!
# >> (If hooks are set up, the unused import gets removed)

# Without pre-commit, the GitHub Actions lint step would catch it.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 16: Branch Protection                            ║
# ║     Show GitHub settings. "Nobody pushes to main directly."             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: Branch Protection + PR Workflow                      ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

# Enable branch protection (requires GitHub web UI or gh CLI):
# GitHub → Settings → Branches → Add rule for "main":
#   ✅ Require status checks to pass
#   ✅ Require pull request reviews

# Now work on a feature branch:

git checkout -b feature/add-evaluation

cat > src/evaluate.py << 'PYEOF'
"""Model evaluation utilities."""
import numpy as np


def accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """Calculate a simple 2x2 confusion matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
PYEOF

cat > tests/test_evaluate.py << 'PYEOF'
"""Tests for evaluation utilities."""
import numpy as np
from src.evaluate import accuracy, confusion_matrix


def test_accuracy_perfect():
    y = np.array([0, 1, 0, 1])
    assert accuracy(y, y) == 1.0


def test_accuracy_half():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert accuracy(y_true, y_pred) == 0.5


def test_confusion_matrix_shape():
    y = np.array([0, 1, 0, 1])
    cm = confusion_matrix(y, y)
    assert cm.shape == (2, 2)


def test_confusion_matrix_perfect():
    y = np.array([0, 1, 0, 1])
    cm = confusion_matrix(y, y)
    assert cm[0, 1] == 0  # no false positives
    assert cm[1, 0] == 0  # no false negatives
PYEOF

# Run tests locally first:

pytest tests/ -v

# >> 11 passed

# Commit and push:

git add -A
git commit -m "Add evaluation module with tests"
git push -u origin feature/add-evaluation

# Create a PR:

gh pr create --title "Add evaluation module" --body "Adds accuracy and confusion matrix utilities with tests."

# >> Creating pull request...
# >> https://github.com/your-name/cicd-demo/pull/1

# Watch CI run on the PR:

gh pr checks

# >> ✓ lint
# >> ✓ test
# >> ✓ train

# If CI passes, merge the PR:

gh pr merge --merge

# >> ✓ Merged pull request #1

git checkout main
git pull

rm src/broken.py 2>/dev/null



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 7: Makefile — Run CI Locally                              ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > Makefile << 'EOF'
.PHONY: setup test lint format ci clean

setup:
	pip install -r requirements.txt
	pip install pytest ruff pre-commit
	pre-commit install
	@echo "Setup complete!"

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

ci: lint test
	@echo "✅ All checks passed — safe to push!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
EOF

# Run CI locally before pushing:

make ci

# >> ruff check .
# >> All checks passed!
# >> pytest tests/ -v
# >> 11 passed
# >> ✅ All checks passed — safe to push!



# ═══════════════════════════════════════════════════════════════════════════
# WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════
#
# What we covered today:
#
#   Act 1:  Setup — a testable ML project
#   Act 2:  Writing tests with pytest
#   Act 3:  Pre-commit hooks (ruff, formatting)
#   Act 4:  GitHub Actions — CI on every push
#   Act 5:  Push and watch CI run
#   Act 6:  Branch protection + PR workflow
#   Act 7:  Makefile — run CI locally
#
# The progression:
#   Manual checks → pre-commit hooks → GitHub Actions → branch protection
#
# Next week: APIs & demos — ship your model to the world!
# ═══════════════════════════════════════════════════════════════════════════

cd ~
deactivate 2>/dev/null
# rm -rf ~/cicd-demo   # uncomment to clean up
