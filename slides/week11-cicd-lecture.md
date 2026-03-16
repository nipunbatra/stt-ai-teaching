---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# CI/CD & Automation

## Week 11 : CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The Journey So Far

| Week | What We Built | Analogy |
|------|--------------|---------|
| Week 9 | **Git** — version control | Time travel for code |
| Week 10 | **Environments** — reproducibility | A time capsule anyone can open |
| Week 11 | **Automation** — CI/CD | Robot guards that never sleep |

We can collaborate and run code anywhere.

**But who checks that our collaborators don't break things?**

---

# The Problem: Humans Are Unreliable

```
Developer pushes code
        |
Someone manually runs tests       <-- forgets sometimes
        |
Someone reviews code style         <-- inconsistent
        |
Someone tests on staging           <-- "we'll do it later"
        |
Deploy to production               <-- breaks at 2am
```

Humans forget. Humans are inconsistent. Humans skip steps under pressure.

**Machines never forget. Let's build robots to do this.**

---

# Story Time: The $125 Million Bug

In 1999, NASA's **Mars Climate Orbiter** burned up in the Martian atmosphere.

**Why?** One team sent thrust data in **pound-force seconds**. Another team expected **newton-seconds**.

A single unit mismatch. $125 million lost.

```python
# A unit test would have caught this
def test_thrust_units():
    result = compute_thrust(input_newtons=4.45)
    assert result.unit == "newton-seconds"
```

**Lesson:** Automated checks catch what humans miss -- even at NASA.

---

<!-- _class: lead -->

# Part 1: Testing
## The Safety Net

---

# Analogy: Airport Security

Think of testing like the layers of airport security:

| Security Layer | Testing Layer | Speed | How Many? |
|---------------|--------------|-------|-----------|
| Metal detector (every bag) | **Unit test** (every function) | Fast | Many |
| Passport check (ID + ticket) | **Integration test** (components together) | Medium | Some |
| Full boarding process | **End-to-end test** (whole system) | Slow | Few |

**You don't skip the metal detector because you have passport control.**
You need all layers -- but more of the fast, cheap ones.

---

# The Testing Pyramid

```
              /\
             /  \           End-to-End
            / E2E\          "Does the whole app work?"
           /------\         Slow, expensive, FEW
          /        \
         /Integration\      "Do components work together?"
        /-------------\     Medium speed, SOME
       /               \
      /   Unit  Tests   \   "Does this function work?"
     /-------------------\  Fast, cheap, MANY
```

**For ML projects:**
- **Unit:** Does `preprocess()` handle NaN?
- **Integration:** Does the training pipeline produce a model file?
- **E2E:** Does the API return predictions given input?

---

# What to Test in ML Code

| Layer | What to Test |
|-------|-------------|
| **Data** | Input shape correct, no NaN after cleaning, correct dtypes |
| **Preprocessing** | Normalization output in [0, 1], encoding maps correctly |
| **Model** | Output shape matches number of classes |
| **Training** | Pipeline runs without error, model file saved |

**You're NOT testing accuracy** -- that belongs in experiment tracking (Week 8).

You're testing that the code **runs correctly and produces the right shapes**.

---

# What NOT to Test in CI

| Don't Do This | Why It Fails |
|--------------|-------------|
| `assert accuracy > 0.85` | Accuracy varies with data, hardware, random seeds |
| `assert loss < 0.1` | Training might not converge in short CI runs |
| `assert model == expected_model` | Model objects aren't deterministically comparable |

**Instead, test properties that are always true:**

```python
assert accuracy > 0.0            # model isn't totally broken
assert loss < initial_loss       # at least 1 step improves
assert preds.shape == (n,)       # output is the right shape
```

---

# Meet pytest

The simplest test in Python -- just 3 lines:

```python
# tests/test_math.py

def test_addition():
    assert 1 + 1 == 2
```

```bash
$ pytest tests/ -v
tests/test_math.py::test_addition PASSED   ✅
```

Change `2` to `3`? You get a **red failure** with a clear error message.

**Convention:** files named `test_*.py`, functions named `test_*`.

<!-- ⌨ TERMINAL -> Act 1: write a test, run pytest, see green and red -->

---

<!-- _class: lead -->

# Part 2: Pre-commit Hooks
## The Local Bouncer

---

# Analogy: "You Forgot the Attachment!"

You know that moment when you write "please find attached" in an email...
and hit send **without the attachment**?

Gmail warns you: *"Did you mean to attach something?"*

**Pre-commit hooks are that warning -- for your code.**

They catch mistakes **before** they leave your laptop.
Much better than finding out in CI, 5 minutes later.

---

# What Is a Git Hook?

A **hook** is a script that Git runs automatically at certain moments.

```
You type: git commit -m "Add feature"
                |
                v
     Pre-commit hook runs automatically
                |
        Pass? --> Commit created  ✅
        Fail? --> Commit BLOCKED  ❌
                  (fix the issue, try again)
```

**No extra steps to remember.** Git does it for you, every time.

---

# Meet ruff: The Fastest Python Linter

**ruff** checks your code style and catches bugs -- written in Rust, blazingly fast.

```python
# Before ruff format
import   os
import numpy as np
x=1+2
import os    # duplicate!
def f( x  ):
    return x +1
```

```python
# After ruff format + ruff check --fix
import numpy as np

x = 1 + 2

def f(x):
    return x + 1
```

Unused import removed. Spacing fixed. Duplicate caught. **In milliseconds.**

---

# The pre-commit Framework

Instead of writing hooks by hand, use the `pre-commit` framework:

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
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
```

```bash
pre-commit install    # one-time setup, then it runs on every commit
```

<!-- ⌨ TERMINAL -> Act 2: commit with bad formatting, watch pre-commit fix it -->

---

<!-- _class: lead -->

# Part 3: CI/CD
## The Cloud Robot

---

# Analogy: The Car Factory Assembly Line

Imagine a car factory:

```
Raw parts --> Welding --> Painting --> Quality Check --> Ship
                                          |
                                    Defect found?
                                    STOP the line!
```

**CI/CD is an assembly line for code:**

```
Push code --> Build env --> Lint --> Test --> All green? Merge!
                                      |
                                 Test fails?
                                 STOP! Merge BLOCKED.
```

No car leaves the factory without passing inspection.
No code reaches `main` without passing CI.

---

# CI vs CD: Simple Definitions

| Term | What It Means | Analogy |
|------|--------------|---------|
| **Continuous Integration (CI)** | Test on every push | Factory quality check |
| **Continuous Delivery (CD)** | Auto-build after tests pass | Package the car |
| **Continuous Deployment** | Auto-deploy to production | Ship it immediately |

```
Push --> CI (test) --> CD (build) --> Deploy (auto)
```

**Most teams do CI. Many do CD. Fewer do continuous deployment.**

For this course, we focus on **CI** -- automated testing on every push.

---

# Story Time: $460 Million in 45 Minutes

In 2012, **Knight Capital Group** deployed new trading software.

They forgot to update **1 of 8 servers.** The old code started buying stocks wildly.

**In 45 minutes, they lost $460 million.** The company went bankrupt.

An automated deployment pipeline would have ensured all servers got the same code.

**Lesson:** "I'll just deploy it manually" is a recipe for disaster.

---

# The CI Pipeline

<img src="images/week11/cicd_pipeline.png" width="800" style="display: block; margin: 0 auto;">

Every push triggers this pipeline. All automated. No human intervention.

---

# GitHub Actions = Your Cloud Robot Butler

You write a **YAML recipe**. GitHub boots a fresh computer in the cloud and follows it.

```
You push code to GitHub
        |
        v
GitHub reads your YAML recipe
        |
        v
Spins up a fresh Ubuntu machine (the "runner")
        |
        v
Follows your steps: install Python, install deps, run pytest
        |
        v
Reports back: ✅ all passed  or  ❌ test_model.py FAILED
```

**You don't manage any servers.** GitHub does it all for free (2,000 mins/month).

<!-- ⌨ TERMINAL -> Act 3: push broken code, watch Actions fail -->

---

<!-- _class: lead -->

# Part 4: GitHub Actions in Practice
## Writing Your First Workflow

---

# Actions Vocabulary

| Term | What It Is | Think of It As... |
|------|-----------|-------------------|
| **Workflow** | A YAML file in `.github/workflows/` | The recipe |
| **Trigger** | What starts it (`push`, `pull_request`) | The doorbell |
| **Job** | A group of steps on one machine | One chef's task |
| **Step** | A single command or action | One instruction |
| **Runner** | The cloud computer | The kitchen |

```
Push (trigger) --> Workflow (recipe) --> Job (chef) --> Steps (instructions)
                                                        on a Runner (kitchen)
```

---

# Your First Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4           # clone your repo
      - uses: actions/setup-python@v5       # install Python
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v               # run your tests
```

**That's it.** 15 lines. Every push now runs your tests automatically.

---

# What You'll See on a PR

When you open a Pull Request, GitHub shows:

- **Green checkmark** ✅ -- all tests passed, safe to merge
- **Red X** ❌ -- something failed, merge is blocked

You click the red X to see **exactly which test failed and why.**

**This is how every real software company works.** No PR merges without green CI.

---

# Branch Protection: Two Keys to Launch

**Analogy:** Nuclear launch requires two keys turned simultaneously.
Your `main` branch should require at least two checks:

1. **CI must pass** -- robot says the code works
2. **A teammate must approve** -- human says the code makes sense

**How to set it up:** GitHub --> Settings --> Branches --> Branch protection rules

```
feature-branch --> PR --> CI runs --> ✅ Pass --> Review --> Merge allowed
                                     ❌ Fail --> Merge BLOCKED
```

**Nobody pushes directly to main.** Everything goes through this gate.

---

# The Complete Workflow

```
1. Branch        git checkout -b feature/add-evaluation

2. Write code    edit src/evaluate.py + tests/test_evaluate.py

3. Commit        git commit  -->  pre-commit formats your code

4. Push          git push -u origin feature/add-evaluation

5. Open PR       gh pr create

6. CI runs       GitHub Actions runs pytest automatically

7. Review        Teammate reads your code, approves

8. Merge!        Code reaches main, safe and tested
```

It feels slow at first. But it catches bugs early and keeps `main` always working.

<!-- ⌨ TERMINAL -> Acts 4-5: set up branch protection, open PR, see checks -->

---

<!-- _class: lead -->

# Wrap Up

---

# Key Takeaways

| Principle | Tool | When It Runs |
|-----------|------|-------------|
| Test early | **pytest** | While you code |
| Catch locally | **pre-commit hooks** | On every `git commit` |
| Verify in the cloud | **GitHub Actions** | On every `git push` |
| Protect main | **Branch protection** | On every PR |

**The mantra:** Automate the boring stuff. Let robots be the safety net.

**Want to go deeper?** Fixtures, parametrize, coverage, matrix builds, caching -- explore them in the companion notebook.

---

# Next Week

```
Week 9:   Version your CODE        (Git)                ✓
Week 10:  Version your ENVIRONMENT  (venv, Docker)      ✓
Week 11:  AUTOMATE everything       (CI/CD)             ✓
Week 12:  Ship it!                  (APIs & Demos)      <-- next
Week 13:  Make it fast and small    (profiling)
```

**You can now write code, share it, reproduce it, and test it automatically.**

**Next week: put your model in front of real users.**
