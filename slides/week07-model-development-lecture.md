---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Model Evaluation & AutoML

## Week 7: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

<!-- _class: lead -->

# Part 1: Refresher

*Where we are and what we know*

---

# The Story So Far

| Week | What We Built | Key Skill |
|------|---------------|-----------|
| 1-2 | Collected and validated 10K movies | Data engineering |
| 3-4 | Labeled with AL + weak supervision | Efficient annotation |
| 5 | Augmented the dataset | More data from existing data |
| 6 | Used Gemini API for multimodal tasks | Foundation models as tools |

**This week**: We train and rigorously evaluate our *own* models, then let AutoML do it for us.

**Why not just use LLMs for everything?**
- Cost at scale (10M predictions/day)
- Latency requirements (< 5ms)
- Privacy (data can't leave your server)
- Interpretability (need to explain *why*)

---

# The Complexity Ladder

```
Level 5: Neural Network         ← Only if huge data + GPU budget
Level 4: Gradient Boosting      ← Often best for tabular (XGBoost, LightGBM)
Level 3: Random Forest          ← Great default, hard to mess up
Level 2: Logistic Regression    ← Start here. Seriously.
Level 1: Dummy (majority class) ← Your baseline floor
```

**Rule**: Climb one step at a time. Stop when gains are marginal.

**The dummy baseline matters**: If 70% of movies succeed, predicting "success" always = 70%. Any real model *must* beat this.

---

# Bias-Variance Tradeoff

<img src="images/week07/bias_variance_tradeoff.png" width="650" style="display: block; margin: 0 auto;">

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

**Simple model** = high bias, low variance (underfitting)
**Complex model** = low bias, high variance (overfitting)

---

# Overfitting vs Underfitting

<img src="images/week07/overfitting_underfitting.png" width="750" style="display: block; margin: 0 auto;">

---

# Diagnosing Your Model

| Train Acc | Test Acc | Gap | Diagnosis | Action |
|-----------|----------|-----|-----------|--------|
| 70% | 68% | 2% | **Underfitting** | More complex model, better features |
| 85% | 83% | 2% | **Good fit** | Ship it |
| 95% | 80% | 15% | **Mild overfitting** | Regularize, more data |
| 99% | 65% | 34% | **Severe overfitting** | Simplify drastically |

<div class="insight">

**The train-test gap is your overfitting detector.** Gap > 10% = red flag.

</div>

---

# Regularization: One Slide

**Idea**: Penalize complexity. "Fit the data, but keep weights small."

| Type | What It Does | Code |
|------|--------------|------|
| **L2 (Ridge)** | Shrinks all weights toward zero | `LogisticRegression(C=0.1)` |
| **L1 (Lasso)** | Drives some weights to exactly zero | `LogisticRegression(penalty='l1')` |
| **Tree depth** | Limits how deep trees can grow | `max_depth=5` |
| **Dropout** | Randomly drops neurons during training | `Dropout(0.5)` |

**Smaller C = more regularization** (C = 1/$\lambda$)

OK -- so how do we actually *measure* if our model is good?

---

<!-- _class: lead -->

# Part 2: Cross-Validation

*How to actually trust your numbers*

---

# The Problem

```python
# Run 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 87.3%

# Run 2 (same code, different random seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 79.8%
```

**Which is the real accuracy? 87%? 80%? Something else?**

You wouldn't bet $500M on a single coin flip. Don't bet your model evaluation on a single random split.

---

# Why Does This Happen?

**Different splits create different test sets:**

- **Test Set A**: Mostly "easy" movies (clear hits and flops)
- **Test Set B**: Mostly "hard" movies (borderline cases)

Same model, same training data, wildly different results.

**The fundamental issue**: One test set is a sample. Samples have variance.

---

# K-Fold Cross-Validation: The Fix

<img src="images/week07/cross_validation_kfold.png" width="750" style="display: block; margin: 0 auto;">

**Key insight**: Every data point is used for testing exactly once.

---

# K-Fold: How It Works

**Split data into K equal parts (folds). Rotate which one is the test set.**

- **Fold 1**: Train on folds 2-5, test on fold 1
- **Fold 2**: Train on folds 1,3-5, test on fold 2
- **Fold 3**: Train on folds 1-2,4-5, test on fold 3
- **Fold 4**: Train on folds 1-3,5, test on fold 4
- **Fold 5**: Train on folds 1-4, test on fold 5

**Final score** = Average of all 5 test scores

$$\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}_k \qquad SE = \frac{\sigma}{\sqrt{K}}$$

---

# K-Fold in Code

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

# One line. That's it.
scores = cross_val_score(model, X, y, cv=5)

print(f"Fold scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std:  {scores.std():.3f}")
```

```
Fold scores: [0.823, 0.851, 0.842, 0.815, 0.834]
Mean: 0.833
Std:  0.013
```

---

# How to Report Results

**Wrong**: "Our model achieves 87% accuracy"

**Right**: "Our model achieves **83.3% +/- 1.3%** accuracy (5-fold CV)"

| Std Dev | Interpretation |
|---------|----------------|
| +/- 1% | Very reliable estimate |
| +/- 3% | Reasonable |
| +/- 5% | Noisy -- need more data or folds |
| +/- 10% | Don't trust this number |

The standard deviation tells you how much to trust the mean.

---

# Choosing K

| K | Train Size | Bias | Variance | Speed |
|---|------------|------|----------|-------|
| 2 | 50% | High (less training data) | High | Fast |
| **5** | **80%** | **Low** | **Low** | **Good** |
| 10 | 90% | Very low | Medium (correlated folds) | Slower |
| N (LOO) | N-1 | Lowest | High (!) | Very slow |

**Default**: K=5. Use K=10 if dataset is small. LOO only for < 100 samples.

**Why LOO has high variance**: Each test set has 1 sample. That's a very noisy estimate per fold.

---

# Stratified K-Fold

**Problem**: Our movie data is 70% success, 30% failure.

Random splits might give:
- Fold 1: 75% success (too many)
- Fold 2: 62% success (too few)

**Stratified K-Fold** ensures every fold maintains the 70/30 ratio.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

**Good news**: `cross_val_score` uses stratified folds by default for classifiers.

---

# When Standard K-Fold Breaks

| Data Type | Problem | Solution |
|-----------|---------|----------|
| **Time series** | Training on future, testing on past | `TimeSeriesSplit` |
| **Grouped data** | Same patient in train AND test | `GroupKFold` |
| **Very small** | K folds too small to be useful | `LeaveOneOut` |

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Split 1: Train [2018],       Test [2019]
# Split 2: Train [2018-2019],  Test [2020]
# Split 3: Train [2018-2020],  Test [2021]
# Always: past predicts future. Never the reverse.
```

---

# Data Leakage: The #1 CV Mistake

**Leakage**: Information from the test set "leaks" into training.

```python
# WRONG: Scaler sees ALL data (including test)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)           # <-- Leakage!
scores = cross_val_score(model, X_scaled, y, cv=5)
```

```python
# RIGHT: Use a Pipeline (scaler fits only on training fold)
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
scores = cross_val_score(pipe, X, y, cv=5)   # <-- Clean
```

**The Pipeline ensures preprocessing happens *inside* each fold.**

---

# Other Common Leakage Sources

| Leakage Type | Example | Fix |
|--------------|---------|-----|
| **Preprocessing** | Scaling on full data | Use `Pipeline` |
| **Feature selection** | Selecting features using full data | Select inside CV |
| **Target leakage** | Feature that encodes the label | Remove the feature |
| **Temporal leakage** | Using future data | `TimeSeriesSplit` |
| **Duplicate leakage** | Same sample in train and test | Deduplicate first |

<div class="warning">

**Data leakage gives you optimistic results that won't hold in production.**
Your model looks great in the notebook, fails in the real world.

</div>

---

# Learning Curves

**Plot score vs training set size.** Diagnoses whether you need more data.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=5
)
```

| Shape | Diagnosis | Action |
|-------|-----------|--------|
| Big gap, both rising | **Overfitting** -- more data would help | Collect more data |
| Both flat at low score | **Underfitting** -- more data won't help | More complex model |
| Converged at high score | **Good fit** | You're done |

---

# Validation Curves

**Plot score vs hyperparameter value.** Finds the sweet spot.

<img src="images/week07/validation_curve.png" width="700" style="display: block; margin: 0 auto;">

```python
from sklearn.model_selection import validation_curve

train_scores, val_scores = validation_curve(
    RandomForestClassifier(), X, y,
    param_name="max_depth", param_range=[1, 2, 5, 10, 20, 50], cv=5
)
```

---

# Cross-Validation Summary

| Situation | Use This | Code |
|-----------|----------|------|
| **Classification** | `StratifiedKFold` | `cross_val_score(model, X, y, cv=5)` |
| **Regression** | `KFold` | `cross_val_score(model, X, y, cv=5)` |
| **Time series** | `TimeSeriesSplit` | `TimeSeriesSplit(n_splits=5)` |
| **Grouped data** | `GroupKFold` | `GroupKFold(n_splits=5)` |
| **Avoid leakage** | `Pipeline` | `Pipeline([('scaler', ...), ('model', ...)])` |
| **Find right K** | Validation curve | `validation_curve(...)` |
| **Need more data?** | Learning curve | `learning_curve(...)` |

---

<!-- _class: lead -->

# Part 3: AutoML

*What if the computer did all of this for you?*

---

# The Manual Process We Just Learned

```
Step 1: Pick a model                    (complexity ladder)
Step 2: Choose hyperparameters          (grid/random/Bayesian search)
Step 3: Evaluate properly               (nested cross-validation)
Step 4: Try another model               (repeat steps 1-3)
Step 5: Compare all models              (pick the best)
Step 6: Maybe ensemble the top ones     (combine for better accuracy)
```

**AutoML automates steps 1-6.**

---

# What AutoML Does

<img src="images/week07/automl_pipeline.png" width="750" style="display: block; margin: 0 auto;">

---

# AutoGluon: 3 Lines of Code

```python
from autogluon.tabular import TabularPredictor

# 1. Create predictor
predictor = TabularPredictor(label='success')

# 2. Fit (give it a time budget)
predictor.fit(train_data, time_limit=300)  # 5 minutes

# 3. Predict
predictions = predictor.predict(test_data)
```

**That's it.** No model selection. No hyperparameter tuning. No ensembling.
AutoGluon does all of it.

---

# What Happens Inside

```
AutoGluon: Starting fit...
Preprocessing data...
  15 numeric features, 3 categorical features

Fitting 11 models...
  LightGBM           ✓ (32s)   val_acc=0.851
  CatBoost           ✓ (45s)   val_acc=0.856
  XGBoost            ✓ (38s)   val_acc=0.848
  RandomForest       ✓ (25s)   val_acc=0.832
  ExtraTrees         ✓ (28s)   val_acc=0.828
  NeuralNetTorch     ✓ (65s)   val_acc=0.819
  LogisticRegression ✓ (5s)    val_acc=0.789
  ...

Ensembling top models...  ✓ (15s)
Best: WeightedEnsemble_L2 (val_acc=0.873)
```

---

# AutoGluon Leaderboard

```python
predictor.leaderboard(test_data)
```

```
                   model  score_val  fit_time  pred_time
0    WeightedEnsemble_L2     0.873      180s       0.5s
1              CatBoost     0.856       60s        0.1s
2              LightGBM     0.851       40s        0.1s
3               XGBoost     0.848       55s        0.1s
4          RandomForest     0.832       30s        0.2s
5    LogisticRegression     0.789       10s        0.0s
```

**The ensemble beats every individual model.** That's the power of stacking.

---

# AutoGluon Presets

```python
# Quick exploration (minutes)
predictor.fit(train_data, presets='medium_quality', time_limit=60)

# Balanced (recommended default)
predictor.fit(train_data, presets='good_quality', time_limit=300)

# Production
predictor.fit(train_data, presets='high_quality', time_limit=3600)

# Competition (best possible, very slow)
predictor.fit(train_data, presets='best_quality', time_limit=14400)
```

| Preset | Time | Models | Ensembling |
|--------|------|--------|------------|
| `medium_quality` | ~1 min | 5-6 | Simple |
| `good_quality` | ~5 min | 8-10 | Weighted |
| `high_quality` | ~1 hour | 10+ | Multi-layer stacking |
| `best_quality` | Hours | 15+ | Deep stacking |

---

# When to Use AutoML

<div class="columns">
<div>

**Good for:**
- Tabular data (CSVs, dataframes)
- Quick baselines and upper bounds
- When you lack time or ML expertise
- Kaggle competitions

</div>
<div>

**Be careful when:**
- Model must be interpretable (use LR or DT instead)
- Inference latency matters (ensembles are slow)
- Model must fit on device (ensembles are large)
- Data is non-tabular (images, text, audio)

</div>
</div>

---

# AutoML vs Manual: The Spectrum

| Approach | Effort | Control | Accuracy |
|----------|--------|---------|----------|
| `DummyClassifier()` | Zero | N/A | Baseline |
| `LogisticRegression()` | Low | High | Good |
| `RandomizedSearchCV(RF, ...)` | Medium | High | Better |
| `optuna.optimize(objective, ...)` | Medium | High | Better |
| `TabularPredictor().fit()` | Low | Low | Best |

**They're not mutually exclusive.** Use AutoML to find the ceiling, then manually build an interpretable model that gets close.

---

# The Complete Workflow

```python
# Step 1: Know your floor
dummy = cross_val_score(DummyClassifier(), X, y, cv=5).mean()

# Step 2: Simple interpretable model
lr = cross_val_score(LogisticRegression(), X, y, cv=5).mean()

# Step 3: Strong default with tuning
search = RandomizedSearchCV(RandomForestClassifier(), params, n_iter=60, cv=5)
outer = cross_val_score(search, X, y, cv=5)  # Nested CV

# Step 4: AutoML ceiling
predictor = TabularPredictor(label='target').fit(train_data, time_limit=300)

# Step 5: Decide
# If LR is close to AutoML → deploy LR (interpretable, fast)
# If RF is close to AutoML → deploy RF (good balance)
# If only AutoML is good enough → deploy AutoML (accept complexity)
```

---

<!-- _class: lead -->

# Key Takeaways & Exam Prep

---

# Key Takeaways

| Concept | One-Liner |
|---------|-----------|
| **Cross-validation** | Never trust a single train/test split |
| **Stratified CV** | Preserve class ratios in each fold |
| **Data leakage** | Preprocessing must happen *inside* CV, not before |
| **Learning curves** | Tells you if more data would help |
| **Validation curves** | Tells you the best hyperparameter value |
| **AutoML** | Automates model selection + tuning + ensembling |

---

# Exam Questions

**Q1**: Why use CV instead of a single train/test split?
> Single split can be lucky/unlucky. CV averages over K splits for a reliable estimate with a standard error.

**Q2**: Train accuracy 99%, test accuracy 70%. What's wrong?
> Overfitting. Model memorized training data. Fix: regularize, simplify, more data.

**Q3**: You scale all features, then run `cross_val_score`. What's wrong?
> Data leakage. The scaler saw test fold data. Fix: use a `Pipeline`.

**Q4**: When would you NOT use standard K-fold CV?
> Time series (use TimeSeriesSplit), grouped data (use GroupKFold).

**Q5**: AutoML gets 88% accuracy. Your logistic regression gets 85%. Which do you deploy?
> Depends on context. If interpretability, latency, or model size matter, the 3% gap may not justify AutoML's complexity. There's no universal right answer -- articulate the tradeoffs.

---

# Lab Preview

| Task | Time | What You'll Do |
|------|------|----------------|
| **1. CV Exploration** | 20 min | Single split vs 5-fold: see the variance yourself |
| **2. Validation Curves** | 20 min | Plot `max_depth` vs accuracy for a Random Forest |
| **3. Learning Curves** | 15 min | Diagnose underfitting vs overfitting |
| **4. Data Leakage** | 15 min | Pipeline vs raw scaling: see the difference |
| **5. AutoGluon** | 30 min | Run AutoML, analyze leaderboard |

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

**This week's message:**

> Measure properly (CV). Understand bias-variance tradeoffs.
> Detect leakage with Pipelines. Let AutoML find the ceiling.
> Start simple. Climb the complexity ladder only when justified.

**Next week**: Tuning & Experiment Tracking
