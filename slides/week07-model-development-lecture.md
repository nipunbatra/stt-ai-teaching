---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# AutoML and Transfer Learning

## Week 7: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Three Ways to Build ML Models

| Approach | When to Use | Effort |
|----------|-------------|--------|
| **Traditional ML** | Tabular data, need interpretability | Medium |
| **AutoML** | Tabular data, want best accuracy | Low |
| **Transfer Learning** | Images, text, audio | Low |

<div class="columns">
<div>

**This week:**
- Cross-validation
- Bias-variance tradeoff
- AutoML with AutoGluon
- Transfer learning (vision + text)

</div>
<div>

**Connection to pipeline:**
- Week 3-4: Labeled data
- Week 5: Augmented data
- Week 6: LLM features
- **Week 7: Train models!**

</div>
</div>

---

<!-- _class: section-slide -->

# Part 1: Evaluation Fundamentals

---

# The Problem with One Test Set

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)  # 85%
```

**Is 85% reliable?**
- What if we got "lucky" with that split?
- What if test set was unusually easy?
- Different random seed → different accuracy

**One test set = one coin flip.** We need something more reliable.

---

# K-Fold Cross-Validation

<img src="images/week07/cross_validation_kfold.png" width="750" style="display: block; margin: 0 auto;">

**Each data point is tested exactly once.**

---

# Cross-Validation: The Math

For K-fold CV, the estimated performance is:

$$\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}_k$$

**Standard error** of the estimate:

$$SE = \frac{\sigma}{\sqrt{K}}$$

where $\sigma$ is the standard deviation of fold scores.

**Typical choice**: K = 5 or K = 10

---

# Cross-Validation in Code

```python
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100)

# Run 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Fold scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std:  {scores.std():.3f}")
```

```
Fold scores: [0.82, 0.85, 0.84, 0.81, 0.83]
Mean: 0.830
Std:  0.015
```

**Report as**: 83.0% ± 1.5%

---

# Stratified K-Fold

**Problem**: If classes are imbalanced, random folds may have different class ratios.

**Solution**: Stratified K-Fold ensures each fold has same class distribution.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Each fold has same % of positive/negative
```

**Use stratified CV for classification problems.**

---

# When NOT to Use Standard K-Fold

| Data Type | Problem | Solution |
|-----------|---------|----------|
| **Time series** | Future data leaks into past | TimeSeriesSplit |
| **Grouped data** | Same patient in train & test | GroupKFold |
| **Very small data** | K folds too small | Leave-One-Out CV |

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Split 1: Train on [1], Test on [2]
# Split 2: Train on [1,2], Test on [3]
# Split 3: Train on [1,2,3], Test on [4]
# ...
```

---

<!-- _class: section-slide -->

# Part 2: Bias-Variance Tradeoff

---

# The Fundamental Tradeoff

<img src="images/week07/bias_variance_tradeoff.png" width="700" style="display: block; margin: 0 auto;">

**Total Error = Bias² + Variance + Irreducible Noise**

---

# Understanding Bias and Variance

| | **High Bias** | **High Variance** |
|---|---------------|-------------------|
| **Meaning** | Model too simple | Model too complex |
| **Symptom** | Underfitting | Overfitting |
| **Train error** | High | Low |
| **Test error** | High | High |
| **Example** | Linear model on curved data | Deep tree on small data |

<div class="insight">

**Key insight**: You cannot minimize both simultaneously. The goal is to find the sweet spot.

</div>

---

# Overfitting vs Underfitting

<img src="images/week07/overfitting_underfitting.png" width="750" style="display: block; margin: 0 auto;">

---

# Diagnosing Your Model

```python
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train: {train_acc:.1%}, Test: {test_acc:.1%}")
```

| Train Acc | Test Acc | Diagnosis | Fix |
|-----------|----------|-----------|-----|
| 70% | 68% | **Underfitting** | More complex model |
| 99% | 75% | **Overfitting** | Regularization, more data |
| 85% | 83% | **Good fit** | You're done! |

**Gap between train and test indicates overfitting.**

---

# Reducing Overfitting

1. **More training data** - Best solution if available

2. **Regularization** - Penalize complexity
   ```python
   LogisticRegression(C=0.1)  # Smaller C = more regularization
   ```

3. **Simpler model** - Fewer parameters
   ```python
   DecisionTreeClassifier(max_depth=5)  # Limit tree depth
   ```

4. **Early stopping** - Stop before overfitting
   ```python
   model.fit(X, y, early_stopping_rounds=10)
   ```

5. **Dropout** (neural networks) - Randomly drop neurons

---

# Reducing Underfitting

1. **More complex model**
   ```python
   # From linear to polynomial
   PolynomialFeatures(degree=3)
   ```

2. **More features** - Engineer better inputs

3. **Less regularization**
   ```python
   LogisticRegression(C=10)  # Larger C = less regularization
   ```

4. **Train longer** (neural networks)

5. **Remove noise from data**

---

<!-- _class: section-slide -->

# Part 3: Baseline Models

---

# The Complexity Ladder

```
Complexity vs Accuracy:

     5. Deep Neural Network     ← Only if data is huge
     4. Gradient Boosting       ← Often best for tabular
     3. Random Forest           ← Great default
     2. Logistic Regression     ← Start here
     1. Majority Class          ← Your baseline floor
```

**Rule**: Climb one step at a time. Only go up if the improvement justifies complexity.

---

# Baseline 0: Dummy Classifier

```python
from sklearn.dummy import DummyClassifier

# Always predict most common class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f"Baseline: {dummy.score(X_test, y_test):.1%}")
```

**If 70% of movies succeed**, predicting "success" always = 70% accuracy.

**Any real model must beat this!**

---

# Baseline 1: Logistic Regression

**Idea**: Weighted sum of features → probability

$$P(y=1|x) = \sigma(w_0 + w_1 x_1 + w_2 x_2 + \ldots) = \frac{1}{1 + e^{-(w^T x)}}$$

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.1%}")

# Interpretable: see feature weights
print(f"Weights: {model.coef_}")
```

**Pros**: Fast, interpretable, works well for linearly separable data

---

# Baseline 2: Decision Tree

**Idea**: Sequence of if-else rules

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
```

**Visualization**:
```python
from sklearn.tree import plot_tree
plot_tree(tree, feature_names=feature_names, filled=True)
```

**Pros**: Interpretable, handles non-linear relationships
**Cons**: High variance (overfits easily)

---

# Baseline 3: Random Forest

<img src="images/week07/random_forest_ensemble.png" width="700" style="display: block; margin: 0 auto;">

**Ensemble**: Train many trees, take majority vote.

---

# Random Forest: Key Concepts

**Bagging** (Bootstrap Aggregating):
- Train each tree on random subset of data (with replacement)
- Each tree sees different examples

**Feature randomness**:
- At each split, consider random subset of features
- Trees become decorrelated

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Limit tree depth
    random_state=42
)
rf.fit(X_train, y_train)
```

---

# Feature Importance

```python
# Which features matter most?
importances = rf.feature_importances_

for name, imp in sorted(zip(feature_names, importances),
                        key=lambda x: -x[1])[:5]:
    print(f"{name}: {imp:.3f}")
```

```
budget:        0.25
star_power:    0.18
genre_action:  0.12
runtime:       0.10
is_sequel:     0.08
```

**Useful for**: Feature selection, model interpretation, debugging

---

<!-- _class: section-slide -->

# Part 4: AutoML

---

# The Problem with Manual ML

```
1. Try Logistic Regression... okay
2. Try Random Forest... better
3. Try XGBoost... similar
4. Try different hyperparameters...
5. Try feature engineering...
6. Repeat for days...
```

**AutoML automates this entire process.**

---

# What AutoML Does

<img src="images/week07/automl_search_space.png" width="700" style="display: block; margin: 0 auto;">

Automatically searches through models, hyperparameters, and ensembles.

---

# AutoGluon: 3 Lines of Code

```python
from autogluon.tabular import TabularPredictor

# Just point to your data
predictor = TabularPredictor(label='target_column')
predictor.fit(train_data)

# Predict
predictions = predictor.predict(test_data)
```

**What happens inside**:
1. Auto-detect feature types
2. Train 10+ model types
3. Tune hyperparameters
4. Create stacked ensemble

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

**The ensemble combines the best models!**

---

# AutoGluon with Time Budget

```python
# Quick run (5 minutes)
predictor.fit(train_data, time_limit=300)

# Full run (1 hour)
predictor.fit(train_data, time_limit=3600)
```

| Time | What AutoGluon Does |
|------|---------------------|
| 1 min | Basic models (LR, RF) |
| 5 min | + XGBoost, LightGBM |
| 30 min | + Neural nets, tuning |
| 2+ hours | Full tuning, multi-layer stacking |

---

# AutoGluon Presets

```python
# Different quality levels
predictor.fit(train_data, presets='medium_quality')
```

| Preset | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `best_quality` | Slow | Highest | Competitions |
| `high_quality` | Medium | High | Production |
| `good_quality` | Fast | Good | Prototyping |
| `medium_quality` | Faster | Decent | Quick tests |

---

# When to Use AutoML

**Good for:**
- Tabular data (spreadsheets, CSVs)
- Quick prototyping
- Kaggle competitions
- When you don't have ML expertise

**Be careful:**
- Slow training (minutes to hours)
- Large model files
- Hard to interpret
- May not fit in production

---

<!-- _class: section-slide -->

# Part 5: Transfer Learning

---

# Why Train From Scratch?

| | Train from Scratch | Transfer Learning |
|---|-------------------|-------------------|
| **Data needed** | Millions | Hundreds |
| **Training time** | Days/weeks | Minutes/hours |
| **Hardware** | Multiple GPUs | 1 GPU or CPU |
| **Expertise** | High | Low |

**Key insight**: Someone already trained on massive data. Use their work!

---

# Transfer Learning Concept

<img src="images/week07/transfer_learning_architecture.png" width="750" style="display: block; margin: 0 auto;">

---

# What Pretrained Models Learn

**For Images (ResNet, ViT)**:

| Layer | What It Learns |
|-------|----------------|
| Early | Edges, textures |
| Middle | Shapes, parts |
| Late | Objects, scenes |

**For Text (BERT, RoBERTa)**:

| Layer | What It Learns |
|-------|----------------|
| Early | Word meanings |
| Middle | Syntax, grammar |
| Late | Context, semantics |

**Lower layers = General (reusable)**
**Higher layers = Task-specific (replace)**

---

# Transfer Learning: Two Strategies

| | Feature Extraction | Fine-Tuning |
|---|-------------------|-------------|
| **Pretrained layers** | Frozen | Trained (slowly) |
| **New head** | Trained | Trained |
| **Speed** | Fast | Slower |
| **Data needed** | Less | More |
| **Accuracy** | Good | Better |

**Start with feature extraction. Fine-tune if you need more accuracy.**

---

<!-- _class: section-slide -->

# Part 6: Transfer Learning for Vision

---

# Image Classification with timm

```python
import timm
import torch

# Load pretrained model
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# Freeze all layers except the head
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Now train only the final layer
```

**timm** has 700+ pretrained models!

---

# Popular Vision Models

| Model | Size | Accuracy (ImageNet) | Speed |
|-------|------|---------------------|-------|
| ResNet-50 | 25M | 76% | Fast |
| EfficientNet-B0 | 5M | 77% | Fast |
| ViT-B/16 | 86M | 81% | Medium |
| ConvNeXt-Base | 89M | 84% | Medium |

**Recommendation**:
- Quick prototype → ResNet-50 or EfficientNet-B0
- Best accuracy → ViT or ConvNeXt

---

# Vision Example: Movie Posters

```python
from torchvision import transforms, datasets
import timm

# Load pretrained model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)

# Data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load your data
dataset = datasets.ImageFolder('movie_posters/', transform=transform)
```

---

<!-- _class: section-slide -->

# Part 7: Transfer Learning for Text

---

# Text Classification with Transformers

```python
from transformers import pipeline

# Load pretrained sentiment classifier
classifier = pipeline("sentiment-analysis")

# Use immediately - no training!
result = classifier("This movie was fantastic!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Zero training required** for many tasks!

---

# Fine-Tuning BERT

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Fine-tune on your data
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", num_train_epochs=3),
    train_dataset=train_dataset,
)
trainer.train()
```

---

# Popular Text Models

| Model | Size | Best For |
|-------|------|----------|
| BERT-base | 110M | General NLP tasks |
| DistilBERT | 66M | Faster, 97% of BERT quality |
| RoBERTa | 125M | Better than BERT |
| DeBERTa | 140M | State-of-the-art |

**Recommendation**:
- Quick start → DistilBERT
- Best accuracy → DeBERTa

---

# Zero-Shot Classification

**No training at all** - just describe your classes!

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "The new iPhone has amazing battery life"
labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, labels)
print(result['labels'][0])  # "technology"
```

**Connection to Week 6**: Similar to LLM classification, but smaller/faster models.

---

<!-- _class: section-slide -->

# Part 8: Putting It Together

---

# Decision Flowchart

```
What type of data?
    │
    ├── Tabular (spreadsheet) ──► AutoML (AutoGluon)
    │
    ├── Images ──► Transfer Learning (timm, ResNet, ViT)
    │
    ├── Text ──► Transfer Learning (HuggingFace, BERT)
    │
    └── Audio ──► Transfer Learning (Whisper, Wav2Vec)
```

**Always**:
1. Start with a baseline
2. Use cross-validation
3. Watch for overfitting

---

# Complete Example: Movie Success

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from autogluon.tabular import TabularPredictor

# Load data
movies = pd.read_csv('movies.csv')

# Baseline with cross-validation
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X, y, cv=5)
print(f"RF Baseline: {scores.mean():.1%} ± {scores.std():.1%}")

# AutoML
predictor = TabularPredictor(label='success')
predictor.fit(movies, time_limit=300)
print(predictor.leaderboard())
```

---

# Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Cross-validation** | Always use K-fold, never single split |
| **Bias-variance** | Underfitting vs overfitting |
| **Baselines** | Start simple before complex |
| **AutoML** | Automates model selection for tabular |
| **Transfer learning** | Use pretrained for images/text |
| **Overfitting** | Train acc >> Test acc = problem |

---

# Common Exam Questions

1. **Why use cross-validation instead of single train/test split?**
   - Single split can be lucky/unlucky; CV averages over K splits

2. **High train accuracy, low test accuracy - what's wrong?**
   - Overfitting; model memorized training data

3. **When would you NOT use standard K-fold CV?**
   - Time series (use TimeSeriesSplit), grouped data (use GroupKFold)

4. **What's the difference between feature extraction and fine-tuning?**
   - Feature extraction freezes pretrained weights; fine-tuning updates them

---

# Lab: Hands-On

1. **Cross-validation** (20 min)
   - Compare single split vs 5-fold CV
   - Use StratifiedKFold for imbalanced data

2. **Baseline models** (20 min)
   - Train LR, DT, RF on movie data
   - Compare with cross-validation

3. **AutoGluon** (30 min)
   - Run with different time limits
   - Analyze leaderboard

4. **Transfer learning** (30 min)
   - Text: Sentiment with HuggingFace
   - Vision: Image classification with timm

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

**Key concepts:**
- Cross-validation (K-fold, stratified)
- Bias-variance tradeoff
- AutoML (AutoGluon)
- Transfer learning (vision + text)

**Remember**: Simple first, complex only if needed!
