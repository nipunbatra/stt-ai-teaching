"""
Snorkel Weak Supervision Demo
=============================
This demo shows how to use Snorkel's labeling functions to create
training labels without manual annotation.

Run: python snorkel_weak_supervision.py

Requirements: pip install snorkel pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# Constants for labels
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

# =============================================================================
# Sample Dataset: Movie Reviews
# =============================================================================

data = pd.DataFrame({
    'text': [
        "This movie was absolutely amazing! Best film of the year.",
        "Terrible waste of time. Awful acting and boring plot.",
        "It was okay, nothing special but not bad either.",
        "Oscar-worthy performance! A masterpiece of cinema.",
        "I fell asleep halfway through. Very disappointing.",
        "Great special effects but weak storyline.",
        "A delightful film that the whole family will enjoy!",
        "The worst movie I have ever seen. Total garbage.",
        "Interesting concept but poorly executed.",
        "Five stars! Would definitely recommend to everyone.",
    ],
    'rating': [9.2, 2.1, 5.5, 8.8, 3.0, 6.5, 8.0, 1.5, 4.5, 9.0],
    # Ground truth (for evaluation only - not used in training!)
    'true_label': [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
})

print("=" * 60)
print("SNORKEL WEAK SUPERVISION DEMO")
print("=" * 60)
print(f"\nDataset: {len(data)} movie reviews")
print(data[['text', 'rating']].head())

# =============================================================================
# Define Labeling Functions
# =============================================================================

@labeling_function()
def lf_contains_amazing(x):
    """If review contains 'amazing', likely positive."""
    return POSITIVE if "amazing" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_terrible(x):
    """If review contains 'terrible' or 'awful', likely negative."""
    text = x.text.lower()
    if "terrible" in text or "awful" in text:
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_contains_worst(x):
    """If review contains 'worst' or 'garbage', likely negative."""
    text = x.text.lower()
    if "worst" in text or "garbage" in text:
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_contains_best(x):
    """If review contains 'best' or 'masterpiece', likely positive."""
    text = x.text.lower()
    if "best" in text or "masterpiece" in text:
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_high_rating(x):
    """Movies with rating > 8 are likely positive."""
    if x.rating > 8.0:
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_low_rating(x):
    """Movies with rating < 3 are likely negative."""
    if x.rating < 3.0:
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_contains_recommend(x):
    """If review recommends the movie, likely positive."""
    if "recommend" in x.text.lower():
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_contains_boring(x):
    """If review says boring or disappointing, likely negative."""
    text = x.text.lower()
    if "boring" in text or "disappointing" in text:
        return NEGATIVE
    return ABSTAIN

# =============================================================================
# Apply Labeling Functions
# =============================================================================

lfs = [
    lf_contains_amazing,
    lf_contains_terrible,
    lf_contains_worst,
    lf_contains_best,
    lf_high_rating,
    lf_low_rating,
    lf_contains_recommend,
    lf_contains_boring,
]

print("\n" + "=" * 60)
print("LABELING FUNCTIONS")
print("=" * 60)
for i, lf in enumerate(lfs, 1):
    print(f"{i}. {lf.name}")

# Apply LFs to data
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=data)

print("\n" + "=" * 60)
print("LABEL MATRIX (rows=examples, cols=LFs)")
print("=" * 60)
print("Legend: -1=ABSTAIN, 0=NEGATIVE, 1=POSITIVE\n")
print(L_train)

# =============================================================================
# Analyze Labeling Functions
# =============================================================================

print("\n" + "=" * 60)
print("LABELING FUNCTION ANALYSIS")
print("=" * 60)
lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(lf_analysis)

# =============================================================================
# Train Label Model
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING LABEL MODEL")
print("=" * 60)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

# Get probabilistic labels
probs = label_model.predict_proba(L=L_train)
predictions = label_model.predict(L=L_train)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

results = data.copy()
results['predicted'] = predictions
results['prob_negative'] = probs[:, 0].round(3)
results['prob_positive'] = probs[:, 1].round(3)

print("\nPredictions vs Ground Truth:")
print(results[['text', 'true_label', 'predicted', 'prob_positive']].to_string())

# Calculate accuracy
valid_mask = predictions != ABSTAIN
if valid_mask.sum() > 0:
    accuracy = (predictions[valid_mask] == data['true_label'].values[valid_mask]).mean()
    coverage = valid_mask.mean()
    print(f"\n{'='*60}")
    print(f"Coverage: {coverage:.1%} ({valid_mask.sum()}/{len(data)} examples labeled)")
    print(f"Accuracy on labeled examples: {accuracy:.1%}")
    print("=" * 60)

print("\nKey Insight: We created training labels WITHOUT manual annotation!")
print("These probabilistic labels can now train any downstream classifier.")
