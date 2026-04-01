---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Your Model is Rotting
## Data Drift & Model Monitoring

**Week 10: CS 203 - Software Tools and Techniques for AI**

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# A Story

You build a spam filter in January. It's **95% accurate.** You deploy it. 🎉

Six months later, users complain: *"Why am I getting so much spam?"*

You check: accuracy is now **72%.**

Same code. Same model. Same server. **What happened?**

---

# What Happened: The World Changed

![w:800](images/week10/model_accuracy_decay.png)

**Your model didn't change. The data did.**

---

# ML Models ≠ Regular Software

| Regular software | ML models |
|:--|:--|
| Sorting algorithm from 2010 still works | Spam filter from 2020 is broken |
| `2 + 2 = 4` forever | "Buy crypto now!" wasn't spam in 2020, is spam in 2026 |
| Bugs are in your **code** | Problems are in the **data** |

A model is a **snapshot** of the world at training time.

When the world moves on, the snapshot gets stale.

---

# Before We Go Further: X and Y

In ML, we always have:

- **X = the input** (what the model sees). Could be a number, an image, a text message.
- **Y = the output** (what the model predicts). Could be a price, a label like "spam/not spam", etc.

The model **learns a pattern from X to Y** during training.

Drift means this pattern **stops working** after deployment. But it can break in different ways...

---

# Three Types of Drift

| Type | Plain English | Shorthand |
|:--|:--|:--|
| **Data Drift** | The inputs look different | The pattern of X values changed |
| **Concept Drift** | Same input → different correct answer | The X→Y relationship changed |
| **Label Drift** | The mix of outcomes shifted | The balance of Y values changed |

Let's look at each with **detailed examples** from tables, images, and text.

---

<!-- _class: lead -->

# Type 1: Data Drift
## *The inputs your model sees have changed.*
## The X values shifted, but the rules for X→Y stayed the same.

---

# Data Drift Example 1: Used Car Price Prediction

**The setup:**

- **X** (input) = kilometers driven (one number)
- **Y** (output) = resale price in ₹ lakhs (one number)
- **Model** = `LinearRegression` from sklearn

We train on **relatively new cars** at a dealership (5,000–60,000 km).

---

# The True Relationship

Car prices **don't** drop linearly with km. They follow an **exponential decay:**

- First 30k km: price drops fast (₹8L → ₹4L)
- After 60k km: price barely changes (₹3L → ₹2.5L)

But in the **training range** (5k–60k km), a straight line fits well enough!

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
km_train = np.random.normal(25000, 10000, 500).clip(5000, 60000)
price_train = 8 * np.exp(-km_train / 50000) + np.random.normal(0, 0.3, 500)

model = LinearRegression()
model.fit(km_train.reshape(-1, 1), price_train)
```

**Quick jargon check:** R² (R-squared) measures how well the model fits. 1.0 = perfect, 0 = useless, negative = worse than just guessing the average every time. Clean test: **R² = 0.89** — good!

---

# 4 Years Later: High-Mileage Cars Flood the Market

Same dealership, same city. But the used car market changed — people now sell **older, higher-mileage cars** (80k–150k km).

**This is data drift:** the pattern of inputs shifted over time. The pricing rules didn't change — the same car at the same km still costs the same. But the **mix of cars people are selling has changed.**

---

# What Goes Wrong: The Visualization

![w:850](images/week10/car_price_scatter.png)

---

# The Numbers Tell the Story

| Metric | Clean (low-km) | Drifted (high-km) |
|:--|:--|:--|
| **R²** | 0.89 | **-10.3** |
| **MAE** | ₹0.28 lakh | **₹2.26 lakh** |

**R² = -10.3** means the model is catastrophically wrong — worse than just predicting the average price every time. It even predicts **negative prices** for some cars!

**MAE** (Mean Absolute Error) = average of |predicted - actual|. It went from ₹0.28L to ₹2.26L — the model is off by ₹2+ lakhs on average.

---

# Data Drift Example 2: Digit Recognition (Images)

**The setup:**

- **X** = an image of a handwritten digit (8×8 pixels = 64 numbers)
- **Y** = which digit (0, 1, 2, ... 9)
- **Model** = `RandomForestClassifier`

We train on **clean, well-lit scans** from a lab scanner.

Then the model is deployed on a **phone camera app** — photos are noisy, dark, sometimes blurry.

---

# What Is X? Images Are Just Numbers

An 8×8 image is just **64 numbers** (pixel brightness 0–1):

```
0.0  0.0  0.3  0.9  0.9  0.3  0.0  0.0
0.0  0.5  1.0  0.2  0.2  1.0  0.5  0.0
0.0  0.5  1.0  0.2  0.2  1.0  0.5  0.0
...
```

**X = a vector of 64 numbers.** Each pixel is a "feature."

When the camera is noisy, these 64 numbers change — even though it's the same digit.

---

# Clean vs Drifted Images

![w:800](images/week10/image_drift_example.png)

Same digits! But the pixel values are very different. **The inputs (X) shifted.**

---

# But X Is 64-Dimensional — How to Detect Drift?

We can't plot 64 dimensions. Instead, compute **summary statistics** per image and compare:

![w:800](images/week10/image_drift_pixels.png)

Now we can compare these histograms between training and production — just like any tabular feature! (We'll see formal tests for this soon.)

---

# Image Drift: The Code

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)
digits = load_digits()
X_train, y_train = digits.data[:1200], digits.target[:1200]
X_test,  y_test  = digits.data[1200:], digits.target[1200:]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Clean:   {accuracy_score(y_test, model.predict(X_test)):.1%}")

X_noisy = X_test + np.random.normal(0, 3, X_test.shape)  # add noise
print(f"Noisy:   {accuracy_score(y_test, model.predict(X_noisy)):.1%}")

X_dark = X_test * 0.3   # darken
print(f"Dark:    {accuracy_score(y_test, model.predict(X_dark)):.1%}")
```

---

# Data Drift Example 3: Spam Filter (Text)

**The setup:**

- **X** = a text message (converted to word-frequency scores — a vector of ~5000 numbers)
- **Y** = spam or not spam (binary)
- **Model** = `MultinomialNB` (Naive Bayes)

Trained on **formal office emails.** Deployed on **WhatsApp messages.**

---

# What Is X for Text?

The model doesn't read words directly. It converts text to numbers using **TF-IDF** (Term Frequency–Inverse Document Frequency) — each word gets a score based on how important it is in that document:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_emails)
# X_train is now a matrix: each row = one email, each column = one word's score
```

**X = a vector of 5000 word-frequency scores.**

Email vocabulary: "Dear", "meeting", "invoice", "regards"...
WhatsApp vocabulary: "lol", "brb", "😂", "send"...

Most WhatsApp words **don't exist** in the training vocabulary → model has no signal!

---

# Text Drift: What Changes

![w:800](images/week10/text_drift_example.png)

Message length, word choice, emoji usage — **everything about X is different.**

---

# Data Drift: The Pattern

Every example follows the same pattern:

| | Car Price | Digits | Spam |
|:--|:--|:--|:--|
| **X** | km driven (1 number) | pixels (64 numbers) | word scores (5000 numbers) |
| **Y** | price in ₹ lakhs | digit 0-9 | spam/ham |
| **Training** | Low-km cars | Clean scans | Office emails |
| **Production** | High-km cars | Phone camera | WhatsApp |
| **What shifted** | km range | Brightness, noise | Vocabulary, length |
| **Result** | R²: 0.89 → -10.3 | Accuracy drops | Near-random |

**In every case:** the inputs (X) changed. The model was never trained on this kind of input.

---

<!-- _class: lead -->

# Type 2: Concept Drift
## *Same inputs, different correct answer.*
## The X→Y relationship changed.

---

# Concept Drift: What Makes It Different

In data drift, the **inputs** change but the rules stay the same.

In concept drift, the **rules** change — the same input now has a different correct answer.

| | Data Drift | Concept Drift |
|:--|:--|:--|
| X changed? | Yes | No (or maybe) |
| Y\|X changed? | No | **Yes** |
| Same input → same answer? | Yes | **No!** |

---

# Concept Drift Example 1: Zomato Premium Customers

**The setup:**

- **X** = (orders per week, average spend in ₹) — 2 numbers
- **Y** = premium customer (yes/no)
- **Model** = `DecisionTreeClassifier`

**In January 2024** (before Zomato Gold):
- A customer ordering **3x/week** spending **₹500** = **premium**

**In June 2024** (after Zomato Gold launched):
- *Everyone* orders 3x/week now (Gold gives free delivery)
- The **same customer** is now just... normal

---

# Before Zomato Gold: The Model Works

```python
# Model learned: orders > 3 AND spend > ₹400 → premium
# This boundary worked well in January
```

| Customer | Orders/week | Avg Spend | Label (Jan) |
|:--|:--|:--|:--|
| Priya | 4 | ₹500 | **Premium** |
| Rahul | 2 | ₹200 | Regular |
| Ankit | 5 | ₹600 | **Premium** |
| Neha | 1 | ₹150 | Regular |

The model has **85% accuracy** on the clean test set. Life is good.

---

# After Zomato Gold: Same People, Different Labels

| Customer | Orders/week | Avg Spend | Label (Jan) | Label (Jun) |
|:--|:--|:--|:--|:--|
| Priya | 4 | ₹500 | Premium | **Regular** |
| Rahul | 2 | ₹200 | Regular | Regular |
| Ankit | 5 | ₹600 | Premium | **Regular** |
| Neha | 1 | ₹150 | Regular | Regular |

**Priya's input didn't change.** But she's no longer "premium" because *everyone* orders 4x/week now.

The old rule (orders > 3 = premium) **is no longer true.**

---

# Concept Drift: The Visualization

![w:850](images/week10/concept_drift_detailed.png)

The yellow zone = customers the model **confidently classifies as premium, but are now regular.**

---

# Concept Drift Example 2: The Word "Crypto"

**The setup:**

- **X** = email text (converted to word-frequency scores)
- **Y** = spam or not spam

Consider this exact email in two different years:

> *"Invest in crypto for amazing returns! Limited opportunity!"*

| Year | Same text → Label |
|:--|:--|
| 2018 | **Not spam** (crypto was mainstream investing) |
| 2024 | **Spam** (crypto scams are everywhere) |

**Same X, different correct Y.** The model from 2018 would let this through.

---

# Concept Drift Example 3: "Fast Delivery" Reviews

**The setup:**

- **X** = product review text
- **Y** = helpful / not helpful

| Year | Review text | Label |
|:--|:--|:--|
| 2019 | "Delivery was fast, got it in 2 days" | **Not helpful** (obvious, everyone gets fast delivery) |
| 2021 (COVID) | "Delivery was fast, got it in 2 days" | **Helpful** (rare during COVID, very useful info) |

**Same exact review text → different correct label.** The world context changed.

---

# Why Concept Drift Is the Hardest

**You can't detect it by looking at X!**

| Detection method | Data Drift | Concept Drift |
|:--|:--|:--|
| Compare input distributions | ✅ Works (inputs look different) | ❌ Inputs look the same! |
| Compare image statistics | ✅ Works | ❌ Images look the same! |
| Monitor model accuracy over time | Maybe | ✅ **Only way** |
| Need labeled production data? | No | **Yes** |

**You need ground truth labels from production** to detect concept drift. Without them, you're blind.

---

<!-- _class: lead -->

# Type 3: Label Drift
## *The mix of outcomes shifted.*
## The balance of Y values changed.

---

# Label Drift: The Outcome Proportions Shifted

The **balance of Y values** changed — not the inputs, not the rules, just **how often each outcome occurs:**

![w:750](images/week10/label_drift_example.png)

---

# Label Drift Example 1: UPI Fraud Detection

**The setup:**

- **X** = transaction features (amount, time, merchant, device)
- **Y** = fraudulent (yes/no)

| | Training (2022) | Production (2026) |
|:--|:--|:--|
| Fraud rate | **1%** | **5%** |
| Model's threshold | Tuned to flag top 1% | Still tuned for 1% |

The model was tuned to flag the most suspicious 1% of transactions. But now 5% of transactions are actually fraud — it only catches 1 out of every 5 fraudulent ones!

---

# Label Drift Example 2: Pet Classifier at a Dog Park

**The setup:**

- **X** = photo of a pet
- **Y** = cat / dog / bird

| Class | Training | Production (dog park) |
|:--|:--|:--|
| Cat | 40% | **10%** |
| Dog | 40% | **75%** |
| Bird | 20% | 15% |

The model expects to see cats 40% of the time. When it's uncertain between cat and dog, it often picks "cat" because that was common in training. At the dog park, that's wrong.

---

# Label Drift Example 3: Product Reviews After Redesign

**The setup:**

- **X** = review text
- **Y** = positive / neutral / negative

After a product redesign, customers love the new version:

| Sentiment | Training | Production |
|:--|:--|:--|
| Positive | 30% | **65%** |
| Neutral | 45% | 25% |
| Negative | 25% | **10%** |

Model over-predicts "negative" for borderline reviews because it expects 25% negative, not 10%.

---

# The Three Types: Summary So Far

| | Data Drift | Concept Drift | Label Drift |
|:--|:--|:--|:--|
| **What changed** | Inputs shifted | X→Y rules changed | Outcome mix shifted |
| **Example** | Car km: 30k→120k | Zomato Gold | Fraud 1%→5% |
| **The inputs...** | Look different | Look the same! | May or may not change |
| **The correct answer...** | Same rules apply | **Rules changed** | Same rules, different mix |
| **Hardest to detect?** | No | **Yes** | No |

---

# Quick Check: Which Type?

| Scenario | Type |
|:--|:--|
| Swiggy sees more users from tier-2 cities | Data drift |
| "Work from home" used to predict low productivity, now high | Concept drift |
| Positive reviews went from 70% to 90% after redesign | Label drift |
| Average user age shifted from 25 to 40 | Data drift |
| Same X-ray, but new WHO guidelines change the diagnosis | Concept drift |
| Fraud rate doubled after UPI scam wave | Label drift |

---

<!-- _class: lead -->

# Hands-On: Watch Models Break

---

# Experiment 1: Car Price (Exponential Truth + Linear Model)

```python
np.random.seed(42)
km_train = np.random.normal(25000, 10000, 500)   # new-ish cars
km_drift = np.random.normal(80000, 20000, 200)   # high-mileage cars
model = LinearRegression().fit(km_train.reshape(-1,1), price_train)
```

| Metric | Clean (low-km) | Drifted (high-km) |
|:--|:--|:--|
| **R²** | 0.89 | **-10.3** |
| **MAE** | ₹0.28L | **₹2.26L** |

---

# Experiment 2: Pass/Fail from Study Hours

```python
hours_train = np.random.normal(15, 4, 500)  # regular students
hours_drift = np.random.normal(30, 5, 200)  # coaching program
model = DecisionTreeClassifier(max_depth=4).fit(...)
```

| Metric | Clean | Drifted |
|:--|:--|:--|
| **Accuracy** | 80.5% | **21.5%** |

---

# Experiment 3: Car Price (3 Features)

| Feature | Training avg | Production avg | Changed? |
|:--|:--|:--|:--|
| km_driven | 30,000 | **120,000** | **YES — shifted** |
| age_years | 3 | 3 | No |
| engine_cc | 1,200 | 1,200 | No |

| Metric | Clean | 1-feature drifted |
|:--|:--|:--|
| **R²** | 0.85 | **-30.1** |
| **MAE** | ₹0.5 lakh | **₹5.8 lakh** |

One feature shifted → model **completely broken.** We'll see how to detect this formally soon.

---

# Experiment 4: The Gotcha — Accuracy Can Lie

Loan approval: income drops due to recession.

| | Clean | 1-feat drift | All 3 drifted |
|:--|:--|:--|:--|
| **Accuracy** | 81.5% | 74.5% | **83.5%** |

**Accuracy went UP?!** Yes — when everyone is low-income, the model correctly rejects everyone. High accuracy, useless model.

**Lesson:** Monitor **input distributions**, not just accuracy. We'll see how next.

---

<!-- _class: lead -->

# How to Detect Drift

---

# Detection: The Visual Test

The simplest test: **plot the same feature from training and production.**

```python
plt.hist(train["sqft"], bins=30, alpha=0.5, label="Training", color="steelblue")
plt.hist(prod["sqft"],  bins=30, alpha=0.5, label="Production", color="coral")
plt.legend(); plt.show()
```

If the shapes look different → probably drift. But "looks different" is subjective.

We need a **number.**

---

# Detection: The KS Test

**Idea:** Convert each histogram into a **running total** (called a CDF — Cumulative Distribution Function). A CDF shows "what fraction of values are below X?"

Then find the **biggest gap** between the two CDFs. Bigger gap = more drift.

![w:700](images/week10/ks_test_intuition.png)

---

# KS Test: The Code

```python
from scipy.stats import ks_2samp

stat, p = ks_2samp(train_km, prod_km)

print(f"KS statistic: {stat:.3f}")  # the biggest gap (0 = identical, 1 = completely different)
print(f"p-value: {p:.4f}")
```

**p-value** = "if the two distributions were actually identical, what's the chance of seeing a gap this large by luck?" If p < 0.05, we say the difference is **statistically significant** → drift detected.

**One line** per feature. Loop over all features to check everything.

---

# Detection: PSI (Population Stability Index)

**Idea:** Split data into bins (like a histogram). Compare what % falls in each bin between training and production. Big changes = drift.

![w:750](images/week10/psi_step_by_step.png)

| PSI | Meaning | Action |
|:--|:--|:--|
| < 0.1 | No drift | All good |
| 0.1–0.25 | Moderate | Monitor |
| > 0.25 | Significant | Investigate! |

---

# Detection: Evidently (Automate Everything)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_train, current_data=df_prod)
report.save_html("drift_report.html")
```

**4 lines.** Tests every feature automatically, picks the right test (KS for numbers, chi-squared for categories), generates a full HTML report.

---

# Detection Summary

| What to Detect | Method | Python |
|:--|:--|:--|
| Numeric features shifted? | KS test | `ks_2samp(a, b)` |
| Categorical features shifted? | Chi-squared (compares category counts) | `chi2_contingency(table)` |
| Check everything at once | Evidently (picks the right test for you) | `Report(metrics=[DataDriftPreset()])` |
| Concept drift (X→Y rules changed) | Monitor accuracy over time | Need labeled production data |
| Label drift (outcome mix changed) | Track class proportions | `y_pred.value_counts()` |

---

<!-- _class: lead -->

# How to Fix Drift

---

# The Response Pipeline

![w:850](images/week10/drift_response_pipeline.png)

---

# First: Is It Drift or a Bug?

| Symptom | Drift | Bug |
|:--|:--|:--|
| Feature mean shifted gradually | Probably drift | Unlikely |
| Feature suddenly all zeros | Unlikely | Broken data source |
| All values multiplied by 100 | Unlikely | Unit conversion error |

**Always check the pipeline first.** Retraining on buggy data makes things worse.

---

# Fix Strategies by Drift Type

| Type | Detect | Fix |
|:--|:--|:--|
| **Data Drift** (inputs shifted) | KS test, PSI, Evidently | Retrain with new data, data augmentation, collect more diverse training data |
| **Concept Drift** (X→Y changed) | Monitor accuracy (need labels!) | Sliding window (retrain on only the last N months), update model continuously |
| **Label Drift** (outcome mix shifted) | Track class proportions | Recalibrate thresholds, adjust `class_weight` |

---

# The Universal Fix: Retrain

```python
# Combine old training data with new labeled production data
df_combined = pd.concat([df_train, df_new_labeled])

model_v2 = RandomForestClassifier(random_state=42)
model_v2.fit(df_combined.drop("target", axis=1), df_combined["target"])
```

**Connections to previous weeks:**
- **Docker** (Week 9) → same environment for retraining
- **TrackIO** (Week 8) → compare v1 vs v2 metrics
- **Seeds** (Week 9) → reproducible retraining

---

# Retraining Keeps Your Model Alive

![w:800](images/week10/retrain_recovery.png)

Without monitoring, you don't even know accuracy dropped.

---

<!-- _class: lead -->

# Summary

---

# The Three Types — One Slide

| Type | What Changed | Detect | Fix |
|:--|:--|:--|:--|
| **Data Drift** | Inputs shifted | KS test / Evidently | Retrain |
| **Concept Drift** | X→Y rules changed | Monitor accuracy | Retrain on recent data |
| **Label Drift** | Outcome mix shifted | Track proportions | Recalibrate thresholds |

---

# Key Takeaways

1. **Models rot** — data drift, concept drift, label drift

2. **Data drift:** detect with `ks_2samp()` or Evidently (4 lines)

3. **Concept drift:** hardest — need labeled production data to detect

4. **Label drift:** track class proportions over time, recalibrate

5. **Accuracy can lie** — always monitor distributions, not just metrics

6. **The fix:** Monitor → Detect → Diagnose (bug or drift?) → Retrain

---

# Tools Cheat Sheet

| Task | Tool | One-liner |
|:--|:--|:--|
| Test numeric feature | scipy | `ks_2samp(train, prod)` |
| Test categorical feature | scipy | `chi2_contingency(table)` |
| Test all features | Evidently | `Report(metrics=[DataDriftPreset()])` |
| Set pass/fail thresholds | Evidently | `TestSuite(tests=[...])` |
| Monitor over time | Evidently + cron | Save weekly reports |

---

# References

**Books:**
- Chip Huyen, *Designing ML Systems* (O'Reilly, 2022) — Ch. 8
- Kevin Murphy, *Probabilistic ML: Advanced Topics* (2023) — Ch. 19

**Tutorials:**
- Evidently AI: [What is Data Drift?](https://www.evidentlyai.com/ml-in-production/data-drift)

**Tools:**
- [Evidently](https://github.com/evidentlyai/evidently) — open-source drift detection
- [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

**Paper:** Gama et al., *A Survey on Concept Drift Adaptation* (ACM, 2014)
