---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Data Augmentation

## Week 5 · CS 203: Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

<!-- _class: lead -->

# Part 1: The Data Hunger Problem

*More data from existing data*

---

# Previously on CS 203...

| Week | What We Did | Outcome |
|------|-------------|---------|
| Week 1 | Collected 10,000 movie records | Raw dataset |
| Week 2 | Validated and cleaned the data | Clean dataset |
| Week 3 | Labeled 5,000 movies | Labeled dataset |
| Week 4 | Optimized with AL + weak supervision | Efficient labeling |

**Current state:**
- 5,000 labeled movies
- Model accuracy: 82%
- Netflix wants 90%+
- Labeling budget exhausted!

**Can we improve without more labeling?**

---

# The Data Hunger Problem

**Deep learning models are hungry:**

| Model | Training Data |
|-------|--------------|
| ResNet-50 | 1.2M images |
| GPT-3 | 45TB of text |
| AlphaGo | 30M game positions |

**Your reality:**
- 500 labeled images
- 1,000 text samples
- 100 audio clips

**Solution:** Create more data from existing data!

---

# What is Data Augmentation?

**Data Augmentation**: Transform existing data to create new training examples

**Key Insight**: Some transformations preserve the label

| Transform | Cat image | Still a cat? |
|-----------|-----------|--------------|
| Rotate 10° | Tilted cat | Yes! |
| Flip horizontal | Mirror cat | Yes! |
| Slightly darker | Dim cat | Yes! |
| Add noise | Grainy cat | Yes! |

**Result**: 1 image → 10 training examples (for free!)

---

# The Photographer Analogy

**Teaching someone "what is a cat" with ONE photo:**
- They might think "cat" = this specific pose, lighting, angle

**Teaching with MANY photos:**
- They learn cats can be in different poses
- Cats look similar in different lighting
- Cats can appear anywhere in frame

**Augmentation = Taking many "virtual photos" from one real photo**

---

# Image Augmentation: Real Examples

![w:900](images/week05/digit_augmentation_examples.png)

**Same digit "3" → 10 different training examples!**

---

# Why Augmentation Reduces Overfitting

![w:900](images/week05/augmentation_overfitting.png)

**Without augmentation**: Model memorizes exact pixels
**With augmentation**: Model learns general patterns

---

<!-- _class: lead -->

# Part 2: Image Augmentation

*Geometric and color transforms*

---

# Geometric Transforms

![w:900](images/week05/geometric_transforms.png)

Rotation, flip, translation, scaling, cropping - all preserve the label!

---

# Color/Intensity Transforms

![w:900](images/week05/color_transforms.png)

Brightness, contrast, inversion - simulates different lighting conditions

---

# The "6 vs 9" Problem

![w:800](images/week05/bad_augmentation_6_vs_9.png)

**Critical Rule**: Only augment if transformation preserves the label!

---

# Good vs Bad Augmentation

![w:900](images/week05/good_vs_bad_augmentation.png)

**Always ask**: Does this transformation change what the image represents?

---

# Albumentations: The Go-To Library

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
])

# Apply to image
augmented = transform(image=image)['image']
```

**Why Albumentations?**
- Fast (NumPy/OpenCV optimized)
- Handles bounding boxes and masks
- 60+ transformations built-in

---

# Advanced: Cutout

![w:800](images/week05/cutout_example.png)

**Idea**: Randomly remove patches → forces model to use all features

---

# Advanced: Mixup

![w:900](images/week05/mixup_example.png)

**Idea**: Blend two images AND their labels → smoother decision boundaries

---

# Noise Augmentation

![w:900](images/week05/noise_augmentation.png)

Trains model to be robust to sensor noise and image compression

---

# Blur Augmentation

![w:900](images/week05/blur_augmentation.png)

Light blur is OK, heavy blur loses information!

---

# When NOT to Augment

| Domain | Be Careful With |
|--------|-----------------|
| **Medical imaging** | Flips (anatomy matters) |
| **OCR/Text** | Rotation (text unreadable) |
| **Digits** | Vertical flip (6 ↔ 9) |
| **Fine-grained** | Heavy blur (loses details) |
| **Asymmetric objects** | Flips change meaning |

**Rule of thumb**: If a human can't label the augmented image, don't use it!

---

<!-- _class: lead -->

# Part 3: Text Augmentation

*Preserving meaning while changing words*

---

# Text Augmentation Challenges

**Images**: Continuous pixels, easy to transform
**Text**: Discrete tokens, meaning matters!

| Challenge | Why it's hard |
|-----------|---------------|
| Synonyms | "good" ≠ "great" in all contexts |
| Grammar | Random swaps break sentences |
| Sentiment | Must preserve positive/negative |
| Context | Meaning depends on surrounding words |

---

# Text Augmentation Examples

![w:900](images/week05/text_augmentation_examples.png)

---

# Easy Data Augmentation (EDA)

**4 simple operations:**

| Operation | Example |
|-----------|---------|
| **Synonym Replace** | "The movie was great" → "The film was excellent" |
| **Random Insert** | "I love this" → "I really love this" |
| **Random Swap** | "She likes pizza" → "She pizza likes" |
| **Random Delete** | "This is very good" → "This very good" |

```python
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
text = "The movie was fantastic"
augmented = aug.augment(text)
# "The film was fantastic"
```

---

# Back-Translation

**Idea**: Translate to another language and back

```
English:  "I love machine learning"
    ↓
German:   "Ich liebe maschinelles Lernen"
    ↓
English:  "I love machine learning"  (or slight variation!)
```

**Why it works**: Translation models rephrase naturally

```python
# Using transformers
en_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
de_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

german = en_de("I love this movie")[0]['translation_text']
back = de_en(german)[0]['translation_text']
```

---

# LLM Paraphrasing

**Use GPT/Claude to generate paraphrases:**

```python
prompt = """
Generate 3 paraphrases of this text. Keep the same meaning.

Text: "The model achieved 95% accuracy on the test set."
"""

# Response:
# 1. "The model reached 95% accuracy during testing."
# 2. "On the test data, the model scored 95% accuracy."
# 3. "Testing showed the model was 95% accurate."
```

**Pros**: High quality, natural variations
**Cons**: API cost, slower

---

<!-- _class: lead -->

# Part 4: Audio Augmentation

*Time and frequency transformations*

---

# SpecAugment: Masking Spectrograms

![w:900](images/week05/specaugment_example.png)

**Used by Google's speech recognition and Wav2Vec**

---

# Audio Augmentation with audiomentations

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import librosa

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])

audio, sr = librosa.load('speech.wav', sr=16000)
augmented = augment(samples=audio, sample_rate=sr)
```

**Time domain**: noise, stretch, pitch shift, volume

---

# SpecAugment Code

```python
from torchaudio.transforms import FrequencyMasking, TimeMasking

# Apply to spectrogram
freq_mask = FrequencyMasking(freq_mask_param=30)
time_mask = TimeMasking(time_mask_param=100)

augmented_spec = time_mask(freq_mask(spectrogram))
```

**Intuition**: Force model to use context, not rely on single features

---

<!-- _class: lead -->

# Part 5: Practical Guidelines

*Building your augmentation pipeline*

---

# Start Simple, Measure Impact

**Step 1: Baseline** (no augmentation)
```python
# Train and measure
baseline_acc = train_and_evaluate(augment=None)  # e.g., 75%
```

**Step 2: Add one augmentation**
```python
transform = A.HorizontalFlip(p=0.5)
acc_v1 = train_and_evaluate(augment=transform)  # e.g., 78%
```

**Step 3: Gradually add more**
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])
acc_v2 = train_and_evaluate(augment=transform)  # e.g., 82%
```

---

# Hyperparameters to Tune

| Parameter | What it controls | Starting point |
|-----------|-----------------|----------------|
| **Probability (p)** | How often to apply | 0.5 |
| **Magnitude** | Strength of transform | Low → increase |
| **Num augmentations** | How many at once | 2-3 |

**Example tuning:**
```python
# Start mild
A.Rotate(limit=10, p=0.3)

# Increase if underfitting
A.Rotate(limit=30, p=0.5)

# Decrease if validation gets worse
A.Rotate(limit=15, p=0.3)
```

---

# RandAugment: Automatic Selection

![w:850](images/week05/randaugment_example.png)

Randomly pick N augmentations with magnitude M - simple and effective!

---

# Test-Time Augmentation (TTA)

**Idea**: Augment at inference time, average predictions

```python
def tta_predict(model, image, n_augments=5):
    predictions = [model(image)]  # Original

    for _ in range(n_augments - 1):
        aug_image = augment(image)
        predictions.append(model(aug_image))

    return np.mean(predictions, axis=0)
```

**Result**: Often +1-2% accuracy
**Cost**: N× slower inference

---

# Don't Augment Validation/Test!

**Common mistake:**

```python
# WRONG!
val_transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Don't do this!
])
```

**Correct:**
```python
# Training: with augmentation
train_transform = A.Compose([A.HorizontalFlip(p=0.5), ...])

# Validation/Test: NO augmentation
val_transform = A.Compose([])  # Only normalization if needed
```

**Why?** You want to measure performance on real distribution

---

# Domain-Specific Strategies

| Domain | Recommended | Avoid |
|--------|-------------|-------|
| **Natural images** | Flip, rotate, color jitter | - |
| **Medical** | Mild rotation, brightness | Heavy transforms |
| **Satellite** | Any rotation, color shifts | - |
| **Documents/OCR** | Perspective, shadows | Rotation, flip |
| **Faces** | Limited rotation, brightness | Heavy distortion |

---

<!-- _class: lead -->

# Part 6: Tools & Libraries

*Quick reference*

---

# Libraries Summary

| Modality | Library | Key Features |
|----------|---------|--------------|
| **Images** | Albumentations | Fast, bbox support, 60+ transforms |
| **Images** | torchvision | PyTorch native, RandAugment |
| **Text** | nlpaug | Synonym, BERT, back-translation |
| **Audio** | audiomentations | Time-domain transforms |
| **Audio** | torchaudio | SpecAugment, frequency transforms |
| **All** | Augly (Meta) | Unified API for all modalities |

---

# Quick Start: Image Classification

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

# Quick Start: Text Classification

```python
import nlpaug.augmenter.word as naw

# Option 1: Synonym replacement
aug_syn = naw.SynonymAug(aug_src='wordnet')

# Option 2: Contextual (BERT-based)
aug_bert = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute"
)

# Apply
original = "This movie was fantastic"
augmented = aug_syn.augment(original)
```

---

# Resources

**Papers:**
- AutoAugment (2019) - Learning augmentation policies
- RandAugment (2020) - Simple random augmentation
- SpecAugment (2019) - Audio augmentation
- Mixup (2018) - Blending images and labels

**Libraries:**
- albumentations.ai
- github.com/makcedward/nlpaug
- github.com/iver56/audiomentations

---

# Key Takeaways

1. **Augmentation = free training data**
   - Same images, different views → better generalization

2. **Preserve the label**
   - Flip a cat? Still a cat. Flip a "6"? Now it's a "9"!

3. **Domain-specific choices matter**
   - Images: geometric + color
   - Text: synonyms, paraphrasing
   - Audio: time stretch, pitch shift

4. **Start simple, measure impact**
   - Baseline → add one → measure → iterate

5. **Don't augment validation/test data**

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

**Lab**: Implement augmentation pipeline
Compare model performance with/without augmentation

