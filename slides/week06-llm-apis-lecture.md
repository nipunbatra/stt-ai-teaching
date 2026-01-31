---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Foundation Models in Practice

## Week 6: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The Paradigm Shift

<div class="columns">
<div>

**Before (2010s)**

- One model per task
- Weeks of training
- Thousands of labels needed
- Single modality

</div>
<div>

**Now (2020s)**

- One model, many tasks
- Ready to use via API
- Zero-shot or few examples
- Text + Image + Audio + Video

</div>
</div>

<img src="images/week06/foundation_model_paradigm.png" width="650" style="display: block; margin: 0 auto;">

---

# Today: Hands-On with Gemini API

**Tutorial**: [nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)

| Capability | Examples |
|------------|----------|
| **Text** | Sentiment, NER, summarization |
| **Vision** | Object detection, OCR |
| **Audio** | Transcription |
| **Video** | Scene understanding |
| **Documents** | PDF extraction |

**Get API key**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

# Setup

```python
from google import genai
import os

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
MODEL = "gemini-2.0-flash"
```

```bash
pip install google-genai pillow
export GEMINI_API_KEY='your-key-here'
```

**First call:**
```python
response = client.models.generate_content(
    model=MODEL, contents="What is the capital of France?"
)
print(response.text)  # "Paris"
```

---

<!-- _class: section-slide -->

# Text Understanding

---

# Zero-Shot Classification

```python
prompt = """Classify sentiment: Positive, Negative, or Neutral.
Reply with ONLY the label.

Text: "This product is absolutely amazing!" """

response = client.models.generate_content(model=MODEL, contents=prompt)
print(response.text)  # "Positive"
```

**Zero-shot**: No examples needed, just describe the task.

---

# Few-Shot Classification

```python
prompt = """Classify customer queries:

"How do I reset my password?" → Technical Support
"I was charged twice" → Billing
"What are your hours?" → General Inquiry

Query: "My app keeps crashing"
Category:"""

response = client.models.generate_content(model=MODEL, contents=prompt)
print(response.text)  # "Technical Support"
```

**Few-shot**: Provide examples, model learns the pattern.

---

# Named Entity Recognition

```python
text = "Apple CEO Tim Cook announced $500M investment in California."

prompt = f"Extract entities as JSON (PERSON, ORG, LOCATION, MONEY): {text}"
response = client.models.generate_content(model=MODEL, contents=prompt)
```

```json
{
  "PERSON": ["Tim Cook"],
  "ORG": ["Apple"],
  "LOCATION": ["California"],
  "MONEY": ["$500M"]
}
```

---

<!-- _class: section-slide -->

# Vision

---

# Image Understanding

```python
from PIL import Image

image = Image.open("photo.jpg")

response = client.models.generate_content(
    model=MODEL,
    contents=["Describe this image.", image]
)
print(response.text)
```

<img src="images/week06/vision_capabilities.png" width="600" style="display: block; margin: 0 auto;">

---

# Object Detection

```python
prompt = """Detect objects. Return JSON:
[{"label": "cat", "box_2d": [y0, x0, y1, x1]}]"""

response = client.models.generate_content(
    model=MODEL, contents=[prompt, image]
)
```

```json
[
  {"label": "cat", "box_2d": [116, 85, 1000, 885]},
  {"label": "left eye", "box_2d": [519, 625, 615, 713]}
]
```

Coordinates normalized to 0-1000. See notebook for visualization.

---

# OCR and Math

**Receipt OCR:**
```python
prompt = "Extract store name, items, total as JSON."
response = client.models.generate_content(
    model=MODEL, contents=[prompt, receipt_image]
)
```

**Solve Math from Image:**
```python
response = client.models.generate_content(
    model=MODEL, contents=["Solve step by step.", equation_image]
)
```

Works with handwritten equations, calculus, linear algebra.

---

<!-- _class: section-slide -->

# Audio and Video

---

# Audio and Video

**Audio Transcription:**
```python
audio = client.files.upload(file="speech.mp3")
response = client.models.generate_content(
    model=MODEL, contents=["Transcribe this.", audio]
)
```

**Video Understanding:**
```python
video = client.files.upload(file="scene.mp4")
response = client.models.generate_content(
    model=MODEL, contents=["What happens in this video?", video]
)
```

Files need processing time. See notebook for wait loop.

---

<!-- _class: section-slide -->

# Key Concepts

---

# Temperature

```python
# Deterministic - same output every time
config = {"temperature": 0}

# Creative - varied outputs
config = {"temperature": 1.0}
```

| Temperature | Use Case |
|-------------|----------|
| 0 | Facts, code, classification |
| 0.7 | General tasks |
| 1.0+ | Creative writing |

<img src="images/week06/temperature_sampling.png" width="550" style="display: block; margin: 0 auto;">

---

# Structured JSON Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = client.models.generate_content(
    model=MODEL,
    contents="Extract: Sarah is 34 years old",
    config={
        "response_mime_type": "application/json",
        "response_schema": Person
    }
)
# Guaranteed valid JSON matching schema
```

---

# Prompt Engineering

1. **Be specific**: "Classify as Positive/Negative" not "What do you think?"

2. **Show examples**: Few-shot works better for complex tasks

3. **Specify format**: "Return JSON" or "One word only"

4. **Chain of thought**: "Let's solve step by step"

<img src="images/week06/prompt_engineering_concept.png" width="500" style="display: block; margin: 0 auto;">

---

<!-- _class: section-slide -->

# LLMs in Your ML Pipeline

---

# Connecting to Previous Weeks

| Week | Task | LLM Application |
|------|------|-----------------|
| 3-4 | Data Labeling | Auto-label 10-100x faster |
| 5 | Augmentation | Generate paraphrases |

**Batch labeling** - 50 items in one API call:
```python
prompt = f"""Classify each as Positive/Negative/Neutral:
1. {reviews[0]}
2. {reviews[1]}
...
Return JSON array."""
```

**Text augmentation** - generate variations:
```python
prompt = f'Generate 3 paraphrases of: "{text}"'
```

---

# Cost and Rate Limits

**Free tier**: 15 requests/minute, 1M tokens/day

**Cost tips:**
- Use `gemini-2.0-flash` (fast, cheap) vs `gemini-2.0-pro`
- Batch multiple items per request
- Keep prompts concise
- Cache identical requests

**Tokens**: ~4 characters = 1 token, ~75 words = 100 tokens

---

# Summary

| Topic | Key Takeaway |
|-------|--------------|
| Text | Zero-shot and few-shot classification |
| Vision | Object detection, OCR, VQA |
| Audio/Video | Transcription, scene understanding |
| Structured | Pydantic schemas for JSON |
| Pipeline | Labeling and augmentation at scale |

**Tutorial**: [nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)

---

# Lab

1. **Setup** (10 min) - Get API key, run first cells
2. **Text** (20 min) - Sentiment, NER, summarization
3. **Vision** (30 min) - Detection, OCR, charts
4. **Audio/Video** (20 min) - Transcription
5. **Your App** (40 min) - Build something!

**Start now**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Let's Code!

[Tutorial Notebook](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)
