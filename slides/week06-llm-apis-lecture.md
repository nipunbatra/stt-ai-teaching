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

<img src="images/week06/foundation_model_paradigm.png" width="700" style="display: block; margin: 0 auto;">

---

# Today: Hands-On with Gemini API

**Main Resource**: [Gemini API Multimodal Tutorial](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)

We'll explore:

| Capability | What You'll Do |
|------------|----------------|
| **Text** | Sentiment, NER, summarization |
| **Vision** | Object detection, OCR, VQA |
| **Audio** | Speech transcription |
| **Video** | Scene understanding |
| **Documents** | PDF extraction |

**Get your API key**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

# Setup: 3 Lines of Code

```python
from google import genai
import os

# Initialize client
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# That's it! Now you can use any Gemini model
MODEL = "gemini-2.0-flash"
```

```bash
pip install google-genai pillow matplotlib
export GEMINI_API_KEY='your-key-here'
```

---

# Your First API Call

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is the capital of France?"
)

print(response.text)
# "The capital of France is Paris."
```

**That's all it takes.** No training, no datasets, no infrastructure.

---

<!-- _class: section-slide -->

# Text Understanding

---

# Zero-Shot Sentiment Analysis

```python
texts = [
    "This product is absolutely amazing!",
    "Terrible experience. Waste of money.",
    "It's okay. Nothing special."
]

prompt = """Classify sentiment: Positive, Negative, or Neutral.
Reply with ONLY the label.

Text: {text}"""

for text in texts:
    response = client.models.generate_content(
        model=MODEL, contents=prompt.format(text=text)
    )
    print(f"{text[:40]}... → {response.text.strip()}")
```

```
This product is absolutely amazing!... → Positive
Terrible experience. Waste of money... → Negative
It's okay. Nothing special... → Neutral
```

---

# Few-Shot Classification

```python
prompt = """Classify customer queries:

"How do I reset my password?" → Technical Support
"I was charged twice" → Billing
"What are your hours?" → General Inquiry

Query: "My app keeps crashing when I upload photos"
Category:"""

response = client.models.generate_content(model=MODEL, contents=prompt)
print(response.text)  # "Technical Support"
```

**Few-shot learning**: Provide examples, model learns the pattern.

---

# Named Entity Recognition

```python
text = """Apple CEO Tim Cook announced a $500M investment
in California on December 15, 2024."""

prompt = f"""Extract entities as JSON:
PERSON, ORGANIZATION, LOCATION, MONEY, DATE

Text: {text}"""

response = client.models.generate_content(model=MODEL, contents=prompt)
```

```json
{
  "PERSON": ["Tim Cook"],
  "ORGANIZATION": ["Apple"],
  "LOCATION": ["California"],
  "MONEY": ["$500M"],
  "DATE": ["December 15, 2024"]
}
```

---

<!-- _class: section-slide -->

# Vision: Beyond Text

---

# Image Understanding

```python
from PIL import Image

image = Image.open("cat.jpg")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Describe this image in detail.", image]
)

print(response.text)
# "The image shows a grey tabby cat sitting on a windowsill..."
```

<img src="images/week06/vision_capabilities.png" width="650" style="display: block; margin: 0 auto;">

---

# Object Detection with Bounding Boxes

```python
prompt = """Detect objects and return bounding boxes as JSON:
[{"label": "cat", "box_2d": [y0, x0, y1, x1]}]
Coordinates in range [0, 1000]."""

response = client.models.generate_content(
    model=MODEL,
    contents=[prompt, image]
)
```

```json
[
  {"label": "left eye", "box_2d": [519, 625, 615, 713]},
  {"label": "right eye", "box_2d": [426, 400, 526, 510]},
  {"label": "cat", "box_2d": [116, 85, 1000, 885]}
]
```

**Demo**: See notebook for visualization with `supervision` library.

---

# OCR and Document Understanding

```python
receipt_image = Image.open("receipt.jpg")

prompt = """Extract from this receipt:
- Store name
- Items with prices
- Total amount
Return as JSON."""

response = client.models.generate_content(
    model=MODEL, contents=[prompt, receipt_image]
)
```

```json
{
  "store": "ACME STORE",
  "items": [{"name": "Coffee Beans", "price": 24.99}, ...],
  "total": 47.49
}
```

---

# Solving Math from Images

```python
math_image = Image.open("equation.jpg")

response = client.models.generate_content(
    model=MODEL,
    contents=["Solve this step by step.", math_image]
)
```

The model can:
- Read handwritten equations
- Show step-by-step solutions
- Explain each step
- Handle calculus, linear algebra, statistics

**Demo**: See notebook for least-squares derivation example.

---

<!-- _class: section-slide -->

# Audio & Video

---

# Audio Transcription

```python
# Upload audio file
audio_file = client.files.upload(file="interview.mp3")

# Wait for processing
while audio_file.state == 'PROCESSING':
    time.sleep(1)
    audio_file = client.files.get(name=audio_file.name)

# Transcribe
response = client.models.generate_content(
    model=MODEL,
    contents=["Transcribe this audio.", audio_file]
)

print(response.text)
# "Gemini is Google's latest multimodal AI model..."
```

---

# Video Understanding

```python
# Upload video
video_file = client.files.upload(file="scene.mp4")

# Wait for processing...

# Analyze
response = client.models.generate_content(
    model=MODEL,
    contents=["Describe what happens in this video.", video_file]
)
```

The model can:
- Describe scenes and actions
- Identify objects and people
- Answer questions about video content
- Extract temporal information

---

<!-- _class: section-slide -->

# Key Concepts

---

# Temperature: Controlling Randomness

<img src="images/week06/temperature_sampling.png" width="700" style="display: block; margin: 0 auto;">

```python
# Deterministic (same output every time)
config = {"temperature": 0}

# Creative (varied outputs)
config = {"temperature": 1.0}
```

| T = 0 | Factual answers, code |
| T = 0.7 | General conversation |
| T = 1.0+ | Creative writing |

---

# Structured Output with JSON

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    skills: list[str]

response = client.models.generate_content(
    model=MODEL,
    contents="Extract: Sarah, 34, knows Python and SQL",
    config={
        "response_mime_type": "application/json",
        "response_schema": Person
    }
)

person = Person.model_validate_json(response.text)
print(person.name)  # "Sarah"
```

**Guaranteed valid JSON output.**

---

# Prompt Engineering Tips

<img src="images/week06/prompt_engineering_concept.png" width="600" style="display: block; margin: 0 auto;">

1. **Be specific**: "Classify as Positive/Negative" not "What do you think?"
2. **Show examples**: Few-shot beats zero-shot for complex tasks
3. **Specify format**: "Return JSON" or "Reply with one word"
4. **Think step-by-step**: Add "Let's solve this step by step"

---

<!-- _class: section-slide -->

# Connecting to Your ML Pipeline

---

# LLMs in Your ML Pipeline

| Week | Task | How LLMs Help |
|------|------|---------------|
| 1 | Data Collection | Parse unstructured pages |
| 2 | Data Validation | Fix malformed data |
| 3-4 | **Data Labeling** | Auto-label at 10-100x speed |
| 5 | **Augmentation** | Generate paraphrases |
| 6 | Feature Extraction | Embeddings, classification |

<img src="images/week06/llm_pipeline_integration.png" width="800" style="display: block; margin: 0 auto;">

---

# Example: Batch Labeling

```python
def label_batch(reviews: list[str]) -> list[dict]:
    prompt = f"""Classify each review as Positive/Negative/Neutral.
Return JSON: [{{"text": "...", "sentiment": "..."}}]

Reviews:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(reviews))}"""

    response = client.models.generate_content(
        model=MODEL, contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

# Label 50 reviews in ONE API call
results = label_batch(reviews[:50])
```

---

# Example: Text Augmentation

```python
def augment(text: str, n: int = 3) -> list[str]:
    prompt = f"""Generate {n} paraphrases. Same meaning, different words.
Return as JSON array.

Text: "{text}" """

    response = client.models.generate_content(
        model=MODEL, contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

augment("The movie was fantastic!")
# ["The film was excellent!",
#  "What a wonderful movie!",
#  "I really loved this film!"]
```

---

# Cost Optimization

<img src="images/week06/cost_optimization.png" width="650" style="display: block; margin: 0 auto;">

1. **Use smaller models**: `gemini-2.0-flash` vs `gemini-2.0-pro`
2. **Batch requests**: 50 items in one call, not 50 calls
3. **Shorter prompts**: "Sentiment:" not "Please classify the sentiment..."
4. **Cache responses**: Same input = same output (at T=0)

---

# Summary

**What we covered:**

| Topic | Key Takeaway |
|-------|--------------|
| Text | Zero-shot & few-shot classification |
| Vision | Object detection, OCR, VQA |
| Audio | Transcription with speaker labels |
| Video | Scene understanding |
| Documents | PDF extraction with structure |

**Main Resource**: [nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)

---

# Lab: Hands-On with the Notebook

**What you'll do:**

1. **Setup** (10 min): Get API key, run setup cells
2. **Text Tasks** (20 min): Sentiment, NER, summarization
3. **Vision Tasks** (30 min): Object detection, OCR, chart analysis
4. **Audio/Video** (20 min): Transcription, video understanding
5. **Build Your Own** (40 min): Create an application

**Get your key now**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Let's Code!

Open the notebook and follow along

[Tutorial Notebook](https://nipunbatra.github.io/blog/posts/2025-12-01-gemini-api-multimodal.html)
