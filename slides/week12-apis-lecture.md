---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->

# APIs & Model Demos

## Week 12 · CS 203: Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are

```
Week 8-9:   Version code + environment         ✓
Week 10:    Track experiments                   ✓
Week 11:    Automate with CI/CD                 ✓
Week 12:    SHIP IT                             ← you are here
Week 13:    Make it fast and small
```

Your model trains, tests pass, CI is green.

**But your model lives in a Jupyter notebook. How does anyone USE it?**

---

# From Notebook to Product

```
Jupyter Notebook        →  Nobody can use it (except you)
        ↓
Python Script           →  Developers can run it
        ↓
REST API (FastAPI)      →  Any app can call it
        ↓
Demo (Gradio/Streamlit) →  Anyone with a browser can use it
        ↓
Deployed (Docker+Cloud) →  Available 24/7 worldwide
```

**Today:** Build a REST API and a demo UI for your model.

---

<!-- _class: lead -->

# Part 1: Theory — How Web APIs Work

---

# What Is an API?

**Application Programming Interface** — a contract between two programs.

```
Your App          HTTP Request           ML Service
(frontend,    ─────────────────────>    (your model
 mobile app,   POST /predict             running on
 another       {"budget": 50, ...}       a server)
 service)
              <─────────────────────
               200 OK
               {"prediction": "hit"}
```

**The client sends a request. The server sends a response.**

No shared code. No shared language. Just HTTP + JSON.

---

# HTTP: The Protocol

**Every web request has:**

| Component | Example |
|-----------|---------|
| **Method** | GET, POST, PUT, DELETE |
| **URL** | `http://localhost:8000/predict` |
| **Headers** | `Content-Type: application/json` |
| **Body** (optional) | `{"budget": 50, "runtime": 120}` |

**Every response has:**

| Component | Example |
|-----------|---------|
| **Status code** | 200, 404, 500 |
| **Headers** | `Content-Type: application/json` |
| **Body** | `{"prediction": "hit", "confidence": 0.85}` |

---

# HTTP Methods

| Method | Purpose | Idempotent? | Body? |
|--------|---------|-------------|-------|
| **GET** | Read/retrieve data | Yes | No |
| **POST** | Create/send data | No | Yes |
| **PUT** | Update/replace data | Yes | Yes |
| **DELETE** | Remove data | Yes | No |

**For ML APIs, you mostly use:**
- `GET /` — health check ("is the server alive?")
- `GET /info` — model metadata
- `POST /predict` — send features, get prediction
- `POST /predict/batch` — send many samples

---

# HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| **200** | OK | Request succeeded |
| **201** | Created | New resource created |
| **400** | Bad Request | Malformed request |
| **404** | Not Found | Endpoint doesn't exist |
| **422** | Unprocessable Entity | Valid JSON but invalid data |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Bug in your code |

**Your API should return appropriate status codes.** Don't return 200 for everything!

---

# JSON: The Data Format

```json
{
  "budget": 50.0,
  "runtime": 120,
  "genres": ["action", "drama"],
  "director": {
    "name": "Nolan",
    "experience_years": 25
  }
}
```

**JSON is the universal language of APIs:**
- Human-readable
- Every programming language can parse it
- Python: `json.loads()` / `json.dumps()`
- Maps directly to Python dicts/lists

---

# Online vs Batch Inference

**Online (real-time):**
```
User clicks → API call → Model predicts → Response (< 100ms)
```
- One prediction at a time
- Low latency required
- Example: "Is this transaction fraudulent?" (need answer NOW)

**Batch:**
```
Upload 10,000 rows → Process overnight → Download results
```
- Many predictions at once
- Latency doesn't matter
- Example: "Score all customers for churn risk" (report due Monday)

**Today we build online inference.** Batch = just a script with a loop.

---

# Stateless vs Stateful Services

**Stateless:** Each request is independent. Server doesn't remember previous requests.

```python
# Stateless — GOOD
@app.post("/predict")
def predict(features):
    return model.predict(features)  # same input → same output

# Stateful — BAD for scaling
@app.post("/predict")
def predict(features):
    self.history.append(features)   # server "remembers"
    if len(self.history) > 10:
        # behavior changes based on past requests!
```

**Why stateless?**
- Scale: add 10 servers, any can handle any request
- Restart: no state to lose
- Test: no hidden dependencies between requests

---

<!-- _class: lead -->

# Part 2: FastAPI

*Build an API in minutes*

<!-- ⌨ TERMINAL → Acts 1-3: FastAPI basics, model serving, testing -->

---

# Why FastAPI?

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Speed | WSGI (slower) | ASGI (async, fast) |
| Type validation | Manual | Automatic (Pydantic) |
| Auto-docs | No | Yes (`/docs` Swagger UI) |
| Type hints | Optional | Required and useful |
| Error messages | Generic | Detailed, structured |

**FastAPI = Flask's modern replacement for ML serving.**

---

# Hello World with FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/hello/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}
```

```bash
pip install fastapi uvicorn
uvicorn app:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (auto-generated!)
```

**`--reload`** = auto-restart when code changes. Use during development.

---

# Pydantic: Input Validation

```python
from pydantic import BaseModel, Field

class MovieFeatures(BaseModel):
    budget: float = Field(..., gt=0, description="Budget in millions")
    runtime: float = Field(..., gt=0, le=300, description="Minutes")
    genre_action: int = Field(..., ge=0, le=1)
    genre_comedy: int = Field(..., ge=0, le=1)
    director_experience: int = Field(..., ge=0)

class Prediction(BaseModel):
    success: bool
    confidence: float
    label: str
```

**Send invalid data → automatic 422 error with clear message.**

```json
{"detail": [{"loc": ["body", "budget"], "msg": "ensure this value is greater than 0"}]}
```

No manual `if budget <= 0: return error` code needed!

---

# Serving a Model

```python
from fastapi import FastAPI
import joblib
import numpy as np

# Load model ONCE at startup (not per request!)
model = joblib.load("model.pkl")

app = FastAPI(title="Movie Predictor API")

@app.post("/predict", response_model=Prediction)
def predict(features: MovieFeatures):
    X = np.array([[
        features.budget,
        features.runtime,
        features.genre_action,
        features.genre_comedy,
        features.director_experience,
    ]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return Prediction(
        success=bool(pred),
        confidence=float(max(proba)),
        label="Hit!" if pred else "Flop",
    )
```

---

# Batch Endpoint

```python
@app.post("/predict/batch")
def predict_batch(features_list: list[MovieFeatures]):
    """Predict for multiple movies at once."""
    X = np.array([
        [f.budget, f.runtime, f.genre_action,
         f.genre_comedy, f.director_experience]
        for f in features_list
    ])
    preds = model.predict(X)
    probas = model.predict_proba(X)

    return [
        {
            "success": bool(p),
            "confidence": float(max(pr)),
            "label": "Hit!" if p else "Flop",
        }
        for p, pr in zip(preds, probas)
    ]
```

**Batch is more efficient** — one forward pass instead of N separate ones.

---

# Error Handling

```python
from fastapi import FastAPI, HTTPException

@app.post("/predict")
def predict(features: MovieFeatures):
    try:
        X = np.array([[features.budget, ...]])
        pred = model.predict(X)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/{model_name}")
def get_model(model_name: str):
    if model_name not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    return available_models[model_name]
```

---

# Auto-Generated Documentation

FastAPI creates **Swagger UI** at `/docs` automatically:

```
┌──────────────────────────────────────────┐
│  Movie Predictor API                      │
│                                          │
│  GET  /           Health check           │
│  GET  /info       Model information      │
│  POST /predict    Predict movie success  │
│  POST /predict/batch  Batch prediction   │
│                                          │
│  [Try it out] button for each endpoint   │
│  Shows request/response schemas          │
│  Test with real data in the browser      │
└──────────────────────────────────────────┘
```

**No Postman needed.** No separate docs to maintain. The code IS the documentation.

Also available: ReDoc at `/redoc` (alternative UI).

---

# Testing Your API

```python
# test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "budget": 0.5, "runtime": 0.5,
        "genre_action": 1, "genre_comedy": 0,
        "director_experience": 0.5,
    })
    assert response.status_code == 200
    assert "confidence" in response.json()
    assert 0 <= response.json()["confidence"] <= 1

def test_invalid_input():
    response = client.post("/predict", json={"budget": -1})
    assert response.status_code == 422
```

---

# Model Serialization

**Save and load models for serving:**

```python
# sklearn — use joblib
import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

# PyTorch — save state_dict (not the whole model)
torch.save(model.state_dict(), "model.pth")
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()  # important: set to eval mode!

# ONNX — portable format (Week 13)
torch.onnx.export(model, dummy_input, "model.onnx")
```

| Format | Pros | Cons |
|--------|------|------|
| joblib/pickle | Easy, works with sklearn | Python-only, security risk |
| torch state_dict | Standard for PyTorch | Need model class definition |
| ONNX | Portable, fast inference | Doesn't support all ops |

---

<!-- _class: lead -->

# Part 3: Gradio & Streamlit

*Demo UIs in 10 lines of code*

<!-- ⌨ TERMINAL → Acts 4-5: Gradio and Streamlit demos -->

---

# Gradio: Instant ML Demo

```python
import gradio as gr
import joblib

model = joblib.load("model.pkl")

def predict(budget, runtime, genre):
    features = [budget, runtime, int(genre == "Action")]
    pred = model.predict([features])
    return "Hit!" if pred[0] == 1 else "Flop"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Budget ($M)"),
        gr.Slider(60, 240, label="Runtime (min)"),
        gr.Dropdown(["Action", "Comedy", "Drama"], label="Genre"),
    ],
    outputs=gr.Text(label="Prediction"),
    title="Movie Success Predictor",
)
demo.launch()
```

**Web UI with form inputs, validation, and sharing — in 15 lines.**

---

# Gradio: Rich Input Types

```python
# Image classification
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5),
)

# Text generation
gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter prompt..."),
    outputs=gr.Textbox(lines=5),
)

# Audio classification
gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(source="microphone"),
    outputs=gr.Label(),
)
```

**Gradio handles file upload, microphone access, webcam — all built in.**

---

# Gradio: Sharing

```python
# Local only
demo.launch()
# → http://127.0.0.1:7860

# Public URL (free, 72h)
demo.launch(share=True)
# → https://abc123.gradio.live

# Permanent hosting on Hugging Face Spaces
# 1. Create a Space at huggingface.co/spaces
# 2. Push your code + model
# 3. Free GPU available for ML models!
```

**`share=True`** creates a public URL through Gradio's tunnel service. Great for showing demos to non-technical people.

---

# Streamlit: Dashboard-Style Demo

```python
import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Movie Success Predictor")

col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("Budget ($M)", min_value=0.1, value=50.0)
    runtime = st.slider("Runtime (min)", 60, 240, 120)
with col2:
    genre = st.selectbox("Genre", ["Action", "Comedy", "Drama"])

if st.button("Predict"):
    features = [budget, runtime, int(genre == "Action")]
    pred = model.predict([features])
    if pred[0] == 1:
        st.success("Predicted: Hit!")
    else:
        st.error("Predicted: Flop")
```

```bash
streamlit run app.py
```

---

# Streamlit: Data Dashboards

```python
import streamlit as st
import pandas as pd

st.title("Model Performance Dashboard")

# Sidebar for controls
model_name = st.sidebar.selectbox("Model", ["v1", "v2", "v3"])
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "0.87", "+0.02")
col2.metric("F1 Score", "0.83", "+0.05")
col3.metric("Latency", "12ms", "-3ms")

# Charts
st.line_chart(training_history)
st.bar_chart(feature_importances)

# Data table
st.dataframe(predictions_df)
```

**Streamlit excels at dashboards with charts, tables, and interactive controls.**

---

# Gradio vs Streamlit

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| **Best for** | ML demos, sharing | Dashboards, data apps |
| **Sharing** | `share=True` → public URL | Streamlit Cloud |
| **Rich inputs** | Image, audio, video, 3D | Standard widgets |
| **Layout** | Function-based (input → output) | Script-based (top to bottom) |
| **Rerun model** | On submit | On any widget change |
| **Hugging Face** | First-class support | Supported |

**Quick demo for a meeting?** → Gradio
**Internal dashboard for the team?** → Streamlit

---

<!-- _class: lead -->

# Part 4: Deployment

<!-- ⌨ TERMINAL → Act 6: Docker + deployment options -->

---

# Dockerfile for FastAPI

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t movie-api .
docker run -p 8000:8000 movie-api
curl http://localhost:8000/docs    # Swagger UI works!
```

---

# Deployment Options

| Platform | Cost | Complexity | Best For |
|----------|------|-----------|----------|
| **Hugging Face Spaces** | Free | Very low | Gradio/Streamlit demos |
| **Railway / Render** | Free tier | Low | Simple APIs |
| **Google Cloud Run** | Pay-per-use | Medium | Production APIs |
| **AWS Lambda** | Pay-per-call | Medium | Serverless, bursty traffic |
| **Kubernetes** | $$$ | High | Large-scale production |

**For class projects:** Hugging Face Spaces (free, easy, supports GPU).

**For production:** Cloud Run or similar container service.

---

# Key Takeaways

1. **REST APIs** let any application use your model
   - POST for predictions, GET for health checks
   - Stateless = easy to scale and test
   - Return appropriate status codes

2. **FastAPI** is the modern choice for ML APIs
   - Auto validation (Pydantic), auto docs (`/docs`)
   - TestClient for automated testing
   - Load model once at startup, not per request

3. **Gradio/Streamlit** create demo UIs in minutes
   - Gradio: quick ML demos with `share=True` for public links
   - Streamlit: richer dashboards with charts

4. **Docker** makes your API portable and deployable

**Next week:** Model profiling & quantization — make it fast!

---

<!-- _class: lead -->

# Questions?

**Exam-relevant concepts:**
- HTTP methods (GET/POST/PUT/DELETE) and when to use each
- Status codes: 200, 400, 404, 422, 500
- Online vs batch inference — tradeoffs
- Stateless vs stateful services — why stateless for ML
- JSON as the universal data format
- Model serialization: joblib, torch.save, ONNX
- Pydantic for input validation
