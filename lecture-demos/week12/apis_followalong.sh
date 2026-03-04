#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  APIs & Model Demos — Follow-Along Guide                                ║
# ║  Week 12 · CS 203 · Software Tools and Techniques for AI               ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   Your model trains, tests pass, CI is green. But it lives in a notebook.
#   Today you turn it into something anyone can use: a REST API with FastAPI,
#   a demo with Gradio, and a Docker container ready to deploy.
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
# ║  📽  PROJECTOR → Slides 2-3: From Notebook to Product                   ║
# ║     "Your model lives in a notebook. How does anyone USE it?"           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: Setup — Train and Save a Model                        ~8 min   │
# └──────────────────────────────────────────────────────────────────────────┘

mkdir -p ~/api-demo && cd ~/api-demo

python -m venv .venv
source .venv/bin/activate

pip install scikit-learn numpy fastapi uvicorn joblib gradio streamlit

# Train and save a model:

cat > train.py << 'PYEOF'
"""Train and save the movie predictor model."""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(500, 5)  # budget, runtime, genre_action, genre_comedy, director_exp
y = (X[:, 0] * 2 + X[:, 2] - X[:, 3] + np.random.randn(500) * 0.3 > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")

# Save the model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")

# Also save feature names for documentation
feature_names = ["budget", "runtime", "genre_action", "genre_comedy", "director_experience"]
joblib.dump(feature_names, "feature_names.pkl")
PYEOF

python train.py

# >> Model accuracy: 0.xxx
# >> Model saved to model.pkl



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 4-8: REST Theory (methods, status codes,        ║
# ║     online vs batch, stateless). Then back to terminal.                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: FastAPI — Your First API                              ~15 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > app.py << 'PYEOF'
"""Movie predictor REST API."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Load model at startup (not per request!)
model = joblib.load("model.pkl")

app = FastAPI(
    title="Movie Success Predictor",
    description="Predict whether a movie will be a hit or flop",
    version="1.0.0",
)


class MovieFeatures(BaseModel):
    """Input features for prediction."""
    budget: float = Field(..., gt=0, description="Budget (0-1 scale)")
    runtime: float = Field(..., gt=0, le=1, description="Runtime (0-1 scale)")
    genre_action: int = Field(..., ge=0, le=1, description="1 if action genre")
    genre_comedy: int = Field(..., ge=0, le=1, description="1 if comedy genre")
    director_experience: float = Field(..., ge=0, le=1, description="Director experience (0-1)")


class Prediction(BaseModel):
    """Prediction response."""
    success: bool
    confidence: float
    label: str


@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "random_forest_v1"}


@app.get("/info")
def model_info():
    """Model information."""
    return {
        "model_type": "RandomForestClassifier",
        "n_features": 5,
        "features": ["budget", "runtime", "genre_action", "genre_comedy", "director_experience"],
    }


@app.post("/predict", response_model=Prediction)
def predict(features: MovieFeatures):
    """Predict movie success."""
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
        label="Hit!" if pred == 1 else "Flop",
    )


@app.post("/predict/batch")
def predict_batch(features_list: list[MovieFeatures]):
    """Batch prediction for multiple movies."""
    X = np.array([[f.budget, f.runtime, f.genre_action, f.genre_comedy, f.director_experience] for f in features_list])
    preds = model.predict(X)
    probas = model.predict_proba(X)
    return [
        {"success": bool(p), "confidence": float(max(pr)), "label": "Hit!" if p else "Flop"}
        for p, pr in zip(preds, probas)
    ]
PYEOF

# Start the server (in background):

uvicorn app:app --reload --port 8000 &
SERVER_PID=$!
sleep 2

# Test the health endpoint:

curl -s http://localhost:8000/ | python -m json.tool

# >> {
# >>     "status": "healthy",
# >>     "model": "random_forest_v1"
# >> }

# Test model info:

curl -s http://localhost:8000/info | python -m json.tool

# Test a prediction:

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"budget": 0.8, "runtime": 0.5, "genre_action": 1, "genre_comedy": 0, "director_experience": 0.7}' \
  | python -m json.tool

# >> {
# >>     "success": true,
# >>     "confidence": 0.85,
# >>     "label": "Hit!"
# >> }

# Test batch prediction:

curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"budget": 0.8, "runtime": 0.5, "genre_action": 1, "genre_comedy": 0, "director_experience": 0.7},
    {"budget": 0.1, "runtime": 0.3, "genre_action": 0, "genre_comedy": 1, "director_experience": 0.1}
  ]' | python -m json.tool

# >> [{"success": true, ...}, {"success": false, ...}]

# Test validation — send invalid data:

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"budget": -5}' | python -m json.tool

# >> 422 Unprocessable Entity — Pydantic catches the invalid input!

# Check auto-generated docs:

echo "Open http://localhost:8000/docs in your browser — interactive Swagger UI!"

kill $SERVER_PID 2>/dev/null



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: Testing Your API                                       ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘

pip install httpx pytest

cat > test_api.py << 'PYEOF'
"""Test the prediction API."""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    response = client.post("/predict", json={
        "budget": 0.5, "runtime": 0.5, "genre_action": 1,
        "genre_comedy": 0, "director_experience": 0.5,
    })
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_invalid():
    response = client.post("/predict", json={"budget": -1})
    assert response.status_code == 422
PYEOF

pytest test_api.py -v

# >> test_api.py::test_health PASSED
# >> test_api.py::test_predict PASSED
# >> test_api.py::test_predict_invalid PASSED



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 13-15: Gradio and Streamlit                     ║
# ║     Show the code comparison. Then build both live.                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: Gradio — Demo in 10 Lines                            ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > app_gradio.py << 'PYEOF'
"""Gradio demo for movie predictor."""
import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.pkl")


def predict(budget, runtime, genre_action, genre_comedy, director_exp):
    features = np.array([[budget, runtime, genre_action, genre_comedy, director_exp]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba)
    label = "Hit!" if pred == 1 else "Flop"
    return f"{label} (confidence: {confidence:.1%})"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 1, value=0.5, label="Budget (0=low, 1=high)"),
        gr.Slider(0, 1, value=0.5, label="Runtime (0=short, 1=long)"),
        gr.Checkbox(label="Action Genre"),
        gr.Checkbox(label="Comedy Genre"),
        gr.Slider(0, 1, value=0.5, label="Director Experience"),
    ],
    outputs=gr.Text(label="Prediction"),
    title="Movie Success Predictor",
    description="Predict whether a movie will be a hit or flop based on its features.",
)

if __name__ == "__main__":
    demo.launch()
PYEOF

python app_gradio.py &
GRADIO_PID=$!
sleep 3

# >> Running on local URL:  http://127.0.0.1:7860
# Open in browser — interactive UI!

echo "Open http://localhost:7860 — slide the sliders, click Submit."

kill $GRADIO_PID 2>/dev/null

# For sharing publicly (great for demos!):
# demo.launch(share=True)  → gives you a public *.gradio.live URL



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: Streamlit — Dashboard Demo                            ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > app_streamlit.py << 'PYEOF'
"""Streamlit dashboard for movie predictor."""
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Movie Success Predictor")
st.write("Enter movie features to predict success.")

col1, col2 = st.columns(2)

with col1:
    budget = st.slider("Budget", 0.0, 1.0, 0.5)
    runtime = st.slider("Runtime", 0.0, 1.0, 0.5)
    director_exp = st.slider("Director Experience", 0.0, 1.0, 0.5)

with col2:
    genre_action = st.checkbox("Action Genre")
    genre_comedy = st.checkbox("Comedy Genre")

if st.button("Predict", type="primary"):
    features = np.array([[budget, runtime, int(genre_action), int(genre_comedy), director_exp]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba)

    if pred == 1:
        st.success(f"Predicted: Hit! (confidence: {confidence:.1%})")
        st.balloons()
    else:
        st.error(f"Predicted: Flop (confidence: {confidence:.1%})")

    # Show feature importance
    st.subheader("Feature Importances")
    names = ["Budget", "Runtime", "Action", "Comedy", "Director Exp"]
    importances = model.feature_importances_
    st.bar_chart(dict(zip(names, importances)))
PYEOF

echo "Run: streamlit run app_streamlit.py"
echo "Opens at http://localhost:8501"

# streamlit run app_streamlit.py &
# STREAMLIT_PID=$!
# sleep 3
# kill $STREAMLIT_PID 2>/dev/null



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 17: Dockerizing Your API                         ║
# ║     Show the Dockerfile. Then build and run.                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: Dockerize the API                                    ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

pip freeze > requirements.txt

cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY feature_names.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > .dockerignore << 'EOF'
.venv/
__pycache__/
*.pyc
.git/
app_gradio.py
app_streamlit.py
EOF

# Build and run:

docker build -t movie-api .

docker run -d -p 8000:8000 --name movie-api movie-api

sleep 2

# Test it:

curl -s http://localhost:8000/ | python -m json.tool

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"budget": 0.8, "runtime": 0.5, "genre_action": 1, "genre_comedy": 0, "director_experience": 0.7}' \
  | python -m json.tool

# >> Same results as running locally!

# Stop the container:

docker stop movie-api && docker rm movie-api



# ═══════════════════════════════════════════════════════════════════════════
# WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════
#
# What we covered today:
#
#   Act 1: Train and save a model (joblib)
#   Act 2: FastAPI — REST API with auto-docs
#   Act 3: Testing your API (TestClient)
#   Act 4: Gradio — instant ML demo
#   Act 5: Streamlit — dashboard-style demo
#   Act 6: Docker — containerize the API
#
# The progression:
#   Notebook → script → API → demo → Docker → deploy
#
# Next week: Model profiling & quantization
# ═══════════════════════════════════════════════════════════════════════════

cd ~
deactivate 2>/dev/null
# rm -rf ~/api-demo   # uncomment to clean up
