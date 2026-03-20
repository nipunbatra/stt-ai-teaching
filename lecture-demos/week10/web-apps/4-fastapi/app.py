"""
FastAPI Spam Classifier — The "production API" way.
No UI — just a JSON API that other code can call.

    pip install fastapi uvicorn
    uvicorn app:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "WINNER! Free iPhone!"}'

Or open http://localhost:8000/docs for auto-generated Swagger UI.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Classifier API")

# Load model ONCE at startup
model = joblib.load("../0-model/spam_model.pkl")


class Message(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict


@app.post("/predict", response_model=Prediction)
def predict(msg: Message):
    proba = model.predict_proba([msg.text])[0]
    pred_class = model.predict([msg.text])[0]
    return Prediction(
        label="Spam" if pred_class == 1 else "Not Spam",
        confidence=round(float(max(proba)) * 100, 1),
        probabilities={"Not Spam": round(float(proba[0]), 4),
                       "Spam": round(float(proba[1]), 4)},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
