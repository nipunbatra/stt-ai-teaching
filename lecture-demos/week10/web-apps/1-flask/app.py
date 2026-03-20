"""
Flask Spam Classifier — The "traditional" way.
You write BOTH the Python backend AND the HTML frontend.

    pip install flask
    python app.py

Then open http://localhost:5000
"""
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model ONCE at startup (not inside the route!)
model = joblib.load("../0-model/spam_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        proba = model.predict_proba([text])[0]
        pred_class = model.predict([text])[0]
        prediction = "Spam" if pred_class == 1 else "Not Spam"
        confidence = f"{max(proba) * 100:.1f}%"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           text=text)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
