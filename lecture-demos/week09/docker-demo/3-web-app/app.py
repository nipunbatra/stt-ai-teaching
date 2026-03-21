"""
Gradio Spam Classifier — used for the Docker demo.
This is a self-contained version (model trained inline, no external files).

    python app.py              # run locally
    docker build -t spam-app . # build Docker image
    docker run -p 7860:7860 spam-app  # run in container
"""
import gradio as gr
import joblib
import os

# Train and save model if it doesn't exist yet
MODEL_PATH = "spam_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Training model (first run)...")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    categories = ["rec.sport.baseball", "rec.autos",
                  "talk.politics.guns", "talk.religion.misc"]
    data = fetch_20newsgroups(subset="all", categories=categories,
                              random_state=42)
    labels = [1 if data.target_names[t].startswith("talk") else 0
              for t in data.target]

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, stop_words="english"),
        MultinomialNB()
    )
    pipeline.fit(data.data, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved.")

model = joblib.load(MODEL_PATH)


def classify(text):
    if not text.strip():
        return {"Not Spam": 0.5, "Spam": 0.5}
    proba = model.predict_proba([text])[0]
    return {"Not Spam": float(proba[0]), "Spam": float(proba[1])}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Message", placeholder="Type a message...", lines=4),
    outputs=gr.Label(label="Prediction"),
    title="Spam Classifier",
    description="Is this message spam or not?",
    examples=[
        ["WINNER!! You have been selected to receive a free iPhone! Click NOW!"],
        ["Hey, are we still meeting for lunch tomorrow at noon?"],
        ["URGENT: Your account has been compromised. Send password to verify."],
        ["The baseball game last night was incredible, great home run!"],
    ],
)

if __name__ == "__main__":
    demo.launch()
