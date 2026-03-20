"""
Gradio Spam Classifier — The "ML demo" way.
Define inputs and outputs. Gradio builds the UI.

    pip install gradio
    python app.py

Bonus: Gradio auto-generates an API endpoint too!
Click "Use via API" at the bottom of the interface.
"""
import gradio as gr
import joblib

model = joblib.load("../0-model/spam_model.pkl")


def classify(text):
    """Returns label and confidence dict for Gradio's Label component."""
    proba = model.predict_proba([text])[0]
    return {"Not Spam": float(proba[0]), "Spam": float(proba[1])}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Message", placeholder="Type or paste a message...",
                      lines=4),
    outputs=gr.Label(label="Prediction"),
    title="Spam Classifier",
    description="Enter a message to classify as Spam or Not Spam.",
    examples=[
        ["WINNER!! You have been selected to receive a free iPhone! Click here NOW!"],
        ["Hey, are we still meeting for lunch tomorrow at noon?"],
        ["URGENT: Your account has been compromised. Send your password to verify."],
        ["The baseball game last night was incredible, did you see that home run?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
