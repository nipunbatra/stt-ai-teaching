"""
Streamlit Spam Classifier — The "data science" way.
No HTML needed. Pure Python. Hot-reload.

    pip install streamlit
    streamlit run app.py

Notice: NO templates folder. NO HTML. Just Python.
"""
import streamlit as st
import joblib

# Load model once (Streamlit caches across reruns)
@st.cache_resource
def load_model():
    return joblib.load("../0-model/spam_model.pkl")

model = load_model()

st.title("Spam Classifier")
st.write("Enter a message to classify:")

text = st.text_area("Message", placeholder="Type or paste a message...")

if st.button("Classify"):
    if text.strip():
        proba = model.predict_proba([text])[0]
        pred_class = model.predict([text])[0]
        label = "Spam" if pred_class == 1 else "Not Spam"
        confidence = max(proba) * 100

        if pred_class == 1:
            st.error(f"**{label}** (confidence: {confidence:.1f}%)")
        else:
            st.success(f"**{label}** (confidence: {confidence:.1f}%)")
    else:
        st.warning("Please enter some text.")
