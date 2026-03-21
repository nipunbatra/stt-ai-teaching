"""
Demo 5: Environment variables — configure without changing code.

The same Docker image behaves differently based on ENV vars:
  - MODEL_TYPE: "rf" or "svm"
  - N_ESTIMATORS: number of trees (for RF)
  - APP_TITLE: shown in the Gradio UI

    docker run -p 7861:7860 env-demo
    docker run -p 7861:7860 -e MODEL_TYPE=svm -e APP_TITLE="SVM Classifier" env-demo
    docker run -p 7861:7860 -e N_ESTIMATORS=500 -e APP_TITLE="Big Forest" env-demo

Same image, different behavior. No rebuild needed!
"""
import os
import gradio as gr
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ── Read configuration from environment variables ──
MODEL_TYPE = os.environ.get("MODEL_TYPE", "rf")
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "100"))
APP_TITLE = os.environ.get("APP_TITLE", "Digit Classifier")

print(f"Config: MODEL_TYPE={MODEL_TYPE}, N_ESTIMATORS={N_ESTIMATORS}")
print(f"Title:  {APP_TITLE}")

# ── Train model based on config ──
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

if MODEL_TYPE == "svm":
    model = SVC(probability=True, random_state=42)
    model_name = "SVM"
else:
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
    model_name = f"RandomForest (n={N_ESTIMATORS})"

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")


def classify(image_data):
    """Takes a 8x8 grayscale image (from sketchpad), returns digit prediction."""
    import numpy as np
    from PIL import Image

    if image_data is None:
        return {str(i): 0.0 for i in range(10)}

    # Handle composite sketchpad output (dict with 'composite' key)
    if isinstance(image_data, dict):
        img = image_data.get("composite", image_data.get("image"))
        if img is None:
            return {str(i): 0.0 for i in range(10)}
        image_data = img

    img = Image.fromarray(image_data).convert("L").resize((8, 8))
    pixels = np.array(img).reshape(1, -1).astype(float)
    # Invert (white bg, dark strokes → sklearn digits convention)
    pixels = 16.0 - (pixels / 255.0 * 16.0)

    proba = model.predict_proba(pixels)[0]
    return {str(i): float(proba[i]) for i in range(10)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Sketchpad(label="Draw a digit", canvas_size=(150, 150)),
    outputs=gr.Label(label="Prediction", num_top_classes=3),
    title=APP_TITLE,
    description=f"Model: {model_name} | Accuracy: {accuracy:.1%} | "
                f"Configured via environment variables",
)

if __name__ == "__main__":
    demo.launch()
