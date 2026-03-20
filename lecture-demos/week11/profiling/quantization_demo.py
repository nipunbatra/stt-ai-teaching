"""
Demo: Quantize an sklearn model with ONNX
==========================================
Train a large MLP, convert to ONNX, quantize to INT8,
compare size and accuracy.

    pip install scikit-learn skl2onnx onnxruntime
    python quantization_demo.py
"""
import os
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ── Train a deliberately large model ──
print("Training a large MLP (3 hidden layers of 500 neurons)...")
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(500, 500, 500),
                    max_iter=50, random_state=42)
clf.fit(X_train, y_train)
sklearn_acc = clf.score(X_test, y_test)
print(f"sklearn accuracy: {sklearn_acc:.1%}")

# ── Convert to ONNX (FP32) ──
print("\nConverting to ONNX (FP32)...")
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

fp32_path = "/tmp/model_fp32.onnx"
with open(fp32_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# ── Quantize to INT8 ──
print("Quantizing to INT8...")
from onnxruntime.quantization import quantize_dynamic, QuantType

int8_path = "/tmp/model_int8.onnx"
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)

# ── Compare sizes ──
size_fp32 = os.path.getsize(fp32_path) / (1024 * 1024)
size_int8 = os.path.getsize(int8_path) / (1024 * 1024)

# ── Compare accuracy ──
import onnxruntime as ort

sess_fp32 = ort.InferenceSession(fp32_path)
sess_int8 = ort.InferenceSession(int8_path)

input_name = sess_fp32.get_inputs()[0].name
preds_fp32 = sess_fp32.run(None, {input_name: X_test})[0]
preds_int8 = sess_int8.run(None, {input_name: X_test})[0]

acc_fp32 = np.mean(preds_fp32 == y_test)
acc_int8 = np.mean(preds_int8 == y_test)

# ── Compare speed ──
n_runs = 200

start = time.time()
for _ in range(n_runs):
    sess_fp32.run(None, {input_name: X_test})
time_fp32 = (time.time() - start) / n_runs * 1000

start = time.time()
for _ in range(n_runs):
    sess_int8.run(None, {input_name: X_test})
time_int8 = (time.time() - start) / n_runs * 1000

# ── Results ──
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"  {'':20s} {'FP32':>10s} {'INT8':>10s}")
print(f"  {'Size':20s} {size_fp32:>9.2f}M {size_int8:>9.2f}M")
print(f"  {'Accuracy':20s} {acc_fp32:>9.1%} {acc_int8:>9.1%}")
print(f"  {'Inference (ms)':20s} {time_fp32:>9.2f} {time_int8:>9.2f}")
print(f"\n  Size reduction: {(1 - size_int8/size_fp32)*100:.0f}%")
print(f"  Accuracy drop:  {(acc_fp32 - acc_int8)*100:.1f}%")
print(f"\n→ {(1 - size_int8/size_fp32)*100:.0f}% smaller with minimal accuracy loss!")
print("  This is what quantization buys you.")
