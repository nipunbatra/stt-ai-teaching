"""Generate simple Colab-friendly notebooks for Week 11 concepts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "colab-notebooks"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_notebook(filename: str, title: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": filename},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUTDIR / filename
    path.write_text(json.dumps(notebook, indent=2) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    write_notebook(
        "01-floating-point-basics.ipynb",
        "Floating Point Basics",
        [
            markdown_cell(
                """# Floating Point Basics

This notebook is a first look at how computers store numbers.

Goals:
- see why `0.1 + 0.2` is not exactly `0.3`
- inspect FP32 bit patterns
- compare FP32, FP16, and an INT8-style approximation
"""
            ),
            code_cell(
                """import struct
import numpy as np
import pandas as pd

result = 0.1 + 0.2
print("0.1 + 0.2 =", result)
print("Is it exactly 0.3?", result == 0.3)

for value in [0.1, 0.2, 0.3]:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    print(f"{value:>3}: {bits:032b}")
"""
            ),
            code_cell(
                """weights = np.array([0.12, -1.78, 3.14159, 0.004, -0.56], dtype=np.float32)
fp16 = weights.astype(np.float16)

# A simple fake INT8-style approximation using a scale factor.
scale = 32
int8_values = np.clip(np.round(weights * scale), -128, 127).astype(np.int8)
recovered = int8_values.astype(np.float32) / scale

comparison = pd.DataFrame(
    {
        "fp32": weights,
        "fp16": fp16.astype(np.float32),
        "fake_int8": recovered,
    }
)
comparison["fp16_error"] = (comparison["fp32"] - comparison["fp16"]).abs()
comparison["fake_int8_error"] = (comparison["fp32"] - comparison["fake_int8"]).abs()
comparison
"""
            ),
        ],
    )

    write_notebook(
        "02-parameter-count-and-memory.ipynb",
        "Parameter Count and Memory",
        [
            markdown_cell(
                """# Parameter Count and Memory

This notebook turns parameter counts into approximate memory use.

Goals:
- count model parameters
- estimate size in FP32, FP16, and INT8
- see why bigger models cost more memory
"""
            ),
            code_cell(
                """import pandas as pd
import torch
import torch.nn as nn


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


small = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

medium = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

large = nn.Sequential(
    nn.Linear(64, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

rows = []
for name, model in [("small", small), ("medium", medium), ("large", large)]:
    params = count_params(model)
    rows.append(
        {
            "model": name,
            "parameters": params,
            "fp32_kb": params * 4 / 1024,
            "fp16_kb": params * 2 / 1024,
            "int8_kb": params * 1 / 1024,
        }
    )

pd.DataFrame(rows)
"""
            ),
        ],
    )

    write_notebook(
        "03-profiling-basics.ipynb",
        "Profiling Basics",
        [
            markdown_cell(
                """# Profiling Basics

This notebook shows a classic mistake: loading the model on every request.

Goals:
- use `cProfile` to inspect a slow function
- compare a slow and fast implementation
- see why measuring beats guessing
"""
            ),
            code_cell(
                """import cProfile
import io
import joblib
import timeit
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

artifacts = Path("/tmp/week11_profiling")
artifacts.mkdir(exist_ok=True)

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, artifacts / "model.pkl")
pd.DataFrame(X_train).to_csv(artifacts / "data.csv", index=False)

sample = X_test[0]
"""
            ),
            code_cell(
                """def slow_predict(sample):
    data = pd.read_csv(artifacts / "data.csv")
    model = joblib.load(artifacts / "model.pkl")
    return model.predict([sample])[0], data.shape


cached_model = joblib.load(artifacts / "model.pkl")


def fast_predict(sample):
    return cached_model.predict([sample])[0]


profile_buffer = io.StringIO()
profiler = cProfile.Profile()
profiler.enable()
slow_predict(sample)
profiler.disable()
profiler.print_stats(sort="cumulative")
"""
            ),
            code_cell(
                """slow_time = timeit.timeit(lambda: slow_predict(sample), number=10) / 10
fast_time = timeit.timeit(lambda: fast_predict(sample), number=1000) / 1000

print(f"Slow version: {slow_time * 1000:.2f} ms")
print(f"Fast version: {fast_time * 1000:.2f} ms")
print(f"Speedup: {slow_time / fast_time:.1f}x")
"""
            ),
        ],
    )

    write_notebook(
        "04-batching-benchmark.ipynb",
        "Batching Benchmark",
        [
            markdown_cell(
                """# Batching Benchmark

This notebook measures throughput at different batch sizes.

Goals:
- see why batching often improves throughput
- compare latency per example
- connect the result to deployment choices
"""
            ),
            code_cell(
                """import time

import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
model.eval()

batch_sizes = [1, 8, 32, 128, 512]
rows = []

for batch_size in batch_sizes:
    x = torch.randn(batch_size, 256)

    for _ in range(20):
        with torch.no_grad():
            model(x)

    start = time.perf_counter()
    runs = 200
    for _ in range(runs):
        with torch.no_grad():
            model(x)
    avg_ms = (time.perf_counter() - start) * 1000 / runs

    rows.append(
        {
            "batch_size": batch_size,
            "avg_batch_ms": avg_ms,
            "avg_per_example_ms": avg_ms / batch_size,
            "examples_per_second": batch_size / (avg_ms / 1000),
        }
    )

results = pd.DataFrame(rows)
results
"""
            ),
            code_cell(
                """results.plot(
    x="batch_size",
    y=["avg_per_example_ms", "examples_per_second"],
    subplots=True,
    figsize=(8, 6),
    grid=True,
    title=["Latency per example", "Throughput"],
)
"""
            ),
        ],
    )

    write_notebook(
        "05-pytorch-dynamic-quantization.ipynb",
        "PyTorch Dynamic Quantization",
        [
            markdown_cell(
                """# PyTorch Dynamic Quantization

This notebook trains a tiny MLP and then quantizes it to INT8.

Goals:
- train a simple baseline model
- apply dynamic quantization
- compare size, accuracy, and latency
"""
            ),
            code_cell(
                """import io
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic

torch.manual_seed(42)

digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
"""
            ),
            code_cell(
                """class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


def model_size_kb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024


model = SmallMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(8):
    model.train()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

fp32_acc = evaluate(model)
fp32_size = model_size_kb(model)
"""
            ),
            code_cell(
                """quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
int8_acc = evaluate(quantized_model)
int8_size = model_size_kb(quantized_model)

sample_batch = X_test[:256]

def benchmark(model, runs=300):
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(sample_batch)
    return (time.perf_counter() - start) * 1000 / runs


fp32_ms = benchmark(model)
int8_ms = benchmark(quantized_model)

pd.DataFrame(
    [
        {"model": "fp32", "accuracy": fp32_acc, "size_kb": fp32_size, "latency_ms": fp32_ms},
        {"model": "int8", "accuracy": int8_acc, "size_kb": int8_size, "latency_ms": int8_ms},
    ]
)
"""
            ),
        ],
    )

    write_notebook(
        "06-onnx-export-and-quantization.ipynb",
        "ONNX Export and Quantization",
        [
            markdown_cell(
                """# ONNX Export and Quantization

This notebook exports a scikit-learn model to ONNX and then quantizes it.

Goals:
- export a model to ONNX
- run it with ONNX Runtime
- compare FP32 and INT8 ONNX files
"""
            ),
            code_cell(
                """%pip install -q onnx onnxruntime skl2onnx"""
            ),
            code_cell(
                """import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

workdir = Path("/tmp/week11_onnx")
workdir.mkdir(exist_ok=True)

X, y = load_digits(return_X_y=True)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, random_state=42)
clf.fit(X_train, y_train)
"""
            ),
            code_cell(
                """initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

fp32_path = workdir / "digits_fp32.onnx"
fp32_path.write_bytes(onnx_model.SerializeToString())

int8_path = workdir / "digits_int8.onnx"
quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

fp32_session = ort.InferenceSession(str(fp32_path))
int8_session = ort.InferenceSession(str(int8_path))
input_name = fp32_session.get_inputs()[0].name
"""
            ),
            code_cell(
                """def accuracy(session):
    preds = session.run(None, {input_name: X_test})[0]
    return np.mean(preds == y_test)


def benchmark(session, runs=200):
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: X_test})
    return (time.perf_counter() - start) * 1000 / runs


results = pd.DataFrame(
    [
        {
            "model": "onnx_fp32",
            "size_kb": os.path.getsize(fp32_path) / 1024,
            "accuracy": accuracy(fp32_session),
            "latency_ms": benchmark(fp32_session),
        },
        {
            "model": "onnx_int8",
            "size_kb": os.path.getsize(int8_path) / 1024,
            "accuracy": accuracy(int8_session),
            "latency_ms": benchmark(int8_session),
        },
    ]
)
results
"""
            ),
        ],
    )

    write_notebook(
        "07-pruning-basics.ipynb",
        "Pruning Basics",
        [
            markdown_cell(
                """# Pruning Basics

This notebook removes small weights from a model after training.

Goals:
- train a small MLP
- prune a fraction of its weights
- measure sparsity and accuracy
"""
            ),
            code_cell(
                """import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
"""
            ),
            code_cell(
                """class PruneMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


def sparsity(layer):
    zeros = torch.sum(layer.weight == 0).item()
    total = layer.weight.nelement()
    return zeros / total


model = PruneMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(8):
    model.train()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

before_acc = evaluate(model)
"""
            ),
            code_cell(
                """parameters_to_prune = (
    (model.fc1, "weight"),
    (model.fc2, "weight"),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4,
)

after_acc = evaluate(model)

pd.DataFrame(
    [
        {
            "metric": "accuracy_before",
            "value": before_acc,
        },
        {
            "metric": "accuracy_after",
            "value": after_acc,
        },
        {
            "metric": "fc1_sparsity",
            "value": sparsity(model.fc1),
        },
        {
            "metric": "fc2_sparsity",
            "value": sparsity(model.fc2),
        },
    ]
)
"""
            ),
        ],
    )

    write_notebook(
        "08-distillation-basics.ipynb",
        "Distillation Basics",
        [
            markdown_cell(
                """# Distillation Basics

This notebook teaches a small student model using a larger teacher.

Goals:
- train a teacher on the full dataset
- train two students on a small subset
- compare hard-label training vs distillation
"""
            ),
            code_cell(
                """import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_small = X_train_full[:200]
y_train_small = y_train_full[:200]

full_loader = DataLoader(TensorDataset(X_train_full, y_train_full), batch_size=64, shuffle=True)
small_loader = DataLoader(TensorDataset(X_train_small, y_train_small), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
"""
            ),
            code_cell(
                """class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


teacher = Teacher()
student_hard = Student()
student_distilled = Student()
"""
            ),
            code_cell(
                """teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    teacher.train()
    for xb, yb in full_loader:
        loss = loss_fn(teacher(xb), yb)
        teacher_optimizer.zero_grad()
        loss.backward()
        teacher_optimizer.step()

teacher_acc = evaluate(teacher)
teacher_acc
"""
            ),
            code_cell(
                """hard_optimizer = torch.optim.Adam(student_hard.parameters(), lr=0.01)

for epoch in range(15):
    student_hard.train()
    for xb, yb in small_loader:
        loss = loss_fn(student_hard(xb), yb)
        hard_optimizer.zero_grad()
        loss.backward()
        hard_optimizer.step()

student_hard_acc = evaluate(student_hard)
student_hard_acc
"""
            ),
            code_cell(
                """distill_optimizer = torch.optim.Adam(student_distilled.parameters(), lr=0.01)
temperature = 3.0
alpha = 0.5

teacher.eval()
for epoch in range(15):
    student_distilled.train()
    for xb, yb in small_loader:
        with torch.no_grad():
            teacher_logits = teacher(xb)

        student_logits = student_distilled(xb)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
        hard_loss = loss_fn(student_logits, yb)
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        distill_optimizer.zero_grad()
        loss.backward()
        distill_optimizer.step()

student_distilled_acc = evaluate(student_distilled)

def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


pd.DataFrame(
    [
        {"model": "teacher", "params": count_params(teacher), "accuracy": teacher_acc},
        {"model": "student_hard", "params": count_params(student_hard), "accuracy": student_hard_acc},
        {"model": "student_distilled", "params": count_params(student_distilled), "accuracy": student_distilled_acc},
    ]
)
"""
            ),
        ],
    )

    write_notebook(
        "09-optimization-comparison-dashboard.ipynb",
        "Optimization Comparison Dashboard",
        [
            markdown_cell(
                """# Optimization Comparison Dashboard

This notebook is a simple summary notebook.

Goals:
- compare size, latency, and accuracy in one table
- plot tradeoffs
- encourage students to replace the sample numbers with their own results
"""
            ),
            code_cell(
                """import pandas as pd

results = pd.DataFrame(
    [
        {"variant": "baseline_fp32", "size_kb": 620, "latency_ms": 4.8, "accuracy": 0.969},
        {"variant": "dynamic_int8", "size_kb": 210, "latency_ms": 3.6, "accuracy": 0.964},
        {"variant": "onnx_fp32", "size_kb": 540, "latency_ms": 3.4, "accuracy": 0.967},
        {"variant": "onnx_int8", "size_kb": 180, "latency_ms": 2.8, "accuracy": 0.963},
        {"variant": "pruned_model", "size_kb": 620, "latency_ms": 4.6, "accuracy": 0.958},
        {"variant": "distilled_student", "size_kb": 90, "latency_ms": 1.9, "accuracy": 0.952},
    ]
)
results
"""
            ),
            code_cell(
                """ax = results.plot.scatter(
    x="size_kb",
    y="accuracy",
    s=results["latency_ms"] * 80,
    figsize=(8, 6),
    grid=True,
    title="Accuracy vs size (bubble size shows latency)",
)

for _, row in results.iterrows():
    ax.annotate(row["variant"], (row["size_kb"], row["accuracy"]))
"""
            ),
            code_cell(
                """results.sort_values(["accuracy", "latency_ms"], ascending=[False, True])"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
