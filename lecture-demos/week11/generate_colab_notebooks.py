"""Generate simple, first-year-friendly Colab notebooks for Week 11 concepts.

Each notebook is intentionally small. Code cells stay short. Most of the
learning happens in the *markdown* cells around them — short context, what
to look for in the output, and a small "your turn" prompt.

The notebooks are companions to:
    slides/week11-profiling-quantization-lecture.md

Run:
    python lecture-demos/week11/generate_colab_notebooks.py
"""

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


def write_notebook(filename: str, cells: list[dict]) -> None:
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


# ----------------------------------------------------------------------------
# 01 — Floating point basics
# ----------------------------------------------------------------------------
def nb_floating_point() -> list[dict]:
    return [
        markdown_cell(
            """# 01 — Floating Point Basics

> Companion to **Week 11**, Part 1 of the lecture.

## What you will see

Computers can't store every decimal number perfectly. They use a clever trick
called **floating point** that approximates real numbers using a fixed number
of bits (32 bits = FP32, 16 bits = FP16, 8 bits for an integer = INT8).

By the end of this notebook you will be able to answer:

1. Why is `0.1 + 0.2` *not* exactly `0.3`?
2. What does the bit pattern of an FP32 number look like?
3. How much does an FP32 weight change when we round it to FP16 or "fake INT8"?
"""
        ),
        markdown_cell(
            """## Step 1 — The famous floating-point gotcha

Run the cell below. **Do not be alarmed** if Python tells you that `0.1 + 0.2`
is *not* equal to `0.3`. This is a feature of floating point, not a bug.
"""
        ),
        code_cell(
            """import struct

result = 0.1 + 0.2
print("0.1 + 0.2 =", result)
print("Is it exactly 0.3?", result == 0.3)
"""
        ),
        markdown_cell(
            """**What happened?** Numbers like `0.1`, `0.2`, `0.3` cannot be written
exactly in binary — just like `1/3` cannot be written exactly in decimal.
The result is *very close* to `0.3`, but not exactly `0.3`.

> ⚠️ **Lesson for ML:** never compare floats with `==`. Use `abs(a - b) < 1e-6`
> or `numpy.isclose(a, b)` instead.

## Step 2 — Look at the actual bits

A 32-bit float is stored as a sign bit, an exponent, and a mantissa. We can
peek at the bit pattern using Python's `struct` module.
"""
        ),
        code_cell(
            """for value in [0.0, 1.0, 0.1, 0.2, 0.3, -1.5]:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    print(f"{value:>5}:  {bits:032b}")
"""
        ),
        markdown_cell(
            """Take a moment. Notice that:

- The first bit is the **sign**: 0 for positive, 1 for negative.
- The next 8 bits are the **exponent**.
- The last 23 bits are the **mantissa** (the significant digits).

You do **not** need to memorize this — just know that every float lives inside
32 of these little switches.

## Step 3 — FP32 vs FP16 vs (fake) INT8

Now let's pretend we are quantizing some neural-network weights. We will:

1. Start from a few FP32 values.
2. Round them to FP16.
3. Round them to a simple INT8 approximation (multiply by a scale, round,
   then divide back).
4. Compare how big each rounding error is.
"""
        ),
        code_cell(
            """import numpy as np
import pandas as pd

weights = np.array([0.12, -1.78, 3.14159, 0.004, -0.56], dtype=np.float32)
fp16 = weights.astype(np.float16)

# Simple "fake INT8" using a scale factor
scale = 32
int8_values = np.clip(np.round(weights * scale), -128, 127).astype(np.int8)
recovered = int8_values.astype(np.float32) / scale

table = pd.DataFrame(
    {
        "fp32":      weights,
        "fp16":      fp16.astype(np.float32),
        "fake_int8": recovered,
    }
)
table["fp16_error"]      = (table["fp32"] - table["fp16"]).abs()
table["fake_int8_error"] = (table["fp32"] - table["fake_int8"]).abs()
table
"""
        ),
        markdown_cell(
            """### What to look for in that table

- The **fp16** column should look almost identical to **fp32** — errors near zero.
- The **fake_int8** column should be visibly chunkier — bigger steps between values.
- The **errors** are still very small for inference-style use.

That tiny error is the price you pay for using fewer bits per number. In return
you get a model that is 2× to 4× smaller in memory.

## 🧪 Your turn

Change the `scale` from `32` to `8` and re-run the cell. What happens to the
INT8 errors? Why do you think a smaller scale gives a chunkier approximation?
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 02 — Parameter count and memory
# ----------------------------------------------------------------------------
def nb_parameter_count() -> list[dict]:
    return [
        markdown_cell(
            """# 02 — Parameter Count and Memory

> Companion to **Week 11**, Part 2 of the lecture.

## Why this matters

A neural network is just a big bag of numbers (its **parameters**). The total
memory it takes is roughly:

```
size = number_of_parameters × bytes_per_parameter
```

So once you know how many parameters a model has and how many bytes you store
each one in (FP32 = 4, FP16 = 2, INT8 = 1), you can estimate its size.
"""
        ),
        markdown_cell(
            """## Step 1 — Build three small models

We will define three tiny MLPs of growing size and count their parameters.
"""
        ),
        code_cell(
            """import pandas as pd
import torch
import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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

for name, model in [("small", small), ("medium", medium), ("large", large)]:
    print(f"{name:>6}: {count_params(model):>7,} parameters")
"""
        ),
        markdown_cell(
            """## Step 2 — Convert parameter counts to memory

Each parameter takes 4 bytes in FP32, 2 in FP16, 1 in INT8. We will convert
to kilobytes (KB) so the numbers are easy to read.
"""
        ),
        code_cell(
            """rows = []
for name, model in [("small", small), ("medium", medium), ("large", large)]:
    params = count_params(model)
    rows.append(
        {
            "model":   name,
            "params":  params,
            "fp32_kb": params * 4 / 1024,
            "fp16_kb": params * 2 / 1024,
            "int8_kb": params * 1 / 1024,
        }
    )

pd.DataFrame(rows)
"""
        ),
        markdown_cell(
            """### What to notice

- The **large** model has way more parameters than the small one.
- For the same model, **INT8 is 4× smaller** than FP32 — same model, fewer
  bits per number.
- These savings scale: a 7-billion-parameter LLM is 28 GB in FP32 but only
  ~7 GB in INT8 (and ~3.5 GB in INT4!).

## 🧪 Your turn

Add a fourth model called `huge` with three hidden layers of 1024 units.
- How many parameters does it have?
- How big would it be in FP32? In INT8?
- Could it fit in your laptop's RAM (assume 8 GB)?
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 03 — Profiling basics
# ----------------------------------------------------------------------------
def nb_profiling() -> list[dict]:
    return [
        markdown_cell(
            """# 03 — Profiling Basics

> Companion to **Week 11**, Part 3 of the lecture.

## The story

You wrote a tiny ML web service. Every request takes 2.5 seconds. That feels
slow. **Where** is the time going? Don't guess — *measure*.

In this notebook we will:

1. Train a small classifier and save it to disk.
2. Time things three different ways: `time.time()`, `%timeit`, and `cProfile`.
3. Use `cProfile` to find the *exact* slow line — not "the code", a specific line.
4. Fix the bug and measure the speedup.
"""
        ),
        markdown_cell(
            """## Step 1 — Train and save a small model

This is the kind of artifact a real web service would load on startup.
"""
        ),
        code_cell(
            """from pathlib import Path

import joblib
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
print("Saved model.pkl and data.csv to", artifacts)
"""
        ),
        markdown_cell(
            """## Step 2 — A slow function and a fast function

The **slow** function reloads the CSV and the model on every call. This is a
real bug in many FastAPI / Flask apps — people accidentally put `joblib.load(...)`
*inside* the request handler.

The **fast** function loads the model **once** at module level and reuses it.
"""
        ),
        code_cell(
            """def slow_predict(sample):
    data = pd.read_csv(artifacts / "data.csv")     # reloaded every call!
    model = joblib.load(artifacts / "model.pkl")    # reloaded every call!
    return model.predict([sample])[0]


# Cache the model ONCE, outside the function
cached_model = joblib.load(artifacts / "model.pkl")


def fast_predict(sample):
    return cached_model.predict([sample])[0]
"""
        ),
        markdown_cell(
            """## Step 3 — Three ways to time code

### (a) `time.time()` — the stopwatch

Quick and dirty. Works anywhere. One single run, so the number is noisy.
"""
        ),
        code_cell(
            """import time

start = time.time()
slow_predict(sample)
print(f"slow_predict took {time.time() - start:.3f} s (one run, noisy)")
"""
        ),
        markdown_cell(
            """### (b) IPython `%timeit` magic — the right way in Jupyter

`%timeit` runs the line many times, throws out outliers, and reports a stable
mean ± std. It even picks a sensible number of repetitions for you.

- `%timeit  expr`         → time a single expression
- `%%timeit`              → time the whole cell
- `%timeit -n 100 expr`   → force 100 runs per loop
"""
        ),
        code_cell(
            """%timeit slow_predict(sample)
"""
        ),
        code_cell(
            """%timeit fast_predict(sample)
"""
        ),
        markdown_cell(
            """You should see something like:

```
slow_predict:  ~30 ms ± 2 ms per loop
fast_predict:  ~50 µs ± 5 µs per loop      ← microseconds, not milliseconds
```

That's already a ~600× difference. But `%timeit` only tells us **how long** —
not **why**. For that we need `cProfile`.
"""
        ),
        markdown_cell(
            """### (c) `cProfile` — the itemized receipt

`cProfile` records every function call and how much time it took. The raw
output is huge and noisy, so we route it through `pstats` to:

1. Sort by `cumulative` time (slowest things first).
2. Print only the **top 10** lines.
3. Strip the long Python paths so the function names are readable.
"""
        ),
        code_cell(
            """import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
for _ in range(20):                  # 20 runs so the numbers are stable
    slow_predict(sample)
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("cumulative").print_stats(10)
"""
        ),
        markdown_cell(
            """### How to read the output

Each row is one function. The columns you actually care about:

| Column | Meaning |
|---|---|
| `ncalls` | how many times the function was called |
| `tottime` | time spent **inside** this function (not its children) |
| `cumtime` | time spent inside this function **plus everything it called** |
| `filename:lineno(function)` | which line of which file |

**What to look for:**

- `slow_predict` itself has a huge `cumtime` — it's the entry point.
- Inside it, the biggest `cumtime` rows belong to **`read_csv`** and
  **`joblib.load`** (or `pickle.load`).
- `predict` is barely visible. **It is not the bottleneck.**

> 🩺 **Diagnosis:** ~99% of the time is spent re-loading the CSV and the
> pickled model on every call. The actual ML inference is microseconds.
"""
        ),
        markdown_cell(
            """## Step 4 — Measure the speedup we get from the fix

The fix ("load once at startup") is just two lines moved out of the function.
Let's measure how big that speedup is.
"""
        ),
        code_cell(
            """import timeit

slow_time = timeit.timeit(lambda: slow_predict(sample), number=10) / 10
fast_time = timeit.timeit(lambda: fast_predict(sample), number=1000) / 1000

print(f"slow_predict: {slow_time * 1000:>10.3f} ms per call")
print(f"fast_predict: {fast_time * 1000:>10.3f} ms per call")
print(f"speedup     : {slow_time / fast_time:>10.1f} ×")
"""
        ),
        markdown_cell(
            """### Reflection

You should see a **huge** speedup — typically 100× or more. We did not change
the model. We did not change the prediction logic. We just **moved two lines
out of the request handler**.

> 🧠 **Take-away:** The first place to look for an ML web speedup is almost
> never the model itself — it's loading and copying work you didn't need to redo.

## 🧪 Your turn

1. Write a `medium_predict` that caches the model but still calls
   `pd.read_csv(...)` on every request. Where does it land between slow and fast?
2. Re-profile `medium_predict` with `cProfile`. Does the top of the report change?
3. **Bonus:** time `model.predict([sample])[0]` directly with `%timeit`. How
   much of `fast_predict` is "real" inference vs Python overhead?
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 04 — Batching benchmark
# ----------------------------------------------------------------------------
def nb_batching() -> list[dict]:
    return [
        markdown_cell(
            """# 04 — Batching Benchmark

> Companion to **Week 11**, Part 4 of the lecture.

## The idea in one line

> Calling a model **once** with 100 examples is almost always faster than
> calling it **100 times** with one example each.

This is because every model call has fixed overhead: allocating memory,
launching kernels, Python function calls, etc. That cost is paid *per call*,
not *per example*. Batch your inputs and you pay it once.

Think of it like an Uber Pool: the same driver, more passengers per trip.
"""
        ),
        markdown_cell(
            """## Step 1 — Build a small model

We will use the same model for every experiment so the only thing changing
is the batch size.
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
print("Model built. Parameters:", sum(p.numel() for p in model.parameters()))
"""
        ),
        markdown_cell(
            """## Step 2 — Time the model at different batch sizes

For each batch size we:

1. Build a random input tensor of that size.
2. Warm up (a few calls so any one-time setup doesn't pollute timing).
3. Run the model many times and average.
4. Record latency per batch, latency per example, and throughput.
"""
        ),
        code_cell(
            """batch_sizes = [1, 8, 32, 128, 512]
rows = []

for batch_size in batch_sizes:
    x = torch.randn(batch_size, 256)

    # Warm-up
    for _ in range(20):
        with torch.no_grad():
            model(x)

    # Time it
    start = time.perf_counter()
    runs = 200
    for _ in range(runs):
        with torch.no_grad():
            model(x)
    avg_ms = (time.perf_counter() - start) * 1000 / runs

    rows.append(
        {
            "batch_size":           batch_size,
            "avg_batch_ms":         avg_ms,
            "avg_per_example_ms":   avg_ms / batch_size,
            "examples_per_second":  batch_size / (avg_ms / 1000),
        }
    )

results = pd.DataFrame(rows)
results
"""
        ),
        markdown_cell(
            """### What to look for

- `avg_batch_ms` grows slowly with batch size — not linearly.
- `avg_per_example_ms` should drop sharply as the batch grows.
- `examples_per_second` (throughput) should grow with batch size.

That is the whole point of batching: more work per call = lower per-example
latency = higher throughput.

## Step 3 — Plot it
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
        markdown_cell(
            """## 🧪 Your turn

1. Add a batch size of `2048` to the list. Does throughput keep growing, or
   does it level off?
2. Replace the model with a much bigger one (e.g. hidden size 2048). How does
   the relationship between batch size and per-example latency change?

> 💡 **In production**, the right batch size is the **largest one your memory
> can handle, while still meeting your latency budget.**
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 05 — PyTorch dynamic quantization
# ----------------------------------------------------------------------------
def nb_pytorch_quant() -> list[dict]:
    return [
        markdown_cell(
            """# 05 — PyTorch Dynamic Quantization

> Companion to **Week 11**, Part 5 of the lecture.

## What you will do

Train a tiny MLP on the digits dataset, then **quantize** it to INT8 with
**one line of PyTorch code**, and compare:

| Variant | Size on disk | Accuracy | Speed |
|---|---|---|---|
| FP32 (original) | ? | ? | ? |
| INT8 (quantized) | ? | ? | ? |

You will fill in the `?` boxes from your own measurements.
"""
        ),
        markdown_cell(
            """## Step 1 — Load data and define a model
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
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256)


class SmallMLP(nn.Module):
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
"""
        ),
        markdown_cell(
            """## Step 2 — Helper functions and training loop
"""
        ),
        code_cell(
            """def evaluate(model):
    model.eval()
    correct, total = 0, 0
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

fp32_acc  = evaluate(model)
fp32_size = model_size_kb(model)
print(f"FP32 — accuracy: {fp32_acc:.3f}   size: {fp32_size:.1f} KB")
"""
        ),
        markdown_cell(
            """## Step 3 — Quantize in ONE line

This is the line of code worth remembering:

```python
quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

That's it. We tell PyTorch which kinds of layers to quantize (`nn.Linear`),
and the target precision (`torch.qint8`). PyTorch handles the rest.
"""
        ),
        code_cell(
            """quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

int8_acc  = evaluate(quantized_model)
int8_size = model_size_kb(quantized_model)

sample_batch = X_test[:256]


def benchmark(m, runs=300):
    m.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            m(sample_batch)
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
        markdown_cell(
            """### What you should see — and how to read it honestly

A typical Colab CPU run of this notebook prints something like:

```
   model    accuracy    size_kb    latency_ms
0   fp32    0.955       70.0       0.45
1   int8    0.958       22.0       2.14
```

Three things to notice:

**1. Size went down ~3×** (70 KB → 22 KB). That's the reliable win and it
matches our theory: `nn.Linear` weights now use 1 byte instead of 4.

**2. Accuracy looks *higher* for INT8.** Don't believe it. The test set has
~360 samples, so one prediction flipping is worth `±0.28%`. Re-run the whole
notebook with `torch.manual_seed(0)` vs `torch.manual_seed(7)` and the winner
will swap. The honest read is **same accuracy**.

**3. INT8 is *slower*.** Yes — on a model this small, on CPU. Dynamic
quantization adds per-call overhead (observe activations → pick scale →
quantize → matmul in INT8 → dequantize). On a `64→128→64→10` MLP the matmul
is so cheap that the overhead dominates.

> The **size win is always there**. The **speed win only shows up once the
> matmul is big enough** that scaling overhead disappears in the noise — think
> BERT-sized and above. For tiny MLPs, quantize for *size* only.

## 🧪 Your turn

1. Train for more epochs (e.g. 30 instead of 8). Does the "accuracy difference"
   between FP32 and INT8 stay the same or shrink? (It should be noise either way.)
2. Make the model **much** wider — `nn.Linear(64, 1024)` and `nn.Linear(1024, 1024)`
   in the middle. Re-time both versions. Does INT8 start winning on latency?
3. Print `quantized_model.net[0]` and look at the layer type. What is it
   actually? (Hint: it's no longer `Linear`.)
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 06 — ONNX export and quantization
# ----------------------------------------------------------------------------
def nb_onnx() -> list[dict]:
    return [
        markdown_cell(
            """# 06 — ONNX Export and Quantization

> Companion to **Week 11**, Part 5 of the lecture.

## Why ONNX?

ONNX is the **"PDF of machine learning"**. You write your model in PyTorch or
sklearn, then export it to ONNX, and after that *anyone* can run it — on a
phone, in a browser, in C++, in Java, on a microcontroller. They don't need
PyTorch installed.

In this notebook we will:

1. Train a sklearn MLP on digits.
2. Convert it to ONNX.
3. Run it with ONNX Runtime.
4. Quantize the ONNX file to INT8 and compare.
"""
        ),
        markdown_cell(
            """## Step 1 — Install ONNX libraries (Colab only)
"""
        ),
        code_cell(
            """%pip install -q onnx onnxruntime skl2onnx
"""
        ),
        markdown_cell(
            """## Step 2 — Train a sklearn model
"""
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
print("sklearn accuracy:", clf.score(X_test, y_test))
"""
        ),
        markdown_cell(
            """## Step 3 — Convert to ONNX and quantize

This is the magic. Three lines convert a sklearn model to a portable ONNX
file. Two more lines quantize that file to INT8.
"""
        ),
        code_cell(
            """initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

fp32_path = workdir / "digits_fp32.onnx"
fp32_path.write_bytes(onnx_model.SerializeToString())

int8_path = workdir / "digits_int8.onnx"
quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

print("FP32 file:", fp32_path, f"({os.path.getsize(fp32_path)/1024:.1f} KB)")
print("INT8 file:", int8_path, f"({os.path.getsize(int8_path)/1024:.1f} KB)")
"""
        ),
        markdown_cell(
            """## Step 4 — Run both with ONNX Runtime

Notice: from this point on, **we never import sklearn or PyTorch again**.
Inference is happening through `onnxruntime` only.
"""
        ),
        code_cell(
            """fp32_session = ort.InferenceSession(str(fp32_path))
int8_session = ort.InferenceSession(str(int8_path))
input_name = fp32_session.get_inputs()[0].name


def accuracy(session):
    preds = session.run(None, {input_name: X_test})[0]
    return float(np.mean(preds == y_test))


def benchmark(session, runs=200):
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: X_test})
    return (time.perf_counter() - start) * 1000 / runs


pd.DataFrame(
    [
        {
            "model":      "onnx_fp32",
            "size_kb":    os.path.getsize(fp32_path) / 1024,
            "accuracy":   accuracy(fp32_session),
            "latency_ms": benchmark(fp32_session),
        },
        {
            "model":      "onnx_int8",
            "size_kb":    os.path.getsize(int8_path) / 1024,
            "accuracy":   accuracy(int8_session),
            "latency_ms": benchmark(int8_session),
        },
    ]
)
"""
        ),
        markdown_cell(
            """### What you should see

- The INT8 ONNX file is roughly 3-4x smaller than the FP32 ONNX file.
- Accuracy is nearly identical (often within 0.5 percentage points).
- Latency is the same or slightly faster.

> 🧠 **Take-away:** ONNX gives you portability *and* a clean place to apply
> optimizations. You can quantize, prune, or fuse layers without touching the
> original training code.

## 🧪 Your turn

Try `weight_type=QuantType.QInt8` instead of `QuantType.QUInt8` (signed instead
of unsigned). Does accuracy change? Does file size change?
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 07 — Pruning basics
# ----------------------------------------------------------------------------
def nb_pruning() -> list[dict]:
    return [
        markdown_cell(
            """# 07 — Pruning Basics

> Companion to **Week 11**, Part 6 of the lecture.

## The idea

> Many of the weights in a trained network are *tiny*. Setting them to zero
> barely changes the model's predictions — but the resulting "sparse" model
> compresses much better and can sometimes run faster on supporting hardware.

This notebook trains a small MLP, then **removes 40% of the weights** and
checks how much accuracy is lost.
"""
        ),
        markdown_cell(
            """## Step 1 — Train the model
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
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256)


class PruneMLP(nn.Module):
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
    correct, total = 0, 0
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
print(f"Accuracy BEFORE pruning: {before_acc:.3f}")
print(f"Sparsity of fc1 BEFORE:   {sparsity(model.fc1):.2%}")
"""
        ),
        markdown_cell(
            """## Step 2 — Unstructured pruning

We start with **global unstructured L1 pruning**: PyTorch looks at all the
weights in `fc1` and `fc2` together, finds the 40% with the smallest absolute
value, and replaces them with zero. *Individual edges* are removed — the shape
of the matrix does not change.
"""
        ),
        code_cell(
            """import copy

unstructured_model = copy.deepcopy(model)

parameters_to_prune = (
    (unstructured_model.fc1, "weight"),
    (unstructured_model.fc2, "weight"),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4,
)

unstructured_acc = evaluate(unstructured_model)

pd.DataFrame(
    [
        {"metric": "accuracy_before",        "value": before_acc},
        {"metric": "accuracy_after_unstruct","value": unstructured_acc},
        {"metric": "fc1_sparsity",           "value": sparsity(unstructured_model.fc1)},
        {"metric": "fc2_sparsity",           "value": sparsity(unstructured_model.fc2)},
        {"metric": "fc1_shape (unchanged)",  "value": str(tuple(unstructured_model.fc1.weight.shape))},
    ]
)
"""
        ),
        markdown_cell(
            """### What you should see

- ~40% of the weights in `fc1` and `fc2` are now exactly zero.
- Accuracy is nearly the same as before — usually within 1-2 percentage points.
- **The shape of the weight matrix did not change**: `(128, 64)` is still
  `(128, 64)`. That's the key property of *unstructured* pruning.

> ⚠️ **Important caveat:** unstructured pruning does **not** automatically make
> inference faster. You still multiply the input by a full dense matrix —
> most of the entries just happen to be zero. You need a sparse matrix library
> or special hardware to turn those zeros into actual speed.

## Step 3 — Structured pruning

*Structured* pruning removes whole **rows** (output neurons) or **columns**
(input connections) of a weight matrix, so the matrix is *literally* smaller
afterwards. The matmul is smaller → real speedup on any hardware.

We use `ln_structured` which ranks rows by their L2 norm and drops the 30%
smallest.
"""
        ),
        code_cell(
            """structured_model = copy.deepcopy(model)

# Drop 30% of the OUTPUT channels (rows) of fc1 — dim=0 means along rows
prune.ln_structured(
    structured_model.fc1,
    name="weight",
    amount=0.3,
    n=2,           # L2 norm
    dim=0,         # row-wise = per output neuron
)

structured_acc = evaluate(structured_model)

row_norms = structured_model.fc1.weight.norm(dim=1)
dead_rows = int((row_norms == 0).sum())

pd.DataFrame(
    [
        {"metric": "accuracy_after_structured", "value": structured_acc},
        {"metric": "fc1_total_rows",            "value": structured_model.fc1.weight.shape[0]},
        {"metric": "fc1_zeroed_rows",           "value": dead_rows},
        {"metric": "fc1_sparsity",              "value": sparsity(structured_model.fc1)},
    ]
)
"""
        ),
        markdown_cell(
            """### What to compare

| | Unstructured (step 2) | Structured (step 3) |
|---|---|---|
| What was removed | 40% of individual weights | 30% of whole output neurons of `fc1` |
| Matrix shape after | unchanged | unchanged *tensor*, but 30% of rows are now all zero |
| Accuracy drop | very small | larger (we removed more "all at once") |
| Speedup on a normal CPU | ❌ unless sparse kernels | ✅ if you actually drop the dead rows |

Structured pruning costs more accuracy, but its speedup is real on any
hardware — you can literally rebuild `fc1` as a smaller `nn.Linear(64, 90)`
layer and save the multiplication by the dead rows entirely.

## 🧪 Your turn

1. Push unstructured pruning to the limit: `amount = 0.6`, `0.8`, `0.95`.
   Where is the "accuracy cliff"?
2. In the structured block, raise `amount` to `0.5`, then `0.7`. How quickly
   does structured pruning fall off compared to unstructured?
3. **Bonus:** rebuild `structured_model.fc1` as a smaller `nn.Linear` that
   drops the zeroed rows, and time it vs the original. Do you see a speedup?
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 08 — Distillation basics
# ----------------------------------------------------------------------------
def nb_distillation() -> list[dict]:
    return [
        markdown_cell(
            """# 08 — Knowledge Distillation Basics

> Companion to **Week 11**, Part 6 of the lecture.

## The idea

A big **teacher** model trains a small **student** model. Instead of learning
from raw labels alone, the student learns from the teacher's *soft predictions*
("90% cat, 8% lynx, 2% dog"). This extra information is rich, and lets the
student get away with much fewer parameters.

In this notebook we will:

1. Train a **big teacher** on the full training set.
2. Train a small **student** on a *small* subset using only hard labels.
3. Train another small **student** on the same small subset, but using the
   teacher's soft predictions.
4. Compare the three.
"""
        ),
        markdown_cell(
            """## Step 1 — Set up data and model classes
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

# The students get to see only a SMALL slice of training data.
X_train_small = X_train_full[:200]
y_train_small = y_train_full[:200]

full_loader  = DataLoader(TensorDataset(X_train_full, y_train_full), batch_size=64, shuffle=True)
small_loader = DataLoader(TensorDataset(X_train_small, y_train_small), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256)


class Teacher(nn.Module):
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
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total


def count_params(m):
    return sum(p.numel() for p in m.parameters())


teacher           = Teacher()
student_hard      = Student()
student_distilled = Student()

print("Teacher parameters:", count_params(teacher))
print("Student parameters:", count_params(student_hard))
"""
        ),
        markdown_cell(
            """## Step 2 — Train the big teacher on the full dataset
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
print(f"Teacher accuracy: {teacher_acc:.3f}")
"""
        ),
        markdown_cell(
            """## Step 3 — Train a student WITHOUT distillation

We give it only the **small** training set and hard labels.
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
print(f"Hard-label student accuracy: {student_hard_acc:.3f}")
"""
        ),
        markdown_cell(
            """## Step 4 — Train a second student WITH distillation

Same architecture, same small training set. The only difference: we ALSO
match the teacher's *soft* predictions.

The two ingredients of distillation:

- **temperature** — softens the teacher's probabilities so the small numbers
  ("8% lynx, 2% dog") become visible to the student.
- **alpha** — how much weight to give the soft loss vs the hard loss.
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
print(f"Distilled student accuracy: {student_distilled_acc:.3f}")
"""
        ),
        markdown_cell(
            """## Step 5 — Compare all three
"""
        ),
        code_cell(
            """pd.DataFrame(
    [
        {"model": "teacher",            "params": count_params(teacher),           "accuracy": teacher_acc},
        {"model": "student_hard",       "params": count_params(student_hard),      "accuracy": student_hard_acc},
        {"model": "student_distilled",  "params": count_params(student_distilled), "accuracy": student_distilled_acc},
    ]
)
"""
        ),
        markdown_cell(
            """### What you should see

- The teacher is the most accurate (it had the most data and the most parameters).
- The hard-label student is the **least** accurate.
- The distilled student lands in between, often much closer to the teacher
  than to the hard-label student.

Two punchlines:

1. The **same small architecture** can do dramatically better when trained
   from a teacher's soft labels.
2. The student in production is **as small as the hard-label student** — same
   memory, same latency. You only paid the teacher cost during training.

## 🧪 Your turn

Set `temperature = 1.0` and re-run Step 4. What happens to the distilled
student? Why? (Hint: what does temperature do to the teacher's probabilities?)
"""
        ),
    ]


# ----------------------------------------------------------------------------
# 10 — Lab: profiling & quantization (student exercises)
# ----------------------------------------------------------------------------
def nb_lab() -> list[dict]:
    return [
        markdown_cell(
            """# Week 11 Lab — Profiling and Quantization

> **What this is.** A hands-on lab covering the first five parts of the Week 11
> lecture: floating point, parameters & memory, profiling, batching, and
> quantization (PyTorch + ONNX). Pruning and distillation are **not** in this lab.

## How to work through this

- Read each **Task** box carefully, then fill in the `# TODO:` lines in the
  following code cell. Do not skip the markdown questions.
- Each task ends with a tiny **check** cell that should print `PASS` if your
  answer looks right. Don't edit the check cells.
- At the end you'll produce one small **summary table** that compares all the
  variants you built.

## Before you start

Run the setup cell below once. It installs ONNX libraries, imports everything,
loads the digits dataset, and defines helpers used throughout the lab.
"""
        ),
        markdown_cell(
            """## Setup — run this first
"""
        ),
        code_cell(
            """%pip install -q onnx onnxruntime skl2onnx
"""
        ),
        code_cell(
            """import io
import os
import struct
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
np.random.seed(0)

# Digits dataset (8x8 grayscale, 10 classes) — tiny, CPU-friendly
digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=256)

artifacts = Path("/tmp/week11_lab")
artifacts.mkdir(exist_ok=True)

print("Setup OK — train:", X_train.shape, "test:", X_test.shape)
"""
        ),
        markdown_cell(
            """---

## Part A — Floating Point  *(Lecture Part 1)*

Three quick tasks to build intuition about how decimals are stored.
"""
        ),
        markdown_cell(
            """### Task A1 — Why is `0.1 + 0.2` not `0.3`?

Run the cell. In the markdown cell after it, **write one or two sentences**
explaining the printed result in your own words (hint: look back at the FP32
slides — 23 bits of mantissa can't represent `0.1` exactly).
"""
        ),
        code_cell(
            """a = 0.1 + 0.2
print(f"0.1 + 0.2 = {a!r}")
print(f"Equal to 0.3?  {a == 0.3}")
print(f"Difference   : {a - 0.3:.2e}")
"""
        ),
        markdown_cell(
            """**Your explanation (Task A1):**

> _Replace this line with your 1–2 sentence answer._
"""
        ),
        markdown_cell(
            """### Task A2 — Encode `3.25` in FP32 by hand, verify with numpy

`3.25` in binary is `11.01`. Normalize it and fill in the three fields below.
Then run the check cell to compare against numpy's real FP32 bits.
"""
        ),
        code_cell(
            """# TODO — fill these in by hand using the Lecture Part 1 recipe.
# Hint: 3.25 = 11.01 (binary) = 1.101 × 2^?
sign_bit             = None   # 0 for positive, 1 for negative
real_exponent        = None   # the "? " in 2^?
stored_exponent      = None   # = real_exponent + 127
mantissa_23_bits_str = None   # 23-char string of 0s and 1s, leading "1." dropped
"""
        ),
        code_cell(
            """# --- check cell for A2 (do not edit) ---
def fp32_bits(x: float) -> str:
    return format(struct.unpack("I", struct.pack("f", x))[0], "032b")

real = fp32_bits(3.25)
real_sign = int(real[0])
real_exp  = int(real[1:9], 2) - 127
real_mant = real[9:]

print(f"numpy says  sign={real_sign}, real_exp={real_exp}, mantissa={real_mant}")
print(f"you said    sign={sign_bit}, real_exp={real_exponent}, mantissa={mantissa_23_bits_str}")
assert sign_bit == real_sign, "sign bit wrong"
assert real_exponent == real_exp, "real exponent wrong"
assert stored_exponent == real_exp + 127, "stored exponent wrong"
assert mantissa_23_bits_str == real_mant, "mantissa bits wrong"
print("PASS")
"""
        ),
        markdown_cell(
            """### Task A3 — Build a "fake INT8" quantizer

Finish the function below. It must:

1. Find the scale: `max(|x|) / 127`.
2. Divide by scale and round to the nearest integer.
3. Clamp into `[-127, 127]` so it fits in INT8.
4. Multiply the rounded integers by the scale to "dequantize" back to floats.

Then run the check cell to see the maximum error.
"""
        ),
        code_cell(
            """def fake_int8_quantize(x: np.ndarray) -> tuple[np.ndarray, float]:
    \"\"\"Return (dequantized_values, scale).\"\"\"
    x = np.asarray(x, dtype=np.float32)
    # TODO — replace the three "None" lines
    scale   = None
    q_int8  = None     # step 2 and 3 combined: round, cast to int8, clamp
    x_back  = None     # step 4: back to float
    return x_back, scale
"""
        ),
        code_cell(
            """# --- check cell for A3 (do not edit) ---
rng = np.random.default_rng(0)
w   = rng.normal(0, 0.3, size=500).astype(np.float32)

w_back, scale = fake_int8_quantize(w)

max_err = float(np.max(np.abs(w - w_back)))
print(f"scale     : {scale:.6f}")
print(f"max error : {max_err:.6f}")
print(f"mean error: {float(np.mean(np.abs(w - w_back))):.6f}")
assert max_err < 0.02, "error too large — did you forget to multiply back by the scale?"
print("PASS")
"""
        ),
        markdown_cell(
            """---

## Part B — Parameters and Memory  *(Lecture Part 2)*
"""
        ),
        markdown_cell(
            """### Task B1 — Count parameters manually

Look at `LabMLP` below. **Before** running any code, compute the total number
of parameters **by hand** (hint: every `nn.Linear(a, b)` has `a*b + b` params)
and fill it into the TODO.
"""
        ),
        code_cell(
            """class LabMLP(nn.Module):
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


# TODO — your hand-computed total parameter count:
my_param_count_guess = None
"""
        ),
        code_cell(
            """# --- check cell for B1 ---
model = LabMLP()
real_params = sum(p.numel() for p in model.parameters())
print(f"PyTorch says: {real_params:,}")
print(f"Your guess  : {my_param_count_guess}")
assert my_param_count_guess == real_params, "recount — don't forget the biases!"
print("PASS")
"""
        ),
        markdown_cell(
            """### Task B2 — Memory at FP32 / FP16 / INT8

Fill in the function so it returns the model size in **kilobytes** for a given
bytes-per-parameter. A parameter at FP32 takes 4 bytes, FP16 takes 2, INT8 takes 1.
"""
        ),
        code_cell(
            """def model_size_kb(num_params: int, bytes_per_param: int) -> float:
    # TODO
    return None


# Use it on our LabMLP
print("FP32:", model_size_kb(real_params, 4), "KB")
print("FP16:", model_size_kb(real_params, 2), "KB")
print("INT8:", model_size_kb(real_params, 1), "KB")
"""
        ),
        code_cell(
            """# --- check cell for B2 ---
assert abs(model_size_kb(real_params, 4) - real_params * 4 / 1024) < 1e-9
assert abs(model_size_kb(real_params, 1) - real_params * 1 / 1024) < 1e-9
print("PASS")
"""
        ),
        markdown_cell(
            """### Task B3 — LLaMA-7B in three precisions

LLaMA-7B has **7,000,000,000** parameters. Compute how many **GB** it needs in
FP32, FP16, and INT8 (1 GB = 1024³ bytes ≈ 1.07e9 bytes; use the decimal
definition `10^9` bytes to keep the math simple).
"""
        ),
        code_cell(
            """LLAMA_PARAMS = 7_000_000_000

# TODO — fill in the three numbers (in GB, using 1 GB = 1e9 bytes)
fp32_gb = None
fp16_gb = None
int8_gb = None

print(f"FP32: {fp32_gb} GB")
print(f"FP16: {fp16_gb} GB")
print(f"INT8: {int8_gb} GB")
"""
        ),
        code_cell(
            """# --- check cell for B3 ---
assert fp32_gb == 28, "FP32 should be 28 GB"
assert fp16_gb == 14, "FP16 should be 14 GB"
assert int8_gb == 7,  "INT8 should be 7 GB"
print("PASS")
"""
        ),
        markdown_cell(
            """---

## Part C — Profiling  *(Lecture Part 3)*
"""
        ),
        markdown_cell(
            """### Task C1 — `%timeit` two functions

Both functions below compute the same thing. Use `%timeit` on each and report
*which one is faster* and *by how much*. You do not need to edit the functions.
"""
        ),
        code_cell(
            """def dot_python(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def dot_numpy(a, b):
    return float(np.dot(a, b))


aa = np.random.rand(10_000).astype(np.float32)
bb = np.random.rand(10_000).astype(np.float32)

print("Python loop =", dot_python(aa, bb))
print("NumPy       =", dot_numpy(aa, bb))
"""
        ),
        code_cell(
            """%timeit dot_python(aa, bb)
"""
        ),
        code_cell(
            """%timeit dot_numpy(aa, bb)
"""
        ),
        markdown_cell(
            """**Your answer (Task C1):**

- Which is faster? _____
- Roughly how many times faster? _____
- Why is the faster one faster (one sentence)? _____
"""
        ),
        markdown_cell(
            """### Task C2 — cProfile a slow endpoint

We've deliberately written `slow_endpoint` below to mimic the bug from the
lecture. **Run the cProfile cell, read the top few rows, and identify the
actual bottleneck**.

Hint: look for the function with the biggest `cumtime`.
"""
        ),
        code_cell(
            """import joblib
from sklearn.ensemble import RandomForestClassifier

# Train and save a little RF model + a CSV "config file"
rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X_train.numpy(), y_train.numpy())
joblib.dump(rf, artifacts / "rf.pkl")
pd.DataFrame(X_train.numpy()).to_csv(artifacts / "config.csv", index=False)

sample = X_test[0].numpy()


def slow_endpoint(sample):
    cfg = pd.read_csv(artifacts / "config.csv")       # reloaded every call!
    model = joblib.load(artifacts / "rf.pkl")          # reloaded every call!
    return model.predict([sample])[0]
"""
        ),
        code_cell(
            """import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    slow_endpoint(sample)
profiler.disable()

pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(10)
"""
        ),
        markdown_cell(
            """**Your answer (Task C2):**

- Which function takes the most cumulative time? _____
- Is `model.predict(...)` in the top 3? (yes / no) _____
"""
        ),
        markdown_cell(
            """### Task C3 — Fix the slow endpoint, measure the speedup

Write `fast_endpoint` that keeps the same behavior but loads the model and the
CSV **once** at module level (outside the function). Then compare `%timeit`
numbers.
"""
        ),
        code_cell(
            """# TODO — load these ONCE, outside the function
cached_cfg   = None
cached_model = None


def fast_endpoint(sample):
    # TODO — use cached_model here and return its prediction
    pass
"""
        ),
        code_cell(
            """# --- check cell for C3 ---
pred_slow = slow_endpoint(sample)
pred_fast = fast_endpoint(sample)
assert pred_fast is not None, "fast_endpoint is still returning None"
assert pred_slow == pred_fast, "fast_endpoint returned a different prediction!"
print("Predictions match. Now compare timings below:")
"""
        ),
        code_cell(
            """%timeit slow_endpoint(sample)
"""
        ),
        code_cell(
            """%timeit fast_endpoint(sample)
"""
        ),
        markdown_cell(
            """**Your answer (Task C3):**

- slow_endpoint average: _____ ms
- fast_endpoint average: _____ ms
- Speedup (slow / fast): _____ ×
"""
        ),
        markdown_cell(
            """---

## Part D — Batching  *(Lecture Part 4)*
"""
        ),
        markdown_cell(
            """### Task D1 — Benchmark latency vs batch size

Fill in the loop so `rows` ends up as a list of dicts with these keys:
`batch_size`, `avg_batch_ms`, `avg_per_example_ms`, `examples_per_second`.
"""
        ),
        code_cell(
            """bench_model = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)
bench_model.eval()

batch_sizes = [1, 8, 32, 128, 512]
rows = []

for bs in batch_sizes:
    x = torch.randn(bs, 256)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            bench_model(x)

    # Time it: average of 100 forward passes
    runs = 100
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            bench_model(x)
    avg_batch_ms = (time.perf_counter() - start) * 1000 / runs

    # TODO — compute these from avg_batch_ms and bs
    avg_per_example_ms  = None
    examples_per_second = None

    rows.append({
        "batch_size":          bs,
        "avg_batch_ms":        avg_batch_ms,
        "avg_per_example_ms":  avg_per_example_ms,
        "examples_per_second": examples_per_second,
    })

batch_df = pd.DataFrame(rows)
batch_df
"""
        ),
        code_cell(
            """# --- check cell for D1 ---
assert batch_df["avg_per_example_ms"].isna().sum() == 0, "fill in per-example latency"
assert batch_df["examples_per_second"].isna().sum() == 0, "fill in throughput"
assert batch_df.loc[batch_df["batch_size"] == 512, "avg_per_example_ms"].iloc[0] < \
       batch_df.loc[batch_df["batch_size"] == 1,   "avg_per_example_ms"].iloc[0], \
    "per-example latency at batch=512 should be LOWER than at batch=1"
print("PASS")
"""
        ),
        markdown_cell(
            """### Task D2 — Plot it and interpret
"""
        ),
        code_cell(
            """batch_df.plot(
    x="batch_size",
    y=["avg_per_example_ms", "examples_per_second"],
    subplots=True, figsize=(8, 6), grid=True, logx=True,
    title=["Latency per example (lower is better)", "Throughput (higher is better)"],
)
"""
        ),
        markdown_cell(
            """**Your answer (Task D2):**

- At which batch size does the per-example latency stop dropping significantly? _____
- In one sentence, *why* does batching speed up per-example latency? _____
"""
        ),
        markdown_cell(
            """---

## Part E — PyTorch Dynamic Quantization  *(Lecture Part 5)*
"""
        ),
        markdown_cell(
            """### Task E1 — Train a small MLP and measure the FP32 baseline
"""
        ),
        code_cell(
            """try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic


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


def measure_size_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024


def benchmark_ms(model, x, runs=300):
    model.eval()
    with torch.no_grad():
        for _ in range(20):      # warm-up
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        return (time.perf_counter() - start) * 1000 / runs


torch.manual_seed(0)
fp32_model = LabMLP()
optimizer  = torch.optim.Adam(fp32_model.parameters(), lr=0.01)
loss_fn    = nn.CrossEntropyLoss()

for epoch in range(8):
    fp32_model.train()
    for xb, yb in train_loader:
        loss = loss_fn(fp32_model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

fp32_acc = evaluate(fp32_model)
fp32_kb  = measure_size_kb(fp32_model)
fp32_ms  = benchmark_ms(fp32_model, X_test[:256])

print(f"FP32 — accuracy: {fp32_acc:.3f}   size: {fp32_kb:.1f} KB   latency: {fp32_ms:.3f} ms")
"""
        ),
        markdown_cell(
            """### Task E2 — Apply dynamic quantization in one line
"""
        ),
        code_cell(
            """# TODO — produce int8_model by calling quantize_dynamic on fp32_model.
#        Quantize nn.Linear layers to torch.qint8.
int8_model = None

int8_acc = evaluate(int8_model)
int8_kb  = measure_size_kb(int8_model)
int8_ms  = benchmark_ms(int8_model, X_test[:256])

print(f"INT8 — accuracy: {int8_acc:.3f}   size: {int8_kb:.1f} KB   latency: {int8_ms:.3f} ms")
"""
        ),
        code_cell(
            """# --- check cell for E2 ---
assert int8_model is not None, "you forgot to assign int8_model"
assert int8_kb < fp32_kb, "INT8 model should be smaller on disk"
assert abs(int8_acc - fp32_acc) < 0.05, "accuracy drop is too large — did you quantize the right layers?"
print("PASS")
"""
        ),
        markdown_cell(
            """### Task E3 — Write down what you observed

Reminder from the lecture: on a tiny MLP, dynamic quantization usually makes
the model **smaller** but NOT faster, because the per-call scaling overhead
dominates the matmul itself. Noise on a small test set can even make INT8
look slightly *more* accurate — that's normal.

**Your answers:**

- INT8 size / FP32 size = _____ (expect ~0.3)
- INT8 latency / FP32 latency = _____ (could be anything 0.8–5)
- Is the INT8 accuracy drop bigger than 1%? (yes / no) _____
- In one sentence, why might INT8 be *slower* than FP32 here? _____
"""
        ),
        markdown_cell(
            """---

## Part F — ONNX Export and Quantization  *(Lecture Part 5)*

A separate toolchain that does the same "shrink the weights" trick, but
produces a portable `.onnx` file that any language or device can load.
"""
        ),
        markdown_cell(
            """### Task F1 — Export `fp32_model` to ONNX, run with ONNX Runtime
"""
        ),
        code_cell(
            """import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic as onnx_quantize_dynamic

onnx_fp32 = artifacts / "lab_fp32.onnx"
onnx_int8 = artifacts / "lab_int8.onnx"

# TODO — export fp32_model. Hints:
#   - use torch.onnx.export
#   - dummy input shape: (1, 64)
#   - input_names=["input"], output_names=["logits"]
#   - dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
# Finish the call below.
dummy = torch.randn(1, 64)
torch.onnx.export(
    fp32_model, dummy, str(onnx_fp32),
    input_names=["input"],
    output_names=["logits"],
    # TODO — add dynamic_axes here
)
print("FP32 ONNX size:", round(os.path.getsize(onnx_fp32) / 1024, 1), "KB")
"""
        ),
        code_cell(
            """# Load and run the exported ONNX model
fp32_sess = ort.InferenceSession(str(onnx_fp32))
probs = fp32_sess.run(None, {"input": X_test[:5].numpy()})[0]
print("Output shape:", probs.shape)
print("Predicted classes:", probs.argmax(axis=1).tolist())
print("True classes     :", y_test[:5].tolist())
"""
        ),
        markdown_cell(
            """### Task F2 — Quantize the ONNX file to INT8
"""
        ),
        code_cell(
            """# TODO — call onnx_quantize_dynamic to turn onnx_fp32 into onnx_int8
#        with weight_type=QuantType.QUInt8
# YOUR CODE HERE

print("INT8 ONNX size:", round(os.path.getsize(onnx_int8) / 1024, 1), "KB")
"""
        ),
        code_cell(
            """# --- check cell for F2 ---
assert onnx_int8.exists(), "INT8 ONNX file was not produced"
assert os.path.getsize(onnx_int8) < os.path.getsize(onnx_fp32), \
    "INT8 ONNX file should be smaller than FP32"
print("PASS")
"""
        ),
        markdown_cell(
            """### Task F3 — Benchmark the two ONNX files and record the numbers
"""
        ),
        code_cell(
            """int8_sess = ort.InferenceSession(str(onnx_int8))


def onnx_accuracy(session):
    preds = session.run(None, {"input": X_test.numpy()})[0].argmax(axis=1)
    return float((preds == y_test.numpy()).mean())


def onnx_latency_ms(session, n=200):
    batch = X_test[:256].numpy()
    for _ in range(20):
        session.run(None, {"input": batch})
    start = time.perf_counter()
    for _ in range(n):
        session.run(None, {"input": batch})
    return (time.perf_counter() - start) * 1000 / n


onnx_fp32_kb  = os.path.getsize(onnx_fp32) / 1024
onnx_int8_kb  = os.path.getsize(onnx_int8) / 1024
onnx_fp32_acc = onnx_accuracy(fp32_sess)
onnx_int8_acc = onnx_accuracy(int8_sess)
onnx_fp32_ms  = onnx_latency_ms(fp32_sess)
onnx_int8_ms  = onnx_latency_ms(int8_sess)

print(f"ONNX FP32 — acc {onnx_fp32_acc:.3f}  size {onnx_fp32_kb:.1f} KB  latency {onnx_fp32_ms:.3f} ms")
print(f"ONNX INT8 — acc {onnx_int8_acc:.3f}  size {onnx_int8_kb:.1f} KB  latency {onnx_int8_ms:.3f} ms")
"""
        ),
        markdown_cell(
            """---

## Part G — Summary Table (Submission)

**Fill in** the comparison table below. The FP32 and INT8 (PyTorch) rows come
from Part E; the two ONNX rows come from Part F. Then answer the reflection
questions.
"""
        ),
        code_cell(
            """summary = pd.DataFrame(
    [
        {"variant": "pytorch_fp32", "size_kb": fp32_kb,       "accuracy": fp32_acc,       "latency_ms": fp32_ms},
        {"variant": "pytorch_int8", "size_kb": int8_kb,       "accuracy": int8_acc,       "latency_ms": int8_ms},
        {"variant": "onnx_fp32",    "size_kb": onnx_fp32_kb,  "accuracy": onnx_fp32_acc,  "latency_ms": onnx_fp32_ms},
        {"variant": "onnx_int8",    "size_kb": onnx_int8_kb,  "accuracy": onnx_int8_acc,  "latency_ms": onnx_int8_ms},
    ]
)
summary
"""
        ),
        markdown_cell(
            """### Reflection questions

Write a short answer (1–2 sentences each) under each question.

1. **Which variant has the smallest file size?** Is that the variant you would
   actually ship? Why / why not?

   > _your answer_

2. **Did your INT8 model lose more than 1% accuracy?** Based on the lecture, is
   that a typical result? (one sentence)

   > _your answer_

3. **Suppose you need to deploy this model inside a Java microservice**. Which
   row of the table would you pick, and why (two sentences max)?

   > _your answer_

4. **In Part C you made the slow endpoint ~100× faster without changing the
   model.** In one sentence, what was the actual bottleneck?

   > _your answer_

> 🧠 **Submit:** the completed notebook (`.ipynb`) with every PASS cell green
> and every `> _your answer_` filled in.
"""
        ),
    ]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_notebook("01-floating-point-basics.ipynb",          nb_floating_point())
    write_notebook("02-parameter-count-and-memory.ipynb",     nb_parameter_count())
    write_notebook("03-profiling-basics.ipynb",               nb_profiling())
    write_notebook("04-batching-benchmark.ipynb",             nb_batching())
    write_notebook("05-pytorch-dynamic-quantization.ipynb",   nb_pytorch_quant())
    write_notebook("06-onnx-export-and-quantization.ipynb",   nb_onnx())
    write_notebook("07-pruning-basics.ipynb",                 nb_pruning())
    write_notebook("08-distillation-basics.ipynb",            nb_distillation())
    write_notebook("10-lab-profiling-and-quantization.ipynb", nb_lab())


if __name__ == "__main__":
    main()
