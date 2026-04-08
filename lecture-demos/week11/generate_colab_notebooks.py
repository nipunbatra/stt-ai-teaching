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


if __name__ == "__main__":
    main()
