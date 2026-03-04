#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model Profiling & Quantization — Follow-Along Guide                    ║
# ║  Week 13 · CS 203 · Software Tools and Techniques for AI               ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   Your model is deployed. But inference is slow and the model is huge.
#   Today you learn to profile (find bottlenecks), quantize (shrink the
#   model), and export to ONNX (fast portable inference).
#
# HOW TO USE:
#   1. Open this file in your editor (VS Code, etc.)
#   2. Open a terminal side-by-side
#   3. Copy-paste each command into your terminal, one at a time
#   4. DO NOT run this file as a script — read it and type along
#
# LEGEND:
#   Lines without # prefix     →  commands to type
#   # >> ...                   →  expected output
#   # ...                      →  explanation / narration
#
# ═══════════════════════════════════════════════════════════════════════════



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 2-5: FP32/FP16/INT8, How Quantization Works     ║
# ║     Show the number representations. "Every weight is 4 bytes. Can we   ║
# ║     use fewer?"                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: Setup — A Model to Optimize                           ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘

mkdir -p ~/profiling-demo && cd ~/profiling-demo

python -m venv .venv
source .venv/bin/activate

pip install torch torchvision onnx onnxruntime numpy memory-profiler

# Create a simple image classification model:

cat > model.py << 'PYEOF'
"""A small CNN for demonstration."""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10-like images (3x32x32)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = SimpleCNN()
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (FP32):    {total * 4 / 1024:.1f} KB")
    print(f"Model size (INT8):    {total * 1 / 1024:.1f} KB")
PYEOF

python model.py

# >> Total parameters:     xxx,xxx
# >> Trainable parameters: xxx,xxx
# >> Model size (FP32):    xxx.x KB
# >> Model size (INT8):    xxx.x KB



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 7-8: Profiling — "Measure First"                ║
# ║     "Never optimize without profiling. Don't guess the bottleneck."     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: Basic Profiling — time and cProfile                   ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > profile_basic.py << 'PYEOF'
"""Basic profiling: timing and cProfile."""
import time
import cProfile
import torch
from model import SimpleCNN

model = SimpleCNN()
model.eval()

# ── Method 1: Simple timing ──
input_data = torch.randn(1, 3, 32, 32)

# Warmup
for _ in range(10):
    model(input_data)

# Time single inference
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        output = model(input_data)
elapsed = time.perf_counter() - start
print(f"Average inference: {elapsed / 100 * 1000:.2f} ms")

# ── Method 2: Batch size matters! ──
for batch_size in [1, 8, 32, 128]:
    input_batch = torch.randn(batch_size, 3, 32, 32)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(input_batch)
    start = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            model(input_batch)
    elapsed = time.perf_counter() - start
    per_sample = elapsed / 50 / batch_size * 1000
    print(f"Batch {batch_size:>3d}: {per_sample:.3f} ms/sample, "
          f"{elapsed / 50 * 1000:.2f} ms/batch")

# ── Method 3: cProfile ──
print("\n=== cProfile results ===")
cProfile.run("model(torch.randn(32, 3, 32, 32))", sort="cumulative")
PYEOF

python profile_basic.py

# >> Average inference: x.xx ms
# >> Batch   1: x.xxx ms/sample, x.xx ms/batch
# >> Batch   8: x.xxx ms/sample, x.xx ms/batch
# >> Batch  32: x.xxx ms/sample, x.xx ms/batch
# >> Batch 128: x.xxx ms/sample, x.xx ms/batch
#
# Notice: per-sample time DECREASES with larger batches (parallelism!)



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: PyTorch Profiler                                      ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > profile_torch.py << 'PYEOF'
"""PyTorch profiler — detailed operation-level profiling."""
import torch
from torch.profiler import profile, ProfilerActivity
from model import SimpleCNN

model = SimpleCNN()
model.eval()
input_data = torch.randn(32, 3, 32, 32)

# Profile CPU operations
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            output = model(input_data)

# Top operations by CPU time
print("=== Top operations by CPU time ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Top operations by memory
print("\n=== Top operations by memory ===")
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Group by input shapes
print("\n=== Grouped by input shape ===")
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="cpu_time_total", row_limit=10
))
PYEOF

python profile_torch.py

# >> === Top operations by CPU time ===
# >> Name                    Self CPU    CPU total   # Calls
# >> aten::conv2d            xxx us      xxx us      30
# >> aten::linear            xxx us      xxx us      20
# >> ...
#
# Now you know WHICH operations are slow!



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 10-12: Types of Quantization                    ║
# ║     Dynamic vs Static vs QAT. Start with dynamic (easiest).             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: Dynamic Quantization                                 ~12 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > quantize.py << 'PYEOF'
"""Quantize the model and compare size + speed."""
import os
import time
import torch
from model import SimpleCNN, count_parameters

# ── Original model ──
model = SimpleCNN()
model.eval()

# Save original
torch.save(model.state_dict(), "model_fp32.pth")
fp32_size = os.path.getsize("model_fp32.pth")
print(f"FP32 model size: {fp32_size / 1024:.1f} KB")

# ── Dynamic quantization ──
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantize Linear layers to INT8
    dtype=torch.qint8,
)

# Save quantized
torch.save(quantized_model.state_dict(), "model_int8.pth")
int8_size = os.path.getsize("model_int8.pth")
print(f"INT8 model size:  {int8_size / 1024:.1f} KB")
print(f"Compression:      {fp32_size / int8_size:.1f}x smaller")

# ── Compare inference speed ──
input_data = torch.randn(32, 3, 32, 32)

def benchmark(model, input_data, label, n_runs=200):
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(input_data)
    # Timed
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(input_data)
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"{label}: {elapsed:.2f} ms/batch")
    return elapsed

fp32_time = benchmark(model, input_data, "FP32")
int8_time = benchmark(quantized_model, input_data, "INT8")
print(f"Speedup:          {fp32_time / int8_time:.1f}x faster")

# ── Compare accuracy (with random data) ──
with torch.no_grad():
    fp32_out = model(input_data)
    int8_out = quantized_model(input_data)

# Check if predictions match
fp32_preds = fp32_out.argmax(dim=1)
int8_preds = int8_out.argmax(dim=1)
agreement = (fp32_preds == int8_preds).float().mean()
print(f"\nPrediction agreement: {agreement:.1%}")

# Check numerical difference
max_diff = (fp32_out - int8_out).abs().max().item()
print(f"Max output difference: {max_diff:.6f}")
PYEOF

python quantize.py

# >> FP32 model size: xxx.x KB
# >> INT8 model size:  xxx.x KB
# >> Compression:      x.x smaller
# >> FP32: xx.xx ms/batch
# >> INT8: xx.xx ms/batch
# >> Speedup:          x.x faster
# >> Prediction agreement: 100.0% (or very close)



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: ONNX Export + Inference                               ~10 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > export_onnx.py << 'PYEOF'
"""Export to ONNX and run with ONNX Runtime."""
import os
import time
import numpy as np
import torch
import onnxruntime as ort
from model import SimpleCNN

# ── Export ──
model = SimpleCNN()
model.eval()
dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},  # allow variable batch
    opset_version=17,
)

onnx_size = os.path.getsize("model.onnx")
fp32_size = os.path.getsize("model_fp32.pth")
print(f"ONNX model size: {onnx_size / 1024:.1f} KB")

# ── Inference with ONNX Runtime ──
session = ort.InferenceSession("model.onnx")

# Single sample
input_np = np.random.randn(1, 3, 32, 32).astype(np.float32)
result = session.run(None, {"image": input_np})
print(f"ONNX output shape: {result[0].shape}")

# ── Benchmark: PyTorch vs ONNX Runtime ──
batch = np.random.randn(32, 3, 32, 32).astype(np.float32)
batch_torch = torch.from_numpy(batch)

# PyTorch
for _ in range(20):
    with torch.no_grad():
        model(batch_torch)
start = time.perf_counter()
for _ in range(200):
    with torch.no_grad():
        model(batch_torch)
pytorch_time = (time.perf_counter() - start) / 200 * 1000

# ONNX Runtime
for _ in range(20):
    session.run(None, {"image": batch})
start = time.perf_counter()
for _ in range(200):
    session.run(None, {"image": batch})
onnx_time = (time.perf_counter() - start) / 200 * 1000

print(f"\nPyTorch:      {pytorch_time:.2f} ms/batch")
print(f"ONNX Runtime: {onnx_time:.2f} ms/batch")
print(f"Speedup:      {pytorch_time / onnx_time:.1f}x")

# ── Verify outputs match ──
with torch.no_grad():
    pytorch_out = model(batch_torch).numpy()
onnx_out = session.run(None, {"image": batch})[0]
max_diff = np.abs(pytorch_out - onnx_out).max()
print(f"Max difference: {max_diff:.8f}")
PYEOF

python export_onnx.py

# >> ONNX model size: xxx.x KB
# >> ONNX output shape: (1, 10)
# >> PyTorch:      xx.xx ms/batch
# >> ONNX Runtime: xx.xx ms/batch
# >> Speedup:      x.xx
# >> Max difference: 0.00000xxx



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 14-16: Pruning, Distillation, Pareto Frontier   ║
# ║     These are complementary techniques. Show the accuracy vs size plot.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: Memory Profiling                                      ~8 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > profile_memory.py << 'PYEOF'
"""Profile memory usage during inference."""
import torch
from model import SimpleCNN

model = SimpleCNN()
model.eval()

# ── Method 1: torch.cuda.memory (GPU) or manual tracking (CPU) ──
print("=== Memory tracking ===")

# Track peak memory with a simple approach
import tracemalloc
tracemalloc.start()

input_data = torch.randn(32, 3, 32, 32)
with torch.no_grad():
    output = model(input_data)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory: {current / 1024:.1f} KB")
print(f"Peak memory:    {peak / 1024:.1f} KB")

# ── Method 2: Model size breakdown ──
print("\n=== Model size breakdown ===")
for name, param in model.named_parameters():
    size_kb = param.numel() * param.element_size() / 1024
    print(f"  {name:30s} shape={str(list(param.shape)):20s} {size_kb:.1f} KB")

total_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
print(f"  {'TOTAL':30s} {'':20s} {total_kb:.1f} KB")

# ── Method 3: Activation memory ──
print("\n=== Activation sizes during forward pass ===")
hooks = []
def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            size_kb = output.numel() * output.element_size() / 1024
            print(f"  {name:30s} output={str(list(output.shape)):20s} {size_kb:.1f} KB")
    return hook

for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # leaf modules only
        hooks.append(module.register_forward_hook(hook_fn(name)))

with torch.no_grad():
    _ = model(torch.randn(1, 3, 32, 32))

for h in hooks:
    h.remove()
PYEOF

python profile_memory.py

# >> === Memory tracking ===
# >> Current memory: xxx.x KB
# >> Peak memory:    xxx.x KB
# >>
# >> === Model size breakdown ===
# >>   features.0.weight    [32, 3, 3, 3]       3.4 KB
# >>   ...
# >>
# >> === Activation sizes ===
# >>   features.0           [1, 32, 32, 32]     128.0 KB
# >>   ...



# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 7: Summary — Putting It All Together                      ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘

cat > summary.py << 'PYEOF'
"""Final comparison of all optimization techniques."""
import os

print("=" * 60)
print("MODEL OPTIMIZATION SUMMARY")
print("=" * 60)

files = {
    "FP32 (PyTorch)": "model_fp32.pth",
    "INT8 (Dynamic Quant)": "model_int8.pth",
    "ONNX": "model.onnx",
}

for label, path in files.items():
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {label:25s} {size_kb:>8.1f} KB")

print()
print("Optimization techniques applied:")
print("  1. Dynamic quantization (Linear layers → INT8)")
print("  2. ONNX export (optimized inference runtime)")
print()
print("Further optimizations available:")
print("  3. Static quantization (calibrate with real data)")
print("  4. Pruning (remove unimportant weights)")
print("  5. Knowledge distillation (train small model)")
print("  6. TensorRT / OpenVINO (hardware-specific)")
PYEOF

python summary.py



# ═══════════════════════════════════════════════════════════════════════════
# WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════
#
# What we covered today:
#
#   Act 1: Setup — a model to optimize
#   Act 2: Basic profiling (time, cProfile, batch sizes)
#   Act 3: PyTorch profiler (operation-level detail)
#   Act 4: Dynamic quantization (FP32 → INT8)
#   Act 5: ONNX export + ONNX Runtime inference
#   Act 6: Memory profiling (weights + activations)
#   Act 7: Summary of all techniques
#
# Key rule: PROFILE FIRST, then optimize the actual bottleneck.
#
# The full course progression:
#   Git → venv/Docker → W&B → CI/CD → APIs → Profiling/Quantization
#   (version everything → automate → ship → optimize)
# ═══════════════════════════════════════════════════════════════════════════

cd ~
deactivate 2>/dev/null
# rm -rf ~/profiling-demo   # uncomment to clean up
