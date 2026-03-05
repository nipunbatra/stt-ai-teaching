---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->

# Model Profiling & Quantization

## Week 13 · CS 203: Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are

```
Week 7-8:   Evaluate, tune & track              ✓
Week 9-10:  Version code + environment           ✓
Week 11:    Automate with CI/CD                  ✓
Week 12:    Ship as API/demo                     ✓
Week 13:    Make it FAST and SMALL               ← you are here
```

Your model is deployed. But:
- Inference takes 500ms (user waits too long)
- Model is 2GB (won't fit on mobile/edge)
- GPU costs $3/hour (your startup burns money)

**Can we make it faster and smaller without losing accuracy?**

---

# Why This Matters Now

| Deployment Target | Constraint |
|-------------------|-----------|
| **Mobile phone** | < 50MB model, < 50ms inference |
| **Edge device (IoT)** | < 10MB, runs on CPU only |
| **Web browser** | < 100MB download, JavaScript runtime |
| **Cloud API** | < 100ms latency, GPU costs money |
| **Autonomous vehicle** | < 10ms, safety-critical |

**Training is done once. Inference happens millions of times.**

Optimizing inference has massive ROI.

---

<!-- _class: lead -->

# Part 1: Theory — Numbers in Computers

---

# How Numbers Are Stored

**FP32 (Float32) — Standard precision:**

```
Sign (1 bit) | Exponent (8 bits) | Mantissa (23 bits)
     0        |    01111100       |  01000000000000000000000
     +         ×  2^(-3)          × 1.25  =  0.15625
```

**32 bits = 4 bytes per number.**

A model with 1 billion parameters = **4 GB** just for weights.

---

# Floating Point Representations

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| **FP32** | 32 | 8 bits | 23 bits | ±3.4 × 10³⁸ | ~7 decimal digits |
| **FP16** | 16 | 5 bits | 10 bits | ±65,504 | ~3 decimal digits |
| **BF16** | 16 | 8 bits | 7 bits | ±3.4 × 10³⁸ | ~2 decimal digits |
| **INT8** | 8 | N/A | N/A | -128 to 127 | Exact integer |
| **INT4** | 4 | N/A | N/A | -8 to 7 | Exact integer |

---

# Model Size by Format

| Format | Bytes/param | 100M model | 7B model (LLaMA) |
|--------|------------|------------|-------------------|
| **FP32** | 4 | 400 MB | 28 GB |
| **FP16** | 2 | 200 MB | 14 GB |
| **INT8** | 1 | 100 MB | 7 GB |
| **INT4** | 0.5 | 50 MB | 3.5 GB |

**LLaMA-7B in FP32:** Needs 28 GB VRAM (A100 GPU)
**LLaMA-7B in INT4:** Fits in 4 GB VRAM (runs on a laptop!)

This is why quantization transformed LLM deployment.

---

# FP16 vs BF16

```
FP16:  [1 sign] [5 exponent] [10 mantissa]  → more precision, smaller range
BF16:  [1 sign] [8 exponent] [7 mantissa]   → less precision, same range as FP32
```

**BF16 is preferred for training:**
- Same exponent range as FP32 → no overflow during gradient updates
- Less precision is OK because gradients are noisy anyway

**FP16 is fine for inference:**
- Weights don't change, so overflow isn't a concern
- Extra precision helps prediction quality slightly

---

# How INT8 Quantization Works

Map floating point values to 8-bit integers:

```
FP32 weights:  [-0.8, 0.3, 1.2, -0.1, 0.7]

Step 1: Find range → min=-0.8, max=1.2
Step 2: Compute scale and zero point
        scale = (max - min) / (127 - (-128))
              = 2.0 / 255 = 0.00784
        zero_point = round(-min / scale)
                   = round(0.8 / 0.00784) = 102

Step 3: Quantize
        q(x) = round(x / scale) + zero_point
        INT8 weights:  [0, 140, 255, 89, 191]

Dequantize: float = (int_value - zero_point) × scale
```

---

# Why Does Quantization Work?

Model weights are **not uniformly important:**

```
Weight distribution (typical neural network):

  Count
  |          ████
  |        ████████
  |      ████████████
  |    ████████████████
  |  ████████████████████
  └────────────────────────
  -1.0       0.0       1.0
```

Most weights cluster near zero. Quantization introduces small rounding errors — but the model's predictions barely change.

**Empirically:** INT8 quantization loses < 1% accuracy on most models.

---

# Quantization: Symmetric vs Asymmetric

**Symmetric:** zero_point = 0, range is [-max_abs, +max_abs]
```
scale = max(|min|, |max|) / 127
q(x) = round(x / scale)           # simpler math, slightly less precise
```

**Asymmetric:** zero_point ≠ 0, maps full [min, max] range
```
scale = (max - min) / 255
zero_point = round(-min / scale)
q(x) = round(x / scale) + zero_point   # more precise, more complex
```

**Symmetric** is faster (no zero_point addition). **Asymmetric** is more accurate for skewed distributions (e.g., ReLU outputs are always ≥ 0).

---

# Compute-Bound vs Memory-Bound

**Compute-bound:** GPU is busy doing math.
- Large batch sizes, big matrix multiplications
- **Bottleneck:** arithmetic throughput (FLOPS)
- Fix: use faster hardware, reduce operations

**Memory-bound:** GPU waits for data to arrive from memory.
- Small batch sizes, attention layers, loading weights
- **Bottleneck:** memory bandwidth (GB/s)
- Fix: reduce model size, use faster memory, cache better

---

# The Roofline Model

```
Performance
(GFLOPS)
    │         ╱────────────── Peak compute
    │       ╱
    │     ╱   ← Memory-bound    Compute-bound →
    │   ╱     (slope = bandwidth)
    │ ╱
    │╱
    └──────────────────────────────────
      Operational Intensity (FLOPS/byte)
```

**Operational intensity** = FLOPs per byte loaded from memory.

- Low intensity (loading weights, small batches) → memory-bound
- High intensity (large matmuls) → compute-bound

**Most LLM inference (single token) is memory-bound.** That's why quantization helps so much — fewer bytes to load!

---

<!-- _class: lead -->

# Part 2: Profiling

*Measure before you optimize*

<!-- ⌨ NOTEBOOK → Profiling demos -->

---

# Rule #1: Measure First

**Never optimize without profiling.**

```
"Premature optimization is the root of all evil."
    — Donald Knuth
```

Common mistakes:
- Optimizing code that takes 1% of runtime
- Guessing the bottleneck (usually wrong)
- Adding complexity for negligible speedup

**Profile → Find bottleneck → Optimize → Profile again**

---

# Profiling Tools

| Tool | What It Measures | Best For |
|------|-----------------|----------|
| `time.time()` | Wall clock time | Quick checks |
| `time.perf_counter()` | High-precision time | Benchmarking |
| `cProfile` | Function call counts + time | Python bottlenecks |
| `torch.profiler` | GPU ops, memory, CUDA kernels | PyTorch models |
| `memory_profiler` | Memory usage per line | Memory leaks |
| `py-spy` | Sampling profiler (no code change) | Production |
| `line_profiler` | Time per line | Micro-optimization |

---

# Benchmarking: Getting It Right

```python
import time
import torch

def benchmark(model, input_data, n_runs=100):
    """Measure average inference time."""
    # Warmup (JIT compilation, cache filling)
    for _ in range(10):
        with torch.no_grad():
            model(input_data)

    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(input_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for GPU to finish!
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms
```

**Three rules:** Warmup. Synchronize GPU. Average many runs.

---

# torch.profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            output = model(input_data)

print(prof.key_averages().table(
    sort_by="cpu_time_total", row_limit=10
))
```

```
Name                    CPU total   CUDA total   # Calls
aten::conv2d            12.5ms      8.2ms        30
aten::batch_norm        3.2ms       1.1ms        30
aten::linear            2.8ms       1.5ms        20
aten::relu              1.1ms       0.3ms        30
```

**Now you know WHICH operations are slow.**

---

# Memory Profiling

**Two types of memory to track:**

| Type | What | Grows With |
|------|------|-----------|
| **Weights** | Model parameters | Model size |
| **Activations** | Intermediate outputs | Batch size |

```python
# Weight memory
weight_mem = sum(p.numel() * p.element_size()
                 for p in model.parameters())
print(f"Weights: {weight_mem / 1024**2:.1f} MB")

# Peak GPU memory (during forward pass)
torch.cuda.reset_peak_memory_stats()
output = model(input_batch)
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU: {peak:.1f} MB")
```

---

<!-- _class: lead -->

# Part 3: Quantization

*Make it smaller and faster*

---

# Types of Quantization

| Type | When Applied | Needs Data? | Quality | Speed |
|------|-------------|-------------|---------|-------|
| **Dynamic** | At inference time | No | Good | Medium |
| **Static (PTQ)** | After training | Yes (calibration) | Better | Good |
| **QAT** | During training | Yes (full training) | Best | Best |

**Start with dynamic** (easiest, no calibration data needed).
Move to static/QAT if accuracy drops too much.

---

# Dynamic Quantization (PyTorch)

```python
import torch

# Original model
model = MyModel()
model.eval()

# Quantize Linear and LSTM layers to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8,
)

# That's it! Use quantized_model for inference.
pred = quantized_model(input_data)
```

**One line of code.** Quantizes weights at load time. Activations stay in FP32 but use INT8 kernels for matrix multiplication.

---

# Static Quantization (PTQ)

```python
# 1. Prepare model with observers
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('x86')
prepared = torch.quantization.prepare(model)

# 2. Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:  # ~100-1000 samples
        prepared(batch)

# 3. Convert to quantized model
quantized = torch.quantization.convert(prepared)
```

**Why calibration?** Static quantization also quantizes activations. It needs to observe real activation ranges to choose good scale/zero_point values.

**More accurate than dynamic** but requires a calibration dataset.

---

# Quantization-Aware Training (QAT)

```python
# Insert fake quantization during training
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
prepared = torch.quantization.prepare_qat(model)

# Train normally — fake quantize simulates INT8 errors
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = criterion(prepared(batch), labels)
        loss.backward()
        optimizer.step()

# Convert to actual quantized model
quantized = torch.quantization.convert(prepared)
```

**The model learns to be robust to quantization noise.** Best accuracy, but requires retraining.

---

# Quantization Summary

```
                     Accuracy
                        │
  FP32 (baseline)  ●    │
                        │
  Static quant     ●    │    ← usually < 1% drop
  QAT              ●    │    ← often < 0.5% drop
  Dynamic quant    ●    │    ← depends on model
                        │
  INT4 (aggressive) ●   │    ← 1-3% drop typical
                        │
                   ─────┴────────────
                   4x    3x    2x    1x
                       Model Size Reduction
```

---

<!-- _class: lead -->

# Part 4: ONNX & Other Optimizations

---

# ONNX: Portable Model Format

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=17,
)
```

**ONNX Runtime** — optimized inference engine:
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"image": input_array})
```

**Why ONNX?** Run PyTorch models without PyTorch. Often 2-5x faster.

---

# ONNX Runtime Optimizations

ONNX Runtime automatically applies:

| Optimization | What It Does |
|-------------|-------------|
| **Operator fusion** | Combine Conv+BN+ReLU into one kernel |
| **Constant folding** | Pre-compute constant expressions |
| **Memory planning** | Reuse memory buffers across operators |
| **Parallel execution** | Run independent ops concurrently |

```python
# Enable all optimizations
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model.onnx", opts)
```

**No code changes needed.** Just switching to ONNX Runtime gives you free speedups.

---

# Pruning: Remove Unimportant Weights

```python
import torch.nn.utils.prune as prune

# Remove 30% of weights (set to zero)
prune.l1_unstructured(model.layer1, name="weight", amount=0.3)

# See the mask
print(model.layer1.weight_mask)  # 0s and 1s
```

**Unstructured pruning:** Set individual weights to zero.
- Sparse tensor, but hardware doesn't always accelerate sparse ops.

**Structured pruning:** Remove entire channels/heads.
- Actually reduces computation. Hardware-friendly.

**Typical:** 50-90% of weights can be pruned with < 1% accuracy loss.

---

# Knowledge Distillation

**Train a small "student" to mimic a large "teacher":**

```
Teacher (large, slow, accurate)
    │
    │ soft predictions (probabilities)
    ▼
Student (small, fast, almost as accurate)
```

```python
# Distillation loss
teacher_logits = teacher(x)
student_logits = student(x)

# Soft labels from teacher
T = 4.0  # temperature
soft_loss = KL_divergence(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T * T

# Hard labels from data
hard_loss = cross_entropy(student_logits, labels)

loss = 0.7 * soft_loss + 0.3 * hard_loss
```

---

# Combining Techniques

These optimizations are **complementary:**

```
Large Model (FP32, 1GB, 100ms)
        │
    Distillation → Smaller architecture
        │
    Pruning → Remove 50% of weights
        │
    Quantization → INT8 (4x smaller)
        │
    ONNX Runtime → Fused operators
        │
Small Model (INT8, 30MB, 15ms)
```

**Real-world example:** BERT-base (440MB, 90ms) → DistilBERT quantized (60MB, 8ms) with 97% of the accuracy.

---

# The Pareto Frontier

```
Accuracy
  100% │   ●  FP32 (baseline)
       │  ●    FP16
   99% │  ●     Static INT8
       │ ●       Dynamic INT8
   98% │●         Pruned + INT8
       │
   95% │           ● Heavy pruning + INT4
       │
       └──────────────────────────
       4MB  2MB  1MB  0.5MB  0.2MB
                Model Size
```

**Pick the point on the frontier that meets YOUR requirements.**

- Mobile app → INT8 + pruning (small, fast)
- Cloud API → FP16 (fast, accurate)
- Research → FP32 (maximum accuracy)
- LLM on laptop → INT4 (fits in RAM)

---

# Practical Optimization Workflow

```
1. Profile the model
   → Is it compute-bound or memory-bound?
   → Which operators are slow?

2. Try the easiest optimization first
   → FP16 inference (one flag)
   → Dynamic quantization (one line)
   → ONNX export (standard)

3. Measure the improvement
   → Latency, throughput, model size
   → CHECK ACCURACY hasn't dropped

4. Go further if needed
   → Static quantization (need calibration data)
   → Pruning (structured for real speedup)
   → Distillation (need to train student)
   → QAT (need to retrain)
```

---

# Hardware-Specific Optimizations

| Hardware | Tool | Format |
|----------|------|--------|
| **NVIDIA GPU** | TensorRT | FP16/INT8 + kernel fusion |
| **Intel CPU** | OpenVINO | INT8 + graph optimization |
| **Apple Silicon** | Core ML | ANE-optimized models |
| **Mobile (Android)** | TFLite | INT8 + delegation |
| **Mobile (iOS)** | Core ML | FP16/INT8 |
| **Browser** | ONNX.js / TF.js | WebGL/WASM |

**For this course:** ONNX Runtime (cross-platform) + PyTorch quantization (native).

---

# Key Takeaways

1. **Profile before optimizing** — measure, don't guess
   - `time.perf_counter()` for timing, `torch.profiler` for ops
   - Warmup, sync GPU, average many runs

2. **Number formats matter**
   - FP32 (4B), FP16 (2B), INT8 (1B), INT4 (0.5B)
   - Most models tolerate INT8 with < 1% accuracy loss

3. **Quantization** is the highest-ROI optimization
   - Dynamic: one line, no data needed
   - Static: better quality, needs calibration
   - QAT: best quality, needs retraining

4. **Compute-bound vs memory-bound** determines your strategy
   - Memory-bound → quantize (fewer bytes to load)
   - Compute-bound → prune, distill (fewer operations)

5. **Combine techniques** for maximum effect: distill → prune → quantize → ONNX

---

# The Full Course Arc

```
Week 8:   Git          → version your code
Week 9:   venv/Docker  → version your environment
Week 10:  W&B/Optuna   → version your experiments
Week 11:  CI/CD        → automate quality
Week 12:  FastAPI      → ship your model
Week 13:  Quantization → make it fast and small
```

**You can now build, track, test, deploy, and optimize ML systems.**

---

<!-- _class: lead -->

# Questions?

**Exam-relevant concepts:**
- FP32/FP16/BF16/INT8: bits, bytes, range, precision
- How INT8 quantization works (scale + zero_point)
- Symmetric vs asymmetric quantization
- Dynamic vs static vs QAT quantization
- Compute-bound vs memory-bound: definitions and implications
- The roofline model (conceptual)
- Why quantization works (weight distribution)
- Pruning vs quantization vs distillation — what each does
- Profile first, optimize the bottleneck
- Benchmarking: warmup, GPU sync, average runs
