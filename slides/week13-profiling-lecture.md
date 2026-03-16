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

# The Problem: Your Model is Too Fat and Slow

Your model is deployed (week 12). But look at these numbers:

| Model | FP32 Size | Your Laptop RAM |
|-------|-----------|-----------------|
| **LLaMA-7B** | 28 GB | 16 GB |
| **GPT-3** | 700 GB | 16 GB |
| **BERT-base** | 440 MB | 16 GB |

LLaMA doesn't even *fit* in memory. GPT-3 needs 44 laptops.

**How do we run these?**

```
Week 7-8:   Evaluate & Tune                       ✓
Week 9-11:  Build, version, automate               ✓
Week 12:    Ship it as API/demo                    ✓
Week 13:    Make it FAST and SMALL                 ← you are here
```

---

# Why This Matters

| Deployment Target | Size Constraint | Speed Constraint |
|-------------------|----------------|-----------------|
| **Mobile phone** | < 50 MB | < 50 ms |
| **Edge / IoT** | < 10 MB | CPU only |
| **Cloud API** | Fit in GPU RAM | < 100 ms (GPU costs $$) |
| **LLM on laptop** | Fit in 16 GB RAM | Usable speed |

> **Training happens once. Inference happens millions of times.**

Every millisecond saved, every megabyte trimmed — multiplied by every user, every request, every day.

Today's plan: put your model on a diet.

---

<!-- _class: lead -->

# Part 1: Profiling — The Checkup

Before any diet, you visit the doctor first.

---

# The Itemized Receipt

"My code is slow" is like saying "I spent too much money."

That's not useful. You need the **receipt**:

> "80% of your time is in one matrix multiply."

A **profiler** gives you that receipt.

```
Profile → Find bottleneck → Optimize → Profile again
```

Without profiling, you'll optimize the wrong thing:

> *"Premature optimization is the root of all evil."*
> — Donald Knuth

---

# Compute-Bound vs Memory-Bound

Think of a restaurant kitchen:

| | Compute-Bound | Memory-Bound |
|---|---|---|
| **Analogy** | Chef is too slow cooking | Waiter is too slow bringing ingredients |
| **Bottleneck** | Not enough math power (FLOPS) | Not enough data bandwidth (GB/s) |
| **When?** | Large batches, big matrix multiplies | Small batches, loading model weights |
| **Fix** | Fewer operations, faster hardware | Smaller model, better caching |

**Key insight:** Most LLM inference is **memory-bound**.

The GPU spends more time *loading* weights than *computing* with them.

This is why quantization helps — fewer bytes to load!

---

# Simple Profiling

The simplest profiler: a stopwatch. But do it right.

```python
import time, torch

def benchmark(model, input_data, n_runs=100):
    # Rule 1: Warmup (JIT compilation, cache filling)
    for _ in range(10):
        with torch.no_grad():
            model(input_data)

    # Rule 2: Sync GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(input_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # milliseconds
```

**Three rules:** (1) Warmup. (2) Sync GPU. (3) Average many runs.

For deeper profiling, use `torch.profiler` — see the notebook.

<!-- ⌨ DEMO: Profile a model, find the bottleneck -->

---

<!-- _class: lead -->

# Part 2: Quantization — The Diet

The single highest-ROI optimization you can do.

---

# What is Quantization?

**Analogy: Rounding prices at the grocery store.**

Instead of adding up:
```
$4.99281 + $3.00192 + $7.49837 = $15.49310    (FP32: precise)
```

Just round and add:
```
$5 + $3 + $7 = $15                             (INT8: fast)
```

The total is close enough. And the mental math is *way* faster.

**That's quantization.** Use fewer bits per number.

| Format | Bytes per number | Precision |
|--------|-----------------|-----------|
| **FP32** | 4 bytes | Very high |
| **FP16** | 2 bytes | High |
| **INT8** | 1 byte | Moderate |
| **INT4** | 0.5 bytes | Low |

---

# Model Size by Format

| Format | Bytes/param | 100M param model | 7B param model (LLaMA) |
|--------|------------|------------------|------------------------|
| **FP32** | 4 | 400 MB | **28 GB** |
| **FP16** | 2 | 200 MB | 14 GB |
| **INT8** | 1 | 100 MB | 7 GB |
| **INT4** | 0.5 | 50 MB | **3.5 GB** |

LLaMA-7B in FP32: needs a $10,000 GPU with 32+ GB VRAM.
LLaMA-7B in INT4: **fits on your laptop with 4 GB free.**

Same model. Same knowledge. Just stored more efficiently.

---

# Why Does It Work?

Neural network weights are not spread evenly — most cluster near zero:

```
  Count
  |          ████
  |        ████████
  |      ████████████
  |    ████████████████
  |  ████████████████████
  └────────────────────────
  -1.0       0.0       1.0
```

Rounding these values introduces tiny errors. But the model barely notices.

**Empirically:** INT8 quantization loses **< 1% accuracy** on most models.

The weights that matter most (near zero) are the ones quantization handles best.

---

# Dynamic Quantization: The One-Liner

```python
import torch

model = MyModel()
model.eval()

# One line. That's it.
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# Use it like normal
prediction = quantized_model(input_data)
```

**What you get:** 2-4x smaller model, faster inference on CPU.

**What it costs:** Usually < 1% accuracy drop. No calibration data needed.

There are fancier approaches (static quantization, quantization-aware training) that give better quality but need more work. Details in the notebook.

---

# The LLaMA Story

In February 2023, Meta released **LLaMA** — a powerful language model.

**Problem:** LLaMA-7B needs 28 GB in FP32. Only big GPUs could run it.

Then Georgi Gerganov built **llama.cpp** — a C++ inference engine with **4-bit quantization**.

| | Before (FP32) | After (INT4) |
|---|---|---|
| **Size** | 28 GB | 3.5 GB |
| **Hardware needed** | A100 GPU ($10K+) | MacBook Air |
| **Cost** | Cloud GPU rental | Free (your laptop) |
| **Privacy** | Data sent to cloud | Everything stays local |

Suddenly, a powerful LLM runs on a phone. On a Raspberry Pi.

**Quantization didn't just save money — it democratized AI.**

This is why every LLM you download today (Mistral, Llama 3, Phi) comes in quantized versions (GGUF, GPTQ, AWQ).

<!-- ⌨ DEMO: Quantize a model, compare size and speed -->

---

<!-- _class: lead -->

# Part 3: Other Optimizations

Quantization is the main course. Here are the side dishes.

---

# Pruning — The Jenga Approach

**Analogy:** In Jenga, you remove blocks that aren't structurally important. The tower still stands.

Neural networks are the same. Many weights are close to zero — they barely contribute.

```
Before pruning:  [0.8, 0.001, -0.7, 0.002, 0.9, -0.003]
After pruning:   [0.8,   0,   -0.7,   0,   0.9,    0  ]
```

Set small weights to zero. The model still works.

**Typical result:** 50-90% of weights pruned with < 1% accuracy loss.

**Structured pruning** (removing entire neurons/channels) gives real speedups because hardware can skip whole blocks of computation.

---

# Knowledge Distillation — Master and Apprentice

**Analogy:** A master chef (big model) trains an apprentice (small model).

The apprentice doesn't read the cookbook (raw data) — they watch the master work and learn the *style*: which flavors pair well, when to trust instinct over recipe.

```
Teacher (BERT-base, 440 MB, slow but smart)
    │
    │  "Here's how I'd rate each answer..."
    ▼
Student (DistilBERT, 60 MB, fast and almost as smart)
```

The student learns from the teacher's **soft predictions** — not just "cat" vs "dog", but "90% cat, 8% lynx, 2% dog." That extra information is rich.

**Result:** DistilBERT keeps **97% of BERT's accuracy** at **7x smaller** and **2x faster**.

---

# ONNX — The PDF of Machine Learning

**Analogy:** You wrote a document in Google Docs. Your colleague uses Word. Your client uses Pages. Export as **PDF** — now everyone can read it.

**ONNX** does the same for models. Train in PyTorch, run *anywhere*.

```python
# Export (3 lines)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["image"], output_names=["prediction"])

# Inference without PyTorch (3 lines)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"image": input_array})
```

**Why bother?** ONNX Runtime automatically fuses operations, optimizes memory, and runs 2-5x faster than vanilla PyTorch inference. And you don't need PyTorch installed.

---

# Combining Techniques — The Full Pipeline

These optimizations **stack**:

```
Large Model (FP32, 440 MB, 90 ms)
    │
    ├── Distill    → Smaller architecture (fewer layers)
    ├── Prune      → Remove useless weights
    ├── Quantize   → INT8 (4x smaller per weight)
    └── ONNX       → Fused operators, no framework overhead
    │
Small Model (INT8, 60 MB, 8 ms) — 97% accuracy
```

**Real example:** BERT-base (440 MB, 90 ms) to DistilBERT quantized (60 MB, 8 ms).

Same task. Nearly same accuracy. 7x smaller. 11x faster.

---

# The Pareto Frontier

```
Accuracy
  100% │   ●  FP32 (baseline)
       │  ●    FP16
   99% │  ●     INT8 (static)
       │ ●       INT8 (dynamic)
   98% │●         Pruned + INT8
       │
   95% │           ● INT4 (aggressive)
       │
       └──────────────────────────
      Large                    Small
              Model Size
```

**Pick the point that meets YOUR requirements:**

| Target | Recommended |
|--------|------------|
| Mobile app | INT8 + pruning (small & fast) |
| Cloud API | FP16 (fast & accurate) |
| Research | FP32 (maximum accuracy) |
| LLM on laptop | INT4 (fit in RAM) |

---

# Practical Workflow

```
1. PROFILE the model
   → Where is the time going?
   → Compute-bound or memory-bound?

2. Try the EASIEST optimization first
   → FP16 inference (one flag)
   → Dynamic quantization (one line)
   → ONNX export (standard)

3. MEASURE the improvement
   → Latency, throughput, model size
   → CHECK ACCURACY hasn't dropped!

4. Go FURTHER if needed
   → Static quantization
   → Pruning
   → Distillation
   → Quantization-aware training
```

Start simple. Measure everything. Stop when you hit your target.

---

# Key Takeaways

**1. Profile first, optimize the bottleneck.**
Don't guess. Measure. The bottleneck is never where you think.

**2. Quantization = highest ROI.**
One line of code. 2-4x smaller. Minimal accuracy loss.

**3. Combine techniques for maximum effect.**
Distill + prune + quantize + ONNX = dramatic improvements.

**4. Pick your point on the Pareto frontier.**
There's no single "best" — it depends on your deployment target.

---

# The Full Course Arc

```
Week 7-8:   Evaluate & Tune      → measure your model
Week 9:     Git                   → version your code
Week 10:    Environments          → version your setup
Week 11:    CI/CD                 → automate quality
Week 12:    APIs & Demos          → ship your model
Week 13:    Profiling & Quant     → make it fast and small
```

**You can now build, track, test, deploy, and optimize ML systems.**

From a Jupyter notebook to a production-ready, optimized, deployed model — that's the journey of this course.
