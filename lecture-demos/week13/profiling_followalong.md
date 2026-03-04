---
title: "Model Profiling & Quantization — Follow-Along Guide"
subtitle: "Week 13 · CS 203 · Software Tools and Techniques for AI"
author: "Prof. Nipun Batra · IIT Gandhinagar"
date: "Spring 2026"
geometry: margin=2cm
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{CS 203 — Profiling \& Quantization}
  - \fancyhead[R]{Follow-Along Guide}
  - \usepackage{tcolorbox}
  - \tcbuselibrary{skins,breakable}
  - \newtcolorbox{tipbox}{colback=green!5,colframe=green!50!black,title=Tip,fonttitle=\bfseries,breakable}
  - \newtcolorbox{warningbox}{colback=red!5,colframe=red!50!black,title=Warning,fonttitle=\bfseries,breakable}
  - \newtcolorbox{slidebox}{colback=blue!5,colframe=blue!60!black,breakable}
  - \newtcolorbox{actbox}{colback=gray!8,colframe=gray!60!black,breakable,top=2mm,bottom=2mm}
---

\vspace{-0.5cm}

# How to Use This Guide

\begin{itemize}
\item Open \texttt{profiling\_followalong.sh} in your editor (left half of screen)
\item Open a terminal (right half of screen)
\item Copy-paste each command, one at a time
\item \textbf{Type it yourself} --- that's how you learn
\end{itemize}

**Legend:**  `$` = command to type. `>>` = expected output. Blue boxes = projector slide.

---

\begin{slidebox}\textbf{Projector: Slides 2--5 --- FP32/FP16/INT8, How Quantization Works}
Every weight is 4 bytes (FP32). Quantization maps them to 1 byte (INT8). 4x smaller, 2--3x faster.
\end{slidebox}

\begin{actbox}\textbf{\large Act 1: Setup --- A Model to Optimize \hfill $\sim$5 min}
\end{actbox}

```bash
$ mkdir -p ~/profiling-demo && cd ~/profiling-demo
$ python -m venv .venv && source .venv/bin/activate
$ pip install torch torchvision onnx onnxruntime numpy memory-profiler
$ python model.py
>> Total parameters:     xxx,xxx
>> Model size (FP32):    xxx.x KB
>> Model size (INT8):    xxx.x KB
```

\begin{slidebox}\textbf{Projector: Slides 7--8 --- ``Measure First''}
Never optimize without profiling. Don't guess the bottleneck.
\end{slidebox}

\begin{actbox}\textbf{\large Act 2: Basic Profiling --- time and cProfile \hfill $\sim$10 min}
\end{actbox}

```bash
$ python profile_basic.py
>> Average inference: x.xx ms
>> Batch   1: x.xxx ms/sample
>> Batch  32: x.xxx ms/sample    (much less per sample!)
>> Batch 128: x.xxx ms/sample    (batching = free speedup)
```

\begin{tipbox}
\textbf{Batch size matters.} Per-sample latency decreases with larger batches because GPU/CPU parallelism amortizes overhead. Always benchmark at your target batch size.
\end{tipbox}

\begin{actbox}\textbf{\large Act 3: PyTorch Profiler \hfill $\sim$10 min}
\end{actbox}

```bash
$ python profile_torch.py
>> === Top operations by CPU time ===
>> Name                    Self CPU     CPU total    # Calls
>> aten::conv2d            xxx us       xxx us       30
>> aten::linear            xxx us       xxx us       20
```

Now you know \textbf{which operations} are the bottleneck. Optimize those, not everything.

\newpage

\begin{slidebox}\textbf{Projector: Slides 10--12 --- Types of Quantization}
Dynamic (easiest, at inference time), Static (needs calibration data), QAT (during training, best quality).
\end{slidebox}

\begin{actbox}\textbf{\large Act 4: Dynamic Quantization \hfill $\sim$12 min}
\end{actbox}

```bash
$ python quantize.py
>> FP32 model size: xxx.x KB
>> INT8 model size:  xxx.x KB
>> Compression:      x.x smaller
>> FP32: xx.xx ms/batch
>> INT8: xx.xx ms/batch
>> Speedup:          x.x faster
>> Prediction agreement: 100.0%
```

\begin{tipbox}
Dynamic quantization is \textbf{one line of code}: \texttt{torch.quantization.quantize\_dynamic(model, \{nn.Linear\}, dtype=torch.qint8)}. It quantizes Linear layers from FP32 to INT8.
\end{tipbox}

\begin{actbox}\textbf{\large Act 5: ONNX Export + Inference \hfill $\sim$10 min}
\end{actbox}

```bash
$ python export_onnx.py
>> ONNX model size: xxx.x KB
>> PyTorch:      xx.xx ms/batch
>> ONNX Runtime: xx.xx ms/batch
>> Speedup:      x.xx
>> Max difference: 0.00000xxx
```

ONNX Runtime runs optimized inference \textbf{without needing PyTorch installed}. Great for deployment.

\begin{warningbox}
Always verify that the exported model produces the same outputs as the original! Small numerical differences (${<}10^{-5}$) are expected due to floating-point order of operations.
\end{warningbox}

\newpage

\begin{actbox}\textbf{\large Act 6: Memory Profiling \hfill $\sim$8 min}
\end{actbox}

```bash
$ python profile_memory.py
>> === Model size breakdown ===
>>   features.0.weight    [32, 3, 3, 3]       3.4 KB
>>   classifier.0.weight  [64, 128]           32.0 KB
>>   ...
>> === Activation sizes ===
>>   features.0           [1, 32, 32, 32]     128.0 KB
```

Two kinds of memory: \textbf{weights} (model parameters, fixed) and \textbf{activations} (intermediate outputs, grow with batch size). Quantization shrinks weights; smaller batch sizes reduce activation memory.

\begin{actbox}\textbf{\large Act 7: Summary \hfill $\sim$5 min}
\end{actbox}

```bash
$ python summary.py
>> MODEL OPTIMIZATION SUMMARY
>>   FP32 (PyTorch)         xxx.x KB
>>   INT8 (Dynamic Quant)   xxx.x KB
>>   ONNX                   xxx.x KB
```

---

# Quick Reference

| Technique | Code | Effect |
|-----------|------|--------|
| Count params | `sum(p.numel() for p in model.parameters())` | Know your model size |
| Time inference | `time.perf_counter()` with warmup | Measure latency |
| Profile ops | `torch.profiler.profile()` | Find bottleneck ops |
| Dynamic quant | `torch.quantization.quantize_dynamic(...)` | Shrink Linear layers |
| Export ONNX | `torch.onnx.export(model, input, "m.onnx")` | Portable format |
| ONNX inference | `ort.InferenceSession("m.onnx")` | Fast inference |
| Memory profile | `tracemalloc.start()` | Track memory |

\vspace{0.5cm}

**Exam-relevant concepts:**

- FP32 (32 bits, 4 bytes), FP16 (16 bits, 2 bytes), INT8 (8 bits, 1 byte)
- INT8 quantization: scale + zero\_point mapping; loses $<$1\% accuracy
- Compute-bound (GPU busy doing math) vs memory-bound (GPU waits for data)
- Most LLM inference is memory-bound $\to$ quantization helps most
- Dynamic quantization: at inference, no calibration needed
- Static quantization: uses calibration data, better quality
- Profile first, then optimize the actual bottleneck (not guessing)
- Warmup before benchmarking (JIT, cache effects)
