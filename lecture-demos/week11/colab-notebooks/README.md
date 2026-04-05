# Week 11 Colab Notebooks

These notebooks are simple, CPU-friendly companions for
`slides/week11-profiling-quantization-lecture.md`.

## Notebook list

| Notebook | Concept |
|---|---|
| `01-floating-point-basics.ipynb` | floating point, precision, fake INT8 approximation |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/01-floating-point-basics.ipynb) |  |
| `02-parameter-count-and-memory.ipynb` | parameter counting and memory estimates |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/02-parameter-count-and-memory.ipynb) |  |
| `03-profiling-basics.ipynb` | `cProfile`, timing, slow vs fast prediction path |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/03-profiling-basics.ipynb) |  |
| `04-batching-benchmark.ipynb` | throughput and latency vs batch size |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/04-batching-benchmark.ipynb) |  |
| `05-pytorch-dynamic-quantization.ipynb` | dynamic INT8 quantization in PyTorch |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/05-pytorch-dynamic-quantization.ipynb) |  |
| `06-onnx-export-and-quantization.ipynb` | export to ONNX, run with ONNX Runtime, quantize |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/06-onnx-export-and-quantization.ipynb) |  |
| `07-pruning-basics.ipynb` | pruning and sparsity |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/07-pruning-basics.ipynb) |  |
| `08-distillation-basics.ipynb` | teacher-student training on a small dataset |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/08-distillation-basics.ipynb) |  |
| `09-optimization-comparison-dashboard.ipynb` | compare size, speed, and accuracy |
| [Open in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week11/colab-notebooks/09-optimization-comparison-dashboard.ipynb) |  |

## Notes

- All notebooks are intended to run on free Colab CPU.
- The models and datasets are deliberately small so the notebooks stay first-year friendly.
- The slide section on modern LLM engines is not turned into a beginner Colab notebook here, because those tools need more setup than the rest of the Week 11 material.

## Regenerating the notebooks

```bash
python lecture-demos/week11/generate_colab_notebooks.py
```
