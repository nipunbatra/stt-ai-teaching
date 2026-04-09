# CS 203: Software Tools and Techniques for AI

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar
**Course Website:** [nipunbatra.github.io/stt-ai-26](https://nipunbatra.github.io/stt-ai-26/)
**Slides & Labs:** [nipunbatra.github.io/stt-ai-teaching](https://nipunbatra.github.io/stt-ai-teaching/)

---

## Course Overview

This course teaches the **complete software engineering stack for AI/ML development**—from collecting and labeling data to deploying and monitoring models in production. While most ML courses focus on algorithms, this course focuses on the **engineering practices** that make ML systems work in the real world.

### What Makes This Course Different?

- **End-to-end coverage**: Data → Model → Deployment → Monitoring
- **Hands-on labs**: 3-hour practical sessions each week
- **Industry-relevant tools**: Docker, FastAPI, MLflow, Gradio, GitHub Actions
- **Real project theme**: Build a Netflix movie prediction system across all weeks

---

## What You'll Learn

By the end of this course, you will be able to:

| Module | Skills |
|--------|--------|
| **Data Engineering** | Collect data via APIs/scraping, validate with schemas, set up annotation pipelines, measure labeling quality |
| **Labeling at Scale** | Use active learning, weak supervision, and LLMs to reduce labeling costs by 50-80% |
| **Model Development** | Train models with scikit-learn/PyTorch, use AutoML, fine-tune LLMs with LoRA |
| **LLM Integration** | Call LLM APIs, engineer prompts, build multimodal applications |
| **MLOps** | Containerize with Docker, version data with DVC, track experiments with MLflow |
| **Deployment** | Build APIs with FastAPI, create demos with Streamlit/Gradio, set up CI/CD |
| **Production** | Optimize models for edge devices, profile performance, monitor for drift |

---

## Course Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CS 203: 13-Week Journey                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PART I: DATA ENGINEERING (Weeks 1-5)                                   │
│  ├── Week 1: Data Collection (APIs, Scraping)                           │
│  ├── Week 2: Data Validation (Schemas, Quality)                         │
│  ├── Week 3: Data Labeling (Annotation, IAA)                            │
│  ├── Week 4: Optimizing Labeling (Active Learning, Weak Supervision)    │
│  └── Week 5: Data Augmentation (Image, Text, Audio)                     │
│                                                                         │
│  PART II: MODEL DEVELOPMENT (Weeks 6-8)                                 │
│  ├── Week 6: LLM APIs (Gemini, Prompt Engineering)                      │
│  ├── Week 7: Model Evaluation (CV, Bias-Variance)                       │
│  └── Week 8: Tuning, AutoML & Tracking (Optuna, TrackIO)               │
│                                                                         │
│  PART III: ENGINEERING (Weeks 9-13)                                     │
│  ├── Week 9: Reproducibility (Docker, Seeds, Tracking)                  │
│  ├── Week 10: Data Drift & Monitoring (KS test, PSI)                    │
│  ├── Week 11: Profiling & Quantization (INT8, ONNX)                     │
│  ├── Week 12: Building AI Agents (Tool Calling, Gemma 4)                │
│  └── Week 13: Course Summary & What's Next                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Links

| Resource | Link |
|----------|------|
| **All Slides (HTML/PDF)** | [nipunbatra.github.io/stt-ai-teaching](https://nipunbatra.github.io/stt-ai-teaching/) |
| **Course Syllabus** | [nipunbatra.github.io/stt-ai-26](https://nipunbatra.github.io/stt-ai-26/) |
| **Report Issues** | [GitHub Issues](https://github.com/nipunbatra/stt-ai-teaching/issues) |
| **Diagram Generators** | [diagram-generators/](diagram-generators/) |

---

## Weekly Topics & Tools

| Week | Topic | Key Tools | Slides |
|:----:|-------|-----------|:------:|
| 1 | Data Collection | curl, requests, BeautifulSoup | [PDF](pdf/week01-data-collection-lecture.pdf) |
| 2 | Data Validation | jq, Pydantic, pandas | [PDF](pdf/week02-data-validation-lecture.pdf) |
| 3 | Data Labeling | Label Studio, Cohen's Kappa | [PDF](pdf/week03-data-labeling-lecture.pdf) |
| 4 | Optimizing Labeling | modAL, Snorkel, cleanlab | [PDF](pdf/week04-optimizing-labeling-lecture.pdf) |
| 5 | Data Augmentation | Albumentations, nlpaug | [PDF](pdf/week05-data-augmentation-lecture.pdf) |
| 6 | LLM APIs | Gemini API, prompt engineering | [PDF](pdf/week06-llm-apis-lecture.pdf) |
| 7 | Model Evaluation | scikit-learn, cross-validation | [PDF](pdf/week07-model-development-lecture.pdf) |
| 8 | Tuning & Tracking | Optuna, TrackIO | [PDF](pdf/week08-tuning-tracking-lecture.pdf) |
| 9 | Reproducibility | Docker, seeds, tracking | [PDF](pdf/week09-reproducibility-lecture.pdf) |
| 10 | Data Drift | KS test, PSI, chi-squared | [HTML](html/week10-data-drift-lecture.html) |
| 11 | Profiling & Quantization | cProfile, INT8, ONNX | [PDF](pdf/week11-profiling-quantization-lecture.pdf) |
| 12 | Building AI Agents | Gemma 4, tool calling, agent loop | [PDF](pdf/week12-agents-lecture.pdf) |
| 13 | Course Summary | — | [PDF](pdf/week13-course-summary-lecture.pdf) |

---

## Building the Slides Locally

The slides are written in Markdown using [Marp](https://marp.app/).

### Prerequisites

```bash
# Install Node.js (if not already installed)
# Then install Marp CLI
npm install -g @marp-team/marp-cli
```

### Build Commands

```bash
# Build all slides (PDF and HTML)
make all

# Build specific week
make pdf/week01-data-collection-lecture.pdf

# Clean generated files
make clean

# List available slides
make list
```

### Output Directories

- `pdf/` — PDF versions (for downloading/printing)
- `html/` — HTML versions (for presenting in browser)

---

## Repository Structure

```
stt-ai-teaching/
├── slides/              # Source Markdown files
│   ├── week01-*.md      # Week 1 lecture and lab
│   ├── week02-*.md      # Week 2 lecture and lab
│   └── ...
├── pdf/                 # Generated PDF slides
├── html/                # Generated HTML slides
├── figures/             # Diagrams and images
├── diagram-generators/  # Python scripts to generate diagrams
├── index.qmd            # Quarto source for course website
├── Makefile             # Build automation
└── README.md            # This file
```

---

## Contributing

Found an issue? Please help us improve!

1. Go to [Issues](https://github.com/nipunbatra/stt-ai-teaching/issues)
2. Click "New Issue"
3. Select "Bug Report for Course Content"
4. Describe the issue (typo, code error, outdated content)

---

## License

Course materials by Prof. Nipun Batra, IIT Gandhinagar.
