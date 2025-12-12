# CS 203: Software Tools and Techniques for AI

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar  
**Course Website:** [nipunbatra.github.io/stt-ai-26](https://nipunbatra.github.io/stt-ai-26/)  
**Slides & Labs:** [nipunbatra.github.io/stt-ai-teaching](https://nipunbatra.github.io/stt-ai-teaching/)

This repository contains the lecture slides, lab exercises, and course materials for CS 203. The course covers the end-to-end software engineering stack required for modern AI development, from data collection to deployment and monitoring.

## Course Syllabus (17 Weeks)

| Week | Topic | Key Tools |
| :--- | :--- | :--- |
| 1 | Data Collection | Chrome DevTools, Playwright, BeautifulSoup |
| 2 | Data Validation & Quality | Pydantic, jq, pandas |
| 3 | Data Labeling & Annotation | Label Studio, Cohen's Kappa |
| 4 | Active Learning | modAL, Uncertainty Sampling |
| 5 | Data Augmentation | Albumentations, nlpaug |
| 6 | LLM APIs & Multimodal AI | Gemini API, OpenAI API |
| 7 | Model Development & Training | Scikit-learn, AutoGluon, PyTorch, LLM Fine-tuning |
| 8 | Reproducibility & Environments | Docker, DVC, MLflow |
| 9 | Interactive AI Demos | Streamlit, Gradio, Hugging Face |
| 10 | HTTP, APIs & FastAPI | FastAPI, curl, HTTP |
| 11 | Git, GitHub Actions & CI/CD | GitHub API, PyGithub, Actions |
| 12 | Deployment on Constrained Devices | ONNX, Quantization, Pruning |
| 13 | Profiling & Optimization | PyTorch Profiler, AMP, Distillation |
| 14 | RAG & Vector Databases | ChromaDB, Embeddings, LangChain |
| 15 | Cloud Foundations & Orchestration | Docker Compose, AWS/Render |
| 16 | Model Monitoring & Observability | Evidently AI, Data Drift |
| 17 | LLM Agents & Tool Use | LangGraph, Tool Use |



## Building the Slides

The slides are written in Markdown using [Marp](https://marp.app/).

### Prerequisites

1.  **Node.js**: Install Node.js.
2.  **Marp CLI**: Install globally via npm:
    ```bash
    npm install -g @marp-team/marp-cli
    ```

### Build Commands

We use a `Makefile` to automate the build process.

```bash
# Build all slides (PDF and HTML)
make all

# Clean generated files
make clean

# List available slides
make list
```

Outputs are generated in:
- `pdf/`: PDF versions of slides (for printing/downloading)
- `html/`: HTML versions of slides (for presenting)

## Directory Structure

```
.
├── slides/       # Source Markdown files for lectures and labs
├── pdf/          # Generated PDF slides
├── html/         # Generated HTML slides
├── index.qmd     # Quarto source for the course index page
├── Makefile      # Build automation
└── README.md     # This file
```

## Contributing

**Teaching Assistants:**
If you find an issue with the slides (typos, code errors, outdated content), please **open an issue** using the "Bug Report" template.

1.  Go to the [Issues](https://github.com/nipunbatra/stt-ai-teaching/issues) tab.
2.  Click "New Issue".
3.  Select "Bug Report for Course Content".

## License

Course materials by Prof. Nipun Batra, IIT Gandhinagar.

## Mermaid Diagrams

Mermaid diagrams are supported in **HTML output only**. They are rendered using mermaid.js via a custom Marp engine.

- **HTML slides**: Mermaid diagrams render correctly
- **PDF slides**: Mermaid blocks appear as code (limitation of PDF format)

To view slides with diagrams, use the HTML version or present directly from HTML.
