---
title: "Reproducibility & Environments — Follow-Along Guide"
subtitle: "Week 10 · CS 203 · Software Tools and Techniques for AI"
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
  - \fancyhead[L]{CS 203 — Reproducibility \& Environments}
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
\item Open \texttt{reproducibility\_followalong.sh} in your editor (left half of screen)
\item Open a terminal (right half of screen)
\item Copy-paste each command, one at a time
\item Compare your output with the expected output shown here
\item \textbf{Type it yourself} --- that's how you learn
\end{itemize}

**Legend:**  `$` = command to type (don't type the `$`). `>>` = expected output. Blue boxes = look at the projector slide.

---

\begin{slidebox}\textbf{Projector: Slides 2--4 --- ``Works on My Machine'' + Why It Matters}
Look at the projector. Your friend gets \texttt{ImportError}. Three hours later, still debugging.
\end{slidebox}

\begin{actbox}\textbf{\large Act 1: The Problem --- ``It Works on My Machine'' \hfill $\sim$5 min}
\end{actbox}

Create a simple ML project and try to share it:

```bash
$ mkdir -p ~/repro-demo && cd ~/repro-demo
$ mkdir movie-predictor && cd movie-predictor
```

Write a training script:

```bash
$ cat > train.py << 'PYEOF'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(200, 5)
y = (X[:, 0] + X[:, 2] > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
PYEOF
$ python train.py
>> Accuracy: 0.850   (yours may differ!)
```

Now imagine emailing this to a friend. They get `ModuleNotFoundError: No module named 'sklearn'`. Even if they figure out the package is called `scikit-learn`, they might get a different version. **Your code is versioned (Git). Your environment is not.**

\newpage

\begin{slidebox}\textbf{Projector: Slides 6--9 --- Virtual Environments (rooms in a house)}
Look at the projector. Each project gets its own room with its own packages.
\end{slidebox}

\begin{actbox}\textbf{\large Act 2: Virtual Environments (venv) \hfill $\sim$10 min}
\end{actbox}

Create an isolated Python environment:

```bash
$ python -m venv .venv
$ ls -la .venv/
>> bin/  include/  lib/  pyvenv.cfg
```

Activate it:

```bash
$ source .venv/bin/activate
$ which python
>> /path/to/movie-predictor/.venv/bin/python
```

\begin{tipbox}
Your prompt shows \texttt{(.venv)} when the environment is active. Always check before installing packages!
\end{tipbox}

See what's installed (almost nothing):

```bash
$ pip list
>> pip        xx.x
>> setuptools xx.x
```

Install what we need and save the list:

```bash
$ pip install scikit-learn numpy
$ python train.py
>> Accuracy: 0.850
$ pip freeze > requirements.txt
$ cat requirements.txt
>> joblib==1.4.2
>> numpy==1.26.4
>> scikit-learn==1.5.0
>> scipy==1.13.1
>> threadpoolctl==3.5.0
```

Now anyone can recreate your exact environment:

```bash
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

\newpage

\begin{slidebox}\textbf{Projector: Slides 10--11 --- Good vs Bad requirements.txt}
Look at the projector. Pinned versions = time capsule. Unpinned = ticking bomb.
\end{slidebox}

\begin{actbox}\textbf{\large Act 3: Version Pinning \hfill $\sim$8 min}
\end{actbox}

Bad requirements.txt (no versions):

```
numpy
scikit-learn
```

Good requirements.txt (pinned --- what `pip freeze` gives you):

```
numpy==1.26.4
scikit-learn==1.5.0
```

Prove the environment is isolated:

```bash
$ deactivate
$ python -c "import sklearn" 2>/dev/null || echo "Not found!"
>> Not found!
$ source .venv/bin/activate
$ python -c "import sklearn; print(sklearn.__version__)"
>> 1.5.0
```

Prove we can recreate from scratch:

```bash
$ deactivate && rm -rf .venv
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
$ python train.py
>> Accuracy: 0.850   (same result!)
```

\begin{warningbox}
Never commit \texttt{.venv/} to Git. Commit \texttt{requirements.txt} instead. The venv folder is 50+ MB and platform-specific.
\end{warningbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 13--16 --- Random Seeds}
Look at the projector. ``Which result do you report?'' Three runs, three answers.
\end{slidebox}

\begin{actbox}\textbf{\large Act 4: Reproducible Randomness (Seeds) \hfill $\sim$8 min}
\end{actbox}

Run the script three times --- different results each time:

```bash
$ python train.py && python train.py && python train.py
>> Accuracy: 0.850
>> Accuracy: 0.825
>> Accuracy: 0.875
```

Fix it by adding seeds everywhere:

```bash
$ cat > train.py << 'PYEOF'
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

X = np.random.rand(200, 5)
y = (X[:, 0] + X[:, 2] > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
model = RandomForestClassifier(n_estimators=100, random_state=SEED)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
PYEOF
$ python train.py && python train.py && python train.py
>> Accuracy: 0.925
>> Accuracy: 0.925
>> Accuracy: 0.925
```

\begin{tipbox}
Seeds go in \textbf{two places}: (1) Global --- \texttt{random.seed()}, \texttt{np.random.seed()}, \texttt{torch.manual\_seed()}. (2) Per-function --- \texttt{random\_state=42} in sklearn calls.
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slide 18 --- Virtual Environments Aren't Enough}
Look at the projector. venv handles Python packages. What about the OS, system libraries, Python version itself?
\end{slidebox}

\begin{actbox}\textbf{\large Act 5: Limits of venv --- OS \& System Dependencies \hfill $\sim$3 min}
\end{actbox}

venv + requirements.txt handles Python packages. But not:

- System libraries (libssl, libblas, libffi)
- Python version itself (3.8 vs 3.11)
- OS differences (Mac Accelerate vs Linux OpenBLAS)
- CUDA/GPU drivers

```bash
$ python --version
>> Python 3.x.x   (whatever YOUR system has --- your friend's may differ)
```

**venv isolates Python packages. Docker isolates everything.**

\begin{slidebox}\textbf{Projector: Slides 19--24 --- Docker Concepts, Dockerfile, Commands}
Walk through the Docker concept slides: Image, Container, Dockerfile, Registry.
\end{slidebox}

\begin{actbox}\textbf{\large Act 6: Docker --- Same Result Everywhere \hfill $\sim$15 min}
\end{actbox}

Create a Dockerfile:

```bash
$ cat > Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
CMD ["python", "train.py"]
EOF
$ cat > .dockerignore << 'EOF'
.venv/
__pycache__/
.git/
EOF
```

Build and run:

```bash
$ docker build -t movie-predictor .
$ docker run movie-predictor
>> Accuracy: 0.925
$ docker run movie-predictor
>> Accuracy: 0.925   (same --- on ANY machine with Docker)
```

Explore inside the container:

```bash
$ docker run -it movie-predictor /bin/bash
>> root@abc123:/app# python --version
>> Python 3.10.x
>> root@abc123:/app# exit
```

\begin{tipbox}
Copy \texttt{requirements.txt} before your code in the Dockerfile. Docker caches layers --- if requirements don't change, it skips the slow \texttt{pip install} step.
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slide 25 --- Docker Compose}
Look at the projector. Multiple services defined in one YAML file.
\end{slidebox}

\begin{actbox}\textbf{\large Act 7: Docker Compose --- App + Service \hfill $\sim$8 min}
\end{actbox}

Create a prediction server and compose file:

```bash
$ cat > docker-compose.yml << 'EOF'
services:
  trainer:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./output:/app/output
  predictor:
    build:
      context: .
      dockerfile: Dockerfile.server
    ports:
      - "8000:8000"
EOF
```

Start everything:

```bash
$ docker compose up --build -d
$ docker compose ps
$ curl -s http://localhost:8000 | python -m json.tool
>> {"prediction": 1, "features": [...]}
$ docker compose down
```

\begin{tipbox}
\texttt{docker-compose.yml} is version-controlled. Anyone can run \texttt{docker compose up} and get the exact same multi-service setup.
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 27--30 --- Project Structure + Config}
Look at the projector. Directory tree, config.yaml, .gitignore examples.
\end{slidebox}

\begin{actbox}\textbf{\large Act 8: Project Structure \& Best Practices \hfill $\sim$5 min}
\end{actbox}

Organize the project properly:

```bash
$ mkdir -p src data/raw data/processed models notebooks
$ mv train.py src/train.py
```

Create a config file (no hardcoded values!):

```bash
$ cat > config.yaml << 'EOF'
model:
  type: random_forest
  n_estimators: 100
  seed: 42
data:
  n_samples: 200
  test_size: 0.2
paths:
  model: models/model.pkl
EOF
```

Create `.env` for secrets, `.gitignore`, `Makefile`, and `README.md`:

```bash
$ echo "API_KEY=sk-abc123secret" > .env
$ cat > .gitignore << 'EOF'
.venv/
__pycache__/
data/raw/
models/*.pkl
.env
.DS_Store
EOF
$ cat > Makefile << 'EOF'
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
train:
	python src/train.py
docker:
	docker build -t movie-predictor . && docker run movie-predictor
EOF
```

\begin{warningbox}
Never commit \texttt{.env} files. They contain secrets (API keys, passwords). Add \texttt{.env} to \texttt{.gitignore} and use a \texttt{.env.example} template instead.
\end{warningbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 31--32 --- Reproducibility Checklist}
Walk through the checklist on the projector while running it live.
\end{slidebox}

\begin{actbox}\textbf{\large Act 9: Reproducibility Checklist --- Verify \& Share \hfill $\sim$5 min}
\end{actbox}

Run the checklist for our project:

```bash
$ echo "=== REPRODUCIBILITY CHECKLIST ==="
```

| Check | Status |
|-------|--------|
| Virtual environment (`.venv/`) | Yes |
| `requirements.txt` with pinned versions (`==`) | Yes |
| Random seeds (`random_state`, `np.random.seed`) | Yes |
| `README.md` with setup instructions | Yes |
| `config.yaml` (no hardcoded values) | Yes |
| `.gitignore` (exclude data, models, secrets) | Yes |
| Dockerfile (optional, for full isolation) | Yes |

**The ultimate test:** Can someone else clone your repo and get the exact same results?

1. Push to GitHub
2. Friend clones
3. `make setup && source .venv/bin/activate && make train`
4. Same accuracy? **You're reproducible!**

---

# Quick Reference

| I want to... | Command |
|-------------|---------|
| Create virtual environment | `python -m venv .venv` |
| Activate (Mac/Linux) | `source .venv/bin/activate` |
| Activate (Windows) | `.venv\Scripts\activate` |
| Install packages | `pip install <package>` |
| Save dependencies | `pip freeze > requirements.txt` |
| Install from file | `pip install -r requirements.txt` |
| Deactivate | `deactivate` |
| Build Docker image | `docker build -t name .` |
| Run container | `docker run name` |
| Interactive shell | `docker run -it name /bin/bash` |
| Mount volume | `docker run -v $(pwd)/data:/app/data name` |
| Start Compose | `docker compose up --build -d` |
| Stop Compose | `docker compose down` |

\vspace{0.5cm}

**The Progression:**

Git (version code) $\to$ venv (version environment) $\to$ Docker (version everything) $\to$ W\&B (version experiments)
