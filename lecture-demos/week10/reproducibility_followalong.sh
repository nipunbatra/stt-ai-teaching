#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Reproducibility & Environments — Follow-Along Guide                    ║
# ║  Week 10 · CS 203 · Software Tools and Techniques for AI               ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   Last week you learned to version your code with Git. But your code
#   alone isn't enough — "pip install sklearn" gives a different version
#   on every machine. Today you'll learn to version your ENVIRONMENT:
#   virtual environments, pinned dependencies, random seeds, and Docker.
#
# HOW TO USE:
#   1. Open this file in your editor (VS Code, etc.)
#   2. Open a terminal side-by-side
#   3. Copy-paste each command into your terminal, one at a time
#   4. Compare your output with the expected output shown here
#   5. DO NOT run this file as a script — read it and type along
#
# LEGEND:
#   Lines without # prefix     →  commands to type
#   # >> ...                   →  expected output
#   # ...                      →  explanation / narration
#   # [SLIDE: ...]             →  instructor: show this slide
#
# ═══════════════════════════════════════════════════════════════════════════



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 2-4: "Works on My Machine" + Why It Matters     ║
# ║     Show the ImportError slide. Then say "Let's feel the pain."         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: The Problem — "It Works on My Machine"                ~5 min   │
# └──────────────────────────────────────────────────────────────────────────┘
#
# You've built a movie predictor. Let's share it the naive way.

mkdir -p ~/repro-demo && cd ~/repro-demo
mkdir movie-predictor && cd movie-predictor

# Create a simple ML training script:

cat > train.py << 'PYEOF'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic movie data
np.random.seed(42)
X = np.random.rand(200, 5)  # 200 movies, 5 features
y = (X[:, 0] + X[:, 2] > 1).astype(int)  # success label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
PYEOF

# Try running it:

python train.py

# >> Accuracy: 0.850   (or similar — yours may differ!)

# Now imagine emailing this to a friend. They run it:
#
#   $ python train.py
#   ModuleNotFoundError: No module named 'sklearn'
#
# They install it:
#   $ pip install sklearn
#   ERROR: Could not find a version that satisfies the requirement sklearn
#
# The correct package name is scikit-learn, not sklearn.
# Even if they figure that out, they might get a different version
# with different default behavior.
#
# YOUR CODE IS VERSIONED (Git). YOUR ENVIRONMENT IS NOT.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 6-9: Virtual Environments concept               ║
# ║     Show the "rooms in a house" diagram. Then back to terminal.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: Virtual Environments (venv)                           ~10 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Solution: give this project its own isolated Python environment.

# First, check what Python we're using:

which python
python --version

# Create a virtual environment:

python -m venv .venv

# What just happened? Let's look:

ls -la .venv/

# >> bin/  include/  lib/  pyvenv.cfg
# It created a self-contained Python installation!

# Activate it:

source .venv/bin/activate

# Notice your prompt changed:
# >> (.venv) $

# Check — we're using the LOCAL Python now:

which python

# >> /path/to/movie-predictor/.venv/bin/python

# What's installed? Almost nothing:

pip list

# >> pip        xx.x
# >> setuptools xx.x
# That's it! A clean slate.

# Install what we need:

pip install scikit-learn numpy

# >> Successfully installed ...

# Now our script runs:

python train.py

# >> Accuracy: 0.850

# See everything installed (including sub-dependencies):

pip list

# >> joblib, numpy, scikit-learn, scipy, threadpoolctl, ...
# scikit-learn pulls in several packages automatically.

# Save this exact list:

pip freeze > requirements.txt

cat requirements.txt

# >> joblib==1.4.2
# >> numpy==1.26.4
# >> scikit-learn==1.5.0
# >> scipy==1.13.1
# >> threadpoolctl==3.5.0

# Now ANYONE can recreate your exact environment:
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt

# When you're done working, deactivate:

deactivate

# >> (no more (.venv) prefix)
# But let's re-activate for the rest of the demo:

source .venv/bin/activate



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 10-11: Good vs Bad requirements.txt             ║
# ║     Show side-by-side comparison. Then back to terminal.                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: Version Pinning                                        ~8 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Let's see why pinning matters.

# BAD requirements.txt (no versions):

cat > requirements_bad.txt << 'EOF'
numpy
scikit-learn
EOF

# GOOD requirements.txt (pinned versions — what pip freeze gives you):

cat requirements.txt

# The bad one works TODAY. But 6 months from now, numpy 2.0 might
# break your code. Pinning = time capsule.

# Let's prove the environment is isolated. Deactivate and check:

deactivate
python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "sklearn not found in system Python!"

# Re-activate — it's back:

source .venv/bin/activate
python -c "import sklearn; print(sklearn.__version__)"

# >> 1.5.0  (or whatever version pip freeze showed)

# Let's prove we can recreate the environment from scratch:

deactivate
rm -rf .venv

# Create fresh:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# >> (installs exact same versions)

python train.py

# >> Accuracy: 0.850  (same result — environment is reproducible!)



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 13-16: Random Seeds                             ║
# ║     Show the "which result do you report?" slide. Then terminal.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: Reproducible Randomness (Seeds)                        ~8 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Our environment is reproducible. But are our RESULTS?

# Run train.py three times:

python train.py
python train.py
python train.py

# >> Accuracy: 0.850
# >> Accuracy: 0.825
# >> Accuracy: 0.875
#
# Different every time! train_test_split and RandomForest use
# random numbers internally. Different random draws → different results.

# Fix it — add random seeds to train.py:

cat > train.py << 'PYEOF'
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Reproducibility: set ALL random seeds ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Generate synthetic movie data
X = np.random.rand(200, 5)
y = (X[:, 0] + X[:, 2] > 1).astype(int)

# Pass random_state to every sklearn function that uses randomness
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
model = RandomForestClassifier(n_estimators=100, random_state=SEED)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
PYEOF

# Run three times:

python train.py
python train.py
python train.py

# >> Accuracy: 0.925
# >> Accuracy: 0.925
# >> Accuracy: 0.925
#
# SAME result every time. Reproducible!

# KEY INSIGHT: You need seeds in TWO places:
#   1. Global: random.seed(), np.random.seed(), torch.manual_seed()
#   2. Per-function: random_state=42 in sklearn calls



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 18: Virtual Environments Aren't Enough           ║
# ║     Show the OS dependency problem. Then back to terminal.              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: Limits of venv — "Works on Linux, Not Mac"             ~3 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# venv + requirements.txt handles PYTHON packages.
# But what about:
#
#   - System libraries (libssl, libblas, libffi)
#   - Python version itself (3.8 vs 3.11)
#   - OS differences (Mac uses Accelerate, Linux uses OpenBLAS)
#   - CUDA/GPU drivers
#
# Example: Your Mac requirements.txt has:
#   numpy==1.24.3
# On a fresh Ubuntu server:
#   pip install numpy==1.24.3
#   ERROR: ... missing libopenblas ...
#
# venv isolates Python packages. Docker isolates EVERYTHING.

# Quick check — our venv doesn't control the Python version:

python --version

# >> Python 3.x.x (whatever YOUR system has)
# Your friend might have a completely different Python version.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 19-24: Docker concepts, Dockerfile, commands    ║
# ║     Walk through the Docker concept slides. Then build live.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: Docker — Same Result Everywhere                       ~15 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Docker packages your code + Python + OS + everything into a container.

# First, check Docker is installed:

docker --version

# >> Docker version 24.x.x (or similar)
# If not installed: https://docs.docker.com/get-docker/

# Create a Dockerfile:

cat > Dockerfile << 'EOF'
# Start from an official Python image (includes the OS!)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY train.py .

# Default command when container runs
CMD ["python", "train.py"]
EOF

# Also create a .dockerignore so we don't copy the venv into the container:

cat > .dockerignore << 'EOF'
.venv/
__pycache__/
*.pyc
.git/
EOF

# Build the Docker image:

docker build -t movie-predictor .

# >> [+] Building ...
# >> Step 1/6 : FROM python:3.10-slim
# >> Step 2/6 : WORKDIR /app
# >> ...
# >> Successfully tagged movie-predictor:latest

# Run it:

docker run movie-predictor

# >> Accuracy: 0.925
#
# SAME result as running locally! But this container will produce
# the same result on ANY machine — Mac, Linux, Windows, cloud server.

# Run it again — same result:

docker run movie-predictor

# >> Accuracy: 0.925

# Peek inside the container interactively:

docker run -it movie-predictor /bin/bash

# >> root@abc123:/app# ls
# >> requirements.txt  train.py
# >> root@abc123:/app# python --version
# >> Python 3.10.x
# >> root@abc123:/app# exit

# Share files between your machine and the container using volumes:

mkdir -p output
docker run -v "$(pwd)/output:/app/output" movie-predictor

# The container can now write to output/ on your host machine.

# See your images:

docker images | grep movie

# >> movie-predictor   latest   abc123   ...   ~400MB

# Clean up stopped containers:

docker container prune -f



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 25: Docker Compose                               ║
# ║     Show the multi-service diagram. Then back to terminal.              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 7: Docker Compose — App + Service                         ~8 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Real ML projects often have multiple services:
#   - Your training/prediction app
#   - A database for results
#   - A dashboard
#
# Docker Compose lets you define and run multi-container setups.

# Create a simple web prediction service first:

cat > predict_server.py << 'PYEOF'
"""Simple prediction server using built-in http.server."""
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train model at startup
SEED = 42
np.random.seed(SEED)
X = np.random.rand(200, 5)
y = (X[:, 0] + X[:, 2] > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
model = RandomForestClassifier(n_estimators=100, random_state=SEED)
model.fit(X_train, y_train)
print(f"Model trained. Accuracy: {model.score(X_test, y_test):.3f}")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Predict on a random sample
        sample = np.random.rand(1, 5)
        pred = model.predict(sample)[0]
        result = {"prediction": int(pred), "features": sample[0].tolist()}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

server = HTTPServer(("0.0.0.0", 8000), Handler)
print("Serving on http://0.0.0.0:8000")
server.serve_forever()
PYEOF

# Create a Dockerfile for the server:

cat > Dockerfile.server << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict_server.py .
EXPOSE 8000
CMD ["python", "predict_server.py"]
EOF

# Create docker-compose.yml:

cat > docker-compose.yml << 'EOF'
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

# Start all services:

docker compose up --build -d

# >> Creating movie-predictor-trainer-1   ... done
# >> Creating movie-predictor-predictor-1 ... done

# Check running containers:

docker compose ps

# Test the prediction server:

sleep 3
curl -s http://localhost:8000 | python -m json.tool

# >> {
# >>     "prediction": 1,
# >>     "features": [0.374, ...]
# >> }

# View logs:

docker compose logs trainer

# Stop everything:

docker compose down

# >> Stopping movie-predictor-predictor-1 ... done
# >> Stopping movie-predictor-trainer-1   ... done

# KEY: docker-compose.yml is version-controlled. Anyone can run
# `docker compose up` and get the exact same multi-service setup.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 27-30: Project Structure + Config               ║
# ║     Show the directory tree and config examples. Then terminal.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 8: Project Structure & Best Practices                     ~5 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# A reproducible project needs more than just code and requirements.

# Create proper project structure:

mkdir -p src data/raw data/processed models notebooks

# Move training script to src/:

mv train.py src/train.py

# Create a config file (no hardcoded values!):

cat > config.yaml << 'EOF'
# Project configuration — change settings here, not in code
model:
  type: random_forest
  n_estimators: 100
  seed: 42

data:
  n_samples: 200
  n_features: 5
  test_size: 0.2

paths:
  model: models/model.pkl
  data: data/processed/
EOF

# Create a .env file for secrets (NEVER commit this):

cat > .env << 'EOF'
# API keys, passwords, etc. — NEVER commit this file
DATABASE_URL=postgresql://user:pass@localhost/mldb
API_KEY=sk-abc123secret
EOF

# Create .gitignore:

cat > .gitignore << 'EOF'
# Environment
.venv/
__pycache__/
*.pyc

# Data and models (too large for Git)
data/raw/
models/*.pkl
*.h5

# Secrets
.env

# OS files
.DS_Store

# Docker
output/
EOF

# Create a Makefile for common tasks:

cat > Makefile << 'EOF'
.PHONY: setup train clean docker

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Run: source .venv/bin/activate"

train:
	python src/train.py

clean:
	rm -rf .venv __pycache__ models/*.pkl

docker:
	docker build -t movie-predictor .
	docker run movie-predictor
EOF

# Create a README:

cat > README.md << 'EOF'
# Movie Predictor

Predicts movie success using Random Forest.

## Quick Start

```bash
make setup
source .venv/bin/activate
make train
```

## Docker

```bash
make docker
```

## Configuration

Edit `config.yaml` to change model parameters.
EOF

# Show the final structure:

find . -not -path './.venv/*' -not -path './.git/*' -not -name '*.pyc' | head -30

# >> .
# >> ./src
# >> ./src/train.py
# >> ./data
# >> ./data/raw
# >> ./data/processed
# >> ./models
# >> ./notebooks
# >> ./config.yaml
# >> ./.env
# >> ./.gitignore
# >> ./Makefile
# >> ./README.md
# >> ./requirements.txt
# >> ./Dockerfile
# >> ...



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 31-32: Reproducibility Checklist                ║
# ║     Show the checklist slide. Walk through each item.                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 9: Reproducibility Checklist — Verify & Share             ~5 min  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Let's run through the checklist for our project:

echo "=== REPRODUCIBILITY CHECKLIST ==="
echo ""

# 1. Virtual environment?
echo -n "✅ Virtual environment: "
if [ -d ".venv" ]; then echo "YES (.venv/)"; else echo "❌ NO"; fi

# 2. requirements.txt with pinned versions?
echo -n "✅ requirements.txt:    "
if grep -q "==" requirements.txt 2>/dev/null; then echo "YES (pinned)"; else echo "❌ NO"; fi

# 3. Random seeds?
echo -n "✅ Random seeds:        "
if grep -q "random_state\|random.seed\|np.random.seed" src/train.py 2>/dev/null; then echo "YES"; else echo "❌ NO"; fi

# 4. README?
echo -n "✅ README:              "
if [ -f "README.md" ]; then echo "YES"; else echo "❌ NO"; fi

# 5. Config file?
echo -n "✅ Config file:         "
if [ -f "config.yaml" ]; then echo "YES"; else echo "❌ NO"; fi

# 6. .gitignore?
echo -n "✅ .gitignore:          "
if [ -f ".gitignore" ]; then echo "YES"; else echo "❌ NO"; fi

# 7. Docker?
echo -n "✅ Dockerfile:          "
if [ -f "Dockerfile" ]; then echo "YES"; else echo "❌ NO"; fi

echo ""
echo "=== ALL CHECKS PASSED ==="

# The ultimate test: can someone ELSE reproduce your results?
#
# 1. Push to GitHub
# 2. Friend clones the repo
# 3. Friend runs: make setup && source .venv/bin/activate && make train
# 4. They should see the EXACT SAME accuracy
#
# Or with Docker:
# 1. Friend runs: docker build -t movie-predictor . && docker run movie-predictor
# 2. Same result. No "but it works on my machine!"

# ═══════════════════════════════════════════════════════════════════════════
# WRAP-UP
# ═══════════════════════════════════════════════════════════════════════════
#
# What we covered today:
#
#   Act 1:  The problem — sharing code without environment info
#   Act 2:  venv — isolated Python environments
#   Act 3:  Version pinning — requirements.txt with ==
#   Act 4:  Random seeds — same results every run
#   Act 5:  Limits of venv — OS and system library differences
#   Act 6:  Docker — package everything, run anywhere
#   Act 7:  Docker Compose — multi-service setups
#   Act 8:  Project structure — config, .env, Makefile
#   Act 9:  Reproducibility checklist — verify before sharing
#
# The progression:
#   Git (version code) → venv (version env) → Docker (version everything)
#
# Next week: Experiment tracking with Weights & Biases (W&B)
# ═══════════════════════════════════════════════════════════════════════════

# Clean up demo directory:

cd ~/repro-demo
deactivate 2>/dev/null
# rm -rf ~/repro-demo   # uncomment to clean up
