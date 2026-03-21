# Docker Demos — Week 9: Reproducibility in Practice

Five progressive demos, each teaching one Docker concept.
Run them in order during the lecture.

## Prerequisites

- Docker Desktop installed and **running** (check: `docker --version`)
- Terminal open in this directory: `cd lecture-demos/week09/docker-demo`

---

## Demo 1: Hello Docker (5 min)

**Concept:** Docker runs an isolated Linux environment, not your laptop's Python.

```bash
cd 1-hello

# Build the image
docker build -t hello-docker .

# Run it
docker run hello-docker
```

**What students see:** Python version and OS are different from their laptop.
It says "Linux" even on a Mac! This is a completely isolated environment.

**Teaching point:** The Dockerfile is just 4 lines. `FROM` picks the base,
`COPY` brings your code in, `CMD` says what to run.

**Cleanup:**
```bash
cd ..
```

---

## Demo 2: Dependencies (5 min)

**Concept:** Docker freezes your exact library versions. Everyone gets the same result.

```bash
cd 2-dependencies

# Build (notice: installs sklearn inside the container)
docker build -t train-model .

# Run
docker run train-model
```

**What students see:** The accuracy is always exactly `0.9750` (or whatever it prints).
Same Python 3.10, same sklearn 1.7.2, same random_state=42, same result. Always.

**Teaching point:** `requirements.txt` with pinned versions (`scikit-learn==1.7.2`)
ensures reproducibility. This is why we pin versions!

**Discussion:** "On your laptop, what sklearn version do you have?
Is it the same? Would you get the same accuracy?"

**Cleanup:**
```bash
cd ..
```

---

## Demo 3: Web App + Ports (10 min)

**Concept:** Containers are headless — you need port mapping (`-p`) to access web apps.

```bash
cd 3-web-app

# Build
docker build -t spam-app .

# Run (map container's 7860 to your laptop's 7861)
docker run -p 7861:7860 spam-app
```

Open **http://localhost:7861** — your spam classifier is live!

**Teaching points:**
- `-p 7861:7860` means "laptop port 7861 → container port 7860"
- The container has no screen, no browser — you look through the port window
- `GRADIO_SERVER_NAME="0.0.0.0"` in the Dockerfile lets the outside world connect
  (without it, Gradio only listens to 127.0.0.1 = container talks to itself)

**Experiments to do live:**
1. Open another terminal: `docker ps` — show the running container
2. `docker exec -it <id> bash` — shell into it, run `ls`, `python --version`
3. `Ctrl+C` to stop, try `docker run -d -p 7861:7860 spam-app` (background mode)
4. `docker stop <id>` to clean up

**Cleanup:**
```bash
docker stop $(docker ps -q --filter ancestor=spam-app) 2>/dev/null
cd ..
```

---

## Demo 4: Volumes — The Amnesia Problem (10 min)

**Concept:** Containers are disposable. Files created inside vanish when they stop.
Volumes solve this.

### Part A: Without a volume (data lost!)

```bash
cd 4-volumes

# Build
docker build -t train-save .

# Run WITHOUT volume
docker run train-save

# Check: is there an outputs/ folder on your laptop?
ls outputs/
# → No such file or directory! The model was saved INSIDE the container.
# The container stopped → it's gone forever.
```

### Part B: With a volume (data persists!)

```bash
# Run WITH volume — map container's /app/outputs to laptop's ./outputs
docker run -v $(pwd)/outputs:/app/outputs train-save

# Check again
ls outputs/
# → model.pkl  training_log.txt — they survived!
cat outputs/training_log.txt
```

**Teaching points:**
- `-v $(pwd)/outputs:/app/outputs` = "sync this folder between laptop and container"
- Think of it as plugging a USB drive into the container
- This is how you save trained models, logs, results from Docker runs

**Discussion:** "You trained a model for 3 hours inside Docker. You stop the
container. Where are your weights?" → Gone (unless you used `-v`!)

**Cleanup:**
```bash
rm -rf outputs/
cd ..
```

---

## Demo 5: Environment Variables — Configure Without Rebuilding (10 min)

**Concept:** The same image can behave differently based on environment variables.
No code changes, no rebuild needed.

```bash
cd 5-environment

# Build ONCE
docker build -t env-demo .
```

### Run 1: Default config (RandomForest, 100 trees)

```bash
docker run -p 7861:7860 env-demo
# Open http://localhost:7861
# Title: "Digit Classifier", Model: RandomForest (n=100)
```

Stop it (`Ctrl+C`), then:

### Run 2: Switch to SVM (no rebuild!)

```bash
docker run -p 7861:7860 \
  -e MODEL_TYPE=svm \
  -e APP_TITLE="SVM Digit Classifier" \
  env-demo
# Same image, different model! Title changed too.
```

### Run 3: Bigger forest (no rebuild!)

```bash
docker run -p 7861:7860 \
  -e N_ESTIMATORS=500 \
  -e APP_TITLE="Big Forest (500 trees)" \
  env-demo
# Same image, 500 trees instead of 100.
```

**Teaching points:**
- `-e VAR=value` passes environment variables into the container
- `os.environ.get("VAR", "default")` reads them in Python
- `ENV VAR="default"` in Dockerfile sets defaults
- This is how production apps are configured — database URLs, API keys,
  model paths, feature flags — all via environment variables
- **One image, many configurations. No rebuild needed.**

**Cleanup:**
```bash
docker stop $(docker ps -q --filter ancestor=env-demo) 2>/dev/null
cd ..
```

---

## Summary: What Each Demo Teaches

| Demo | Concept | Key Command |
|------|---------|-------------|
| **1-hello** | Docker runs isolated Linux | `docker build -t name .` / `docker run name` |
| **2-dependencies** | Pinned versions = reproducibility | `requirements.txt` with `==` versions |
| **3-web-app** | Port mapping for web apps | `docker run -p 7861:7860 name` |
| **4-volumes** | Containers have amnesia; volumes persist | `docker run -v $(pwd)/dir:/app/dir name` |
| **5-environment** | Configure without rebuilding | `docker run -e VAR=value name` |

## Quick Reference: The 6 Commands Students Need

```bash
docker build -t my-app .              # Build image from Dockerfile
docker run my-app                      # Run (foreground, no ports)
docker run -d -p 7861:7860 my-app     # Run (background + port mapping)
docker ps                              # List running containers
docker stop <id>                       # Stop a container
docker exec -it <id> bash             # Shell into running container
```

## Cleanup: Remove All Demo Images

```bash
docker rmi hello-docker train-model spam-app train-save env-demo 2>/dev/null
docker system prune -f   # clean up stopped containers
```
