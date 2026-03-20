# Docker Demo — Week 9

Follow along with the lecture slides. Each step matches a slide.

## Step 0: Run locally first (no Docker)

```bash
pip install -r requirements.txt
python app.py
# → open http://localhost:7860
```

## Step 1: Check Docker is running

```bash
docker --version
```

## Step 2: Build the image

```bash
docker build -t spam-app .
```

## Step 3: Run the container

```bash
docker run -p 7860:7860 spam-app
# → open http://localhost:7860
```

## Step 4: Run in background

```bash
docker run -d -p 7860:7860 spam-app
```

## Step 5: See what's running

```bash
docker ps
```

## Step 6: Stop it

```bash
docker stop <container_id>
```

## Experiment 1: The Amnesia Test

Stop the container, start a new one. Any files created inside are gone.

## Experiment 2: Code Change

Edit `app.py` on your laptop (change the title). Refresh browser. Nothing changes!
You must rebuild: `docker build -t spam-app . && docker run -p 7860:7860 spam-app`

## Experiment 3: Volumes

```bash
docker run -v $(pwd):/app -p 7860:7860 spam-app
```

Now edits on your laptop are reflected inside the container.

## Experiment 4: Shell into the container

```bash
docker exec -it <container_id> bash
ls
python --version
exit
```
