---
title: "APIs & Model Demos — Follow-Along Guide"
subtitle: "Week 12 · CS 203 · Software Tools and Techniques for AI"
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
  - \fancyhead[L]{CS 203 — APIs \& Model Demos}
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
\item Open \texttt{apis\_followalong.sh} in your editor (left half of screen)
\item Open a terminal (right half of screen)
\item Copy-paste each command, one at a time
\item \textbf{Type it yourself} --- that's how you learn
\end{itemize}

**Legend:**  `$` = command to type. `>>` = expected output. Blue boxes = projector slide.

---

\begin{slidebox}\textbf{Projector: Slides 2--3 --- From Notebook to Product}
Your model lives in a notebook. Nobody can use it. Today we fix that: API $\to$ demo $\to$ Docker.
\end{slidebox}

\begin{actbox}\textbf{\large Act 1: Train and Save a Model \hfill $\sim$8 min}
\end{actbox}

```bash
$ mkdir -p ~/api-demo && cd ~/api-demo
$ python -m venv .venv && source .venv/bin/activate
$ pip install scikit-learn numpy fastapi uvicorn joblib gradio streamlit
$ python train.py
>> Model accuracy: 0.xxx
>> Model saved to model.pkl
```

\begin{tipbox}
\texttt{joblib.dump(model, "model.pkl")} saves the trained model. \texttt{joblib.load("model.pkl")} loads it back. Load once at server startup, not per request.
\end{tipbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 4--8 --- REST Theory}
HTTP methods: GET (read), POST (send). Status codes: 200 (ok), 422 (bad input), 500 (error). Stateless = no memory between requests.
\end{slidebox}

\begin{actbox}\textbf{\large Act 2: FastAPI --- Your First API \hfill $\sim$15 min}
\end{actbox}

Create `app.py` with FastAPI:

```python
@app.post("/predict", response_model=Prediction)
def predict(features: MovieFeatures):
    X = np.array([[features.budget, ...]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return Prediction(success=bool(pred), ...)
```

Start and test:

```bash
$ uvicorn app:app --reload --port 8000 &
$ curl -s http://localhost:8000/ | python -m json.tool
>> {"status": "healthy", "model": "random_forest_v1"}
$ curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"budget": 0.8, "runtime": 0.5, "genre_action": 1, ...}'
>> {"success": true, "confidence": 0.85, "label": "Hit!"}
```

Open \texttt{http://localhost:8000/docs} --- interactive Swagger UI, auto-generated!

\begin{actbox}\textbf{\large Act 3: Testing Your API \hfill $\sim$5 min}
\end{actbox}

```bash
$ pytest test_api.py -v
>> test_api.py::test_health PASSED
>> test_api.py::test_predict PASSED
>> test_api.py::test_predict_invalid PASSED
```

\begin{warningbox}
Always test the 422 case --- send invalid input and verify the API rejects it. Pydantic handles validation automatically with FastAPI.
\end{warningbox}

\newpage

\begin{slidebox}\textbf{Projector: Slides 13--15 --- Gradio and Streamlit}
Gradio = instant ML demo in 10 lines. Streamlit = richer dashboard.
\end{slidebox}

\begin{actbox}\textbf{\large Act 4: Gradio --- Demo in 10 Lines \hfill $\sim$10 min}
\end{actbox}

```python
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(0, 1, label="Budget"), ...],
    outputs=gr.Text(label="Prediction"),
)
demo.launch()
```

```bash
$ python app_gradio.py
>> Running on http://127.0.0.1:7860
```

\begin{tipbox}
Add \texttt{share=True} to \texttt{demo.launch()} to get a public URL --- great for sharing demos with non-technical people.
\end{tipbox}

\begin{actbox}\textbf{\large Act 5: Streamlit --- Dashboard Demo \hfill $\sim$10 min}
\end{actbox}

```bash
$ streamlit run app_streamlit.py
>> Opens at http://localhost:8501
```

Streamlit reruns the entire script on each interaction. Gradio calls your function on submit.

\begin{actbox}\textbf{\large Act 6: Dockerize the API \hfill $\sim$10 min}
\end{actbox}

```bash
$ docker build -t movie-api .
$ docker run -d -p 8000:8000 --name movie-api movie-api
$ curl -s http://localhost:8000/predict -X POST \
    -H "Content-Type: application/json" \
    -d '{"budget": 0.8, ...}'
>> Same result --- runs identically in Docker!
```

\newpage

# Quick Reference

| I want to... | Command |
|-------------|---------|
| Start FastAPI server | `uvicorn app:app --reload` |
| View auto-docs | Open `http://localhost:8000/docs` |
| Test with curl | `curl -X POST url -H "Content-Type: application/json" -d '{...}'` |
| Run Gradio demo | `python app_gradio.py` |
| Run Streamlit | `streamlit run app.py` |
| Save sklearn model | `joblib.dump(model, "model.pkl")` |
| Load sklearn model | `model = joblib.load("model.pkl")` |
| Build Docker image | `docker build -t name .` |
| Run container | `docker run -p 8000:8000 name` |

\vspace{0.5cm}

**Exam-relevant concepts:**

- HTTP methods: GET (read), POST (create/predict), PUT (update), DELETE (remove)
- Status codes: 200 (OK), 400 (bad request), 404 (not found), 422 (validation), 500 (server error)
- Online inference: one prediction, low latency ($<$100ms)
- Batch inference: many predictions, latency doesn't matter
- Stateless services: no memory between requests $\to$ easy to scale
- Model serialization: joblib (sklearn), pickle, torch.save (PyTorch)
