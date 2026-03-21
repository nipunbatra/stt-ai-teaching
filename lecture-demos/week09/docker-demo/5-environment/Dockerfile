FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Default configuration — override with `docker run -e VAR=value`
ENV MODEL_TYPE="rf"
ENV N_ESTIMATORS="100"
ENV APP_TITLE="Digit Classifier"
ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860
CMD ["python", "app.py"]
