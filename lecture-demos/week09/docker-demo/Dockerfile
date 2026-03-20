# 1. Start with Python + Linux
FROM python:3.10-slim

# 2. Create a workspace folder
WORKDIR /app

# 3. Bring the ingredient list
COPY requirements.txt .

# 4. Install packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Bring our code
COPY app.py .

# 6. Let the outside world connect (Gradio defaults to 127.0.0.1)
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 7. Open the port
EXPOSE 7860

# 8. What runs when the container starts
CMD ["python", "app.py"]
