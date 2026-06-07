# Churn scoring API — reproducible container.
# Build:  docker build -t d2c-churn-api .
# Run:    docker run -p 8000:8000 d2c-churn-api
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application, training script, and data.
COPY train_model.py .
COPY app ./app
COPY data ./data

# Train the model at build time so the image is self-contained and ready to serve.
RUN python train_model.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
