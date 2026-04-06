# Deployment Guide

Complete guide to deploying ML/NLP models in production.

---

## Table of Contents

- [FastAPI Deployment](#fastapi-deployment)
- [Streamlit Deployment](#streamlit-deployment)
- [Docker Deployment](#docker-deployment)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [API Testing](#api-testing)
- [Cloud Deployment Overview](#cloud-deployment-overview)
- [Common Pitfalls](#common-pitfalls)

---

## FastAPI Deployment

FastAPI is the recommended framework for serving ML models as REST APIs. It provides automatic OpenAPI docs, request validation via Pydantic, and async support.

### Step 1: Define Request/Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions")

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top_predictions: List[dict]
    model_version: str
```

### Step 2: Create the FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib

# Global model reference
model = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global model, vectorizer
    model = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    yield
    model = None
    vectorizer = None

app = FastAPI(
    title="ML Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = vectorizer.transform([request.text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    classes = model.classes_
    top_k_idx = probabilities.argsort()[-request.top_k:][::-1]
    top_predictions = [
        {"label": classes[i], "confidence": round(float(probabilities[i]), 4)}
        for i in top_k_idx
    ]
    
    return PredictionResponse(
        label=prediction,
        confidence=round(float(max(probabilities)), 4),
        top_predictions=top_predictions,
        model_version="1.0.0"
    )
```

### Step 3: Run the Server

```bash
# Development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production (with multiple workers)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 4: Access Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Streamlit Deployment

Streamlit is used for interactive ML dashboards and demos.

### Basic ML Dashboard Template

```python
import streamlit as st
import requests
import json

st.set_page_config(page_title="ML Demo", layout="wide")
st.title("ML Model Demo")

# Sidebar configuration
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
top_k = st.sidebar.slider("Top K Predictions", 1, 10, 3)

# Main input
text_input = st.text_area("Enter text for prediction:", height=150)

if st.button("Predict", type="primary"):
    if text_input.strip():
        with st.spinner("Running prediction..."):
            response = requests.post(
                f"{api_url}/predict",
                json={"text": text_input, "top_k": top_k}
            )
            if response.status_code == 200:
                result = response.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Label", result["label"])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                st.subheader("Top Predictions")
                for pred in result["top_predictions"]:
                    st.progress(pred["confidence"], text=f"{pred['label']}: {pred['confidence']:.1%}")
            else:
                st.error(f"API error: {response.status_code}")
    else:
        st.warning("Please enter text.")
```

### Run Streamlit

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## Docker Deployment

### Dockerfile for ML Applications

```dockerfile
# Multi-stage build for smaller image
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY artifacts/ ./artifacts/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/artifacts/model.pkl
      - LOG_LEVEL=info
    volumes:
      - ./artifacts:/app/artifacts:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

### Build and Run

```bash
# Build
docker build -t ml-app .

# Run
docker run -p 8000:8000 ml-app

# With docker-compose
docker-compose up -d

# View logs
docker-compose logs -f api
```

---

## CI/CD with GitHub Actions

### .github/workflows/ml-pipeline.yml

```yaml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Lint check
        run: |
          pip install ruff
          ruff check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t ml-app:${{ github.sha }} .
      
      - name: Run smoke test
        run: |
          docker run -d -p 8000:8000 --name test-app ml-app:${{ github.sha }}
          sleep 5
          curl -f http://localhost:8000/health || exit 1
          docker stop test-app
```

---

## API Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test input for the model", "top_k": 3}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["first text", "second text"]}'
```

### Using Python requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Sample input text", "top_k": 5}
)
print(response.json())

# Batch with error handling
try:
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json={"texts": ["text1", "text2", "text3"]},
        timeout=30
    )
    response.raise_for_status()
    results = response.json()
except requests.exceptions.RequestException as e:
    print(f"API error: {e}")
```

### Using pytest

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={"text": "test input", "top_k": 3})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()

def test_predict_empty_text():
    response = client.post("/predict", json={"text": "", "top_k": 3})
    assert response.status_code == 422  # Validation error
```

---

## Cloud Deployment Overview

### AWS

- **SageMaker**: Managed ML model hosting with auto-scaling
- **ECS/Fargate**: Run Docker containers serverlessly
- **Lambda**: Serverless inference for low-traffic models
- **EC2**: Full control over instance for GPU workloads

### GCP

- **Vertex AI**: Managed ML platform with model endpoints
- **Cloud Run**: Serverless containers (good for FastAPI)
- **GKE**: Kubernetes for complex multi-model deployments

### Azure

- **Azure ML**: Managed endpoints with blue/green deployment
- **Container Apps**: Serverless containers
- **AKS**: Kubernetes for production workloads

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Loading model on every request | Load model once at startup using lifespan events |
| No input validation | Use Pydantic models with field constraints |
| No health check endpoint | Add `/health` endpoint for monitoring |
| Blocking async endpoints | Use `async def` only for I/O-bound ops; ML inference is CPU-bound, use sync |
| No request timeout | Set timeout in uvicorn and client-side |
| Missing CORS headers | Add `CORSMiddleware` for browser-based clients |
| No model versioning | Include `model_version` in responses |
| Large Docker images | Use multi-stage builds and slim base images |
| Running as root in Docker | Create and use a non-root user |
| No graceful shutdown | Use lifespan events for cleanup |
