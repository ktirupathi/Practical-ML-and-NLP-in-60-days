# FastAPI Deployment Guide for ML Models

This guide covers everything you need to deploy a production-ready machine learning model as a REST API using FastAPI. By the end you will have a fully functional text classification API with authentication, rate limiting, logging, and error handling.

**Used in:** Day 25 — FastAPI for ML, Day 28 — Resume Screening, Day 29 — Ticket Router, Day 47 — Financial Risk

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pydantic Models](#pydantic-models)
- [Model Loading with Lifespan Events](#model-loading-with-lifespan-events)
- [Complete Working Example](#complete-working-example)
- [Error Handling and Logging](#error-handling-and-logging)
- [API Testing](#api-testing)
- [Authentication (API Key)](#authentication-api-key)
- [Rate Limiting](#rate-limiting)
- [Production Deployment Tips](#production-deployment-tips)

---

## Project Structure

```
ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic request/response models
│   │   └── ml_model.py       # ML model wrapper class
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py        # /predict endpoints
│   │   └── health.py         # /health endpoint
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py           # API key authentication
│   │   └── rate_limit.py     # Rate limiting
│   └── core/
│       ├── __init__.py
│       └── config.py         # App configuration
├── models/                    # Saved ML model artifacts
│   ├── classifier.pkl
│   └── vectorizer.pkl
├── tests/
│   ├── test_predict.py
│   └── test_health.py
├── requirements.txt
├── Dockerfile
└── .env
```

---

## Installation

```bash
pip install fastapi uvicorn[standard] pydantic python-dotenv \
            scikit-learn joblib slowapi redis python-multipart \
            transformers torch sentence-transformers
```

`requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
pydantic-settings==2.3.0
python-dotenv==1.0.1
scikit-learn==1.5.0
joblib==1.4.2
slowapi==0.1.9
redis==5.0.6
python-multipart==0.0.9
httpx==0.27.0           # for testing
pytest==8.2.2
```

---

## Pydantic Models

`app/models/schemas.py`:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class SentimentLabel(str, Enum):
    positive = "positive"
    neutral  = "neutral"
    negative = "negative"


class TextInput(BaseModel):
    """Single text classification request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Text to classify",
        examples=["The company reported record profits this quarter."]
    )
    model_version: Optional[str] = Field(
        default="v1",
        description="Model version to use for inference"
    )

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty after stripping whitespace")
        return v


class BatchTextInput(BaseModel):
    """Batch classification request (up to 32 texts)."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to classify"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty")
        return [t.strip() for t in v]


class PredictionResult(BaseModel):
    """Single prediction result."""
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    processing_time_ms: float


class BatchPredictionResult(BaseModel):
    """Batch prediction results."""
    results: List[PredictionResult]
    total_texts: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: str
    request_id: Optional[str] = None
```

---

## Model Loading with Lifespan Events

`app/models/ml_model.py`:

```python
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    Wrapper for a scikit-learn sentiment classifier.
    Thread-safe for FastAPI's async environment.
    """

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.version = "unloaded"
        self._load_time: Optional[float] = None

    def load(self, model_path: str, vectorizer_path: str, version: str = "v1") -> None:
        """Load model artifacts from disk."""
        start = time.perf_counter()

        model_file = Path(model_path)
        vec_file   = Path(vectorizer_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vec_file.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        self.model      = joblib.load(model_file)
        self.vectorizer = joblib.load(vec_file)
        self.version    = version
        self._load_time = time.perf_counter() - start

        logger.info(
            "Model loaded successfully",
            extra={"version": version, "load_time_s": round(self._load_time, 3)}
        )

    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str) -> dict:
        """Run inference on a single text. Returns label + probabilities."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call .load() first.")

        start = time.perf_counter()

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "label": self.LABEL_MAP[label_idx],
            "confidence": float(proba[label_idx]),
            "probabilities": {
                self.LABEL_MAP[i]: float(p)
                for i, p in enumerate(proba)
            },
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def predict_batch(self, texts: list) -> list:
        """Run inference on a batch of texts."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")

        start = time.perf_counter()
        X = self.vectorizer.transform(texts)
        probas = self.model.predict_proba(X)
        total_ms = (time.perf_counter() - start) * 1000

        results = []
        for i, proba in enumerate(probas):
            label_idx = int(np.argmax(proba))
            results.append({
                "label": self.LABEL_MAP[label_idx],
                "confidence": float(proba[label_idx]),
                "probabilities": {
                    self.LABEL_MAP[j]: float(p)
                    for j, p in enumerate(proba)
                },
                "processing_time_ms": round(total_ms / len(texts), 2),
            })
        return results


# Module-level singleton
classifier = SentimentClassifier()
```

---

## Complete Working Example

`app/core/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ML Sentiment API"
    app_version: str = "1.0.0"
    debug: bool = False
    model_path: str = "models/classifier.pkl"
    vectorizer_path: str = "models/vectorizer.pkl"
    model_version: str = "v1"
    api_key: str = "changeme-secret-key"
    rate_limit_per_minute: int = 60
    redis_url: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"


settings = Settings()
```

`app/main.py`:

```python
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.ml_model import classifier
from app.routers import predict, health

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App startup / shutdown (lifespan) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML model artifacts on startup.
    Release resources on shutdown.
    Using the modern lifespan approach (replaces on_event).
    """
    logger.info("Starting up — loading model artifacts...")
    try:
        classifier.load(
            model_path=settings.model_path,
            vectorizer_path=settings.vectorizer_path,
            version=settings.model_version,
        )
        logger.info(f"Model v{settings.model_version} loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}. API will start but /predict will fail.")

    app.state.start_time = time.time()
    yield  # <-- application runs here

    logger.info("Shutting down...")


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready sentiment classification API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Add X-Request-ID and X-Process-Time headers to every response."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 2))

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} [{elapsed_ms:.1f}ms] req={request_id}"
    )
    return response


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled exception [req={request_id}]: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
        }
    )


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
```

`app/routers/health.py`:

```python
import time
from fastapi import APIRouter
from app.models.ml_model import classifier
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request=None):
    """Simple liveness check for load balancers."""
    from app.main import app
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded(),
        model_version=classifier.version,
        uptime_seconds=round(uptime, 1),
    )


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe — only ready when model is loaded."""
    if not classifier.is_loaded():
        return {"status": "not_ready", "reason": "model_not_loaded"}, 503
    return {"status": "ready"}
```

`app/routers/predict.py`:

```python
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.schemas import (
    TextInput, BatchTextInput,
    PredictionResult, BatchPredictionResult
)
from app.models.ml_model import classifier
from app.middleware.auth import verify_api_key
import time

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predict",
    response_model=PredictionResult,
    dependencies=[Depends(verify_api_key)],
    summary="Classify sentiment of a single text",
)
async def predict_single(request: Request, payload: TextInput):
    """
    Classify the sentiment of a single financial or general text.

    Returns label (positive/neutral/negative), confidence score,
    and full probability distribution.
    """
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        result = classifier.predict(payload.text)
        logger.debug(
            f"Prediction: {result['label']} ({result['confidence']:.3f}) "
            f"req={getattr(request.state, 'request_id', 'unknown')}"
        )
        return PredictionResult(**result)
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    dependencies=[Depends(verify_api_key)],
    summary="Classify sentiment of multiple texts",
)
async def predict_batch(request: Request, payload: BatchTextInput):
    """
    Classify sentiment for a batch of texts (up to 32 at once).
    More efficient than calling /predict individually for each text.
    """
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        start = time.perf_counter()
        results = classifier.predict_batch(payload.texts)
        total_ms = (time.perf_counter() - start) * 1000

        return BatchPredictionResult(
            results=[PredictionResult(**r) for r in results],
            total_texts=len(payload.texts),
            total_processing_time_ms=round(total_ms, 2),
        )
    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
```

---

## Authentication (API Key)

`app/middleware/auth.py`:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Validate API key from X-API-Key header.
    In production, look this up in a database or secrets manager.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header.",
        )
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key
```

---

## Rate Limiting

`app/middleware/rate_limit.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

# Create rate limiter (uses in-memory storage by default)
# For production, use Redis: Limiter(key_func=..., storage_uri="redis://...")
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "error_code": "RATE_LIMIT_EXCEEDED",
        },
        headers={"Retry-After": "60"},
    )
```

Add to `main.py`:

```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.middleware.rate_limit import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
```

Add to individual route with decorator:

```python
from app.middleware.rate_limit import limiter

@router.post("/predict")
@limiter.limit("30/minute")   # 30 requests per minute per IP
async def predict_single(request: Request, payload: TextInput):
    ...
```

---

## Running the API

```bash
# Development (hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (multiple workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn (recommended for production)
pip install gunicorn
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile -
```

---

## API Testing

### curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-secret-key" \
  -d '{"text": "The company reported record profits this quarter."}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-secret-key" \
  -d '{"texts": ["Profits surged 45%", "Revenue fell short of expectations", "The board met on Tuesday"]}'
```

### Python requests

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS  = {"X-API-Key": "changeme-secret-key", "Content-Type": "application/json"}

# Single prediction
response = requests.post(
    f"{BASE_URL}/api/v1/predict",
    json={"text": "The company reported record profits this quarter."},
    headers=HEADERS,
)
print(response.json())
# {'label': 'positive', 'confidence': 0.91, 'probabilities': {...}, 'processing_time_ms': 3.2}

# Batch prediction
texts = [
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    "The company will close three manufacturing plants.",
    "Board members were elected at the annual general meeting.",
]
response = requests.post(
    f"{BASE_URL}/api/v1/predict/batch",
    json={"texts": texts},
    headers=HEADERS,
)
batch_result = response.json()
for text, result in zip(texts, batch_result["results"]):
    print(f"[{result['label']:8s}] {text[:60]}")
```

### pytest integration tests

```python
# tests/test_predict.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "changeme-secret-key"}


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_requires_auth():
    resp = client.post("/api/v1/predict", json={"text": "test"})
    assert resp.status_code == 401


def test_predict_single():
    resp = client.post(
        "/api/v1/predict",
        json={"text": "Company profits surged to record highs."},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ["positive", "neutral", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_batch():
    resp = client.post(
        "/api/v1/predict/batch",
        json={"texts": ["Good news", "Bad news", "No news"]},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_texts"] == 3
    assert len(data["results"]) == 3


def test_predict_empty_text_rejected():
    resp = client.post(
        "/api/v1/predict",
        json={"text": "   "},
        headers=HEADERS,
    )
    assert resp.status_code == 422   # Pydantic validation error
```

---

## Production Deployment Tips

### 1. Use environment variables for secrets

```bash
# .env
API_KEY=super-secret-key-from-secrets-manager
MODEL_PATH=/app/models/classifier.pkl
VECTORIZER_PATH=/app/models/vectorizer.pkl
DEBUG=false
```

### 2. Add structured logging for observability

```python
import structlog

log = structlog.get_logger()
log.info("prediction_made", label="positive", confidence=0.91, model_version="v2")
```

### 3. Health checks for Kubernetes

```yaml
# kubernetes deployment snippet
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 5
```

### 4. Model versioning strategy

```python
# Support multiple model versions via URL path
@router.post("/api/v1/predict")   # current stable
@router.post("/api/v2/predict")   # new model

# Or via header
model_version = request.headers.get("X-Model-Version", "v1")
```

### 5. Response caching for identical inputs

```python
import hashlib, json
from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_predict(text: str) -> dict:
    return classifier.predict(text)
```

### 6. Async model inference for I/O-bound models

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def predict_async(text: str) -> dict:
    """Run CPU-bound model inference in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, classifier.predict, text)
```
