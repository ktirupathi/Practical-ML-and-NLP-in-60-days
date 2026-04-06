# Day 25: FastAPI for ML Serving

## Learning Objectives

- Build a REST API with FastAPI that loads and serves a trained ML model
- Define request and response schemas with Pydantic for automatic validation
- Handle batch predictions, health checks, and error responses gracefully
- Understand async endpoints and how FastAPI leverages ASGI for concurrency
- Generate interactive API documentation (Swagger UI) with zero extra effort

## Key Concepts

### Why FastAPI for ML

FastAPI has become the go-to framework for serving ML models in Python. It is built on
Starlette (async ASGI) and Pydantic (data validation), delivering both high throughput and
developer ergonomics. Type hints drive automatic request parsing, validation, and
documentation generation. Compared to Flask, FastAPI offers native async support, built-in
OpenAPI docs, and significantly faster request handling under concurrent load --- all of which
matter when your model endpoint is hit by production traffic.

### Pydantic Models and Validation

Every ML endpoint needs a clear contract: what fields does the request contain, what types are
they, and what does the response look like? Pydantic models let you declare this contract as
a plain Python class. FastAPI validates incoming JSON against the schema before your handler
runs, returning a detailed 422 error if anything is wrong. This eliminates an entire category
of runtime bugs and makes the API self-documenting.

### Deployment Patterns

For development you run `uvicorn app:app --reload`. In production you typically put Uvicorn
workers behind Gunicorn (`gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4`) and front
it with NGINX or a cloud load balancer. The model is loaded once at startup using a lifespan
event so it stays in memory across requests.

## Practical Example

```python
# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Pydantic schemas ---
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Four numeric features for the Iris model"
    )

class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# --- Application lifespan: load model once at startup ---
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["clf"] = joblib.load("model.joblib")
    yield
    ml_model.clear()

app = FastAPI(title="Iris Prediction API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded="clf" in ml_model,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        clf = ml_model["clf"]
        pred = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0].tolist()
        return PredictionResponse(prediction=pred, probability=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictionRequest]):
    X = np.array([r.features for r in requests])
    clf = ml_model["clf"]
    preds = clf.predict(X)
    probas = clf.predict_proba(X)
    return [
        PredictionResponse(prediction=int(p), probability=prob.tolist())
        for p, prob in zip(preds, probas)
    ]

# Run with: uvicorn app:app --reload
# Docs at: http://localhost:8000/docs
```

## Resources

- [FastAPI official documentation](https://fastapi.tiangolo.com/)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Deploying ML models with FastAPI (Real Python)](https://realpython.com/fastapi-python-web-apis/)

## Up Next

**Day 26 -- Streamlit Dashboards:** Build interactive ML dashboards that let non-technical stakeholders explore model results.
