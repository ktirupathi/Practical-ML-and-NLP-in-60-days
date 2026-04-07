# Week 4: MLOps Foundations — Tracking, Serving, and Deployment

*Days 22-28: From notebook to production with experiment tracking, APIs, and containers*

---

## 1. Experiment Tracking with MLflow

### What is it
MLflow is an open-source platform for managing the ML lifecycle. It logs parameters, metrics, artifacts, and model versions for every experiment run, making it possible to compare, reproduce, and share results across a team.

### Why it matters
Without experiment tracking, you lose track of which hyperparameters produced which results. Teams waste weeks re-running experiments because nobody recorded the configuration. MLflow provides a single source of truth for all experiments.

### Python code
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X, y = make_classification(n_samples=2000, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("rf_classification")

for n_est, depth in [(100, 5), (200, 10), (300, 15)]:
    with mlflow.start_run(run_name=f"rf_depth{depth}"):
        mlflow.log_params({"n_estimators": n_est, "max_depth": depth})
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "f1": f1_score(y_test, model.predict(X_test)),
            "roc_auc": roc_auc_score(y_test, y_prob),
        })
        mlflow.sklearn.log_model(model, "model")
# View results: mlflow ui -> http://localhost:5000
```

### Common mistakes
- Not setting an experiment name, so all runs go into the default experiment
- Logging metrics outside the run context (the run is already closed)
- Storing large datasets as artifacts instead of using a data versioning tool

### Interview questions
- **Q: What is the difference between MLflow Tracking, Models, and Registry?** A: Tracking logs parameters/metrics/artifacts. Models packages models in a standard format. Registry manages model versions and stage transitions (staging, production, archived).
- **Q: How do you compare experiments in MLflow?** A: Use the MLflow UI to filter, sort, and compare runs side by side. Programmatically, use `mlflow.search_runs()` to query run data as a DataFrame.
- **Q: How does MLflow differ from Weights and Biases?** A: MLflow is open-source and self-hosted. W&B is a managed SaaS with richer visualization and collaboration. MLflow gives more control; W&B gives more convenience.

---

## 2. Data Versioning with DVC

### What is it
DVC (Data Version Control) extends Git to handle large files, datasets, and ML pipelines. It tracks data files with lightweight metafiles (.dvc) stored in Git while the actual data lives in remote storage (S3, GCS, Azure Blob, or local).

### Why it matters
Git cannot handle large data files, and data changes silently break models. DVC lets you version datasets the same way you version code: every commit corresponds to a specific version of both the code and the data it was trained on.

### Setup and usage
```bash
# Initialize DVC in a Git repo
# dvc init
# dvc remote add -d myremote s3://my-bucket/dvc-store

# Track a data file
# dvc add data/train.csv
# git add data/train.csv.dvc data/.gitignore
# git commit -m "Track training data with DVC"
# dvc push              # Upload data to remote
# dvc pull              # Download data on another machine
```

### Pipeline definition (dvc.yaml)
```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps: [src/preprocess.py, data/raw.csv]
    outs: [data/processed.csv]
  train:
    cmd: python src/train.py
    deps: [src/train.py, data/processed.csv]
    outs: [models/model.pkl]
    metrics: [{metrics.json: {cache: false}}]
# Run pipeline: dvc repro
# Compare metrics across commits: dvc metrics diff
```

### Common mistakes
- Committing large data files directly to Git instead of using DVC
- Forgetting to push data to remote storage (teammates run dvc pull and get nothing)
- Not adding .dvc files to Git (losing track of data versions)

### Interview questions
- **Q: How does DVC track data without storing it in Git?** A: DVC creates small .dvc metafiles containing a hash of the data. These metafiles are stored in Git. The actual data is stored in a configured remote. The hash links a Git commit to a specific data version.
- **Q: What is `dvc repro` and why is it useful?** A: It reproduces a pipeline by running only the stages whose dependencies have changed. This saves time and ensures reproducibility.
- **Q: How do you roll back to a previous data version?** A: `git checkout <commit> -- data/train.csv.dvc` then `dvc checkout`. This retrieves the data that corresponds to that Git commit.

---

## 3. Model Serialization (pickle, joblib, ONNX)

### What is it
Model serialization converts a trained model object into a byte stream that can be saved to disk and loaded later for inference. pickle and joblib are Python-native formats. ONNX (Open Neural Network Exchange) is a cross-platform format that enables running models in different runtimes.

### Why it matters
You cannot retrain a model for every prediction. Serialization lets you train once and deploy everywhere. ONNX enables using a model trained in Python inside a Java or C++ application with optimized inference.

### Python code
```python
import pickle, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
loaded = pickle.load(open("model.pkl", "rb"))

# joblib (better for large numpy arrays)
joblib.dump(model, "model.joblib")
loaded = joblib.load("model.joblib")

# ONNX (cross-platform)
from skl2onnx import to_onnx
import onnxruntime as rt

onx = to_onnx(model, X[:1].astype(np.float32))
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
sess = rt.InferenceSession("model.onnx")
onnx_pred = sess.run(None, {sess.get_inputs()[0].name: X[:5].astype(np.float32)})
```

### Common mistakes
- Using pickle for untrusted data (security risk — pickle can execute arbitrary code)
- Not versioning the model file alongside the code that produced it
- Serializing preprocessing steps separately from the model (use a Pipeline instead)

### Interview questions
- **Q: When would you use ONNX over pickle?** A: When you need inference in a non-Python environment (C++, Java, JavaScript), or when you want optimized inference with ONNX Runtime and hardware-specific acceleration.
- **Q: What is the security risk of pickle?** A: pickle can execute arbitrary Python code during deserialization. Never unpickle data from untrusted sources. Use safetensors or ONNX for models shared publicly.
- **Q: How do you handle model versioning?** A: Store models with metadata (training date, metrics, data version, code commit) using MLflow Model Registry or a naming convention like `model_v2.3_auc0.92.joblib`.

---

## 4. FastAPI for ML Serving

### What is it
FastAPI is a modern Python web framework for building REST APIs. It uses Python type hints for automatic request validation, generates interactive API documentation (Swagger), and supports async request handling. It is the most popular choice for serving ML models as HTTP endpoints.

### Why it matters
Models are useless unless other systems can call them. A FastAPI endpoint lets any application send a request with features and receive predictions in milliseconds. It is the bridge between data science and production.

### Python code
```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, numpy as np
from typing import List

app = FastAPI(title="ML Prediction API", version="1.0")
model = joblib.load("model.joblib")  # Load once at startup

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=10, max_length=10)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "1.0"

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0].max())
    return PredictionResponse(prediction=prediction, probability=probability)

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

```python
# client.py
import requests
resp = requests.post("http://localhost:8000/predict",
    json={"features": [0.5, -1.2, 0.3, 0.8, -0.5, 1.1, 0.2, -0.7, 0.9, 0.1]})
print(resp.json())  # {"prediction": 1, "probability": 0.87, "model_version": "1.0"}
```

### Common mistakes
- Loading the model inside the request handler (reloads on every request)
- Not adding input validation (Pydantic handles this automatically in FastAPI)
- Not adding a health check endpoint (required for load balancers and Kubernetes)

### Interview questions
- **Q: Why use FastAPI over Flask for ML serving?** A: FastAPI has automatic request validation via Pydantic, built-in async support, auto-generated OpenAPI docs, and is significantly faster due to Starlette and Uvicorn.
- **Q: How do you handle model updates without downtime?** A: Blue-green deployment (two instances, switch traffic), canary deployment (gradual shift), or load the new model in a background thread and swap the reference atomically.
- **Q: How do you handle batch predictions efficiently?** A: Accept a list of inputs in one request, vectorize inference, and return all predictions at once. This amortizes HTTP overhead and leverages numpy vectorization.

---

## 5. Streamlit Dashboards

### What is it
Streamlit is a Python framework for building interactive ML dashboards with pure Python — no HTML, CSS, or JavaScript. You write a script, and Streamlit converts it into a web app with widgets, charts, and real-time interactivity.

### Why it matters
Stakeholders cannot run Jupyter notebooks. Streamlit lets data scientists share interactive demos, exploratory tools, and model explanations with non-technical users in minutes.

### Python code
```python
# streamlit_app.py
import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("ML Model Explorer")

n_est = st.sidebar.slider("Number of Trees", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
model.fit(X_tr, y_tr)

col1, col2 = st.columns(2)
col1.metric("Train Accuracy", f"{model.score(X_tr, y_tr):.4f}")
col2.metric("Test Accuracy", f"{model.score(X_te, y_te):.4f}")

st.bar_chart(pd.DataFrame({"Importance": model.feature_importances_},
             index=[f"F{i}" for i in range(10)]))

# Run: streamlit run streamlit_app.py
```

### Common mistakes
- Putting expensive computations outside of `st.cache_data` or `st.cache_resource`
- Building complex multi-page apps when a proper framework (Dash, React) would be better
- Not handling user input errors (invalid formats crash the app)

### Interview questions
- **Q: What is the difference between `st.cache_data` and `st.cache_resource`?** A: `cache_data` is for serializable data (DataFrames, arrays) and copies per caller. `cache_resource` is for non-serializable objects (models, DB connections) and shares a single instance.
- **Q: When would you choose Streamlit vs Gradio vs Dash?** A: Streamlit for general ML dashboards. Gradio for quick model demos with input/output interfaces. Dash for production dashboards with complex layouts.
- **Q: How do you deploy a Streamlit app?** A: Streamlit Community Cloud for public repos, or containerize with Docker and deploy to any cloud platform (AWS ECS, GCP Cloud Run, Azure Container Apps).

---

## 6. Docker for ML

### What is it
Docker packages an application and all its dependencies into a container — a lightweight, portable, reproducible runtime environment. For ML, this means your model, code, Python version, and every library are bundled into a single image that runs identically everywhere.

### Why it matters
"It works on my machine" is the most expensive sentence in ML engineering. Docker eliminates environment discrepancies between development, staging, and production. It also enables horizontal scaling and CI/CD pipelines.

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.joblib .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: "3.8"
services:
  ml-api:
    build: .
    ports: ["8000:8000"]
    environment: [MODEL_PATH=/app/model.joblib, LOG_LEVEL=INFO]
    restart: unless-stopped
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --port 5000
# Build: docker build -t ml-api:v1 .
# Run:   docker run -p 8000:8000 ml-api:v1
```

### Common mistakes
- Not using `.dockerignore` to exclude data files, notebooks, and virtual environments
- Installing dependencies and copying code in the same layer (busts cache on every change)
- Running containers as root in production (security risk)

### Interview questions
- **Q: What is the difference between a Docker image and a container?** A: An image is a read-only template with the application and dependencies. A container is a running instance of an image with its own writable layer.
- **Q: Why copy requirements.txt before the rest of the code?** A: Docker caches layers. If requirements.txt has not changed, pip install is skipped on rebuild. Copying code last means code changes do not trigger a slow dependency install.
- **Q: How do you reduce Docker image size for ML?** A: Use slim base images, multi-stage builds, `--no-cache-dir` for pip, `.dockerignore`, and avoid installing dev dependencies.

---

## Week 4 Assignment

See [assignments/week-04-unsupervised-feature-engineering/](../../assignments/week-04-unsupervised-feature-engineering/) for the full assignment with MLflow tracking exercises, FastAPI endpoint development, Docker containerization tasks, and an end-to-end deployment project.
