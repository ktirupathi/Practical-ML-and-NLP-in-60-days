# Day 24: Model Serialization

## Learning Objectives

- Serialize and deserialize models using `pickle` and `joblib`
- Export scikit-learn and PyTorch models to the ONNX format for cross-framework inference
- Understand the security implications of unpickling untrusted files
- Compare model formats on file size, load speed, and portability
- Choose the right serialization strategy for development vs. production

## Key Concepts

### pickle and joblib

Python's built-in `pickle` module can serialize almost any Python object, including trained
models. `joblib` is a drop-in alternative optimized for objects containing large NumPy arrays
--- it compresses data and memory-maps arrays during loading, which can dramatically reduce
both file size and deserialization time. Both formats are Python-specific: the consumer must
have the same library versions installed, and loading a pickle from an untrusted source can
execute arbitrary code, so treat pickle files with the same caution as executable scripts.

### ONNX --- The Open Neural Network Exchange

ONNX defines a framework-agnostic graph representation for ML models. Once exported, an ONNX
model can be served by the ONNX Runtime in C++, C#, Java, or JavaScript --- no Python
required. This makes ONNX ideal for production deployments where latency matters or the
serving stack is not Python-based. Libraries like `skl2onnx` convert scikit-learn pipelines,
while PyTorch provides `torch.onnx.export` natively.

### Choosing a Format

For quick prototyping and notebook workflows, joblib is the pragmatic choice. For production
microservices that already run Python, joblib behind a FastAPI endpoint works well. When you
need language-agnostic, hardware-accelerated inference, ONNX is the clear winner. Some teams
use joblib during development and export to ONNX for deployment, getting the best of both
worlds.

## Practical Example

```python
import numpy as np
import pickle
import joblib
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

# Train a sample model
X, y = make_classification(n_samples=10_000, n_features=20, random_state=42)
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# --- 1. pickle ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    model_pkl = pickle.load(f)

# --- 2. joblib (compressed) ---
joblib.dump(model, "model.joblib", compress=3)
model_jl = joblib.load("model.joblib")

# --- 3. ONNX export ---
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

initial_type = [("X", FloatTensorType([None, 20]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
onnx_preds = session.run(None, {input_name: X[:5].astype(np.float32)})[0]

# --- Compare file sizes ---
import os
for path in ["model.pkl", "model.joblib", "model.onnx"]:
    size_kb = os.path.getsize(path) / 1024
    print(f"{path:20s} -> {size_kb:8.1f} KB")

# --- Benchmark inference speed ---
for name, predictor in [("sklearn", model_jl), ("onnx", session)]:
    start = time.perf_counter()
    for _ in range(100):
        if name == "onnx":
            session.run(None, {input_name: X[:100].astype(np.float32)})
        else:
            predictor.predict(X[:100])
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s for 100 batches")
```

## Resources

- [joblib persistence documentation](https://joblib.readthedocs.io/en/latest/persistence.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [skl2onnx converter guide](https://onnx.ai/sklearn-onnx/)

## Up Next

**Day 25 -- FastAPI for ML:** Build a REST API that serves your serialized model behind a clean, validated endpoint.
