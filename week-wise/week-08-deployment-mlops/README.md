# Week 8: Deployment, MLOps, and Career

*Days 50-60: Production systems, monitoring, optimization, and interview preparation*

---

## 1. ML System Design

### What is it
ML system design is the practice of architecting end-to-end ML systems that work reliably at scale. It covers data pipelines, feature engineering, model training, serving infrastructure, monitoring, and iteration cycles.

### Why it matters
Building a model in a notebook is 10% of the work. The other 90% is: reliable data pipelines, feature stores, model serving with low latency, monitoring for drift, A/B testing, and graceful degradation when models fail.

### Key concepts
- **Batch vs real-time serving**: Batch = precompute predictions periodically; real-time = compute on-demand per request
- **Feature stores**: Centralized storage ensuring consistent features between training and serving
- **Model serving patterns**: Direct embedding, model-as-service, sidecar pattern
- **Fallback strategies**: Rule-based fallback when model confidence is low or model is unavailable

### Interview: System design framework
1. **Clarify requirements**: What metric matters? Latency budget? Scale?
2. **Data pipeline**: Sources, ingestion, validation, feature engineering
3. **Model selection**: Baseline first, then iterate. Consider inference cost.
4. **Serving**: Batch or real-time? Caching? Load balancing?
5. **Monitoring**: Input drift, prediction drift, business metric tracking
6. **Iteration**: A/B testing, retraining triggers, feedback loops

### Common mistakes
- Over-engineering: starting with a complex architecture when a simple model + cron job suffices
- Ignoring data quality: garbage in, garbage out at scale
- No fallback: what happens when the model service is down?

### Interview questions
- **Q: Design a recommendation system for an e-commerce platform.** A: Candidate retrieval (ANN on user/item embeddings) then ranking (LTR model with features). Batch compute daily recommendations, real-time adjust for session context.
- **Q: How would you handle cold-start users?** A: Content-based features (demographics, browse history), popularity-based fallback, explore/exploit strategy.
- **Q: How do you decide between batch and real-time inference?** A: Batch if latency tolerance > minutes and data changes slowly; real-time if sub-second response needed or features change per-request.

---

## 2. Model Monitoring

### What is it
Model monitoring detects when a deployed model starts degrading. This happens due to **data drift** (input distribution changes), **concept drift** (relationship between inputs and target changes), or **model staleness** (the world changes but the model does not).

### Why it matters
Models degrade silently. A fraud detection model trained on 2023 data may miss new fraud patterns in 2024. Without monitoring, you only discover this when business metrics drop — weeks or months too late.

### Math: Drift detection

**Population Stability Index (PSI):**

PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

PSI < 0.1: no significant drift. PSI 0.1-0.25: moderate drift. PSI > 0.25: significant drift.

**Kolmogorov-Smirnov test:** Measures maximum distance between two cumulative distributions.

### Python code
```python
import numpy as np
from scipy import stats

def detect_drift(reference, current, feature_names, alpha=0.05):
    """Detect data drift using KS test per feature."""
    results = {}
    for i, name in enumerate(feature_names):
        stat, p_value = stats.ks_2samp(reference[:, i], current[:, i])
        results[name] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift": p_value < alpha
        }
    return results

# Example
np.random.seed(42)
reference = np.random.randn(1000, 3)  # training distribution
current = np.random.randn(1000, 3)
current[:, 0] += 0.5  # inject drift in feature 0

drift = detect_drift(reference, current, ["feature_0", "feature_1", "feature_2"])
for name, result in drift.items():
    status = "DRIFT" if result["drift"] else "OK"
    print(f"{name}: KS={result['ks_statistic']}, p={result['p_value']} [{status}]")
```

### Common mistakes
- Monitoring only model accuracy without tracking input distributions
- Setting drift thresholds too sensitive (false alarms) or too lenient (missing real drift)
- Not having an automated retraining pipeline when drift is detected

### Interview questions
- **Q: What is the difference between data drift and concept drift?** A: Data drift = P(X) changes; concept drift = P(Y|X) changes. Data drift is easier to detect.
- **Q: How often should you retrain a model?** A: Depends on domain: daily for ad click prediction, monthly for loan default, only on drift for stable domains.
- **Q: How do you monitor an NLP model in production?** A: Track input text length distribution, vocabulary coverage, embedding drift (centroid shift), confidence score distribution.

---

## 3. A/B Testing for ML Models

### What is it
A/B testing compares two model versions by randomly splitting live traffic between them and measuring which performs better on a business metric. It provides statistical evidence that a new model is genuinely better, not just lucky.

### Why it matters
Offline metrics (AUC, F1) do not always correlate with online business metrics (revenue, engagement). A model with higher offline accuracy might actually hurt user experience. A/B testing bridges this gap.

### Python code
```python
from scipy import stats
import numpy as np

def ab_test(control, treatment, alpha=0.05):
    """Run A/B test with two-sample t-test."""
    t_stat, p_value = stats.ttest_ind(control, treatment)
    
    lift = (np.mean(treatment) - np.mean(control)) / np.mean(control)
    
    return {
        "control_mean": round(np.mean(control), 4),
        "treatment_mean": round(np.mean(treatment), 4),
        "lift": f"{lift:.2%}",
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "winner": "treatment" if (p_value < alpha and lift > 0) else "control"
    }

# Example: comparing click-through rates
np.random.seed(42)
control_ctr = np.random.binomial(1, 0.10, 5000)    # 10% baseline CTR
treatment_ctr = np.random.binomial(1, 0.112, 5000)  # 11.2% new model CTR

result = ab_test(control_ctr, treatment_ctr)
for k, v in result.items():
    print(f"{k}: {v}")
```

### Interview questions
- **Q: How do you determine sample size for an A/B test?** A: Based on baseline rate, minimum detectable effect, significance level (alpha=0.05), and power (0.80). Use power analysis calculation.
- **Q: What is the multiple comparison problem?** A: Testing many metrics simultaneously inflates false positive rate. Use Bonferroni correction or control FDR.
- **Q: What are alternatives to A/B testing?** A: Multi-armed bandits (adaptive allocation), interleaving (for ranking), and causal inference methods.

---

## 4. Performance Optimization

### What is it
Model optimization reduces inference latency, memory usage, and compute cost without significantly degrading accuracy. Techniques include quantization (reducing numerical precision), distillation (training a smaller model to mimic a larger one), and ONNX export (optimized runtime).

### Key techniques

| Technique | Speedup | Accuracy Loss | When to Use |
|-----------|---------|--------------|-------------|
| **FP16 (half precision)** | 2x | <0.1% | GPU inference |
| **INT8 quantization** | 2-4x | 0.5-1% | CPU/edge inference |
| **Knowledge distillation** | 3-10x | 1-3% | Deploy smaller model |
| **ONNX Runtime** | 1.5-3x | 0% | Cross-platform optimization |
| **Pruning** | 2-5x | 0.5-2% | Sparse hardware support |

### Python code: ONNX export
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Create dummy input
dummy = tokenizer("Sample text for export", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
    opset_version=14
)
print("Exported to model.onnx")

# Inference with ONNX Runtime
# import onnxruntime as ort
# session = ort.InferenceSession("model.onnx")
# outputs = session.run(None, {"input_ids": input_ids, "attention_mask": mask})
```

### Interview questions
- **Q: What is quantization?** A: Reducing numerical precision (FP32 to INT8) to decrease model size and speed up inference. Post-training quantization is easiest; quantization-aware training is more accurate.
- **Q: When would you use knowledge distillation?** A: When you need a much smaller model for edge/mobile deployment. Train a "student" model to match the "teacher" model's soft predictions.
- **Q: How do you benchmark inference performance?** A: Measure P50, P95, P99 latency over 1000+ requests. Measure throughput (requests/second). Test under realistic batch sizes.

---

## 5. Security for ML APIs

### What is it
Securing ML APIs involves authentication (who can access), authorization (what they can do), input validation (prevent adversarial inputs), rate limiting (prevent abuse), and audit logging (track predictions).

### Python code: JWT auth in FastAPI
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time

app = FastAPI()
security = HTTPBearer()
SECRET = "your-secret-key"

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, SECRET, algorithms=["HS256"])
        if payload["exp"] < time.time():
            raise HTTPException(401, "Token expired")
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

@app.post("/predict")
async def predict(data: dict, user=Depends(verify_token)):
    return {"prediction": "positive", "user": user["sub"]}
```

### Interview questions
- **Q: What is model extraction attack?** A: An attacker queries your API thousands of times to build a clone of your model. Mitigate with rate limiting and monitoring query patterns.
- **Q: How do you prevent adversarial inputs?** A: Input validation (length, character set), anomaly detection on input features, and confidence thresholds.

---

## 6. Interview Preparation

### ML System Design Patterns

| Problem | Retrieval | Ranking | Serving |
|---------|-----------|---------|---------|
| **Recommendation** | ANN on embeddings | LTR with features | Batch + real-time |
| **Search** | BM25 + semantic | Cross-encoder reranker | Real-time |
| **Fraud Detection** | Rule-based filter | Gradient boosting | Real-time streaming |
| **Content Moderation** | Keyword filter | Multi-modal classifier | Real-time |

### Common coding interview topics
1. Implement logistic regression from scratch (sigmoid, cross-entropy, gradient descent)
2. Implement K-means clustering from scratch
3. Write a data preprocessing pipeline with sklearn
4. Build a simple recommendation system with collaborative filtering
5. Implement A/B test statistical significance calculation

### Tips
- Always start with a simple baseline before proposing complex solutions
- Discuss trade-offs explicitly (latency vs accuracy, cost vs performance)
- Mention monitoring, fallbacks, and iteration — production thinking impresses interviewers

---

## Week 8 Assignment

See [assignments/week-08-deployment-mlops/](../../assignments/week-08-deployment-mlops/) for the full assignment.
