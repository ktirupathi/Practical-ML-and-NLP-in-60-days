# Day 51: ML System Design Principles

## Learning Objectives

- Understand the components of a production ML system beyond the model itself
- Design systems that handle batch vs. real-time inference with appropriate trade-offs
- Explain the role of feature stores in decoupling feature engineering from model training and serving
- Apply scalability patterns: horizontal scaling, caching, model sharding, and async processing
- Sketch end-to-end ML architectures for common use cases (recommendation, search, fraud detection)

## Key Concepts

Building a machine learning model is only a small fraction of a production ML system.
Google's famous paper "Hidden Technical Debt in Machine Learning Systems" showed that
model code represents perhaps 5% of a real ML system -- the rest includes data
collection, feature engineering, data validation, serving infrastructure, monitoring,
and configuration management. Understanding these components and how they interact is
essential for any ML engineer working on production systems. A well-designed ML system
must handle data pipelines, model training, model serving, monitoring, and retraining
in a cohesive and reliable manner.

A critical design decision is the inference pattern. Batch inference precomputes
predictions on a schedule (e.g., nightly recommendations for all users) and stores
results for fast lookup. It is simple, cost-effective, and works when predictions do
not need to reflect real-time context. Real-time inference generates predictions
on-demand per request, which is necessary for tasks like fraud detection or
personalized search ranking where the input is dynamic. Many systems use a hybrid
approach: batch-compute what you can, and use real-time inference only where freshness
matters.

Feature stores (like Feast, Tecton, or Hopsworks) solve the problem of feature
consistency between training and serving. Without a feature store, teams often compute
features differently in training pipelines (Python/Spark) and serving pipelines
(Java/Go), leading to training-serving skew. A feature store provides a single
definition of each feature, computes and stores features centrally, and serves them
with low latency at prediction time. It also enables feature reuse across teams and
models, reducing duplicate engineering effort.

## Practical Example

```python
"""
ML System Design: a simplified feature store and serving pattern.
This demonstrates the concept -- production systems use Feast, Tecton, etc.
"""
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Simple feature store simulation
class FeatureStore:
    def __init__(self):
        self.offline_store: Dict[str, Dict] = {}  # For training
        self.online_store: Dict[str, Dict] = {}    # For serving (low latency)

    def register_features(self, entity_id: str, features: Dict[str, Any]):
        """Write features to both offline and online stores."""
        timestamp = time.time()
        record = {"features": features, "timestamp": timestamp}
        self.offline_store.setdefault(entity_id, []).append(record)
        self.online_store[entity_id] = record  # Only latest for online

    def get_online_features(self, entity_id: str) -> Optional[Dict]:
        """Low-latency feature retrieval for real-time serving."""
        record = self.online_store.get(entity_id)
        return record["features"] if record else None

    def get_training_data(self) -> list:
        """Retrieve historical features for model training."""
        rows = []
        for entity_id, records in self.offline_store.items():
            for record in records:
                rows.append({"entity_id": entity_id, **record["features"]})
        return rows

# Serving layer: batch + real-time hybrid
class ModelServer:
    def __init__(self, feature_store, model=None):
        self.feature_store = feature_store
        self.model = model
        self.prediction_cache: Dict[str, Any] = {}

    def batch_predict(self, entity_ids: list):
        """Pre-compute predictions for known entities."""
        for eid in entity_ids:
            features = self.feature_store.get_online_features(eid)
            if features:
                prediction = self._run_model(features)
                self.prediction_cache[eid] = prediction
        print(f"Batch predicted {len(self.prediction_cache)} entities")

    def serve(self, entity_id: str, real_time_features: Dict = None):
        """Hybrid serving: use cache if available, else compute live."""
        if entity_id in self.prediction_cache and not real_time_features:
            return {"source": "cache", "prediction": self.prediction_cache[entity_id]}

        features = self.feature_store.get_online_features(entity_id) or {}
        if real_time_features:
            features.update(real_time_features)
        prediction = self._run_model(features)
        return {"source": "real-time", "prediction": prediction}

    def _run_model(self, features):
        # Placeholder: in production, this calls the actual model
        score = sum(features.values()) / max(len(features), 1)
        return round(score, 4)

# Demo
store = FeatureStore()
store.register_features("user_123", {"avg_spend": 45.0, "visit_freq": 3.2, "tenure_days": 365})
store.register_features("user_456", {"avg_spend": 120.0, "visit_freq": 1.1, "tenure_days": 30})

server = ModelServer(store)
server.batch_predict(["user_123", "user_456"])

print(server.serve("user_123"))  # From cache
print(server.serve("user_789", {"avg_spend": 80.0}))  # Real-time fallback
```

## Resources

- [Hidden Technical Debt in Machine Learning Systems (Google, 2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Feast: open source feature store](https://feast.dev/)
- [Designing Machine Learning Systems by Chip Huyen (O'Reilly)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

## Next Day Preview

Day 52 covers model monitoring -- detecting data drift, concept drift, and performance degradation in deployed models.
