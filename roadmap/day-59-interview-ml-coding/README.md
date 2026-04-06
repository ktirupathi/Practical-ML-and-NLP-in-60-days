# Day 59: Interview Prep — ML Coding

## Learning Objectives

- Implement common ML algorithms from scratch
- Solve data manipulation challenges with pandas efficiently
- Write clean, testable ML pipeline code under time pressure
- Handle edge cases in ML implementations
- Practice the coding patterns most commonly tested in ML interviews

## Key Concepts

ML coding interviews test two things: your understanding of ML fundamentals (can you implement logistic regression from scratch?) and your practical engineering skills (can you write clean data pipelines?). Unlike standard software engineering interviews, ML interviews focus on numerical computing, data manipulation, and statistical reasoning.

The most commonly tested implementations from scratch include: **linear regression** (normal equation and gradient descent), **logistic regression** (sigmoid, cross-entropy loss, gradient descent), **k-means clustering** (initialization, assignment, update loop), **k-nearest neighbors** (distance computation, voting), and **decision tree** (information gain, recursive splitting). You do not need to memorize these, but you should understand the math well enough to implement them with NumPy.

For pandas challenges, expect questions about groupby operations, window functions, merging strategies, handling missing data, and pivot tables. Speed matters — practice until common operations are second nature. Also practice writing sklearn-compatible custom transformers with `fit` and `transform` methods.

## Hands-On Exercise

```python
import numpy as np

# Implement Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear = X @ self.weights + self.bias
            predictions = self._sigmoid(linear)

            dw = (1 / n_samples) * (X.T @ (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Test it
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegressionScratch(lr=0.1, n_iters=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")


# Implement K-Means from scratch
class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self
```

## Resources

- [ML Interview Coding Questions Collection](https://github.com/khangich/machine-learning-interview)
- [Pandas Exercises](https://github.com/guipsamora/pandas_exercises)
- [NumPy ML Implementations](https://github.com/ddbourgin/numpy-ml)

## Next Day Preview

Tomorrow is **Graduation Day** — wrapping up the 60-day journey and planning your next steps in ML/NLP.
