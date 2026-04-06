# Day 22: Experiment Tracking with MLflow

## Learning Objectives

- Set up MLflow tracking server and understand the concepts of experiments, runs, and artifacts
- Log parameters, metrics, and models programmatically during training
- Compare runs in the MLflow UI to identify the best-performing configuration
- Register models in the MLflow Model Registry for staging and production promotion
- Integrate MLflow tracking into scikit-learn and other training loops

## Key Concepts

### The Experiment Tracking Problem

Machine learning development is inherently iterative. You tweak hyperparameters, swap
algorithms, change feature sets, and retrain dozens of times. Without a systematic way to
record what you tried and what happened, you end up with a mess of notebooks, spreadsheets,
and hazy memories. MLflow Tracking solves this by providing a lightweight API that logs every
detail of a run --- parameters, metrics over time, artifacts like plots and serialized models
--- into a central store that you can query and compare later.

### Experiments, Runs, and the Model Registry

An **experiment** groups related runs (e.g., "churn-prediction-v2"). Each **run** captures one
training attempt: its parameters, metric curves, and output artifacts. The MLflow UI lets you
sort, filter, and visualize runs side by side. Once you find a winning model, the **Model
Registry** gives it a name, version number, and stage label (Staging, Production, Archived),
turning ad-hoc experimentation into a governed promotion workflow.

### Beyond Tracking

MLflow also offers a Projects component for reproducible packaging and a Models component for
multi-framework deployment, but Tracking and the Registry are the foundation. Master these
two and every subsequent experiment you run will be fully auditable and reproducible.

## Practical Example

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Point to a local tracking URI (or a remote server URL)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sweep over a few hyperparameter combos
for n_trees in [50, 100, 200]:
    for max_depth in [3, 5, None]:
        with mlflow.start_run(run_name=f"rf-{n_trees}-{max_depth}"):
            # Log parameters
            mlflow.log_param("n_estimators", n_trees)
            mlflow.log_param("max_depth", max_depth)

            clf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=max_depth, random_state=42
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)

            # Log the model artifact
            mlflow.sklearn.log_model(clf, "model")

            print(f"n_trees={n_trees}, max_depth={max_depth} -> acc={acc:.3f}")

# After reviewing runs in the UI, register the best model:
# mlflow.register_model("runs:/<RUN_ID>/model", "IrisClassifier")
```

Run `mlflow ui` in the terminal to open the dashboard at `http://localhost:5000`.

## Resources

- [MLflow Tracking documentation](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry guide](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow quickstart tutorial](https://mlflow.org/docs/latest/quickstart.html)

## Up Next

**Day 23 -- Data Versioning:** Discover how DVC brings Git-like version control to datasets and ML pipelines.
