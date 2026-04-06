# Day 23: Data Versioning with DVC

## Learning Objectives

- Install DVC and initialize it inside a Git repository
- Track large data files and model artifacts without bloating the Git history
- Define reproducible ML pipelines with `dvc.yaml` and `dvc.lock`
- Configure remote storage (S3, GCS, or local) for sharing versioned data across teams
- Move between data versions using Git branches and `dvc checkout`

## Key Concepts

### The Data Versioning Problem

Git handles code beautifully, but it was never designed for multi-gigabyte datasets or binary
model files. Committing large files directly into Git makes cloning painfully slow and bloats
the repository. DVC (Data Version Control) solves this by storing lightweight `.dvc` metafiles
in Git while pushing the actual data to a configurable remote (S3, GCS, Azure Blob, SSH, or
even a local directory). Checking out a branch automatically brings the matching dataset,
giving you full reproducibility without the storage overhead.

### DVC Pipelines

Beyond file tracking, DVC lets you encode your ML workflow as a directed acyclic graph of
stages in a `dvc.yaml` file. Each stage declares its command, dependencies, and outputs.
Running `dvc repro` executes only the stages whose inputs have changed, much like `make` but
purpose-built for data science. The companion `dvc.lock` file records exact hashes of every
dependency and output, guaranteeing bit-for-bit reproducibility.

### Remote Storage and Collaboration

A DVC remote is the shared data warehouse for your team. After configuring a remote with
`dvc remote add`, anyone with access can `dvc pull` to fetch the exact dataset version that
matches their current Git commit. This decouples data sharing from Git hosting limits and
makes onboarding new team members straightforward.

## Practical Example

```python
# --- Shell commands (run in terminal) ---
# Initialize DVC in an existing Git repo
# $ dvc init
# $ git add .dvc .dvcignore
# $ git commit -m "Initialize DVC"

# Track a data file
# $ dvc add data/training_data.csv
# $ git add data/training_data.csv.dvc data/.gitignore
# $ git commit -m "Track training data v1"

# Configure a local remote (use s3://bucket for AWS)
# $ dvc remote add -d myremote /tmp/dvc-storage
# $ dvc push

# --- dvc.yaml pipeline definition ---
# stages:
#   preprocess:
#     cmd: python preprocess.py
#     deps:
#       - data/training_data.csv
#       - preprocess.py
#     outs:
#       - data/processed.csv
#   train:
#     cmd: python train.py
#     deps:
#       - data/processed.csv
#       - train.py
#     outs:
#       - models/model.pkl
#     metrics:
#       - metrics.json:
#           cache: false

# --- Python helper: compare metrics across versions ---
import json
import subprocess

def get_metrics(revision: str) -> dict:
    """Retrieve DVC metrics for a specific Git revision."""
    result = subprocess.run(
        ["dvc", "metrics", "show", "--rev", revision, "--json"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

# Compare current branch with a previous tag
current = get_metrics("HEAD")
baseline = get_metrics("v1.0")

for file, vals in current.items():
    for metric, value in vals.items():
        old = baseline.get(file, {}).get(metric, "N/A")
        print(f"{metric}: {old} -> {value}")
```

## Resources

- [DVC official documentation](https://dvc.org/doc)
- [DVC pipelines tutorial](https://dvc.org/doc/start/data-management/data-pipelines)
- [Iterative.ai blog -- versioning data and models](https://iterative.ai/blog)

## Up Next

**Day 24 -- Model Serialization:** Learn the trade-offs between pickle, joblib, and ONNX for saving and exporting trained models.
