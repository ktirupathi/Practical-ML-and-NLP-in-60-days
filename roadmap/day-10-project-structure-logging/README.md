# Day 10: Project Structure and Logging

## Overview

A well-organized project is easier to debug, test, and hand off to teammates.
Today covers modular directory layouts for ML projects, Python's built-in
logging module, configuration management with YAML and environment variables,
and structured exception handling that keeps pipelines running reliably.

---

## Learning Objectives

- Organize an ML project into data, src, notebooks, configs, and tests folders.
- Replace `print()` debugging with Python's `logging` module at multiple levels.
- Manage hyperparameters and paths with YAML config files.
- Use environment variables and `.env` files for secrets and deployment settings.
- Implement try/except patterns that log errors and fail gracefully.

---

## Key Concepts

### Modular Project Layout

A typical ML project outgrows a single notebook within days. Adopting a standard
layout early -- `src/` for reusable modules, `notebooks/` for exploration,
`configs/` for YAML parameters, `data/` for raw and processed data, and `tests/`
for unit tests -- keeps everything discoverable. Tools like Cookiecutter Data
Science provide templates, but the key habit is separating experiment code
(notebooks) from production code (modules) from configuration (YAML/env).

### Python Logging

The `logging` module lets you emit messages at five severity levels: DEBUG, INFO,
WARNING, ERROR, and CRITICAL. Unlike `print()`, logging lets you route messages
to files, set format strings with timestamps, and control verbosity per module.
In a training script, `logger.info("Epoch %d loss: %.4f", epoch, loss)` gives
you a timestamped audit trail without touching stdout. In production, you can
ship these logs to ELK, CloudWatch, or any aggregation service.

### Configuration and Exception Handling

Hardcoded paths and hyperparameters scattered across scripts make experiments
impossible to reproduce. A single `config.yaml` that lists `learning_rate`,
`batch_size`, `data_path`, and similar parameters serves as a single source of
truth. For secrets (API keys, database URLs), use environment variables loaded
via `python-dotenv`. Wrap I/O and model calls in try/except blocks that log the
traceback and either retry or exit cleanly, so a single corrupt file does not
crash an overnight training run.

---

## Practical Example

```python
# 10_project_logging.py
"""Demonstrate logging, YAML config loading, and exception handling."""

import logging
import os
import yaml
from pathlib import Path

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("ml_pipeline")

# --- Config loading ---
CONFIG_STR = """
project_name: sales_forecast
data:
  raw_path: data/raw/sales.csv
  processed_path: data/processed/sales_clean.parquet
model:
  type: lightgbm
  learning_rate: 0.05
  n_estimators: 500
  max_depth: 6
"""

config = yaml.safe_load(CONFIG_STR)
logger.info("Loaded config for project: %s", config["project_name"])
logger.info("Model type: %s, lr: %s", config["model"]["type"], config["model"]["learning_rate"])

# --- Graceful exception handling ---
def load_data(path: str):
    """Simulate data loading with error handling."""
    logger.info("Loading data from %s", path)
    if not Path(path).exists():
        logger.warning("File not found: %s. Returning empty DataFrame.", path)
        import pandas as pd
        return pd.DataFrame()
    import pandas as pd
    return pd.read_csv(path)

try:
    df = load_data(config["data"]["raw_path"])
    logger.info("Data shape: %s", df.shape)
except Exception:
    logger.exception("Failed to load data. Aborting pipeline.")
    raise

# --- Environment variables for secrets ---
db_url = os.getenv("DATABASE_URL", "sqlite:///local.db")
logger.info("Using database: %s", db_url)

logger.info("Pipeline setup complete.")
```

---

## Resources

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## Up Next

**Day 11 -- Linear and Logistic Regression:** OLS, regularization (L1/L2/ElasticNet), logistic regression, and coefficient interpretation.
