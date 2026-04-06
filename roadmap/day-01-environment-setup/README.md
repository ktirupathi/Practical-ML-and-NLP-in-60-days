# Day 1: Environment Setup

## Overview

Before writing any machine learning code, you need a reliable, reproducible
development environment. Today covers installing Python 3.9+, creating isolated
virtual environments, initializing Git for version control, configuring VS Code,
and managing packages with pip.

---

## Learning Objectives

- Install Python 3.9+ and verify the installation from the command line.
- Create and activate a virtual environment using `venv`.
- Initialize a Git repository and make a first commit.
- Configure VS Code with the Python extension, linter, and formatter.
- Use `pip` and `requirements.txt` to manage project dependencies.

---

## Key Concepts

### Why Virtual Environments Matter

Every Python project should live inside its own virtual environment. Virtual
environments isolate your project's dependencies from the system Python and from
other projects. Without them, upgrading a library for one project can silently
break another. The built-in `venv` module creates a lightweight directory that
contains a private copy of the Python interpreter and a clean `site-packages`
folder.

### Git for Reproducibility

Version control is not optional in machine learning. Experiments evolve quickly,
and you need the ability to roll back to a working state, compare results across
commits, and collaborate with others. A well-structured `.gitignore` keeps large
data files and environment folders out of the repository while tracking code,
configuration, and notebooks.

### VS Code as an ML Workbench

VS Code with the official Python extension gives you IntelliSense, integrated
debugging, Jupyter notebook support, and terminal access in one window. Adding
a formatter like Black and a linter like Flake8 keeps code consistent from day
one, which pays dividends as projects grow.

---

## Practical Example

```python
# 01_verify_environment.py
"""Verify that the development environment is set up correctly."""

import sys
import subprocess

# Check Python version
major, minor = sys.version_info[:2]
assert major == 3 and minor >= 9, f"Need Python 3.9+, got {major}.{minor}"
print(f"Python version : {sys.version}")

# Check key packages
required = ["numpy", "pandas", "scikit-learn", "matplotlib"]
for pkg in required:
    try:
        mod = __import__(pkg)
        print(f"{pkg:20s} : {mod.__version__}")
    except ImportError:
        print(f"{pkg:20s} : NOT INSTALLED")

# Check Git
result = subprocess.run(["git", "--version"], capture_output=True, text=True)
print(f"Git                  : {result.stdout.strip()}")

# Quick virtual-env sanity check
print(f"Virtual env prefix   : {sys.prefix}")
print(f"Base prefix          : {sys.base_prefix}")
if sys.prefix != sys.base_prefix:
    print("Virtual environment is ACTIVE.")
else:
    print("WARNING: No virtual environment detected.")
```

---

## Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [Git Handbook (GitHub)](https://guides.github.com/introduction/git-handbook/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)

---

## Up Next

**Day 2 -- Python for ML:** NumPy arrays, Pandas DataFrames, vectorized operations, and advanced indexing.
