# Day 1: Environment Setup

> **Phase:** Foundations | **Week:** 1 | **Estimated Time:** 3-4 hours

## What You'll Learn Today
- Install and configure Anaconda/Miniconda for Python environment management
- Create isolated virtual environments using both conda and venv
- Manage packages with pip and conda, understanding dependency resolution
- Launch and navigate Jupyter Lab and Jupyter Notebook
- Configure VS Code for ML development with essential extensions
- Understand project directory conventions for ML projects
- Verify your entire stack with an end-to-end smoke test

---

## 1. What Is Environment Setup?

A reproducible development environment is the bedrock of any professional ML project. "It works on my machine" is not an acceptable answer in a team or production setting. Environment setup means creating an isolated, version-controlled space where your Python interpreter, libraries, and system dependencies are pinned and shareable.

Python's packaging ecosystem has evolved dramatically. Today the two dominant approaches are **conda** (from Anaconda/Miniconda) and **venv + pip** (Python's built-in). Conda manages both Python packages and non-Python system-level dependencies (like MKL for NumPy), while venv is lighter and ships with Python itself. In ML, conda is especially useful because libraries like TensorFlow, PyTorch, and scikit-learn have compiled C/CUDA extensions that benefit from conda's binary package management.

Jupyter Notebook (and its successor Jupyter Lab) is the standard interactive computing environment for ML prototyping. It combines live code, equations, visualizations, and narrative text in a single document — making it ideal for exploratory data analysis and communicating results. Understanding how Jupyter connects to a specific conda environment (kernel) is essential to avoid the classic "I installed the library but Jupyter can't find it" problem.

---

## 2. Why Is It Used?

**Reproducibility:** Data science results must be reproducible. An environment file (`environment.yml` or `requirements.txt`) pins exact versions so a colleague — or your future self — can recreate the exact stack months later.

**Isolation:** Different projects may require incompatible library versions. scikit-learn 0.24 and 1.3 have breaking API changes. Environments prevent version conflicts from cascading across projects.

**Production parity:** When you deploy a model, the production Docker container will use the same `requirements.txt` you developed with. Clean, explicit environments make deployment less surprising.

**Collaboration:** Teams use environment files in version control (git) so every member works with identical dependencies, eliminating "works for me" debugging sessions.

---

## 3. Real-World Example

A data scientist at a fintech company is building a credit-scoring model. She works on three projects simultaneously: an old fraud-detection pipeline (Python 3.8, scikit-learn 0.24), a new NLP classifier (Python 3.10, transformers 4.35), and a time-series forecasting model (Python 3.9, Prophet 1.1). Without isolated environments, installing one project's dependencies would break another's. With conda environments named `fraud-v1`, `nlp-classifier`, and `forecasting`, she switches contexts with a single command and each project runs independently without conflicts.

---

## 4. Intuition

Think of a conda environment like a shipping container. Each container has its own OS-level tools, libraries, and runtime. Ships can carry many containers, and they never interfere with each other. Your laptop is the ship; each conda environment is a sealed container. When you `conda activate my-env`, you step inside that container. When you deactivate, you step out. Nothing inside one container leaks into another.

---

## 5. Mathematical Intuition

Environment setup is more engineering than math, but version resolution is a **constraint satisfaction problem**:

```
Given:
  Package A requires: numpy >= 1.20, < 2.0
  Package B requires: numpy >= 1.18
  Package C requires: numpy == 1.21.0

Find: numpy version X such that:
  X >= 1.20 AND X < 2.0    (constraint from A)
  X >= 1.18                 (constraint from B)
  X == 1.21.0               (constraint from C)

Solution: X = 1.21.0  (satisfies all three constraints)

If Package C required numpy == 1.19.0:
  1.19.0 >= 1.20 → FALSE  → UNSATISFIABLE

pip/conda raises a ResolutionError or installs a conflicting version.
This is why "dependency hell" happens — the intersection of all constraints is empty.
```

---

## 6. Worked Example

Setting up a complete ML environment step by step:

```
Step 1: Download Miniconda
  https://docs.conda.io/en/latest/miniconda.html
  Choose Python 3.11, 64-bit for your OS.

Step 2: Verify base install
  $ conda --version
  conda 23.11.0

Step 3: Create environment
  $ conda create -n ml60days python=3.11 -y
  Creates: ~/.conda/envs/ml60days/

Step 4: Activate
  $ conda activate ml60days
  (ml60days) $     ← prefix shows active environment

Step 5: Install core ML stack
  $ pip install numpy pandas scikit-learn matplotlib seaborn \
                scipy jupyterlab ipykernel

Step 6: Register as Jupyter kernel
  $ python -m ipykernel install --user \
      --name ml60days \
      --display-name "Python (ML 60 Days)"

Step 7: Launch Jupyter Lab
  $ jupyter lab

Step 8: Export for sharing
  $ pip freeze > requirements.txt
  # OR
  $ conda env export > environment.yml
```

---

## 7. Python Implementation

```python
# smoke_test.py  —  verify the entire ML stack
# Run with:  python smoke_test.py

import sys
import importlib

print("=" * 55)
print("  ML Environment Smoke Test")
print("=" * 55)

# ── 1. Python version ────────────────────────────────────────
print(f"\nPython: {sys.version}")
assert sys.version_info >= (3, 9), "Need Python 3.9+"

# ── 2. Library checks ────────────────────────────────────────
required = {
    "numpy":      "1.24.0",
    "pandas":     "1.5.0",
    "sklearn":    "1.2.0",
    "matplotlib": "3.6.0",
    "seaborn":    "0.12.0",
    "scipy":      "1.10.0",
    "IPython":    None,
}

all_ok = True
print("\nPackage versions:")
for lib, min_ver in required.items():
    try:
        mod = importlib.import_module(lib)
        ver = getattr(mod, "__version__", "unknown")
        if min_ver and ver != "unknown":
            from packaging.version import Version
            ok = Version(ver) >= Version(min_ver)
            status = "OK  " if ok else f"WARN (need >={min_ver})"
            if not ok:
                all_ok = False
        else:
            status = "OK  "
        print(f"  [{status}] {lib:<15} {ver}")
    except ImportError:
        print(f"  [FAIL] {lib:<15} NOT INSTALLED")
        all_ok = False

# ── 3. NumPy sanity ──────────────────────────────────────────
import numpy as np
arr = np.arange(1, 6, dtype=float)
assert arr.mean() == 3.0
assert arr.std().round(4) == 1.4142
print("\nNumPy  : PASSED  (mean, std correct)")

# ── 4. Pandas sanity ─────────────────────────────────────────
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
assert df["b"].sum() == 60
print("Pandas : PASSED  (DataFrame sum correct)")

# ── 5. scikit-learn sanity ───────────────────────────────────
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
lr = LinearRegression().fit(X, y)
assert abs(lr.predict([[6]])[0] - 12.0) < 0.001
print("sklearn: PASSED  (LinearRegression prediction correct)")

# ── 6. Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
if all_ok:
    print("  All checks PASSED. Environment is ready!")
else:
    print("  Some checks FAILED.")
    print("  Run: pip install -r requirements.txt")
print("=" * 55)
```

---

## 8. Visualization

```python
# env_dashboard.py  —  visual summary of your environment
import importlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

packages = [
    ("numpy",      "NumPy"),
    ("pandas",     "Pandas"),
    ("sklearn",    "scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("seaborn",    "Seaborn"),
    ("scipy",      "SciPy"),
    ("xgboost",    "XGBoost"),
    ("lightgbm",   "LightGBM"),
    ("IPython",    "IPython"),
]

labels, versions, colors = [], [], []
for mod_name, display in packages:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "installed")
        labels.append(display)
        versions.append(ver)
        colors.append("#27ae60")
    except ImportError:
        labels.append(display)
        versions.append("NOT INSTALLED")
        colors.append("#e74c3c")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, [1] * len(labels), color=colors, height=0.6, edgecolor="white")

for bar, ver in zip(bars, versions):
    ax.text(0.5, bar.get_y() + bar.get_height() / 2,
            ver, va="center", ha="center",
            fontsize=10, fontweight="bold", color="white")

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("ML Environment Dashboard", fontsize=14, fontweight="bold", pad=15)

ok_patch   = mpatches.Patch(color="#27ae60", label="Installed")
fail_patch = mpatches.Patch(color="#e74c3c", label="Missing")
ax.legend(handles=[ok_patch, fail_patch], loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("env_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: env_dashboard.png")
```

---

## 9. Common Mistakes

- **Forgetting to activate the environment**: Running `pip install` without activating installs into base (or the wrong env). Always confirm the `(env-name)` prefix is visible in your terminal.
- **Mixing conda and pip carelessly**: conda and pip resolve dependencies independently. Install as much as possible from one source; prefer `conda install` first, then `pip` for anything conda lacks. Never run `conda install` after `pip` for the same package.
- **Not registering the Jupyter kernel**: Installing packages in a conda env but skipping `python -m ipykernel install` means Jupyter uses the wrong Python and can't find your packages.
- **Committing the env directory to git**: The `env/` or `venv/` folder contains gigabytes of compiled binaries. Add it to `.gitignore`. Only commit `requirements.txt` or `environment.yml`.
- **Using `pip freeze` in a polluted environment**: `pip freeze` dumps every transitive dependency. Use `pip-tools` (`pip-compile`) or manually maintain `requirements.in` for clean top-level dependency tracking.
- **Not pinning the Python version**: Sharing `environment.yml` without `python=3.11` means conda may choose a different Python on a different OS, causing subtle breakage.
- **Never testing the environment on a clean machine**: Always validate your environment file with Docker or a fresh VM before sharing with a team.

---

## 10. Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | What is the difference between conda and pip? | conda manages both Python and non-Python (C, CUDA) dependencies with pre-built binaries; pip manages Python packages only and may require local compilation |
| 2 | Why use virtual environments? | Isolation prevents version conflicts across projects; reproducibility lets teammates recreate the exact stack |
| 3 | How do you share your environment? | `pip freeze > requirements.txt` or `conda env export > environment.yml`; commit that file to git |
| 4 | What is a Jupyter kernel? | The process that executes code in a notebook; each env can register as a named kernel so Jupyter can find the right Python |
| 5 | How do you register a conda env as a Jupyter kernel? | `python -m ipykernel install --user --name <env> --display-name "Name"` |
| 6 | Difference between `requirements.txt` and `environment.yml`? | `requirements.txt` is pip-specific; `environment.yml` is conda-specific and can pin Python version, channels, and pip packages together |
| 7 | How do you check the active environment? | `conda info --envs` (star marks active), or check terminal prefix, or `import sys; sys.executable` |
| 8 | What is `pip-compile` and why use it? | From pip-tools; compiles high-level `requirements.in` into a fully pinned `requirements.txt` — separates direct from transitive deps |
| 9 | How do you delete a conda environment? | `conda env remove -n <env_name>` |
| 10 | What belongs in `.gitignore` for Python projects? | `env/`, `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.env` (secrets) |

---

## Exercises

1. Create a conda environment named `day01-test` with Python 3.11. Install `numpy`, `pandas`, `matplotlib`, register it as a Jupyter kernel, launch Jupyter Lab, and verify `np.__version__` prints correctly from within a notebook.
2. Write a complete `environment.yml` for the `ml60days` environment. Delete the environment, recreate it from the YAML using `conda env create -f environment.yml`, and run the smoke test to confirm it is identical.
3. Create a `requirements.in` file with only top-level dependencies. Install `pip-tools` and run `pip-compile requirements.in` to generate a pinned `requirements.txt`. Compare the line count with `pip freeze` output and explain the difference.

---

## Key Takeaways
- Isolated environments prevent dependency conflicts and make experiments reproducible across machines
- Conda is preferred for ML because it handles binary (C/CUDA) deps; pip is universal but Python-only
- Always register conda environments as Jupyter kernels; otherwise Jupyter uses the wrong Python
- Commit `requirements.txt` or `environment.yml` to git — never the `env/` directory itself
- A smoke test script that validates library versions saves hours of "it worked yesterday" debugging

---

## What's Next

**Day 2** dives into Python for ML — specifically the NumPy and Pandas APIs that underpin every ML library. You will learn vectorized array operations, broadcasting, DataFrame manipulation, and why Python loops are 100x slower than NumPy. Bring your Day 1 environment because you will be running a lot of code.
