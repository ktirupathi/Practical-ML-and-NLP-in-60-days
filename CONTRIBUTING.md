# Contributing to Practical ML and NLP in 60 Days

Thank you for your interest in contributing to this project! This guide will help you get started.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding New Projects](#adding-new-projects)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code of Conduct](#code-of-conduct)

## How to Contribute

There are many ways to contribute to this repository:

1. **Fix bugs** - Browse open issues labeled `bug` and submit a fix.
2. **Improve documentation** - Clarify explanations, fix typos, or add missing context to existing notebooks.
3. **Add new projects** - Propose and implement new day projects that align with the curriculum.
4. **Enhance existing projects** - Improve code quality, add visualizations, or extend analysis in current notebooks.
5. **Review pull requests** - Help review and test contributions from other community members.
6. **Share feedback** - Open an issue to suggest improvements to the learning path or project structure.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A virtual environment tool (venv, conda, or similar)

### Setup Steps

1. **Fork the repository** on GitHub.

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/Practical-ML-and-NLP-in-60-days.git
   cd Practical-ML-and-NLP-in-60-days
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install spaCy language model** (if working on NLP projects):
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

7. **Launch Jupyter** to work on notebooks:
   ```bash
   jupyter notebook
   ```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use meaningful variable and function names. Avoid single-letter names except for loop counters.
- Add type hints to all function signatures.
- Keep functions focused and under 50 lines where possible.
- Use f-strings for string formatting.

### Jupyter Notebooks

- Begin every notebook with a markdown cell containing a title, objective, and brief description.
- Organize notebooks into clear sections with markdown headers.
- Include comments explaining the "why" behind non-obvious steps.
- Restart the kernel and run all cells before committing to ensure the notebook runs end-to-end.
- Clear all cell outputs before committing unless the output is essential for understanding.
- Keep cell output concise. Avoid printing entire large DataFrames; use `.head()`, `.shape`, or `.describe()` instead.

### General

- Write docstrings for all public functions and classes using Google-style format.
- Keep imports organized: standard library first, then third-party, then local modules.
- Do not commit API keys, credentials, or secrets. Use environment variables instead.
- Remove debugging print statements before submitting.

## Adding New Projects

When proposing a new project for a specific day, the following requirements must be met:

### Dataset Requirements

- The project **must use a real-world dataset** with a minimum of **5,000 rows**.
- Datasets must come from reputable sources such as:
  - [Kaggle](https://www.kaggle.com/datasets)
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
  - [Hugging Face Datasets](https://huggingface.co/datasets)
  - [data.gov](https://data.gov)
  - [Google Dataset Search](https://datasetsearch.research.google.com/)
- Include the dataset source URL and license information in the notebook.
- Do not commit large dataset files to the repository. Provide download instructions or use a library to fetch the data programmatically.

### Project Structure

Each day's project should follow this structure:

```
Day_XX_Topic_Name/
    Day_XX_Topic_Name.ipynb    # Main notebook
    README.md                   # Brief description, dataset info, key learnings
    utils.py                    # Helper functions (if needed)
    requirements.txt            # Any additional dependencies beyond the base
```

### Content Requirements

- State the learning objective clearly at the top of the notebook.
- Include exploratory data analysis (EDA) with at least 3 visualizations.
- Explain each modeling decision with context.
- Evaluate results using appropriate metrics and discuss findings.
- Summarize key takeaways in a final section.

## Pull Request Process

1. **Ensure your branch is up to date** with the `main` branch:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run any tests** to make sure nothing is broken:
   ```bash
   pytest
   ```

3. **Push your branch** and open a pull request against `main`.

4. **Fill out the PR template** completely, including:
   - A description of what was changed and why.
   - The checklist of requirements.
   - Any relevant issue numbers.

5. **Wait for review**. A maintainer will review your PR. You may be asked to make changes. Please respond to feedback promptly.

6. **Merge**. Once approved, a maintainer will merge your PR. Do not merge your own pull requests.

### PR Requirements

- All notebooks must run from top to bottom without errors.
- No large files (over 10 MB) should be committed.
- All new dependencies must be added to `requirements.txt` with pinned or minimum versions.
- The PR description must clearly explain the purpose of the changes.

## Issue Reporting

### Bug Reports

When reporting a bug, please include:

- A clear and descriptive title.
- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- Your environment details (OS, Python version, relevant package versions).
- Any error messages or tracebacks.

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) when creating a new issue.

### Feature Requests

For feature requests, please include:

- A description of the problem or gap in the current curriculum.
- Your proposed solution.
- Why this addition would benefit learners.

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) when creating a new issue.

### Project Ideas

Have an idea for a new day project? Use the [Project Idea template](.github/ISSUE_TEMPLATE/project_idea.md) and include:

- The topic and which day/phase it fits into.
- A proposed dataset (must meet the 5,000+ row requirement).
- The dataset source and license.
- What learners will gain from the project.

## Code of Conduct

This project is committed to providing a welcoming, inclusive, and harassment-free experience for everyone. By participating, you agree to the following:

- **Be respectful.** Treat everyone with dignity. Disagreements are fine; personal attacks are not.
- **Be constructive.** Provide helpful feedback. When critiquing, suggest improvements rather than just pointing out flaws.
- **Be inclusive.** Use welcoming language. Be mindful of different experience levels, backgrounds, and perspectives.
- **No harassment.** Harassment of any kind, including but not limited to offensive comments, intimidation, or unwelcome attention, will not be tolerated.
- **Report issues.** If you experience or witness unacceptable behavior, please open an issue or contact a maintainer directly.

Violations of this code of conduct may result in removal from the project. Maintainers reserve the right to remove, edit, or reject contributions that do not align with these standards.

---

Thank you for helping make this project better for everyone learning ML and NLP!
