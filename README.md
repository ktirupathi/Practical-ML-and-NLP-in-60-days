<div align="center">

# 🚀 60 Days of End-to-End Machine Learning & NLP

### *From Zero to Production-Ready ML/NLP Engineer in 60 Days*

[![GitHub Stars](https://img.shields.io/github/stars/ktirupathi/practical-ml-and-nlp-in-60-days?style=for-the-badge&logo=github&color=yellow)](https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ktirupathi/practical-ml-and-nlp-in-60-days?style=for-the-badge&logo=github&color=blue)](https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days/network/members)
[![License](https://img.shields.io/github/license/ktirupathi/practical-ml-and-nlp-in-60-days?style=for-the-badge&color=green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co)

---

**10 production-grade ML & NLP projects** | **Real-world datasets** | **End-to-end pipelines** | **FastAPI deployments**

*No Titanic. No Iris. No toy datasets. Only real-world, portfolio-ready projects.*

[Get Started](#-quick-start) · [60-Day Roadmap](#-60-day-roadmap) · [Projects](#-project-showcase) · [Datasets](#-dataset-catalog) · [Contributing](#-contributing)

---

### If this repo helps you, please give it a star — it helps others find it too!

</div>

---

## Table of Contents

- [Who Is This For](#-who-is-this-for)
- [What Makes This Different](#-what-makes-this-different)
- [Learning Outcomes](#-learning-outcomes)
- [Quick Start](#-quick-start)
- [60-Day Roadmap](#-60-day-roadmap)
- [Project Showcase](#-project-showcase)
- [Dataset Catalog](#-dataset-catalog)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Progress Tracker](#-progress-tracker)
- [Deployment Guide](#-deployment-guide)
- [Contributing](#-contributing)
- [License](#-license)

---

## Who Is This For

| Level | Description |
|-------|-------------|
| **Career Switchers** | Professionals transitioning into ML/NLP with a need for portfolio projects |
| **Junior ML Engineers** | Engineers wanting to level up from tutorials to production-grade systems |
| **Data Scientists** | DS professionals who want to build end-to-end deployment pipelines |
| **Backend Developers** | Developers adding ML/NLP capabilities to their skillset |
| **Graduate Students** | MS/PhD students bridging the gap between research and industry |

---

## What Makes This Different

| Feature | Typical ML Repos | This Repo |
|---------|------------------|-----------|
| **Datasets** | Titanic, Iris, MNIST | Real-world Kaggle/HuggingFace datasets (5,000+ rows each) |
| **Pipeline** | Jupyter notebook only | Full production pipeline with logging, validation, API |
| **Deployment** | None | FastAPI + Streamlit + Docker |
| **Code Quality** | Scripts | Modular OOP with exception handling |
| **Documentation** | Minimal | Complete README per project with architecture diagrams |
| **Scope** | Single model training | Data ingestion to validation to training to evaluation to deployment |

---

## Learning Outcomes

After completing this 60-day program, you will be able to:

- Build **end-to-end ML pipelines** from data ingestion to production deployment
- Design **NLP systems** using transformers, embeddings, and RAG architectures
- Deploy models via **FastAPI REST APIs** with proper error handling
- Build interactive **Streamlit dashboards** for model demos
- Containerize ML applications with **Docker**
- Implement **MLOps best practices**: logging, config management, experiment tracking
- Work with **real-world messy data**: missing values, class imbalance, mixed types
- Create a **portfolio of 10 production-grade projects** for job applications

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days.git
cd practical-ml-and-nlp-in-60-days

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Start with Day 1
cd roadmap/day-01-environment-setup
```

> **Start Here:** If you are new, begin with the [60-Day Roadmap](#-60-day-roadmap) and follow it sequentially. Each day builds on the previous one.

---

## 60-Day Roadmap

### Phase 1: Foundation (Days 1-10)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 1 | Environment Setup | Python, venv, Git, VS Code | Configured dev environment |
| 2 | Python for ML | NumPy, Pandas deep dive | Data manipulation exercises |
| 3 | Statistics for ML | Distributions, hypothesis testing | Statistical analysis notebook |
| 4 | Linear Algebra Essentials | Vectors, matrices, eigenvalues | LA for ML notebook |
| 5 | Data Visualization | Matplotlib, Seaborn, Plotly | EDA visualization toolkit |
| 6 | Data Preprocessing | Missing values, encoding, scaling | Preprocessing pipeline |
| 7 | Feature Engineering | Feature creation, selection, importance | Feature engineering toolkit |
| 8 | EDA Masterclass | Automated EDA, profiling | Full EDA on real dataset |
| 9 | Data Validation | Great Expectations, schema validation | Data validation pipeline |
| 10 | Project Structure and Logging | Modular code, logging, config | ML project template |

### Phase 2: Machine Learning (Days 11-20)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 11 | Linear and Logistic Regression | Regularization, interpretation | Regression pipeline |
| 12 | Decision Trees and Random Forests | Bagging, feature importance | Tree-based pipeline |
| 13 | Gradient Boosting | XGBoost, LightGBM, CatBoost | Boosting comparison notebook |
| 14 | SVM and KNN | Kernel methods, distance metrics | Classification pipeline |
| 15 | Clustering | K-Means, DBSCAN, hierarchical | Customer segmentation |
| 16 | Dimensionality Reduction | PCA, t-SNE, UMAP | Visualization pipeline |
| 17 | Model Evaluation | Cross-val, metrics, calibration | Evaluation framework |
| 18 | Hyperparameter Tuning | Optuna, GridSearch, Bayesian | Tuning pipeline |
| 19 | Imbalanced Learning | SMOTE, class weights, threshold | Imbalanced data pipeline |
| 20 | **PROJECT 1: Sales Forecasting** | End-to-end ML pipeline | Deployed forecasting system |

### Phase 3: Advanced ML Pipelines (Days 21-30)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 21 | ML Pipeline Design | sklearn Pipeline, ColumnTransformer | Modular pipeline |
| 22 | Experiment Tracking | MLflow, logging experiments | Experiment tracker |
| 23 | Data Versioning | DVC, data pipelines | Versioned data pipeline |
| 24 | Model Serialization | Pickle, joblib, ONNX | Model export pipeline |
| 25 | FastAPI for ML | REST API, request validation | Model serving API |
| 26 | Streamlit Dashboards | Interactive ML apps | Dashboard prototype |
| 27 | Docker for ML | Containerization, docker-compose | Dockerized ML app |
| 28 | **PROJECT 2: AI Resume Screening** | Classification + deployment | Deployed screening system |
| 29 | **PROJECT 3: Customer Support Ticket Router** | Multi-class + API | Deployed ticket router |
| 30 | Testing and CI/CD | pytest, GitHub Actions | Tested ML pipeline |

### Phase 4: NLP Foundations (Days 31-40)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 31 | Text Preprocessing | Tokenization, lemmatization, regex | NLP preprocessing toolkit |
| 32 | Text Representation | BoW, TF-IDF, n-grams | Text vectorization pipeline |
| 33 | Text Classification | Naive Bayes, SVM for text | Text classifier |
| 34 | Named Entity Recognition | spaCy NER, custom entities | NER pipeline |
| 35 | Topic Modeling | LDA, BERTopic | Topic analysis notebook |
| 36 | Text Summarization | Extractive and abstractive | Summarization pipeline |
| 37 | **PROJECT 4: Email Intent Detection** | Intent classification + API | Deployed intent detector |
| 38 | **PROJECT 5: Multi-label Document Classifier** | Multi-label + threshold tuning | Deployed classifier |
| 39 | **PROJECT 6: Product Review Intelligence** | Aspect-based analysis | Review analysis engine |
| 40 | NLP Evaluation | BLEU, ROUGE, F1 for NLP | Evaluation toolkit |

### Phase 5: Transformers and Embeddings (Days 41-50)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 41 | Word Embeddings | Word2Vec, GloVe, FastText | Embedding exploration |
| 42 | Transformer Architecture | Attention, positional encoding | Transformer deep dive |
| 43 | BERT and Fine-tuning | HuggingFace, transfer learning | Fine-tuned BERT model |
| 44 | Sentence Embeddings | Sentence-BERT, similarity | Embedding pipeline |
| 45 | Vector Databases | FAISS, ChromaDB | Vector search system |
| 46 | RAG Architecture | Retrieval-augmented generation | RAG prototype |
| 47 | **PROJECT 7: Financial News Risk Analyzer** | Transformers + risk scoring | Deployed risk analyzer |
| 48 | **PROJECT 8: Semantic Search Engine** | Embeddings + vector DB | Deployed search engine |
| 49 | **PROJECT 9: Knowledge Base Chatbot (RAG)** | RAG + LLM integration | Deployed RAG chatbot |
| 50 | Advanced Fine-tuning | LoRA, QLoRA, PEFT | Fine-tuning notebook |

### Phase 6: Capstone and Production (Days 51-60)

| Day | Topic | Key Concepts | Deliverable |
|-----|-------|-------------|-------------|
| 51 | System Design for ML | Architecture, scalability | Design documents |
| 52 | Model Monitoring | Drift detection, alerts | Monitoring pipeline |
| 53 | A/B Testing for ML | Statistical testing, rollout | A/B test framework |
| 54 | **PROJECT 10: Enterprise Document Classification** | Full enterprise pipeline | Deployed system |
| 55 | Performance Optimization | Model compression, caching | Optimized models |
| 56 | Security for ML APIs | Auth, rate limiting, validation | Secured API |
| 57 | Portfolio Preparation | Documentation, demos | Portfolio-ready projects |
| 58 | Interview Prep: ML System Design | Common ML design problems | Design solutions |
| 59 | Interview Prep: ML Coding | ML coding challenges | Coding solutions |
| 60 | Graduation and Next Steps | Career roadmap | Complete portfolio |

---

## Project Showcase

### Overview

| # | Project | Domain | ML Type | Dataset | Deployment |
|---|---------|--------|---------|---------|------------|
| 1 | [Sales Forecasting ML System](projects/01-sales-forecasting/) | Retail | Regression / Time Series | Walmart Sales (421K rows) | FastAPI |
| 2 | [AI Resume Screening System](projects/02-ai-resume-screening/) | HR Tech | Multi-class Classification | Resume Dataset (2,484 resumes) | FastAPI + Streamlit |
| 3 | [Customer Support Ticket Router](projects/03-support-ticket-router/) | Customer Service | Multi-class NLP | Customer Support Tickets (100K+) | FastAPI |
| 4 | [Email Intent Detection](projects/04-email-intent-detection/) | Enterprise | Intent Classification | Enron Email (500K+ emails) | FastAPI |
| 5 | [Multi-label Document Classifier](projects/05-multilabel-document-classifier/) | Legal / Enterprise | Multi-label Classification | EUR-Lex (57K documents) | FastAPI |
| 6 | [Product Review Intelligence](projects/06-product-review-intelligence/) | E-Commerce | Aspect-based Sentiment | Amazon Reviews (34M+ reviews) | FastAPI + Streamlit |
| 7 | [Financial News Risk Analyzer](projects/07-financial-news-risk-analyzer/) | Finance | Sequence Classification | Financial PhraseBank + Reuters | FastAPI |
| 8 | [Semantic Search Engine](projects/08-semantic-search-engine/) | Information Retrieval | Embedding + Vector Search | MS MARCO (8.8M passages) | FastAPI + Streamlit |
| 9 | [Knowledge Base Chatbot (RAG)](projects/09-knowledge-base-chatbot-rag/) | Enterprise AI | RAG + LLM | SQuAD 2.0 + Wikipedia (150K+ QA pairs) | FastAPI + Streamlit |
| 10 | [Enterprise Document Classification](projects/10-enterprise-document-classification/) | Enterprise | Hierarchical Classification | RVL-CDIP (400K documents) | FastAPI + Docker |

---

## Dataset Catalog

All datasets are **real, publicly available, and contain 5,000+ rows**.

| # | Dataset | Source | Rows | Features | Download |
|---|---------|--------|------|----------|----------|
| 1 | Walmart Store Sales | Kaggle | 421,570 | 16 | [Link](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) |
| 2 | Resume Dataset | Kaggle | 2,484 resumes | 4 categories | [Link](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) |
| 3 | Bitext Customer Support | HuggingFace | 100,000+ | 8 | [Link](https://huggingface.co/datasets/Bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| 4 | Enron Email Dataset | CMU | 500,000+ | 6 | [Link](https://www.cs.cmu.edu/~enron/) |
| 5 | EUR-Lex (EURLEX57K) | Research | 57,000 | Multi-label | [Link](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/) |
| 6 | Amazon Product Reviews | HuggingFace | 34M+ | 9 | [Link](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| 7 | Financial PhraseBank | HuggingFace | 4,846 | 2 | [Link](https://huggingface.co/datasets/financial_phrasebank) |
| 8 | MS MARCO | Microsoft | 8.8M passages | 4 | [Link](https://microsoft.github.io/msmarco/) |
| 9 | SQuAD 2.0 | Stanford | 150,000+ | 5 | [Link](https://rajpurkar.github.io/SQuAD-explorer/) |
| 10 | RVL-CDIP | HuggingFace | 400,000 | 16 classes | [Link](https://huggingface.co/datasets/rvl_cdip) |

---

## Repository Structure

```
practical-ml-and-nlp-in-60-days/
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── PROGRESS_TRACKER.md
│
├── roadmap/
│   ├── day-01-environment-setup/
│   ├── day-02-python-for-ml/
│   ├── ...
│   └── day-60-graduation/
│
├── projects/
│   ├── 01-sales-forecasting/
│   │   ├── README.md
│   │   ├── dataset_link.md
│   │   ├── notebooks/
│   │   │   └── eda.ipynb
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── components/
│   │   │   │   ├── data_ingestion.py
│   │   │   │   ├── data_validation.py
│   │   │   │   ├── data_transformation.py
│   │   │   │   ├── model_trainer.py
│   │   │   │   └── model_evaluation.py
│   │   │   ├── pipeline/
│   │   │   │   ├── training_pipeline.py
│   │   │   │   └── prediction_pipeline.py
│   │   │   ├── utils/
│   │   │   │   └── common.py
│   │   │   └── config/
│   │   │       └── configuration.py
│   │   ├── app.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── requirements.txt
│   │   └── logs/
│   │       └── .gitkeep
│   │
│   ├── 02-ai-resume-screening/
│   ├── 03-support-ticket-router/
│   ├── ...
│   └── 10-enterprise-document-classification/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── project_idea.md
│   └── PULL_REQUEST_TEMPLATE.md
│
└── assets/
    └── images/
        └── .gitkeep
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.9+ |
| **ML Frameworks** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch, TensorFlow |
| **NLP** | HuggingFace Transformers, spaCy, NLTK, Sentence-Transformers |
| **Vector Databases** | FAISS, ChromaDB |
| **APIs** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Containers** | Docker, docker-compose |
| **Experiment Tracking** | MLflow |
| **Data Validation** | Great Expectations, Pydantic |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Testing** | pytest |
| **Version Control** | Git, DVC |

---

## Progress Tracker

Track your 60-day journey! See the full checklist in [`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md).

```
Phase 1: Foundation         [░░░░░░░░░░] 0/10
Phase 2: Machine Learning   [░░░░░░░░░░] 0/10
Phase 3: Advanced Pipelines [░░░░░░░░░░] 0/10
Phase 4: NLP Foundations    [░░░░░░░░░░] 0/10
Phase 5: Transformers       [░░░░░░░░░░] 0/10
Phase 6: Capstone           [░░░░░░░░░░] 0/10
```

---

## Deployment Guide

### FastAPI (All Projects)

```bash
cd projects/01-sales-forecasting
pip install -r requirements.txt
python train.py          # Train the model
uvicorn app:app --reload # Start API server
```

### Streamlit (Projects 2, 6, 8, 9)

```bash
cd projects/02-ai-resume-screening
streamlit run streamlit_app.py
```

### Docker (Projects 8, 10)

```bash
cd projects/10-enterprise-document-classification
docker build -t doc-classifier .
docker run -p 8000:8000 doc-classifier
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Ways to contribute:**
- Add new projects with real datasets
- Improve existing project documentation
- Fix bugs or optimize code
- Add test coverage
- Improve deployment configurations

---

## Star This Repo

If you find this repository helpful:

1. **Star** this repo to bookmark it and help others discover it
2. **Fork** it to customize the learning path
3. **Share** it with your network

<div align="center">

### [Click here to Star this repository](https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days)

*Your star helps ML engineers worldwide find quality learning resources.*

</div>

---

## Share

If this repo helped you, share it!

[![Twitter](https://img.shields.io/badge/Share_on-Twitter-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%2060-day%20ML%20and%20NLP%20learning%20path%20with%2010%20production-grade%20projects!&url=https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days)
[![LinkedIn](https://img.shields.io/badge/Share_on-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with dedication for the ML community**

*Start your journey today. Star the repo. Build real projects. Land your dream ML role.*

</div>
