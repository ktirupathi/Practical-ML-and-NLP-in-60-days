# Day 39: PROJECT 6 — Product Review Intelligence Engine

## Learning Objectives

- Build an aspect-based sentiment analysis system from real product reviews
- Work with the Amazon Product Reviews 2023 dataset (34M+ reviews)
- Extract product aspects and analyze sentiment per aspect
- Combine TF-IDF text features with numerical features for classification
- Deploy with both FastAPI and Streamlit interfaces

## Project Overview

The Product Review Intelligence Engine analyzes customer reviews to extract actionable insights. Given a product review, the system determines overall sentiment (positive/neutral/negative) and identifies specific product aspects mentioned (e.g., battery, display, price, shipping) along with the sentiment toward each aspect.

This project uses the **Amazon Product Reviews 2023** dataset from HuggingFace, focusing on the Electronics category. We sample 100K reviews for training. The dataset contains real reviews with ratings, text, timestamps, helpful votes, and verified purchase flags.

The model combines TF-IDF features from review text with engineered numerical features (review length, word count, helpful vote ratio, verified purchase flag) to predict sentiment. Aspect extraction uses keyword matching with context-window sentiment scoring.

## Key Steps

1. **Data Ingestion**: Load Electronics subset from HuggingFace, sample 100K reviews
2. **Data Validation**: Check rating distribution, null values, text quality
3. **Data Transformation**: Map ratings to 3 sentiment classes, compute TF-IDF + numerical features
4. **Model Training**: Train LightGBM, XGBoost, LogisticRegression; select best by weighted F1
5. **Aspect Extraction**: Keyword-based aspect detection with local sentiment scoring
6. **Deployment**: FastAPI API + Streamlit dashboard

## Getting Started

```bash
cd projects/06-product-review-intelligence
pip install -r requirements.txt
python train.py
uvicorn app:app --reload

# Or launch the dashboard
streamlit run streamlit_app.py
```

## Project Link

Full project code: [projects/06-product-review-intelligence/](../../projects/06-product-review-intelligence/)

## Resources

- [Amazon Reviews 2023 on HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [Aspect-Based Sentiment Analysis Survey](https://arxiv.org/abs/2203.01054)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## Next Day Preview

Tomorrow we cover **NLP Evaluation** — learning BLEU, ROUGE, F1, and other metrics essential for evaluating NLP systems.
