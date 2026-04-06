# Day 20: PROJECT 1 — Sales Forecasting ML System

## Learning Objectives

- Build a complete end-to-end ML pipeline from data ingestion to deployment
- Work with the Walmart Store Sales dataset (421K+ rows, 16 features)
- Implement modular project structure with logging and exception handling
- Train and compare XGBoost, LightGBM, and Random Forest models
- Deploy the model via FastAPI REST API

## Project Overview

This is your first complete production-grade ML project. You will build a sales forecasting system using the **Walmart Store Sales** dataset from Kaggle. The system predicts weekly sales for each store-department combination based on features like temperature, fuel prices, CPI, unemployment, markdowns, and holiday indicators.

The project follows the modular structure introduced on Day 10: separate components for data ingestion, validation, transformation, model training, and evaluation. Each component has its own class with logging and error handling. The training and prediction pipelines orchestrate these components.

The end result is a trained model served via a FastAPI REST API that accepts store and feature data and returns sales predictions.

## Key Steps

1. **Data Ingestion**: Download the Walmart Sales dataset, perform chronological train/test split
2. **Data Validation**: Check schema, data types, missing values, value ranges
3. **Data Transformation**: Extract date features (month, week, day of week), scale numericals, encode categoricals
4. **Model Training**: Train XGBoost, LightGBM, Random Forest; select best by RMSE
5. **Model Evaluation**: Compute RMSE, MAE, R2, MAPE on test set
6. **Deployment**: Serve predictions via FastAPI with `/predict` endpoint

## Getting Started

```bash
cd projects/01-sales-forecasting
pip install -r requirements.txt

# Download dataset from Kaggle first, then:
python train.py --data-path data/walmart_sales.csv

# Start API
uvicorn app:app --reload
```

## Project Link

Full project code: [projects/01-sales-forecasting/](../../projects/01-sales-forecasting/)

## Resources

- [Walmart Sales Dataset on Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

## Next Day Preview

Tomorrow we learn **ML Pipeline Design** with sklearn Pipeline and ColumnTransformer for cleaner, more maintainable preprocessing code.
