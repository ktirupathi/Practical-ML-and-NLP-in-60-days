# Walmart Store Sales Forecasting Dataset

## Dataset Information
- **Name:** Walmart Sales
- **Source:** [Kaggle - Walmart Sales](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
- **Size:** 421,000+ rows, 16 features
- **Format:** CSV

## Schema

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store number (1-45) |
| Dept | int | Department number |
| Date | string | Week of sales (MM/DD/YYYY) |
| Weekly_Sales | float | Sales for the given store/department/week |
| IsHoliday | bool | Whether the week includes a holiday |
| Type | string | Store type (A, B, C) |
| Size | int | Store size in square feet |
| Temperature | float | Average regional temperature (Fahrenheit) |
| Fuel_Price | float | Regional fuel price (USD) |
| MarkDown1 | float | Anonymized promotional markdown 1 |
| MarkDown2 | float | Anonymized promotional markdown 2 |
| MarkDown3 | float | Anonymized promotional markdown 3 |
| MarkDown4 | float | Anonymized promotional markdown 4 |
| MarkDown5 | float | Anonymized promotional markdown 5 |
| CPI | float | Consumer Price Index |
| Unemployment | float | Regional unemployment rate |

## Preprocessing Steps

1. **Date Feature Extraction:** Parse the Date column to extract Year, Month, Week of Year, Day of Week, and Quarter.
2. **Missing Value Imputation:** MarkDown columns have significant missing values; impute with median. Other numerical columns imputed with median.
3. **Categorical Encoding:** Ordinal-encode `Type` (A=0, B=1, C=2) and binary-encode `IsHoliday`.
4. **Numerical Scaling:** StandardScaler applied to continuous features (Temperature, Fuel_Price, CPI, Unemployment, Size, MarkDown1-5).
5. **Outlier Handling:** Weekly_Sales outliers clipped at 1st and 99th percentiles.
6. **Feature Engineering:** Store-Department interaction feature, holiday proximity indicator, rolling sales statistics (if historical data available).
7. **Train/Test Split:** 80/20 split stratified by Store to maintain store distribution in both sets.
