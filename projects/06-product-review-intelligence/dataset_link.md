# Dataset: Amazon Reviews 2023

## Source
- **Name**: Amazon Reviews 2023 (McAuley Lab)
- **Link**: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Size**: 34M+ reviews across multiple product categories

## Schema

| Column             | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `rating`           | float   | Star rating (1.0 to 5.0)                         |
| `title`            | string  | Review title/headline                             |
| `text`             | string  | Full review body text                             |
| `asin`             | string  | Amazon Standard Identification Number (product)   |
| `parent_asin`      | string  | Parent product ASIN (for product variants)        |
| `user_id`          | string  | Anonymized reviewer identifier                    |
| `timestamp`        | int     | Unix timestamp of the review                      |
| `helpful_vote`     | int     | Number of helpful votes received                  |
| `verified_purchase` | bool   | Whether the purchase was verified                 |

## Why This Dataset

1. **Scale**: 34M+ reviews provide a robust foundation for training sentiment classifiers.
2. **Real-world noise**: Reviews contain typos, slang, sarcasm, and mixed sentiments -- ideal for building resilient NLP models.
3. **Rich metadata**: Ratings, helpful votes, and verified purchase flags enable multi-signal feature engineering.
4. **Category segmentation**: Reviews are grouped by product category (Electronics, Books, etc.), allowing domain-specific analysis.
5. **Aspect diversity**: Product reviews naturally mention specific aspects (battery, screen, price, shipping) making them perfect for aspect-based sentiment analysis.

## Preprocessing Steps

1. **Subset selection**: Load the Electronics category subset to keep the project manageable (~5M reviews), then sample 100K reviews.
2. **Null handling**: Drop rows where `text` is null or empty. Fill missing `title` with empty string.
3. **Deduplication**: Remove exact duplicate reviews (same user_id + text).
4. **Sentiment labeling**: Map star ratings to sentiment classes:
   - 1-2 stars -> `negative`
   - 3 stars -> `neutral`
   - 4-5 stars -> `positive`
5. **Text cleaning**: Lowercase, remove HTML tags, normalize whitespace, strip excessive punctuation.
6. **Feature engineering**:
   - `review_length`: character count of review text
   - `word_count`: number of words
   - `helpful_ratio`: helpful_vote normalized across dataset
   - `is_verified`: binary flag from verified_purchase
7. **TF-IDF vectorization**: Fit on cleaned review text with max 10,000 features, bigrams included.
8. **Train/test split**: 80/20 stratified split preserving sentiment class distribution.
