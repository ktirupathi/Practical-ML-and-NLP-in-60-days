# Dataset Catalog

All datasets used in this repository are **real, publicly available**, and relevant to production ML/NLP systems.

---

## Project Datasets

### 1. Walmart Store Sales

| Property | Value |
|----------|-------|
| **Source** | Kaggle |
| **Rows** | 421,570 |
| **Columns** | 16 |
| **Link** | [Download](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) |
| **Used In** | Project 1: Sales Forecasting |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store number (1-45) |
| Dept | int | Department number |
| Date | date | Week of sales |
| Weekly_Sales | float | Sales for the given store-dept-week |
| IsHoliday | bool | Whether the week includes a holiday |
| Temperature | float | Average temperature in the region |
| Fuel_Price | float | Cost of fuel in the region |
| MarkDown1-5 | float | Anonymized promotional markdowns |
| CPI | float | Consumer Price Index |
| Unemployment | float | Unemployment rate |
| Type | char | Store type (A, B, C) |
| Size | int | Store size in sq ft |

**Sample Rows:**

| Store | Dept | Date | Weekly_Sales | IsHoliday | Temperature | Type |
|-------|------|------|-------------|-----------|-------------|------|
| 1 | 1 | 2010-02-05 | 24924.50 | False | 42.31 | A |
| 1 | 1 | 2010-02-12 | 46039.49 | True | 38.51 | A |
| 1 | 2 | 2010-02-05 | 50605.27 | False | 42.31 | A |

**Preprocessing:** Handle missing MarkDown values (NaN before 2011), parse dates, create lag features, encode store type.

**Feature Engineering Ideas:** Rolling mean sales, holiday proximity features, department-level aggregations, seasonal decomposition, interaction features (Type x IsHoliday).

---

### 2. Resume Dataset

| Property | Value |
|----------|-------|
| **Source** | Kaggle |
| **Rows** | 2,484 resumes |
| **Columns** | 2 (Category, Resume) |
| **Link** | [Download](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) |
| **Used In** | Project 2: AI Resume Screening |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| Category | string | Job category (25 unique) |
| Resume | string | Full resume text (HTML content) |

**Categories include:** Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer, Sales, Health and Fitness, Civil Engineer, Java Developer, Business Analyst, SAP Developer, Automation Testing, Electrical Engineering, Operations Manager, Python Developer, DevOps Engineer, Network Security Engineer, PMO, Database, Hadoop, ETL Developer, DotNet Developer, Blockchain, Testing.

**Preprocessing:** Strip HTML tags, remove URLs and email addresses, remove special characters, lowercase, lemmatize. Handle class imbalance (some categories have <50 samples).

**Feature Engineering Ideas:** Resume length, keyword density per category, skills extraction, education level detection, experience year extraction.

---

### 3. Bitext Customer Support Dataset

| Property | Value |
|----------|-------|
| **Source** | HuggingFace |
| **Rows** | 100,000+ |
| **Columns** | 8 |
| **Link** | [Download](https://huggingface.co/datasets/Bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| **Used In** | Project 3: Support Ticket Router |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| instruction | string | Customer query text |
| intent | string | Intent label (27 unique) |
| category | string | Category label (11 unique) |
| response | string | Ideal agent response |

**Sample Rows:**

| instruction | intent | category |
|-------------|--------|----------|
| I need to cancel my subscription | cancel_order | ORDER |
| How do I track my delivery? | track_order | SHIPPING |
| My payment was charged twice | payment_issue | BILLING |

**Preprocessing:** Clean text, map intents to categories, stratified split by intent.

---

### 4. Enron Email Dataset

| Property | Value |
|----------|-------|
| **Source** | CMU |
| **Rows** | 500,000+ emails |
| **Columns** | 6 |
| **Link** | [Download](https://www.cs.cmu.edu/~enron/) |
| **Used In** | Project 4: Email Intent Detection |

**Schema:** Message-ID, Date, From, To, Subject, Body

**Preprocessing:** Parse email headers, remove quoted replies, strip signatures and disclaimers, remove forwarded headers.

---

### 5. EUR-Lex (EURLEX57K)

| Property | Value |
|----------|-------|
| **Source** | NLP Group, Athens |
| **Rows** | 57,000 documents |
| **Labels** | 4,271 EUROVOC concepts |
| **Link** | [Download](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/) |
| **Used In** | Project 5: Multi-label Document Classifier |

**Schema:** Document text (title + header + recitals), EUROVOC concept labels (multi-label)

**Preprocessing:** Concatenate text fields, filter labels by minimum frequency, create multi-label binary matrix.

---

### 6. Amazon Product Reviews 2023

| Property | Value |
|----------|-------|
| **Source** | HuggingFace |
| **Rows** | 34M+ (full), 100K sampled |
| **Columns** | 9 |
| **Link** | [Download](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| **Used In** | Project 6: Product Review Intelligence |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| rating | float | Star rating (1-5) |
| title | string | Review title |
| text | string | Review body |
| asin | string | Product ID |
| parent_asin | string | Parent product ID |
| user_id | string | Reviewer ID |
| timestamp | int | Unix timestamp |
| helpful_vote | int | Number of helpful votes |
| verified_purchase | bool | Verified purchase flag |

**Preprocessing:** Sample Electronics category, map ratings to sentiment (1-2=negative, 3=neutral, 4-5=positive), clean HTML, compute review length features.

---

### 7. Financial PhraseBank

| Property | Value |
|----------|-------|
| **Source** | HuggingFace |
| **Rows** | 4,846 sentences |
| **Columns** | 2 |
| **Link** | [Download](https://huggingface.co/datasets/financial_phrasebank) |
| **Used In** | Project 7: Financial News Risk Analyzer |

**Schema:** sentence (financial news text), label (positive/negative/neutral)

**Agreement levels:** sentences_allagree (all annotators agree), sentences_75agree, sentences_66agree, sentences_50agree. Higher agreement = cleaner labels.

**Preprocessing:** Select agreement level, stratified split, tokenize with FinBERT tokenizer.

---

### 8. MS MARCO

| Property | Value |
|----------|-------|
| **Source** | Microsoft Research |
| **Rows** | 8.8M passages, 1M queries |
| **Columns** | 4 |
| **Link** | [Download](https://microsoft.github.io/msmarco/) |
| **Used In** | Project 8: Semantic Search Engine |

**Schema:** query_id, query_text, passage_id, passage_text, relevance_label

**Preprocessing:** Sample subset for demo (50K-100K passages), deduplicate, validate text quality.

---

### 9. SQuAD 2.0

| Property | Value |
|----------|-------|
| **Source** | Stanford NLP |
| **Rows** | 150,000+ QA pairs |
| **Columns** | 5 |
| **Link** | [Download](https://rajpurkar.github.io/SQuAD-explorer/) |
| **Used In** | Project 9: Knowledge Base Chatbot (RAG) |

**Schema:** id, question, context (Wikipedia paragraph), answers (list of answer spans), is_impossible (bool for unanswerable questions)

**Preprocessing:** Extract unique context paragraphs, chunk for vector storage, generate embeddings.

---

### 10. RVL-CDIP

| Property | Value |
|----------|-------|
| **Source** | HuggingFace |
| **Rows** | 400,000 document images |
| **Classes** | 16 |
| **Link** | [Download](https://huggingface.co/datasets/rvl_cdip) |
| **Used In** | Project 10: Enterprise Document Classification |

**16 Classes:** letter, form, email, handwritten, advertisement, scientific_report, scientific_publication, specification, file_folder, news_article, budget, invoice, presentation, questionnaire, resume, memo.

**Preprocessing:** Resize images, convert grayscale to RGB, normalize pixel values, apply AutoImageProcessor.

---

## Assignment Datasets

### 11. California Housing (sklearn)

| Property | Value |
|----------|-------|
| **Rows** | 20,640 | **Columns** | 8 features + target |
| **Access** | `from sklearn.datasets import fetch_california_housing` |
| **Used In** | Week 1 Assignment, Week 2 Assignment |

### 12. Bank Marketing (UCI)

| Property | Value |
|----------|-------|
| **Rows** | 45,211 | **Columns** | 17 |
| **Link** | [Download](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| **Used In** | Week 2 Assignment |

### 13. Adult Census Income (UCI)

| Property | Value |
|----------|-------|
| **Rows** | 48,842 | **Columns** | 14 |
| **Link** | [Download](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Used In** | Week 2 Assignment |

### 14. Credit Card Fraud (Kaggle)

| Property | Value |
|----------|-------|
| **Rows** | 284,807 | **Columns** | 31 |
| **Link** | [Download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Used In** | Week 3 Assignment |

### 15. Telco Customer Churn (Kaggle)

| Property | Value |
|----------|-------|
| **Rows** | 7,043 | **Columns** | 21 |
| **Link** | [Download](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Used In** | Week 3 Assignment |

### 16. 20 Newsgroups (sklearn)

| Property | Value |
|----------|-------|
| **Rows** | 18,846 documents | **Categories** | 20 |
| **Access** | `from sklearn.datasets import fetch_20newsgroups` |
| **Used In** | Week 5 Assignment |

### 17. IMDB Reviews (HuggingFace)

| Property | Value |
|----------|-------|
| **Rows** | 50,000 reviews | **Classes** | 2 (pos/neg) |
| **Access** | `from datasets import load_dataset; ds = load_dataset("imdb")` |
| **Used In** | Week 6 Assignment |

### 18. BBC News (Kaggle)

| Property | Value |
|----------|-------|
| **Rows** | 2,225 articles | **Categories** | 5 |
| **Link** | [Download](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification) |
| **Used In** | Week 5 Assignment |

### 19. CoNLL-2003 NER (HuggingFace)

| Property | Value |
|----------|-------|
| **Rows** | 20,744 sentences | **Entity types** | PER, ORG, LOC, MISC |
| **Access** | `from datasets import load_dataset; ds = load_dataset("conll2003")` |
| **Used In** | Week 5 Assignment |

### 20. CNN/DailyMail Summarization (HuggingFace)

| Property | Value |
|----------|-------|
| **Rows** | 311,971 articles | **Columns** | article, highlights |
| **Access** | `from datasets import load_dataset; ds = load_dataset("cnn_dailymail", "3.0.0")` |
| **Used In** | Week 6 Assignment |
