# Day 47: Project 7 -- Financial News Risk Analyzer

## Learning Objectives

- Design an end-to-end NLP pipeline that extracts risk signals from financial news articles
- Apply sentiment analysis and named entity recognition to financial text
- Build a risk scoring system that aggregates multiple NLP signals into a single metric
- Handle domain-specific challenges: financial jargon, negation, and temporal context
- Structure a project with clear separation between data ingestion, processing, and presentation

## Key Concepts

The Financial News Risk Analyzer is a practical project that combines several NLP
techniques learned over the past weeks to solve a real-world problem. Financial
institutions need to monitor news for early warning signals of risk -- credit
downgrades, regulatory actions, supply chain disruptions, and market volatility.
This project builds a system that ingests financial news, extracts entities (companies,
people, regulators), performs domain-adapted sentiment analysis, and computes a
composite risk score for each entity mentioned.

The core pipeline consists of three stages. First, a data ingestion layer collects
and deduplicates news articles. Second, an NLP processing layer runs named entity
recognition to identify companies and financial instruments, sentiment analysis
calibrated for financial language (where "volatile" and "restructuring" carry
different weight than in general text), and keyword-based risk category tagging
(credit risk, operational risk, market risk, regulatory risk). Third, a scoring
layer aggregates per-entity signals over a configurable time window.

This project reinforces skills from Days 41-46: word embeddings for feature
representation, transformer-based models for classification, and proper evaluation
methodology. It also introduces domain adaptation -- financial text has unique
characteristics that general-purpose models handle poorly without calibration.
FinBERT, a BERT model fine-tuned on financial communications, significantly
outperforms vanilla BERT for financial sentiment.

## Practical Example

```python
"""
Financial News Risk Analyzer -- Project skeleton
Full project code is in the project folder (see link below).
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy

# Load domain-specific models
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)
nlp = spacy.load("en_core_web_sm")

# Sample financial news
articles = [
    "Apple Inc. reported record quarterly revenue of $124 billion, beating analyst estimates.",
    "The SEC launched an investigation into Goldman Sachs over potential compliance violations.",
    "Tesla shares plunged 12% after supply chain disruptions halted production in Shanghai.",
]

# Risk analysis pipeline
RISK_KEYWORDS = {
    "regulatory": ["investigation", "sec", "compliance", "violation", "fine", "penalty"],
    "market": ["plunged", "crashed", "volatility", "selloff", "downturn"],
    "operational": ["disruption", "halt", "outage", "recall", "shortage"],
}

def analyze_article(text):
    # Extract entities
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "ORG"]

    # Financial sentiment
    sentiment = finbert(text[:512])[0]

    # Risk category detection
    text_lower = text.lower()
    risk_categories = []
    for category, keywords in RISK_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            risk_categories.append(category)

    # Composite risk score (0 = no risk, 1 = high risk)
    base_score = 0.0
    if sentiment["label"] == "negative":
        base_score = sentiment["score"]
    elif sentiment["label"] == "positive":
        base_score = 1.0 - sentiment["score"]
    risk_score = min(1.0, base_score + 0.1 * len(risk_categories))

    return {
        "entities": entities,
        "sentiment": sentiment,
        "risk_categories": risk_categories,
        "risk_score": round(risk_score, 3),
    }

for article in articles:
    result = analyze_article(article)
    print(f"Text: {article[:70]}...")
    print(f"  Entities: {result['entities']}")
    print(f"  Sentiment: {result['sentiment']['label']} ({result['sentiment']['score']:.3f})")
    print(f"  Risk categories: {result['risk_categories']}")
    print(f"  Risk score: {result['risk_score']}")
    print()
```

## Resources

- [FinBERT: Financial Sentiment Analysis with BERT](https://huggingface.co/ProsusAI/finbert)
- [SEC EDGAR full-text search for financial filings](https://efts.sec.gov/LATEST/search-index?q=%22risk%22)
- [Financial NLP resources and datasets](https://github.com/icoxfog417/awesome-financial-nlp)

## Next Day Preview

Day 48 starts Project 8: Semantic Search Engine, building a full-stack search system powered by sentence embeddings and vector databases.
