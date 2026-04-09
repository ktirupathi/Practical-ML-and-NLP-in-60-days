# Day 31: Text Preprocessing Pipeline

> **Phase 4 – NLP Foundations** | Week 5 | Estimated Time: 3-4 hours

## What You'll Learn
- Build a robust text preprocessing pipeline from raw text to clean tokens
- Apply regex for noise removal, tokenization, lemmatization, and stopword filtering
- Use NLTK and spaCy to implement production-ready preprocessing
- Understand when to apply each step and when NOT to

---

## 1. What Is Text Preprocessing?

Text preprocessing is the transformation of raw, unstructured text into a clean, normalized form that machine learning models can effectively process. Raw text contains noise: HTML tags, punctuation, casing inconsistencies, morphological variants of the same word, and irrelevant filler words.

A preprocessing **pipeline** is an ordered sequence of transformations applied consistently to every piece of text—training and inference alike. Consistency between training and serving is one of the most critical (and most often violated) rules in NLP engineering.

---

## 2. Why Used?

| Problem | Preprocessing Solution |
|---|---|
| "Running", "runs", "ran" treated as different features | Lemmatization / Stemming |
| "the", "a", "is" dominate frequency counts | Stopword removal |
| HTML/URL noise pollutes feature space | Regex cleaning |
| Case mismatch ("Apple" vs "apple") | Lowercasing |
| Punctuation exploding vocabulary | Punctuation removal |

Without preprocessing, models learn spurious patterns from noise rather than signal. A BoW model trained on uncleaned text might learn that `"FREE!!!"` and `"FREE"` and `"free"` are three different, unrelated features.

---

## 3. Real-World Example

An email spam classifier trained without preprocessing might learn that "FREE!!!" and "free" are two different tokens. With lowercasing and punctuation removal, they collapse to one feature. A customer support ticket router that lemmatizes "running", "runs", "ran" to "run" generalizes better across ticket phrasings.

In a search engine, query "best running shoes" and document "top shoes for runners" need to share tokens after lemmatization ("run" + "shoe") to compute relevance correctly.

---

## 4. Intuition

Think of preprocessing as standardizing the alphabet before reading. If every author used a different font, script, and capitalization style, pattern recognition would be nearly impossible. Preprocessing creates a **canonical form** so that semantically identical expressions map to identical features.

The key trade-off: aggressive preprocessing removes noise but can also remove signal. Named entities, negations ("not good"), and domain jargon may be harmed by over-preprocessing. Always validate preprocessing choices on your specific task using held-out evaluation data.

---

## 5. Mathematical Intuition

```
Vocabulary size V grows with corpus size N:
  V ≈ O(N^β),  0 < β < 1  (Heaps' Law: β ≈ 0.5–0.7)

After lemmatization, V_lemma < V_raw
  Reduction ratio: V_lemma / V_raw ≈ 0.6–0.8 in English

After stopword removal (top-k words by frequency):
  Effective vocabulary = V - |stopwords ∩ V|
  Typically removes 50-200 high-frequency words

TF impact after preprocessing:
  tf(t, d) = count(t in d) / |d|
  Removing stopwords → denominator |d| shrinks
                     → tf values increase for content words
                     → IDF differences become more discriminative

Stemming error modes:
  Over-stemming: "university" and "universe" → "univers" (false conflation)
  Under-stemming: "alumnus" and "alumni" not merged (missed conflation)
```

---

## 6. Worked Example

Raw text:
```
"Running the 3 top-rated NLP models... Visit http://example.com for FREE!!!"
```

Step-by-step:
1. Lowercase → `"running the 3 top-rated nlp models... visit http://example.com for free!!!"`
2. Remove URLs → `"running the 3 top-rated nlp models... visit  for free!!!"`
3. Remove numbers → `"running the top-rated nlp models... visit  for free!!!"`
4. Remove punctuation → `"running the top rated nlp models  visit  for free"`
5. Tokenize → `["running", "the", "top", "rated", "nlp", "models", "visit", "for", "free"]`
6. Remove stopwords → `["running", "top", "rated", "nlp", "models", "visit", "free"]`
7. Lemmatize → `["run", "top", "rate", "nlp", "model", "visit", "free"]`

Result: 7 clean tokens from a noisy 13-token input.

---

## 7. Python Implementation

```python
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Load spaCy model: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Remove URLs, HTML, numbers, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"<.*?>", "", text)                      # HTML tags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)            # Non-ASCII
    text = re.sub(r"\d+", "", text)                        # Numbers
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def nltk_pipeline(text: str) -> list:
    """NLTK-based: clean → tokenize → remove stopwords → lemmatize."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def spacy_pipeline(text: str) -> list:
    """spaCy-based: clean → parse → filter → return lemmas."""
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return tokens


def preprocess_corpus(texts: list, method: str = "spacy") -> list:
    """Apply pipeline to a list of texts. Returns list of token lists."""
    pipeline = spacy_pipeline if method == "spacy" else nltk_pipeline
    return [pipeline(text) for text in texts]


def pipeline_to_string(tokens: list) -> str:
    """Join tokens back to string for vectorizers."""
    return " ".join(tokens)


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Running the 3 top-rated NLP models... Visit http://example.com for FREE!!!",
        "The cats are sitting on the mats. They were running fast!",
        "<p>Customer complaints: The products arrived damaged and broken!</p>",
        "I can NOT believe how terrible this experience was. Not good at all.",
    ]

    print("=" * 60)
    print("spaCy Pipeline")
    print("=" * 60)
    for text in samples:
        result = spacy_pipeline(text)
        print(f"Input : {text[:65]}")
        print(f"Output: {result}")
        print()

    print("=" * 60)
    print("NLTK Pipeline")
    print("=" * 60)
    for text in samples:
        result = nltk_pipeline(text)
        print(f"Input : {text[:65]}")
        print(f"Output: {result}")
        print()

    # Vocabulary reduction demo
    import random, string as s
    corpus = [" ".join(random.choices(
        ["running runs run cats cat sitting sits stood stand the a is was were"],
        k=50)) for _ in range(100)]
    raw_vocab = set(w for doc in corpus for w in doc.split())
    processed = preprocess_corpus(corpus)
    proc_vocab = set(t for doc in processed for t in doc)
    print(f"Raw vocabulary size   : {len(raw_vocab)}")
    print(f"Clean vocabulary size : {len(proc_vocab)}")
    print(f"Reduction             : {1 - len(proc_vocab)/len(raw_vocab):.1%}")
```

---

## 8. Visualization

```
Raw Text Input
     │
     ▼
┌─────────────────┐
│  Lowercase      │  "Hello" → "hello"
└────────┬────────┘
         ▼
┌─────────────────┐
│ Remove URLs/    │  "http://..." → ""
│ HTML Tags       │  "<b>text</b>" → "text"
└────────┬────────┘
         ▼
┌─────────────────┐
│ Remove Numbers  │  "3rd" → "rd" (or keep if needed)
│ & Punctuation   │  "FREE!!!" → "free"
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tokenize        │  "hello world" → ["hello", "world"]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Remove          │  ["the","cat","ran"] → ["cat","ran"]
│ Stopwords       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Lemmatize /     │  "running" → "run"
│ Stem            │  "studies" → "study"
└────────┬────────┘
         ▼
Clean Token List
["run", "top", "rate", "nlp", "model"]
     │
     ▼
Feature Extraction (TF-IDF, BoW, Embeddings)
```

---

## 9. Common Mistakes

1. **Inconsistent pipelines** – Training data preprocessed differently than inference data causes distribution shift. Always use the same pipeline object.
2. **Over-aggressive stopword removal** – Removing negations ("not", "no", "never") destroys sentiment signal. Customize your stopword list.
3. **Lemmatizing named entities** – "Paris" should not become "pari". Use spaCy NER to protect entities before lemmatization.
4. **Stemming in production** – Stemming is faster but crude (Porter: "studies" → "studi"). Lemmatization is linguistically correct; prefer it for production.
5. **Forgetting encoding** – Always decode bytes as UTF-8 before preprocessing; handle encoding errors explicitly.
6. **Lowercasing all text** – Breaks NER; "US" (country) becomes "us" (pronoun). Consider task-specific casing rules.
7. **Not validating the pipeline** – Always inspect 50+ preprocessed samples manually before training.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the difference between stemming and lemmatization? | Stemming chops word endings heuristically (fast, imprecise). Lemmatization uses vocabulary/morphological analysis to return the dictionary base form (slower, linguistically correct). |
| 2 | When should you NOT remove stopwords? | Sentiment analysis (negations "not good"), question answering (wh-words), and tasks where function words carry syntactic meaning. |
| 3 | What is tokenization? | Splitting text into atomic units (tokens): words, subwords, or characters. The choice of tokenization granularity affects downstream model vocabulary. |
| 4 | Why lowercase text? | Reduces vocabulary size; "Apple" and "apple" should map to the same feature in most tasks. Caution: breaks NER. |
| 5 | What is a preprocessing pipeline? | An ordered sequence of transformations applied identically to training and inference data. Implemented as a class or function to ensure consistency. |
| 6 | How do you handle emojis in text preprocessing? | Either remove them (`re.sub(r'[^\x00-\x7F]+', '', text)`) or convert to text descriptions using the `emoji` library, depending on task value. |
| 7 | What is Heaps' Law? | Vocabulary size V grows as O(N^β) with corpus size N (β ≈ 0.5–0.7). Preprocessing reduces V by collapsing morphological variants. |
| 8 | What spaCy model should you use for production English NLP? | `en_core_web_sm` for speed; `en_core_web_trf` (transformer-based) for accuracy. Choose based on latency requirements. |
| 9 | How does preprocessing affect TF-IDF? | Reducing vocabulary via lemmatization/stopword removal makes TF-IDF vectors denser and more discriminative; fewer spurious high-IDF noise tokens. |
| 10 | What is the risk of using a generic stopword list? | Domain-specific important words may be on the generic list (e.g., "will" is a stopword but also a proper name; "cancer" might appear on a bio-medical stoplist accidentally). |

---

## Exercises

1. Build a pipeline that preserves named entities (do not lowercase or lemmatize them).
2. Compare vocabulary sizes before and after preprocessing on a 1000-document corpus using the 20 Newsgroups dataset.
3. Implement a custom stopword list for a specific domain (e.g., legal or medical text).
4. Benchmark spaCy vs NLTK pipeline speed on 10,000 documents using `timeit`.
5. Add an emoji-to-text conversion step to the pipeline using the `emoji` library.
6. Write a unit test that verifies your pipeline produces identical output when called twice on the same input (idempotency).

---

## Key Takeaways

- Preprocessing is the foundation of all classical NLP; garbage in, garbage out.
- spaCy is faster and more accurate for production; NLTK is better for learning concepts.
- Always apply the **same** pipeline to training and inference data—inconsistency is the #1 source of NLP bugs.
- Aggressive preprocessing removes noise but may remove signal; tune per task with evaluation metrics.
- Lemmatization > Stemming for most production NLP tasks where interpretability matters.
- Document your preprocessing decisions: what was removed and why.
