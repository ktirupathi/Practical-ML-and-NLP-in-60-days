# Day 31: Text Preprocessing for NLP

## Learning Objectives

- Tokenize text at word and sentence level using NLTK and spaCy
- Reduce words to their base forms with stemming (Porter, Snowball) and lemmatization
- Remove stop words, punctuation, and noise using regex and built-in stop-word lists
- Build a reusable text cleaning function suitable for downstream ML pipelines
- Understand when aggressive preprocessing helps and when it hurts

## Key Concepts

### Tokenization

Tokenization splits a raw string into individual units --- tokens --- that a model can
process. Word tokenization ("The cat sat" -> ["The", "cat", "sat"]) is the most common, but
sentence tokenization is useful for tasks like summarization. Whitespace splitting is a naive
baseline; real tokenizers handle contractions ("don't" -> "do", "n't"), hyphenated words,
URLs, and emojis correctly. NLTK's `word_tokenize` and spaCy's tokenizer are the two
workhorses in Python.

### Stemming and Lemmatization

Both techniques reduce inflected words to a common root. **Stemming** chops suffixes with
hand-crafted rules (Porter: "running" -> "run", "studies" -> "studi") and is fast but
sometimes produces non-words. **Lemmatization** uses vocabulary and morphological analysis to
return valid dictionary forms ("better" -> "good", "studies" -> "study") but requires POS
tags and is slower. For bag-of-words models, stemming is often good enough; for tasks where
meaning matters, lemmatization is preferred.

### Text Cleaning and Stop Words

Real-world text is noisy: mixed case, HTML tags, special characters, repeated whitespace, and
domain-specific junk. A cleaning pipeline typically lowercases, strips HTML, removes or
normalizes URLs and emails, eliminates punctuation, and filters out stop words (common words
like "the", "is", "and" that carry little semantic weight). The right level of cleaning
depends on the task: aggressive cleaning suits topic modeling, while sentiment analysis may
need to preserve negations and punctuation.

## Practical Example

```python
import re
import nltk
import spacy

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

# --- Raw text ---
raw = """
<p>Dr. Smith's 2nd paper on deep-learning (2023) was AMAZING!!!
Check it out: https://example.com/paper. It's a must-read.</p>
"""

# --- Step 1: HTML and URL removal ---
def strip_html(text):
    return re.sub(r"<[^>]+>", "", text)

def remove_urls(text):
    return re.sub(r"https?://\S+", "", text)

# --- Step 2: Lowercasing and punctuation removal ---
def clean_text(text):
    text = strip_html(text)
    text = remove_urls(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

cleaned = clean_text(raw)
print("Cleaned:", cleaned)
# "dr smith s 2nd paper on deep learning 2023 was amazing it s a must read"

# --- Step 3: Tokenization ---
tokens = word_tokenize(cleaned)
print("Tokens:", tokens)

# --- Step 4: Stop word removal ---
stop_words = set(stopwords.words("english"))
filtered = [t for t in tokens if t not in stop_words]
print("Filtered:", filtered)

# --- Step 5a: Stemming ---
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in filtered]
print("Stemmed:", stemmed)

# --- Step 5b: Lemmatization ---
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in filtered]
print("Lemmatized:", lemmatized)

# --- spaCy all-in-one ---
doc = nlp(raw)
spacy_tokens = [
    token.lemma_.lower()
    for token in doc
    if not token.is_stop and not token.is_punct and not token.like_url
]
print("spaCy lemmas:", spacy_tokens)

# --- Reusable preprocessing function ---
def preprocess(text: str, use_lemma: bool = True) -> str:
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if use_lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

print(preprocess(raw))
```

## Resources

- [NLTK tokenization documentation](https://www.nltk.org/api/nltk.tokenize.html)
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features)
- [Regex tutorial for text cleaning](https://docs.python.org/3/howto/regex.html)

## Up Next

**Day 32 -- Text Representation:** Convert cleaned text into numerical feature vectors using Bag of Words, TF-IDF, and n-grams.
