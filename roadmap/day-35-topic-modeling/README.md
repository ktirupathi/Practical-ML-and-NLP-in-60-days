# Day 35: Topic Modeling

## Learning Objectives

- Extract latent topics from a document corpus using LDA and NMF
- Use BERTopic for embedding-based topic modeling with pretrained transformers
- Visualize topics with word clouds, pyLDAvis, and BERTopic's built-in charts
- Evaluate topic quality using coherence scores (C_v, U_Mass)
- Choose the optimal number of topics using coherence-based model selection

## Key Concepts

### LDA and NMF

**Latent Dirichlet Allocation (LDA)** is a probabilistic generative model that assumes each
document is a mixture of topics and each topic is a distribution over words. It uses
variational inference or Gibbs sampling to discover these latent distributions from the
observed word counts. **Non-negative Matrix Factorization (NMF)** takes a linear algebra
approach: it factorizes the TF-IDF matrix into two non-negative matrices, one mapping
documents to topics and the other mapping topics to words. NMF tends to produce more focused,
less overlapping topics and is faster to train, while LDA provides a proper probabilistic
framework.

### BERTopic

BERTopic is a modern topic modeling library that replaces the bag-of-words assumption with
dense sentence embeddings. It first encodes documents with a transformer (e.g., all-MiniLM),
reduces dimensionality with UMAP, clusters with HDBSCAN, and then extracts topic
representations using a class-based TF-IDF variant. Because it operates on semantic
embeddings, BERTopic captures meaning that word-count models miss, handles short texts better,
and produces more interpretable topics.

### Evaluating Topic Quality

A good topic model produces coherent, distinct, and interpretable topics. **Coherence scores**
(C_v, U_Mass, NPMI) measure how often the top words in a topic co-occur in the corpus ---
higher coherence means more interpretable topics. Plotting coherence against the number of
topics helps you find the sweet spot. Human evaluation (asking domain experts to rate topic
labels) remains the gold standard for interpretability.

## Practical Example

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np

# Load data
docs = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data[:2000]

# --- 1. LDA ---
count_vec = CountVectorizer(max_features=5000, stop_words="english")
X_counts = count_vec.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=10, random_state=42, n_jobs=-1)
lda.fit(X_counts)

feature_names = count_vec.get_feature_names_out()

def print_topics(model, feature_names, n_words=8):
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
        print(f"  Topic {idx}: {', '.join(top_words)}")

print("LDA Topics:")
print_topics(lda, feature_names)

# --- 2. NMF ---
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf_vec.fit_transform(docs)

nmf = NMF(n_components=10, random_state=42)
nmf.fit(X_tfidf)

print("\nNMF Topics:")
print_topics(nmf, tfidf_vec.get_feature_names_out())

# --- 3. BERTopic ---
from bertopic import BERTopic

topic_model = BERTopic(nr_topics=10, verbose=False)
topics, probs = topic_model.fit_transform(docs)

print("\nBERTopic Topics:")
for topic_id, info in topic_model.get_topic_info().head(11).iterrows():
    if info["Topic"] == -1:
        continue
    words = topic_model.get_topic(info["Topic"])
    top_words = ", ".join([w for w, _ in words[:6]])
    print(f"  Topic {info['Topic']}: {top_words}")

# --- 4. Coherence evaluation (using gensim) ---
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

tokenized = [doc.lower().split() for doc in docs]
dictionary = Dictionary(tokenized)

coherence_scores = []
for k in range(5, 25, 5):
    lda_k = LatentDirichletAllocation(n_components=k, random_state=42)
    lda_k.fit(X_counts)
    topics_list = []
    for topic in lda_k.components_:
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        topics_list.append(top_words)
    cm = CoherenceModel(
        topics=topics_list, texts=tokenized, dictionary=dictionary, coherence="c_v"
    )
    score = cm.get_coherence()
    coherence_scores.append((k, score))
    print(f"k={k:2d}  C_v={score:.4f}")
```

## Resources

- [scikit-learn LDA documentation](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda)
- [BERTopic documentation](https://maartengr.github.io/BERTopic/)
- [Gensim coherence model](https://radimrehurek.com/gensim/models/coherencemodel.html)

## Up Next

**Day 36 -- Text Summarization:** Learn extractive and abstractive approaches to automatically condense long documents.
