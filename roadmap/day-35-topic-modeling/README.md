# Day 35: Topic Modeling — LDA and BERTopic

> **Phase 4 – NLP Foundations** | Week 5 | Estimated Time: 3-4 hours

## What You'll Learn
- Understand Latent Dirichlet Allocation (LDA) and the Dirichlet distribution
- Implement LDA with sklearn and gensim, evaluate with coherence score
- Use BERTopic for modern transformer-based topic modeling
- Visualize topic distributions and word clouds

---

## 1. What Is Topic Modeling?

Topic modeling is an **unsupervised** technique that discovers hidden thematic structure in a corpus of documents. Given a collection of documents, it answers: "What are the main themes, and which themes does each document discuss?"

A **topic** is represented as a probability distribution over words. A **document** is represented as a mixture of topics.

Unlike classification, topics are not predefined—they emerge from statistical patterns in word co-occurrence across documents.

---

## 2. Why Used?

| Application | Value |
|---|---|
| News aggregation | Automatically group articles by theme |
| Customer feedback | Discover recurring complaint themes |
| Research analysis | Map the landscape of a scientific field |
| Legal document review | Identify relevant document clusters |
| Content recommendation | Recommend articles on similar topics |

Topic modeling scales to millions of documents without requiring any labels—making it invaluable for exploratory analysis of large text corpora.

---

## 3. Real-World Example

The New York Times uses topic models to organize its archive of 150 years of articles. LDA discovers topics like "politics/elections", "sports/baseball", "finance/markets" purely from word patterns, without any human labeling. Journalists use these topics for search, recommendation, and trend analysis.

---

## 4. Intuition

**LDA Intuition**: Imagine each document was written by an author who:
1. Randomly chose a mixture of topics (e.g., 70% tech, 30% business)
2. For each word, randomly chose a topic from their mixture
3. Randomly chose a word from that topic's vocabulary distribution

LDA inverts this process: given the observed words, infer the latent topics and document-topic mixtures that most likely generated the corpus.

**BERTopic Intuition**: Encode documents as dense semantic vectors using BERT, cluster them with HDBSCAN, then extract representative words per cluster using class-based TF-IDF (c-TF-IDF).

---

## 5. Mathematical Intuition

```
=== Dirichlet Distribution ===
Dir(α) over K-simplex (K topics or V words):
  PDF: p(θ) ∝ ∏_k θ_k^(α_k - 1)
  
  α > 1: distribution concentrated at center (uniform mixture)
  α < 1: distribution concentrated at corners (sparse topics)
  α = 1: uniform over simplex

=== LDA Generative Process ===
For each document d:
  θ_d ~ Dirichlet(α)                      # topic distribution for d
  For each word position n:
    z_dn ~ Categorical(θ_d)               # choose topic z
    w_dn ~ Categorical(φ_{z_dn})          # choose word from topic

For each topic k:
  φ_k ~ Dirichlet(β)                      # word distribution for topic k

=== Inference ===
Variational Bayes or Gibbs Sampling to approximate:
  P(z | w, α, β)  ← intractable to compute exactly

=== Coherence Score (C_v) ===
Measures how semantically similar the top words of a topic are:
  Higher coherence → more interpretable topic
  Typical range: -1 to 1, good topics score > 0.4

=== BERTopic c-TF-IDF ===
For each topic cluster c:
  c-tf-idf(t, c) = tf(t, c) * log(1 + |C| / tf(t))
  where |C| = total number of documents, tf(t) = term frequency in all docs
```

---

## 6. Worked Example

```
Corpus of 4 documents about tech and sports:
  d1: "machine learning algorithms neural networks deep"
  d2: "football soccer goal penalty stadium fans"
  d3: "deep learning neural network computer vision"
  d4: "basketball basketball shooting dribbling court"

LDA discovers 2 topics:
  Topic 0 (Tech): {machine:0.3, learning:0.25, neural:0.2, deep:0.15, network:0.1}
  Topic 1 (Sports): {football:0.25, soccer:0.2, basketball:0.2, goal:0.15, fans:0.1}

Document-topic distributions:
  d1: [Topic 0: 0.95, Topic 1: 0.05]  ← mostly tech
  d4: [Topic 0: 0.05, Topic 1: 0.95]  ← mostly sports
```

---

## 7. Python Implementation

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import warnings
warnings.filterwarnings("ignore")

# ── Dataset ────────────────────────────────────────────────────────────────
categories = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]
newsgroups = fetch_20newsgroups(subset="train", categories=categories,
                                remove=("headers", "footers", "quotes"))
texts = newsgroups.data[:500]  # subset for speed
print(f"Corpus size: {len(texts)} documents")

# ── LDA with sklearn ───────────────────────────────────────────────────────
vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=5,
    max_features=5000,
    stop_words="english",
)
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

N_TOPICS = 4
lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=20,
    learning_method="online",
    random_state=42,
    doc_topic_prior=0.1,    # α: sparse document-topic distributions
    topic_word_prior=0.01,  # β: sparse topic-word distributions
)
lda.fit(X)

def print_top_words(model, feature_names, n_top=10):
    print("\n=== LDA Topics ===")
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top-1:-1]]
        print(f"Topic {topic_idx}: {' | '.join(top_words)}")

print_top_words(lda, feature_names)

# Document-topic distributions
doc_topics = lda.transform(X)
print("\n=== Sample Document-Topic Distributions ===")
for i in range(3):
    dominant_topic = doc_topics[i].argmax()
    print(f"Doc {i}: {doc_topics[i].round(3)}  → Dominant Topic {dominant_topic}")
    print(f"  Text: {texts[i][:80]}...")

# ── LDA with gensim + coherence score ─────────────────────────────────────
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    import re

    def tokenize(text):
        return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower())
                if w not in {"the", "and", "for", "this", "that", "with"}]

    tokenized = [tokenize(t) for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.9)
    corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized]

    lda_gensim = LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=N_TOPICS,
        passes=10,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    coherence_model = CoherenceModel(
        model=lda_gensim,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = coherence_model.get_coherence()
    print(f"\n=== Gensim LDA Coherence Score (C_v) ===")
    print(f"Coherence: {coherence_score:.4f}  (higher is better, good > 0.4)")

    print("\n=== Gensim Topics ===")
    for i, topic in lda_gensim.print_topics(num_topics=N_TOPICS, num_words=8):
        print(f"Topic {i}: {topic}")

except ImportError:
    print("gensim not installed; skipping coherence score demo")

# ── BERTopic (optional, requires bertopic) ─────────────────────────────────
try:
    from bertopic import BERTopic

    topic_model = BERTopic(
        nr_topics=N_TOPICS,
        min_topic_size=10,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts)
    freq = topic_model.get_topic_info()
    print("\n=== BERTopic Results ===")
    print(freq.head(N_TOPICS + 1))
    for topic_id in range(N_TOPICS):
        words = topic_model.get_topic(topic_id)
        if words:
            top = [w for w, _ in words[:5]]
            print(f"BERTopic {topic_id}: {top}")
except ImportError:
    print("bertopic not installed; skipping BERTopic demo")
```

---

## 8. Visualization

```
LDA Plate Notation:
┌─────────────────────────────────────────────┐
│ Corpus                                      │
│  ┌──────────────────────────┐               │
│  │ Document d               │               │
│  │   θ_d ~ Dir(α)           │  ← topic mix │
│  │   ┌──────────────────┐   │               │
│  │   │ Word n           │   │               │
│  │   │   z_dn ~ Cat(θ_d)│   │  ← topic z   │
│  │   │   w_dn ~ Cat(φ_z)│   │  ← word w    │
│  │   └──────────────────┘   │               │
│  └──────────────────────────┘               │
│                                             │
│  φ_k ~ Dir(β)  for each topic k             │
└─────────────────────────────────────────────┘

Topic-Word Matrix (φ):
           machine  learning  neural  football  goal
  Topic 0: [0.30,    0.25,    0.20,    0.01,   0.01]  ← Tech
  Topic 1: [0.01,    0.02,    0.01,    0.25,   0.20]  ← Sports

BERTopic Pipeline:
  Docs → BERT embeddings → UMAP (dim reduction) → HDBSCAN (clustering) → c-TF-IDF (keywords)
```

---

## 9. Common Mistakes

1. **Choosing number of topics arbitrarily** – Use coherence score to select optimal K; plot coherence vs K and look for the elbow.
2. **Not removing stopwords for LDA** – Common words like "the", "is" dominate topics. Use `stop_words='english'` in CountVectorizer.
3. **Using TF-IDF with LDA** – LDA expects raw count matrices, not TF-IDF. Use CountVectorizer, not TfidfVectorizer.
4. **Setting min_df too low** – Rare words create noise topics. Set `min_df=5` or higher.
5. **Ignoring topic coherence** – A model with high perplexity but low coherence produces uninterpretable topics.
6. **Not visualizing topics** – Always manually inspect top words per topic; metrics alone don't capture interpretability.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is LDA and what does it model? | Latent Dirichlet Allocation models documents as mixtures of topics, where each topic is a distribution over words. Both distributions use Dirichlet priors. |
| 2 | What is the Dirichlet distribution's role in LDA? | It is a prior over probability distributions (simplices). It controls how sparse/uniform document-topic and topic-word distributions are. |
| 3 | What does a high α value in LDA mean? | Documents will have more uniform topic mixtures (spread across many topics). Low α → documents concentrate on fewer topics. |
| 4 | How do you evaluate topic model quality? | Coherence score (C_v, NPMI) measures semantic similarity of top topic words. Perplexity measures held-out likelihood but doesn't correlate with interpretability. |
| 5 | What is the difference between LDA and NMF for topic modeling? | LDA is probabilistic with Dirichlet priors; NMF (Non-negative Matrix Factorization) is deterministic and faster, also produces interpretable topics. |
| 6 | How does BERTopic differ from LDA? | BERTopic uses contextual embeddings (BERT) for document representations, HDBSCAN for clustering, and c-TF-IDF for topic keywords. No distributional assumptions. |
| 7 | What is c-TF-IDF in BERTopic? | Class-based TF-IDF: treats each topic cluster as a "document" and finds words that are important within a cluster relative to the whole corpus. |
| 8 | How many topics should you choose for LDA? | Use coherence score: train models for K=5,10,15,...,50, plot coherence vs K, choose the elbow. Typical range: 10-100 for large corpora. |
| 9 | Can you update an LDA model with new documents? | Yes, gensim's LDA supports online learning (`update()`) for new documents without full retraining. |
| 10 | What is the difference between LDA's α and β parameters? | α is the document-topic prior (controls how many topics per document); β is the topic-word prior (controls how many words per topic). |

---

## Exercises

1. Train LDA on the 20 Newsgroups dataset with K=4,8,16 and plot coherence score vs K.
2. Visualize LDA topics using pyLDAvis for interactive exploration.
3. Implement dynamic topic modeling to track how topics evolve over time in a news corpus.
4. Compare LDA vs NMF (sklearn) on topic coherence and interpretability.
5. Use BERTopic on a Twitter dataset and visualize topic clusters with UMAP.

---

## Key Takeaways

- LDA models documents as mixtures of topics, topics as distributions over words.
- The Dirichlet prior controls sparsity: low α → few topics per doc, low β → few words per topic.
- Use CountVectorizer (not TF-IDF) as input to LDA.
- Coherence score (C_v > 0.4) is a better evaluation metric than perplexity for interpretability.
- BERTopic uses transformer embeddings + HDBSCAN + c-TF-IDF for more coherent modern topics.
- Always manually inspect top words per topic; automated metrics don't capture all interpretability aspects.
