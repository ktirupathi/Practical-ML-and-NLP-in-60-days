# Day 44: Sentence Embeddings

## Learning Objectives

- Understand why averaging word embeddings is insufficient for capturing sentence-level semantics
- Explain how Sentence-BERT uses siamese networks and contrastive learning to produce sentence embeddings
- Compute sentence similarity scores using cosine similarity on dense embeddings
- Apply sentence embeddings to semantic textual similarity, clustering, and retrieval tasks
- Choose the right sentence embedding model for your use case based on benchmarks

## Key Concepts

Sentence embeddings map entire sentences to fixed-dimensional vectors such that
semantically similar sentences are close together in the embedding space. While BERT
produces token-level embeddings, using its [CLS] token or mean pooling directly for
sentence similarity yields poor results -- often worse than GloVe averages. Sentence-BERT
(Reimers and Gurevych, 2019) solved this by fine-tuning BERT with a siamese network
architecture, where two sentences pass through the same BERT encoder and are trained
using contrastive objectives to produce meaningful sentence-level representations.

The training process uses pairs or triplets of sentences with known similarity labels.
For NLI-based training, the model learns that entailment pairs should be close while
contradiction pairs should be far apart. For semantic textual similarity (STS), the
model directly regresses on human-annotated similarity scores. The result is an encoder
that produces embeddings where cosine similarity correlates strongly with human
judgments of semantic similarity, enabling efficient comparison of millions of sentence
pairs without cross-encoding each pair through BERT.

Modern sentence embedding models like all-MiniLM-L6-v2 offer an excellent balance of
speed and quality, producing 384-dimensional embeddings roughly 5x faster than BERT-base.
For multilingual applications, paraphrase-multilingual-MiniLM supports 50+ languages.
These embeddings are the foundation for semantic search, duplicate detection, clustering,
and retrieval-augmented generation systems that we will build in the coming days.

## Practical Example

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pretrained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode sentences
sentences = [
    "The cat sits on the mat.",
    "A kitten is resting on a rug.",
    "The stock market crashed yesterday.",
    "Financial markets experienced a sharp decline.",
    "I love programming in Python.",
]

embeddings = model.encode(sentences, convert_to_tensor=True)
print(f"Embedding shape: {embeddings.shape}")  # (5, 384)

# Compute pairwise cosine similarity
cosine_scores = util.cos_sim(embeddings, embeddings)
print("\nPairwise similarity matrix:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  [{i}] vs [{j}]: {cosine_scores[i][j]:.4f}")
        print(f"    '{sentences[i][:40]}...' vs '{sentences[j][:40]}...'")

# Semantic search: find most similar to a query
query = "What programming languages do you enjoy?"
query_embedding = model.encode(query, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, embeddings, top_k=3)[0]

print(f"\nQuery: '{query}'")
print("Top matches:")
for hit in hits:
    print(f"  Score: {hit['score']:.4f} | {sentences[hit['corpus_id']]}")

# Clustering similar sentences
from sklearn.cluster import KMeans

embeddings_np = embeddings.cpu().numpy()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings_np)
for label, sentence in zip(labels, sentences):
    print(f"  Cluster {label}: {sentence}")
```

## Resources

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SentenceTransformers documentation](https://www.sbert.net/)
- [MTEB Leaderboard: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

## Next Day Preview

Day 45 introduces vector databases like FAISS and ChromaDB for efficiently storing and searching over millions of embeddings.
