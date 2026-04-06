# Day 48: Project 8 -- Semantic Search Engine

## Learning Objectives

- Build a complete semantic search system from document ingestion to query results
- Implement embedding-based retrieval with re-ranking for improved precision
- Design a search API with proper query handling, pagination, and response formatting
- Compare semantic search against keyword-based search (BM25) and implement hybrid approaches
- Optimize search latency and relevance through index tuning and embedding selection

## Key Concepts

A semantic search engine goes beyond keyword matching by understanding the meaning of
queries and documents. Traditional search engines like Elasticsearch rely on BM25, a
term-frequency-based algorithm that fails when users phrase queries differently from
the document text (vocabulary mismatch). Semantic search solves this by encoding both
queries and documents into dense vector embeddings, where similar meanings map to
nearby points regardless of the specific words used. "How to fix a flat tire" matches
"tire puncture repair guide" even though they share few words.

The architecture of a production semantic search system has two phases: offline indexing
and online retrieval. During indexing, documents are chunked, embedded using a sentence
transformer model, and stored in a vector database with metadata. During retrieval, the
user query is embedded with the same model, an ANN search retrieves the top-k candidates,
and an optional re-ranker (a cross-encoder that scores query-document pairs jointly)
reorders results for higher precision. This two-stage approach balances speed (fast
ANN retrieval) with accuracy (expensive but precise re-ranking on a small candidate set).

Hybrid search combines BM25 and semantic scores to get the best of both worlds.
Exact keyword matches (product IDs, proper nouns) benefit from BM25, while semantic
understanding helps with paraphrased and conceptual queries. Reciprocal Rank Fusion
(RRF) is a simple and effective method to merge ranked lists from different retrieval
methods without needing to normalize their scores.

## Practical Example

```python
"""
Semantic Search Engine -- Project skeleton
Full project code is in the project folder (see link below).
"""
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np

# Initialize models
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Sample document corpus
corpus = [
    {"id": "1", "title": "Python Lists", "text": "Lists are mutable ordered collections in Python. You can append, remove, and sort elements."},
    {"id": "2", "title": "Python Dictionaries", "text": "Dictionaries store key-value pairs. Keys must be hashable. Lookup is O(1) average case."},
    {"id": "3", "title": "Python Sets", "text": "Sets are unordered collections of unique elements. They support union, intersection, and difference."},
    {"id": "4", "title": "NumPy Arrays", "text": "NumPy arrays are fixed-type, contiguous memory blocks for numerical computing. Much faster than lists."},
    {"id": "5", "title": "Pandas DataFrames", "text": "DataFrames are tabular data structures with labeled axes. Built on top of NumPy for data analysis."},
]

# Step 1: Index documents
client = chromadb.Client()
collection = client.create_collection("search_index")

texts = [f"{doc['title']}: {doc['text']}" for doc in corpus]
embeddings = bi_encoder.encode(texts).tolist()

collection.add(
    ids=[doc["id"] for doc in corpus],
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"title": doc["title"]} for doc in corpus],
)

# Step 2: Search with bi-encoder retrieval + cross-encoder re-ranking
def search(query, top_k=3, rerank=True):
    # Stage 1: Fast retrieval
    query_embedding = bi_encoder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k * 2)

    candidates = results["documents"][0]
    candidate_ids = results["ids"][0]

    if rerank and candidates:
        # Stage 2: Precise re-ranking
        pairs = [(query, doc) for doc in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates, candidate_ids), reverse=True)
        return [(cid, score, doc) for score, doc, cid in ranked[:top_k]]

    return [(cid, 0.0, doc) for cid, doc in zip(candidate_ids, candidates[:top_k])]

# Demo queries
queries = [
    "fast numerical computation",
    "how to store unique items",
    "tabular data analysis tool",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = search(query, top_k=2)
    for doc_id, score, text in results:
        print(f"  [{doc_id}] (score: {score:.3f}) {text[:80]}...")
```

## Resources

- [Sentence-Transformers: Semantic Search tutorial](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [MS MARCO passage ranking dataset](https://microsoft.github.io/msmarco/)
- [Hybrid search with Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

## Next Day Preview

Day 49 begins Project 9: Knowledge Base Chatbot using RAG to build an interactive question-answering system over custom documents.
