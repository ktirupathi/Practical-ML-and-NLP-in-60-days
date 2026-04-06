# Day 45: Vector Databases

## Learning Objectives

- Understand why brute-force similarity search does not scale and how approximate nearest neighbor (ANN) algorithms solve this
- Compare vector index types: flat, IVF (Inverted File), PQ (Product Quantization), and HNSW
- Build and query a FAISS index for fast similarity search over embeddings
- Use ChromaDB as a persistent vector store with metadata filtering
- Make informed decisions about index type, memory, and accuracy trade-offs

## Key Concepts

As embedding-based applications scale to millions or billions of vectors, exact nearest
neighbor search becomes prohibitively slow (O(n) per query). Vector databases solve this
with approximate nearest neighbor (ANN) algorithms that trade a small amount of accuracy
for dramatic speedups. FAISS (Facebook AI Similarity Search) is the most widely used
library, offering several index types: IndexFlatL2 for exact search (useful as a
baseline), IndexIVFFlat which partitions the space into Voronoi cells and only searches
nearby partitions, and IndexIVFPQ which adds product quantization to compress vectors
and reduce memory by 10-100x.

HNSW (Hierarchical Navigable Small World) graphs offer another approach, building a
multi-layer graph structure that enables logarithmic search time. HNSW typically
achieves higher recall than IVF at the same speed but uses more memory. The choice
between IVF and HNSW depends on your dataset size, memory budget, and latency
requirements. For most applications under 10 million vectors, HNSW provides the best
quality-speed trade-off.

ChromaDB and similar managed vector databases (Pinecone, Weaviate, Qdrant) add
persistence, metadata filtering, and API layers on top of these indexing algorithms.
ChromaDB is particularly popular for prototyping because it runs locally, supports
automatic embedding generation, and allows filtering results by metadata fields.
These databases form the storage backbone of semantic search engines and RAG systems.

## Practical Example

```python
import numpy as np
import faiss
import time

# Generate synthetic embeddings (simulating 100K document embeddings)
np.random.seed(42)
dimension = 384
num_vectors = 100_000
num_queries = 10

database_vectors = np.random.randn(num_vectors, dimension).astype("float32")
query_vectors = np.random.randn(num_queries, dimension).astype("float32")

# 1. Exact search (baseline)
index_flat = faiss.IndexFlatL2(dimension)
index_flat.add(database_vectors)

start = time.time()
D_exact, I_exact = index_flat.search(query_vectors, k=5)
flat_time = time.time() - start
print(f"Flat index search: {flat_time*1000:.1f}ms")

# 2. IVF index (approximate, much faster at scale)
nlist = 100  # number of Voronoi cells
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index_ivf.train(database_vectors)
index_ivf.add(database_vectors)
index_ivf.nprobe = 10  # search 10 out of 100 cells

start = time.time()
D_ivf, I_ivf = index_ivf.search(query_vectors, k=5)
ivf_time = time.time() - start
print(f"IVF index search:  {ivf_time*1000:.1f}ms")

# Measure recall (how many true neighbors were found)
recall = np.mean([len(set(I_ivf[i]) & set(I_exact[i])) / 5 for i in range(num_queries)])
print(f"IVF recall@5: {recall:.2%}")

# 3. ChromaDB for persistent storage with metadata
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents with metadata
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "The stock market rallied after the Fed announcement.",
    ],
    metadatas=[
        {"topic": "ml", "year": 2024},
        {"topic": "ml", "year": 2023},
        {"topic": "finance", "year": 2024},
    ],
)

# Query with metadata filtering
results = collection.query(
    query_texts=["deep learning models"],
    n_results=2,
    where={"topic": "ml"},
)
print(f"\nChromaDB results: {results['documents']}")
```

## Resources

- [FAISS wiki and tutorials](https://github.com/facebookresearch/faiss/wiki)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [ANN Benchmarks: comparing vector search algorithms](https://ann-benchmarks.com/)

## Next Day Preview

Day 46 ties embeddings and vector databases together with Retrieval-Augmented Generation (RAG), the architecture behind knowledge-grounded LLM applications.
