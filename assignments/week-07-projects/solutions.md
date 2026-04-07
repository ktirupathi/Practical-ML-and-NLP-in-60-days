# Week 7: NLP Projects - Solutions

---

## Part 1: Conceptual Answers

### A1. Sentence Embeddings vs. Word Embeddings
Averaging word embeddings loses word order ("dog bites man" = "man bites dog") and dilutes meaning with frequent, uninformative words. SBERT fine-tunes a pre-trained BERT using a siamese/triplet network structure on Natural Language Inference (NLI) data. The training objective uses contrastive loss: similar sentence pairs should have high cosine similarity, dissimilar pairs should have low similarity. SBERT produces fixed-size sentence embeddings that can be compared efficiently with cosine similarity, unlike raw BERT which requires feeding both sentences through the model simultaneously.

### A2. Cosine Similarity for Semantic Search
Cosine similarity measures the angle between vectors, ignoring magnitude. Two sentences with the same meaning but different lengths may have embedding vectors with different magnitudes (longer sentences accumulate more from the model). Euclidean distance penalizes this magnitude difference, making two semantically identical sentences appear distant simply because one produces a higher-norm vector. Cosine similarity normalizes this away, focusing purely on directional similarity in the embedding space.

### A3. FAISS Fundamentals
FAISS is a library optimized for dense vector similarity search. `IndexFlatL2` performs brute-force exact search with O(n*d) per query -- guaranteed to find the true nearest neighbor but slow for large datasets. `IndexIVFFlat` partitions vectors into Voronoi cells using k-means clustering, then only searches a subset of cells -- much faster but approximate. Choose Flat for datasets under ~100K vectors or when perfect recall is required. Choose IVF for datasets over 100K vectors where speed matters and slight recall loss is acceptable.

### A4. Approximate Nearest Neighbors
IVF partitions the vector space into `nlist` Voronoi cells via k-means on the training data. Each cell's centroid acts as a representative. At query time, the query is compared to all centroids, and only vectors in the `nprobe` nearest cells are searched exhaustively. With `nprobe=1`, only one cell is searched (fast but may miss neighbors in adjacent cells). With `nprobe=nlist`, all cells are searched (exact search, slow). Typical practice: set `nprobe` to 5-10% of `nlist` for a good balance.

### A5. ChromaDB Architecture
ChromaDB is a purpose-built vector database with a document-centric API. Unlike FAISS (a low-level library for vector operations), ChromaDB provides: (1) automatic embedding generation -- pass raw text and it embeds for you, (2) metadata storage and filtering -- attach and query by arbitrary metadata, (3) persistence -- data survives restarts, (4) document CRUD operations -- add, update, delete by ID, (5) built-in distance metrics and result formatting. FAISS is lower-level and faster for pure vector operations but requires you to manage documents, metadata, and persistence yourself.

### A6. RAG Architecture
**Indexing:** Documents are chunked, embedded, and stored in a vector database. **Retrieval:** Given a user query, the system encodes the query, searches the vector store for relevant chunks, and retrieves the top-k results. **Generation:** Retrieved chunks are inserted into a prompt template alongside the query, and an LLM generates an answer grounded in the retrieved context. RAG reduces hallucination because the LLM has access to factual source material rather than relying solely on parametric memory. The model can cite specific passages, and answers can be verified against the retrieved documents.

### A7. Chunking Strategies for RAG
**Fixed-size with overlap:** Simple and predictable, but may split sentences or ideas mid-thought. **Sentence-based:** Preserves sentence integrity but chunk sizes vary widely. **Semantic chunking:** Groups semantically related sentences (using embedding similarity between consecutive sentences), producing coherent chunks but is more complex. Small chunks (100-200 tokens) give precise retrieval but may lack context. Large chunks (500-1000 tokens) provide more context but may include irrelevant information and dilute the embedding. Overlap (10-20%) mitigates boundary effects.

### A8. Embedding Model Selection
`all-MiniLM-L6-v2`: Fastest inference (6 layers), smallest memory footprint, good for prototyping and resource-constrained environments. `all-mpnet-base-v2`: Higher quality embeddings, better on benchmarks (MTEB), but 5x slower. OpenAI `ada-002`: Highest dimensionality (1536), strong performance, but requires API calls (latency, cost, data privacy concerns). To benchmark for a domain: create a test set of query-document pairs with relevance labels, measure Recall@K and MRR, and compare inference throughput. Domain-specific data may favor one model unexpectedly.

### A9. RAG Failure Modes
**(a) Retrieval misses:** The embedding model does not capture the semantic relationship between query and relevant document. Mitigation: use hybrid search (combine sparse BM25 with dense retrieval), fine-tune the embedding model on domain data, or use query expansion. **(b) LLM ignores context:** The model relies on its parametric knowledge instead of the provided context. Mitigation: use explicit instructions ("Answer ONLY based on the provided context"), place context prominently in the prompt, or use models fine-tuned for RAG. **(c) Over-reliance on irrelevant context:** Poor retrieval returns tangentially related text that misleads the LLM. Mitigation: use re-ranking to improve precision, set a similarity threshold to filter low-quality results, or retrieve more candidates and let the LLM evaluate relevance.

### A10. Domain-Specific Fine-Tuning
FinBERT outperforms general BERT because financial text has domain-specific vocabulary ("bullish", "EBITDA", "yield curve") and different sentiment cues ("restructuring" is negative in finance but neutral generally). Domain-adaptive pre-training involves further training BERT's MLM objective on a large unlabeled financial corpus before fine-tuning on labeled sentiment data. This two-stage approach adapts the model's language understanding to the domain. Risks include: catastrophic forgetting of general language knowledge, overfitting to domain-specific patterns that do not generalize, and the computational cost of additional pre-training. Mitigation: use a small learning rate and monitor validation perplexity.

---

## Part 2: Coding Solutions

### CQ1. Encode Sentences with SBERT

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "The match was exciting to watch.",
]

embeddings = model.encode(sentences)
sim_matrix = cosine_similarity(embeddings)

print("Pairwise cosine similarity:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  '{sentences[i]}' <-> '{sentences[j]}': {sim_matrix[i][j]:.4f}")

# Find most similar pair
upper_tri = np.triu(sim_matrix, k=1)
i, j = np.unravel_index(upper_tri.argmax(), upper_tri.shape)
print(f"\nMost similar pair: [{i}] and [{j}] with similarity {sim_matrix[i][j]:.4f}")
```

---

### CQ2. Build a FAISS Flat Index

```python
import faiss
import numpy as np

d = 384
n = 1000
np.random.seed(42)
embeddings = np.random.randn(n, d).astype("float32")

index = faiss.IndexFlatL2(d)
index.add(embeddings)
print(f"Total vectors in index: {index.ntotal}")

query = np.random.randn(1, d).astype("float32")
distances, indices = index.search(query, k=5)

print(f"Top-5 indices: {indices[0]}")
print(f"Top-5 distances: {distances[0]}")
```

---

### CQ3. FAISS IVF Index with Training

```python
import faiss
import numpy as np
import time

d = 384
n = 100000
nlist = 100
np.random.seed(42)
data = np.random.randn(n, d).astype("float32")
query = np.random.randn(5, d).astype("float32")

# Flat index baseline
flat_index = faiss.IndexFlatL2(d)
flat_index.add(data)
t0 = time.time()
D_flat, I_flat = flat_index.search(query, k=10)
flat_time = time.time() - t0
print(f"Flat search time: {flat_time:.4f}s")

# IVF index
quantizer = faiss.IndexFlatL2(d)
ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist)
ivf_index.train(data)
ivf_index.add(data)

for nprobe in [1, 10, 50]:
    ivf_index.nprobe = nprobe
    t0 = time.time()
    D_ivf, I_ivf = ivf_index.search(query, k=10)
    ivf_time = time.time() - t0

    # Compute recall against exact results
    recall = np.mean([len(set(I_ivf[i]) & set(I_flat[i])) / 10 for i in range(len(query))])
    print(f"nprobe={nprobe:2d}: time={ivf_time:.4f}s, recall@10={recall:.2f}")
```

---

### CQ4. ChromaDB Collection Management

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("articles")

documents = [
    "Machine learning transforms healthcare diagnostics",
    "New quantum computing breakthrough at MIT",
    "Stock market reaches all-time high amid tech rally",
    "Climate change accelerates Arctic ice melting",
    "SpaceX launches new satellite constellation",
    "CRISPR gene therapy shows promise for rare diseases",
    "Artificial intelligence in autonomous vehicles",
    "Global supply chain disruptions continue",
    "Renewable energy investment surges worldwide",
    "Cybersecurity threats increase for financial institutions",
]
categories = ["tech", "tech", "finance", "science", "tech",
              "health", "tech", "business", "energy", "finance"]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"category": c, "year": 2024} for c in categories],
)

results = collection.query(query_texts=["machine learning advances"], n_results=3)
print("Query results:", results["documents"])

filtered = collection.query(
    query_texts=["technology news"],
    n_results=5,
    where={"category": "tech"},
)
print("Filtered results:", filtered["documents"])

collection.delete(ids=["doc_0"])
print(f"Count after deletion: {collection.count()}")
```

---

### CQ5. Build a Simple RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
import chromadb

documents = [
    "Python was created by Guido van Rossum and released in 1991.",
    "PyTorch is an open-source machine learning framework by Meta.",
    "FAISS is a library for efficient similarity search by Facebook Research.",
    "ChromaDB is an open-source vector database for AI applications.",
    "Transformers were introduced in the paper Attention Is All You Need in 2017.",
]

client = chromadb.Client()
collection = client.create_collection("knowledge_base")
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
)

def rag_query(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    context = "\n".join(results["documents"][0])

    prompt = f"""Answer the question based ONLY on the following context.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
    print(prompt)
    # In production, send `prompt` to an LLM (e.g., OpenAI API)
    return prompt, results

prompt, results = rag_query("Who created Python?")
print(f"\nRetrieved {len(results['documents'][0])} documents")
print(f"Top result: {results['documents'][0][0]}")
```

---

### CQ6. Semantic Search with Re-Ranking

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

corpus = [
    "Python is a programming language known for its simplicity.",
    "Machine learning automates analytical model building.",
    "Deep learning is part of machine learning using neural networks.",
    "Natural language processing enables computers to understand text.",
    "FAISS is used for efficient similarity search in large datasets.",
    "Data science combines statistics, computing, and domain expertise.",
    "Neural networks are inspired by the human brain structure.",
    "Transfer learning reuses pretrained models for new tasks.",
]

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

corpus_embs = bi_encoder.encode(corpus, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(corpus_embs)

index = faiss.IndexFlatIP(corpus_embs.shape[1])
index.add(corpus_embs)

query = "How do neural networks learn?"
query_emb = bi_encoder.encode([query], convert_to_numpy=True).astype("float32")
faiss.normalize_L2(query_emb)
scores, indices = index.search(query_emb, k=5)

candidates = [corpus[i] for i in indices[0]]
print("Bi-encoder ranking:")
for i, (doc, score) in enumerate(zip(candidates, scores[0])):
    print(f"  {i+1}. [{score:.4f}] {doc}")

pairs = [[query, doc] for doc in candidates]
ce_scores = cross_encoder.predict(pairs)
reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)

print("\nCross-encoder re-ranking:")
for i, (doc, score) in enumerate(reranked):
    print(f"  {i+1}. [{score:.4f}] {doc}")
```

---

### CQ7. Document Chunking for RAG

```python
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=500, overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build overlap: keep trailing sentences that fit within overlap size
            overlap_chunk = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [{"text": c, "index": i, "length": len(c)} for i, c in enumerate(chunks)]

text = "This is the first sentence. Here is the second one. " * 20
result = chunk_text(text, chunk_size=200, overlap=50)
for chunk in result:
    print(f"Chunk {chunk['index']}: {chunk['length']} chars - {chunk['text'][:60]}...")
```

---

### CQ8. Evaluate Retrieval Quality

```python
import numpy as np

def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(relevant) & set(retrieved_k)) / k

def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(relevant) & set(retrieved_k)) / len(relevant) if relevant else 0.0

def mean_reciprocal_rank(relevant, retrieved):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / np.log2(i + 2)

    ideal_rels = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

# Example
relevant = {1, 3, 5, 7}
retrieved = [2, 1, 5, 8, 3, 9, 7, 10]

print(f"Precision@5: {precision_at_k(relevant, retrieved, 5):.4f}")
print(f"Recall@5:    {recall_at_k(relevant, retrieved, 5):.4f}")
print(f"MRR:         {mean_reciprocal_rank(relevant, retrieved):.4f}")
print(f"nDCG@5:      {ndcg_at_k(relevant, retrieved, 5):.4f}")
```

---

### CQ9. FinBERT Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

labels = ["positive", "negative", "neutral"]

headlines = [
    "Company reports record quarterly earnings beating all estimates",
    "Stock plunges 20% after failed merger announcement",
    "Federal Reserve keeps interest rates unchanged",
    "New CEO announces major restructuring plan with layoffs",
    "Tech sector shows moderate growth in Q3 results",
]

for headline in headlines:
    inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_idx].item()
    print(f"[{labels[pred_idx]:>8s}] ({confidence:.2%}) {headline}")
```

---

### CQ10. Persistent ChromaDB with Custom Embeddings

```python
import chromadb
from chromadb.utils import embedding_functions
import os

PERSIST_DIR = "./chroma_persistent_db"

sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=sbert_ef,
)

if collection.count() == 0:
    docs = [f"Document {i}: This is sample content about topic {i % 10}." for i in range(100)]
    collection.add(
        documents=docs,
        ids=[f"doc_{i}" for i in range(100)],
        metadatas=[{"topic": i % 10} for i in range(100)],
    )
    print(f"Added {collection.count()} documents.")

# Demonstrate persistence
del client
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection("documents", embedding_function=sbert_ef)
print(f"After reopening: {collection.count()} documents")

results = collection.query(query_texts=["content about topic 5"], n_results=3)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{dist:.4f}] {doc}")
```
