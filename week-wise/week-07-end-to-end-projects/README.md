# Week 7: End-to-End Projects

*Days 43-49: Sentence embeddings, vector databases, RAG, and production projects*

---

## 1. Sentence Embeddings (Sentence-BERT)

### What is it
Sentence embeddings map entire sentences or paragraphs to fixed-size dense vectors, enabling semantic similarity comparison. Sentence-BERT (SBERT) fine-tunes BERT using a siamese/triplet network to produce embeddings where semantically similar sentences are close in vector space.

### Why it matters
Standard BERT requires passing both sentences through the model simultaneously for comparison (cross-encoder), which is O(n^2) for n sentences. Sentence-BERT produces independent embeddings, enabling fast similarity search over millions of documents using cosine similarity.

### Python code
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the weather today?",
    "Steps to recover account access",
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (4, 384)

# Compute pairwise similarity
sim_matrix = cosine_similarity(embeddings)
print("Similarity matrix:")
for i, s in enumerate(sentences):
    for j, t in enumerate(sentences):
        if i < j:
            print(f"  '{s[:30]}...' vs '{t[:30]}...': {sim_matrix[i][j]:.3f}")
```

### Common mistakes
- Using BERT [CLS] token directly as sentence embedding (poor quality without SBERT fine-tuning)
- Not normalizing embeddings before cosine similarity (most SBERT models already L2-normalize)
- Choosing embedding model without considering domain (domain-specific models outperform generic ones)

### Interview questions
- **Q: What is the difference between a bi-encoder and cross-encoder?** A: Bi-encoder produces independent embeddings for fast retrieval; cross-encoder processes both inputs together for higher accuracy but is slower.
- **Q: Why not just average BERT token embeddings?** A: Averaging produces poor sentence representations because BERT was not trained for this; SBERT is specifically fine-tuned with contrastive loss.
- **Q: How do you choose an embedding model?** A: Consider domain match, embedding dimension (latency vs quality trade-off), and benchmark scores on MTEB leaderboard.

---

## 2. Vector Databases (FAISS, ChromaDB)

### What is it
Vector databases store and index high-dimensional vectors for fast approximate nearest neighbor (ANN) search. FAISS (Facebook AI Similarity Search) provides low-level, highly optimized index types. ChromaDB provides a higher-level API with metadata filtering and persistence.

### Why it matters
Brute-force cosine similarity over 1M vectors with 384 dimensions takes seconds. FAISS with an IVF index reduces this to milliseconds by partitioning the vector space and only searching relevant partitions.

### Key concepts: FAISS index types
- **IndexFlatIP**: Exact inner product search (no approximation, slow for large N)
- **IndexIVFFlat**: Inverted file index — partitions space into nlist clusters, searches nprobe nearest clusters
- **IndexIVFPQ**: Product quantization compresses vectors for memory efficiency

### Python code
```python
import faiss
import numpy as np

# Generate sample data
d = 384  # embedding dimension
n = 10000  # number of vectors
embeddings = np.random.randn(n, d).astype("float32")
faiss.normalize_L2(embeddings)  # for cosine similarity via inner product

# Build IVF index
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # search 10 nearest clusters

# Search
query = np.random.randn(1, d).astype("float32")
faiss.normalize_L2(query)
distances, indices = index.search(query, k=5)
print(f"Top 5 indices: {indices[0]}")
print(f"Top 5 scores: {distances[0]}")
```

### ChromaDB example
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

collection.add(
    documents=["ML is great", "NLP uses transformers", "Python is popular"],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[{"topic": "ml"}, {"topic": "nlp"}, {"topic": "programming"}]
)

results = collection.query(query_texts=["deep learning models"], n_results=2)
print(results["documents"])
```

### Interview questions
- **Q: When would you use FAISS vs ChromaDB?** A: FAISS for maximum performance and custom index tuning at scale; ChromaDB for simpler API, metadata filtering, and persistence out-of-the-box.
- **Q: What is the trade-off in IVF nprobe?** A: Higher nprobe = higher recall but slower search. Typical values: nprobe = sqrt(nlist).
- **Q: How does product quantization work?** A: Splits vectors into sub-vectors, quantizes each independently, reducing memory from O(n*d) to O(n*m*log(k)) where m is sub-vector count.

---

## 3. RAG Architecture

### What is it
Retrieval-Augmented Generation (RAG) combines a retriever (searches a knowledge base for relevant documents) with a generator (LLM that produces answers using retrieved context). This grounds LLM responses in factual data and reduces hallucination.

### Why it matters
LLMs have knowledge cutoffs and cannot access proprietary data. RAG lets you build chatbots that answer questions about your company's documentation, legal contracts, or internal knowledge bases without fine-tuning the LLM.

### Architecture
```
User Query → Embedding Model → Vector Search → Top-K Documents
                                                      ↓
                                        Prompt Template + Context
                                                      ↓
                                              LLM Generation
                                                      ↓
                                                   Answer
```

### Key concepts
- **Chunking**: Split documents into 256-512 token chunks with 50-100 token overlap
- **Retrieval**: Encode query, search vector DB for top-k similar chunks
- **Prompt template**: "Given the following context: {chunks}\n\nAnswer the question: {query}"
- **Evaluation**: Retrieval accuracy (is the answer in retrieved docs?) + generation quality (ROUGE, human eval)

### Python code
```python
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Build knowledge base
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("kb")

documents = [
    "Python was created by Guido van Rossum in 1991.",
    "BERT was introduced by Google in 2018.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Docker containers package applications with their dependencies.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 2. Retrieve relevant context
query = "Who created Python?"
results = collection.query(query_texts=[query], n_results=2)
context = "\n".join(results["documents"][0])

# 3. Generate answer (using LLM — simplified here)
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
print(prompt)
# In practice: response = llm.generate(prompt)
```

### Common mistakes
- Chunk size too large (LLM context window overflow) or too small (missing context)
- Not handling cases where retrieved documents are irrelevant (add a relevance threshold)
- Evaluating only generation quality without measuring retrieval accuracy

### Interview questions
- **Q: How do you handle documents that are too long for the LLM context window?** A: Chunk documents with overlap, retrieve only top-k relevant chunks, and use map-reduce or iterative refinement for long answers.
- **Q: How do you evaluate a RAG system?** A: Separately evaluate retrieval (recall@k, MRR) and generation (ROUGE, faithfulness, human eval).
- **Q: What is the difference between RAG and fine-tuning?** A: RAG retrieves external knowledge at inference time (dynamic, no retraining); fine-tuning bakes knowledge into model weights (static, requires retraining for updates).

---

## 4. Advanced Fine-tuning (LoRA, QLoRA, PEFT)

### What is it
Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA (Low-Rank Adaptation) fine-tune only a small number of additional parameters instead of the full model. LoRA adds low-rank decomposition matrices to attention layers, reducing trainable parameters by 10-100x.

### Why it matters
Fine-tuning a 7B parameter model requires 28GB+ of GPU memory. LoRA reduces this to 4-8GB by freezing original weights and training only small adapter matrices (rank 8-64). QLoRA further reduces memory by quantizing the base model to 4-bit.

### Math: LoRA
Original weight update: W_new = W + delta_W

LoRA decomposes: delta_W = B * A where A is (d x r), B is (r x d), and r << d

If d=4096 and r=8: instead of 4096*4096 = 16.7M params, only 2 * 4096 * 8 = 65K params.

### Python code
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,              # rank
    lora_alpha=32,     # scaling factor
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # which layers to adapt
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 296,450 || all params: 67,251,714 || trainable%: 0.44%
```

### Interview questions
- **Q: Why does LoRA work despite training so few parameters?** A: The weight updates during fine-tuning are typically low-rank; LoRA exploits this by constraining updates to a low-rank subspace.
- **Q: What is the difference between LoRA and QLoRA?** A: QLoRA quantizes the base model to 4-bit (reducing memory 4x) while keeping LoRA adapters in higher precision.
- **Q: How do you choose the rank r?** A: Start with r=8 for classification, r=16-64 for generation. Higher rank = more capacity but more parameters.

---

## Week 7 Assignment

See [assignments/week-07-projects/](../../assignments/week-07-projects/) for the full assignment.
