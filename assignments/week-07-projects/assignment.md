# Week 7: NLP Projects - Semantic Search and RAG

## Topics Covered
Sentence Embeddings, SBERT, FAISS, ChromaDB, Retrieval-Augmented Generation (RAG), FinBERT

---

## Part 1: Conceptual Questions (10 Questions)

### Q1. Sentence Embeddings vs. Word Embeddings
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Explain why averaging word embeddings (e.g., Word2Vec) to create sentence embeddings is suboptimal. How do models like Sentence-BERT (SBERT) produce better sentence-level representations? What training objective does SBERT use?

---

### Q2. Cosine Similarity for Semantic Search
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Why is cosine similarity preferred over Euclidean distance for comparing sentence embeddings? In what scenario might Euclidean distance give misleading results even when two sentences are semantically similar?

---

### Q3. FAISS Fundamentals
**Difficulty:** Easy | **Estimated Time:** 5 minutes

What is FAISS (Facebook AI Similarity Search)? Explain the difference between `IndexFlatL2` (exact search) and `IndexIVFFlat` (approximate search). When would you choose one over the other?

---

### Q4. Approximate Nearest Neighbors
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Explain the Inverted File Index (IVF) approach used in FAISS. How does it partition the vector space using Voronoi cells? What is the `nprobe` parameter and how does it affect the speed-accuracy trade-off?

---

### Q5. ChromaDB Architecture
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Describe ChromaDB's architecture as a vector database. How does it differ from FAISS? What advantages does ChromaDB offer for metadata filtering, persistence, and document management that raw FAISS does not?

---

### Q6. RAG Architecture
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Draw and explain the Retrieval-Augmented Generation (RAG) pipeline. What are the three main stages (indexing, retrieval, generation)? Why does RAG reduce hallucination compared to using an LLM alone?

---

### Q7. Chunking Strategies for RAG
**Difficulty:** Medium | **Estimated Time:** 10 minutes

When building a RAG system, documents must be split into chunks before embedding. Compare these chunking strategies: (a) fixed-size with overlap, (b) sentence-based, (c) semantic chunking. What are the trade-offs of chunk size on retrieval quality?

---

### Q8. Embedding Model Selection
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Compare these embedding models for semantic search: (a) `all-MiniLM-L6-v2` (22M params, 384 dims), (b) `all-mpnet-base-v2` (109M params, 768 dims), (c) OpenAI `text-embedding-ada-002` (1536 dims). Discuss trade-offs between dimensionality, inference speed, and retrieval quality. How do you benchmark embedding models for a specific domain?

---

### Q9. RAG Failure Modes
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Describe three common failure modes of RAG systems: (a) retrieval misses relevant context, (b) retrieved context is relevant but the LLM ignores it, (c) the LLM over-relies on irrelevant retrieved content. For each, propose a mitigation strategy.

---

### Q10. Domain-Specific Fine-Tuning
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Explain why FinBERT outperforms general-purpose BERT for financial sentiment analysis. What is domain-adaptive pre-training? Describe the process of further pre-training BERT on a domain-specific corpus before fine-tuning on a labeled dataset. What risks does this introduce?

---

## Part 2: Coding Questions (10 Questions)

### CQ1. Encode Sentences with SBERT
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Use the `sentence-transformers` library to encode sentences and compute pairwise cosine similarity.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "The match was exciting to watch.",
]

# TODO: Encode all sentences
# TODO: Compute pairwise cosine similarity matrix
# TODO: Find the most similar pair of sentences
```

---

### CQ2. Build a FAISS Flat Index
**Difficulty:** Easy | **Estimated Time:** 15 minutes

Create a FAISS index, add document embeddings, and perform a nearest-neighbor query.

```python
import faiss
import numpy as np

# TODO: Generate or load 1000 document embeddings of dimension 384
# TODO: Create an IndexFlatL2 index
# TODO: Add all embeddings to the index
# TODO: Query with a new embedding and retrieve top-5 results
# TODO: Print distances and indices
```

---

### CQ3. FAISS IVF Index with Training
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Build an IVF index that requires training, and compare search speed with a flat index.

```python
import faiss
import numpy as np
import time

d = 384
n = 100000
nlist = 100  # Number of Voronoi cells

# TODO: Generate random data for benchmarking
# TODO: Create IndexIVFFlat with a quantizer
# TODO: Train the index on the data
# TODO: Add vectors to the index
# TODO: Search with nprobe=1, 10, 50 and compare speed and recall
```

---

### CQ4. ChromaDB Collection Management
**Difficulty:** Easy | **Estimated Time:** 15 minutes

Create a ChromaDB collection, add documents with metadata, and query with filters.

```python
import chromadb

client = chromadb.Client()

# TODO: Create a collection named "articles"
# TODO: Add 10 documents with metadata (category, date, source)
# TODO: Query for similar documents to "machine learning advances"
# TODO: Query with metadata filter: category == "technology"
# TODO: Delete a document and verify the count
```

---

### CQ5. Build a Simple RAG Pipeline
**Difficulty:** Hard | **Estimated Time:** 30 minutes

Build a RAG pipeline using SBERT for retrieval and a prompt template for generation.

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Knowledge base documents
documents = [
    "Python was created by Guido van Rossum and released in 1991.",
    "PyTorch is an open-source machine learning framework by Meta.",
    "FAISS is a library for efficient similarity search by Facebook Research.",
    "ChromaDB is an open-source vector database for AI applications.",
    "Transformers were introduced in the paper Attention Is All You Need in 2017.",
]

# TODO: Embed and store documents in ChromaDB
# TODO: Given a query, retrieve top-3 relevant documents
# TODO: Format a prompt with retrieved context
# TODO: (Simulated) generate an answer using the context
# The prompt template should include: context, question, instruction to answer only from context
```

---

### CQ6. Semantic Search with Re-Ranking
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Implement a two-stage retrieval pipeline: fast FAISS retrieval followed by cross-encoder re-ranking.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

# TODO: Use bi-encoder (all-MiniLM-L6-v2) to encode corpus and build FAISS index
# TODO: Retrieve top-20 candidates with FAISS
# TODO: Re-rank top-20 using cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
# TODO: Return top-5 after re-ranking
# TODO: Compare rankings before and after re-ranking
```

---

### CQ7. Document Chunking for RAG
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Implement a text chunking function with configurable size and overlap.

```python
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    Respect sentence boundaries when possible.
    """
    # TODO: Split text into sentences
    # TODO: Group sentences into chunks respecting chunk_size
    # TODO: Add overlap between consecutive chunks
    # TODO: Return list of chunk strings with metadata (start_idx, end_idx)
    pass
```

---

### CQ8. Evaluate Retrieval Quality
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Compute retrieval evaluation metrics: Precision@K, Recall@K, MRR, and nDCG.

```python
import numpy as np

def precision_at_k(relevant, retrieved, k):
    """Fraction of retrieved docs in top-k that are relevant."""
    # TODO: Implement
    pass

def recall_at_k(relevant, retrieved, k):
    """Fraction of relevant docs found in top-k."""
    # TODO: Implement
    pass

def mean_reciprocal_rank(relevant, retrieved):
    """1 / rank of the first relevant document."""
    # TODO: Implement
    pass

def ndcg_at_k(relevant, retrieved, k):
    """Normalized Discounted Cumulative Gain."""
    # TODO: Implement
    pass
```

---

### CQ9. FinBERT Sentiment Analysis
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Use FinBERT to classify financial news headlines as positive, negative, or neutral.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# TODO: Load ProsusAI/finbert model and tokenizer
# TODO: Classify these headlines:
headlines = [
    "Company reports record quarterly earnings beating all estimates",
    "Stock plunges 20% after failed merger announcement",
    "Federal Reserve keeps interest rates unchanged",
    "New CEO announces major restructuring plan with layoffs",
    "Tech sector shows moderate growth in Q3 results",
]
# TODO: Print predicted label and confidence for each headline
```

---

### CQ10. Persistent ChromaDB with Custom Embeddings
**Difficulty:** Hard | **Estimated Time:** 25 minutes

Create a persistent ChromaDB collection using a custom SBERT embedding function.

```python
import chromadb
from chromadb.utils import embedding_functions

# TODO: Create a persistent client (data stored to disk)
# TODO: Configure SBERT embedding function (all-MiniLM-L6-v2)
# TODO: Create collection with the custom embedding function
# TODO: Add 100 documents from a dataset
# TODO: Demonstrate persistence: close and reopen the client, verify data exists
# TODO: Perform a query and show results with distances
```

---

## Part 3: Case Studies (2 Studies)

### Case Study 1: Enterprise Knowledge Base
**Scenario:** A company with 50,000 internal documents (policies, manuals, reports in PDF and DOCX formats) wants to build an AI-powered knowledge base. Employees should be able to ask natural language questions and get accurate answers with source citations.

**Tasks:**
1. Design the document ingestion pipeline: extraction, cleaning, chunking.
2. Choose an embedding model and vector store. Justify your choices for this scale.
3. Design the RAG pipeline. How do you ensure answers are grounded in retrieved context?
4. How would you handle document updates and versioning in the vector store?
5. Propose an evaluation strategy: how do you measure retrieval quality and answer accuracy without extensive labeled data?

---

### Case Study 2: Semantic Code Search Engine
**Scenario:** A large engineering organization wants to search across 10 million code snippets (Python, Java, JavaScript) using natural language queries like "function that sorts a list using quicksort" or "retry logic with exponential backoff."

**Tasks:**
1. What embedding model would you use for code? Compare CodeBERT, UniXcoder, and general text embeddings.
2. How would you handle the structure of code differently from natural text during chunking?
3. Design the indexing pipeline: parse code into functions/classes, extract docstrings, embed.
4. How would you handle multi-language search (query in English, code in any language)?
5. Discuss evaluation: what metrics would you use, and how would you collect relevance judgments?

---

## Part 4: Practical Assignments (3 Assignments)

### Assignment 1: Build Semantic Search with SBERT + FAISS
**Objective:** Build a complete semantic search system for a dataset of news articles.

**Dataset:** [AG News](https://huggingface.co/datasets/ag_news) (120K news articles in 4 categories)

**Requirements:**
1. Load the AG News dataset and use the test set (7,600 articles) for the search corpus.
2. Encode all articles using `all-MiniLM-L6-v2` from `sentence-transformers`.
3. Build a FAISS index (`IndexIVFFlat` with `nlist=50`) and add all embeddings.
4. Implement a search function that takes a natural language query and returns top-10 results.
5. Build a simple evaluation: manually write 10 queries with expected categories, measure if top results are from the correct category.
6. Compare search times and quality between `IndexFlatL2` and `IndexIVFFlat` (nprobe=1, 5, 10).
7. Visualize the embedding space using UMAP, colored by category.

**Hints:**
- Batch encoding with `model.encode(texts, batch_size=64, show_progress_bar=True)`.
- Normalize embeddings for cosine similarity: `faiss.normalize_L2(embeddings)`.
- Use `IndexFlatIP` (inner product) after normalization for cosine similarity search.

**Deliverables:** Jupyter notebook, UMAP visualization, search quality comparison table.

---

### Assignment 2: Implement RAG Pipeline with ChromaDB
**Objective:** Build a question-answering RAG pipeline over a document collection.

**Dataset:** [SQuAD 2.0](https://huggingface.co/datasets/squad_v2) (use the context paragraphs as the knowledge base)

**Requirements:**
1. Extract unique context paragraphs from SQuAD 2.0 (approximately 19K paragraphs).
2. Chunk long paragraphs (those over 500 characters) with 100-character overlap.
3. Store all chunks in ChromaDB with metadata (title, paragraph_id).
4. Build a retrieval function: given a question, retrieve top-5 relevant chunks.
5. Build a prompt template that includes retrieved context and the question.
6. Evaluate retrieval by checking if the paragraph containing the gold answer is in the top-5.
7. Report Recall@1, Recall@5, and Mean Reciprocal Rank on 500 randomly sampled questions.

**Hints:**
- ChromaDB's default embedding function uses `all-MiniLM-L6-v2`.
- Use `collection.query(query_texts=[question], n_results=5)` for retrieval.
- SQuAD provides the answer span and the gold context -- use this for evaluation.

**Deliverables:** Jupyter notebook, retrieval metrics report, example queries with retrieved context.

---

### Assignment 3: Financial Sentiment Analysis with FinBERT
**Objective:** Analyze sentiment in financial news and correlate with stock price movements.

**Dataset:** [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) (4,845 sentences, 3 sentiment classes)

**Requirements:**
1. Load the Financial PhraseBank dataset (use "sentences_allagree" subset for high-quality labels).
2. Evaluate the pre-trained `ProsusAI/finbert` model on this dataset (zero-shot).
3. Fine-tune FinBERT on an 80/20 train/test split for 3 epochs (lr=2e-5).
4. Compare zero-shot vs. fine-tuned performance (accuracy, per-class F1).
5. Analyze errors: which sentiment class is hardest to predict? Show confusion matrix.
6. Bonus: collect 20 recent financial headlines (from any source), run FinBERT on them, and qualitatively assess the predictions.

**Hints:**
- FinBERT outputs labels: "positive", "negative", "neutral".
- Financial PhraseBank labels map to the same three classes.
- Use `Trainer` API with `compute_metrics` for streamlined training.
- The "neutral" class is typically the most challenging due to ambiguity.

**Deliverables:** Jupyter notebook, confusion matrices (zero-shot and fine-tuned), performance comparison table.
