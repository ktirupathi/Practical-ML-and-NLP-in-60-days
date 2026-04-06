# MS MARCO Dataset — Semantic Search Engine

## Dataset Overview

**Name:** MS MARCO (Microsoft Machine Reading Comprehension)
**Source:** https://microsoft.github.io/msmarco/
**Direct Downloads:**
- Passages: https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
- Queries (train): https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
- QRELs (relevance labels): https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv

## Schema

### Passages (collection.tsv)
| Column    | Type   | Description                        |
|-----------|--------|------------------------------------|
| pid       | int    | Unique passage identifier          |
| passage   | string | The passage text (avg ~60 tokens)  |

- Total: ~8.8 million passages
- Format: TSV (tab-separated), no header row

### Queries (queries.train.tsv)
| Column   | Type   | Description              |
|----------|--------|--------------------------|
| qid      | int    | Unique query identifier  |
| query    | string | Natural language query   |

- Total: ~1 million queries
- Format: TSV, no header row

### Relevance Labels (qrels.train.tsv)
| Column    | Type | Description                                  |
|-----------|------|----------------------------------------------|
| qid       | int  | Query identifier                             |
| 0         | int  | Always 0 (placeholder for TREC format)       |
| pid       | int  | Relevant passage identifier                  |
| relevance | int  | Relevance score (1 = relevant in MS MARCO)   |

## Why This Dataset Is Useful

1. **Scale:** 8.8M passages provide a realistic information retrieval benchmark, unlike
   toy datasets that don't expose real-world performance bottlenecks.
2. **Real Queries:** Queries come from Bing search logs — they reflect how people
   actually phrase questions (typos, ambiguity, varied intent).
3. **Sparse Labels:** Each query typically has only 1 relevant passage labeled. This
   makes evaluation challenging and mirrors production search scenarios where relevance
   judgments are expensive to obtain.
4. **Industry Standard:** MS MARCO is the de-facto benchmark for passage retrieval
   research. Results are directly comparable to published work.
5. **Embedding-Friendly:** Passages are short enough to embed with standard models
   (e.g., all-MiniLM-L6-v2 with 256-token max) without truncation issues.

## Preprocessing Steps

1. **Download & Extract:** Download the tarball and extract the TSV files.
2. **Sample for Demo:** For local development, we sample 100K passages (configurable)
   to keep index building fast and memory reasonable.
3. **Text Cleaning:**
   - Strip leading/trailing whitespace
   - Remove passages shorter than 10 characters (noise)
   - Remove exact duplicate passages
   - Normalize unicode (NFKC)
4. **Embedding Generation:**
   - Encode passages with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
   - Batch encoding with GPU if available, CPU fallback
5. **Index Building:**
   - FAISS Flat (exact search) for small collections
   - FAISS IVF+PQ for large-scale approximate search
   - ChromaDB as a managed alternative with built-in persistence
6. **Query Processing:**
   - Same text cleaning as passages
   - Encode with the same model
   - Search the FAISS/ChromaDB index and return top-k results
