# Project 8: Semantic Search Engine

End-to-end semantic search over MS MARCO passages using sentence embeddings and
vector databases (FAISS + ChromaDB).

## Architecture

```
Query Text
    |
    v
Sentence Encoder (all-MiniLM-L6-v2, 384-dim)
    |
    v
Vector Search (FAISS IVF/PQ or ChromaDB)
    |
    v
Ranked Passages + Similarity Scores
```

## Project Structure

```
08-semantic-search-engine/
├── app.py                  # FastAPI REST API (/search endpoint)
├── streamlit_app.py        # Streamlit search UI
├── train.py                # Build the search index
├── predict.py              # CLI search tool
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt
├── dataset_link.md
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Load MS MARCO passages
│   │   ├── data_validation.py      # Validate passage data
│   │   ├── data_transformation.py  # Generate embeddings
│   │   ├── model_trainer.py        # Build FAISS/ChromaDB index
│   │   └── model_evaluation.py     # MRR, NDCG, Recall evaluation
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrate index building
│   │   └── prediction_pipeline.py  # Query encoding + search
│   ├── utils/
│   │   └── common.py               # Shared utilities
│   └── config/
│       └── configuration.py        # Centralized configuration
├── logs/
└── notebooks/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Search Index

```bash
# Downloads a sample of MS MARCO, generates embeddings, builds FAISS index
python train.py

# With custom settings
python train.py --num-passages 50000 --batch-size 512 --index-type ivfpq
```

This will:
- Download MS MARCO passage collection (or use cached version)
- Sample passages for the demo (default: 100,000)
- Generate 384-dim embeddings with all-MiniLM-L6-v2
- Build a FAISS index and optionally a ChromaDB collection
- Save everything to `artifacts/`

### 3. Search via CLI

```bash
python predict.py "what is machine learning" --top-k 5
python predict.py "how does photosynthesis work" --top-k 10
```

### 4. Launch the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

API usage:

```bash
# Search
curl "http://localhost:8000/search?query=what+is+python&top_k=5"

# Health check
curl "http://localhost:8000/health"
```

**Swagger docs:** http://localhost:8000/docs

### 5. Launch the Streamlit UI

```bash
streamlit run streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser. Features:
- Text input for natural-language queries
- Adjustable top-k slider (1-50)
- Results displayed with passage text, similarity score, and passage ID
- Search latency displayed per query

## Docker Deployment

### Build

```bash
docker build -t semantic-search .
```

### Run (FastAPI)

```bash
docker run -p 8000:8000 semantic-search
```

### Run (Streamlit)

```bash
docker run -p 8501:8501 semantic-search streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Compose (both services)

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
  ui:
    build: .
    command: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    volumes:
      - ./artifacts:/app/artifacts
    depends_on:
      - api
```

## Evaluation Metrics

| Metric     | Description                                    |
|------------|------------------------------------------------|
| MRR@10     | Mean Reciprocal Rank at 10                     |
| Recall@k   | Fraction of relevant docs retrieved in top-k   |
| NDCG@10    | Normalized Discounted Cumulative Gain at 10    |
| Latency    | p50, p95, p99 query latency in milliseconds    |

Run evaluation:

```bash
python -c "from src.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run(evaluate=True)"
```

## Configuration

All paths and hyperparameters are centralized in `src/config/configuration.py`.
Key settings:

| Setting             | Default                          | Description                   |
|---------------------|----------------------------------|-------------------------------|
| MODEL_NAME          | all-MiniLM-L6-v2                 | Sentence transformer model    |
| EMBEDDING_DIM       | 384                              | Embedding dimensionality      |
| NUM_PASSAGES        | 100000                           | Passages to index (demo)      |
| INDEX_TYPE          | flat                             | FAISS index type              |
| NLIST               | 100                              | IVF clusters                  |
| M_PQ                | 48                               | PQ sub-quantizers             |
| NPROBE              | 10                               | IVF probes at search time     |
| CHROMA_COLLECTION   | msmarco_passages                 | ChromaDB collection name      |

## Model Details

**all-MiniLM-L6-v2** (from sentence-transformers):
- 384-dimensional dense embeddings
- 22M parameters
- Trained on 1B+ sentence pairs
- Excellent speed/quality tradeoff for semantic search
- Max sequence length: 256 tokens

## Dataset

MS MARCO Passage Ranking — see [dataset_link.md](dataset_link.md) for full details.
