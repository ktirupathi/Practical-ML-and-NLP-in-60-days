# Project 9: Knowledge Base Chatbot with RAG

A production-ready Retrieval-Augmented Generation (RAG) chatbot that builds a knowledge base from SQuAD 2.0 Wikipedia passages and answers questions using retrieved context combined with a generative language model.

## RAG Architecture

```
User Question
      |
      v
+------------------+
| Query Embedding  |  (sentence-transformers: all-MiniLM-L6-v2)
+------------------+
      |
      v
+------------------+
| Vector Store     |  (ChromaDB with cosine similarity)
| Retrieval (top-k)|
+------------------+
      |
      v
+------------------+
| Context Assembly |  Top-k retrieved chunks concatenated
+------------------+
      |
      v
+------------------+
| LLM Generator   |  (google/flan-t5-base via HuggingFace)
| Prompt:          |
| "Context: ...    |
|  Question: ...   |
|  Answer: "       |
+------------------+
      |
      v
  Generated Answer + Source Passages + Confidence Score
```

### How RAG Works

1. **Indexing Phase (offline):**
   - Load SQuAD 2.0 from HuggingFace datasets
   - Extract and deduplicate Wikipedia context paragraphs
   - Chunk paragraphs into smaller segments (256 tokens with 50-token overlap)
   - Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Store embeddings + metadata in ChromaDB

2. **Query Phase (online):**
   - Embed the user question with the same sentence-transformer
   - Retrieve top-k most similar chunks from ChromaDB
   - Construct a prompt with retrieved context + question
   - Generate answer using `google/flan-t5-base`
   - Return answer, source passages, and confidence score

### Why RAG Over Pure LLM?

- **Grounded answers:** Responses are backed by specific source documents
- **Reduced hallucination:** The model is constrained to information in retrieved passages
- **Updatable knowledge:** Add or remove documents without retraining the LLM
- **Transparent:** Users can inspect which source passages informed the answer
- **Efficient:** No need for expensive fine-tuning; works with smaller models

## Project Structure

```
09-knowledge-base-chatbot-rag/
|-- dataset_link.md          # Dataset documentation
|-- README.md                # This file
|-- requirements.txt         # Python dependencies
|-- train.py                 # Build the knowledge base
|-- predict.py               # CLI question-answering tool
|-- app.py                   # FastAPI REST API
|-- streamlit_app.py         # Streamlit chatbot UI
|-- src/
|   |-- __init__.py
|   |-- components/
|   |   |-- __init__.py
|   |   |-- data_ingestion.py       # Load SQuAD 2.0 dataset
|   |   |-- data_validation.py      # Validate context paragraphs
|   |   |-- data_transformation.py  # Chunk text, embed, build vector store
|   |   |-- model_trainer.py        # Set up RAG retriever + generator
|   |   |-- model_evaluation.py     # Evaluate RAG with ROUGE/BLEU/EM/F1
|   |-- pipeline/
|   |   |-- __init__.py
|   |   |-- training_pipeline.py    # Orchestrate knowledge base construction
|   |   |-- prediction_pipeline.py  # Question -> retrieval -> generation
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- common.py               # Shared utilities
|   |-- config/
|       |-- __init__.py
|       |-- configuration.py        # Centralized configuration
|-- logs/
|   |-- .gitkeep
|-- notebooks/
    |-- .gitkeep
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Knowledge Base

```bash
python train.py
```

This will:
- Download SQuAD 2.0 from HuggingFace
- Extract ~20K unique Wikipedia context paragraphs
- Validate and clean the data
- Chunk paragraphs and generate embeddings
- Build the ChromaDB vector store
- Set up and evaluate the RAG pipeline

Artifacts are saved to `artifacts/` by default.

### 3. Ask Questions (CLI)

```bash
python predict.py --question "What is the capital of France?"
python predict.py  # Interactive mode
```

### 4. Run the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive API docs.

**Endpoints:**
- `POST /ask` — Submit a question, receive answer + source passages + confidence
- `GET /health` — Health check

### 5. Run the Streamlit Chatbot

```bash
streamlit run streamlit_app.py
```

Features:
- Chat-style conversational UI
- Source document display for each answer
- Confidence meter visualization
- Chat history with export option
- Adjustable retrieval parameters in the sidebar

## Configuration

Edit `src/config/configuration.py` to adjust:

| Parameter              | Default                              | Description                          |
|------------------------|--------------------------------------|--------------------------------------|
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`                  | Sentence transformer for embeddings  |
| `GENERATOR_MODEL`      | `google/flan-t5-base`               | LLM for answer generation            |
| `CHUNK_SIZE`           | `256`                                | Tokens per chunk                     |
| `CHUNK_OVERLAP`        | `50`                                 | Overlap between chunks               |
| `TOP_K`                | `5`                                  | Number of chunks to retrieve         |
| `MAX_ANSWER_LENGTH`    | `256`                                | Max tokens in generated answer       |
| `CHROMA_PERSIST_DIR`   | `artifacts/chroma_db`                | ChromaDB storage directory           |
| `MAX_CONTEXTS`         | `5000`                               | Max contexts to index (for speed)    |

## Evaluation Metrics

The evaluation module computes:

- **Retrieval Accuracy:** Fraction of questions where the correct context is in top-k results
- **Exact Match (EM):** Fraction of answers that exactly match ground truth
- **F1 Score:** Token-level overlap between predicted and ground truth answers
- **ROUGE-L:** Longest common subsequence overlap
- **BLEU:** N-gram precision with brevity penalty

## Tech Stack

- **Embedding:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB
- **Generator:** HuggingFace Transformers (flan-t5-base)
- **Orchestration:** LangChain
- **API:** FastAPI
- **UI:** Streamlit
- **Dataset:** HuggingFace Datasets (SQuAD 2.0)
