# Dataset: SQuAD 2.0 + Wikipedia Passages

## Source
- **Name:** Stanford Question Answering Dataset 2.0 (SQuAD 2.0)
- **Link:** https://rajpurkar.github.io/SQuAD-explorer/
- **HuggingFace:** `rajpurkar/squad_v2`

## Schema

Each example in SQuAD 2.0 contains the following fields:

| Field        | Type   | Description                                                    |
|--------------|--------|----------------------------------------------------------------|
| `id`         | string | Unique identifier for each question-answer pair                |
| `title`      | string | Wikipedia article title the context paragraph belongs to       |
| `context`    | string | A paragraph from a Wikipedia article used as the knowledge base|
| `question`   | string | A question about the context paragraph                         |
| `answers`    | dict   | Contains `text` (list of answer strings) and `answer_start` (list of character offsets). Empty for unanswerable questions. |

## Dataset Statistics

- **150K+** question-answer pairs
- **Training set:** ~130K examples (~87K answerable, ~43K unanswerable)
- **Validation set:** ~12K examples
- Contexts drawn from **500+ Wikipedia articles**
- Unique context paragraphs: ~20K+

## Why SQuAD 2.0 Is Useful for RAG

1. **High-Quality Context Paragraphs:** Each context is a clean, well-formed paragraph from Wikipedia, making them ideal as documents in a retrieval-augmented generation knowledge base.

2. **Ground Truth QA Pairs:** The question-answer pairs serve as a gold standard for evaluating both the retrieval component (does the retriever find the right context?) and the generation component (does the generator produce the correct answer?).

3. **Unanswerable Questions:** SQuAD 2.0 includes questions that cannot be answered from the given context. This is critical for evaluating whether a RAG system can correctly abstain or say "I don't know" rather than hallucinating.

4. **Diverse Topics:** The Wikipedia passages span hundreds of articles across many domains, testing the system's ability to retrieve from a heterogeneous knowledge base.

5. **Standardized Benchmarks:** Well-established evaluation metrics (Exact Match, F1) make it easy to compare RAG system performance against published baselines.

## Preprocessing for RAG

### Step 1: Extract Unique Context Paragraphs
- SQuAD has many questions per context paragraph, so we deduplicate contexts.
- Each unique context becomes a "document" in our knowledge base.

### Step 2: Chunking
- Long paragraphs are split into smaller chunks (e.g., 256-512 tokens) with overlap.
- This improves retrieval precision since embeddings work better on focused text.

### Step 3: Embedding Generation
- Each chunk is embedded using a sentence-transformer model (e.g., `all-MiniLM-L6-v2`).
- Embeddings are stored in a vector database (ChromaDB) for fast similarity search.

### Step 4: Index Construction
- ChromaDB stores embeddings with metadata (title, original paragraph ID).
- At query time, the user's question is embedded and top-k similar chunks are retrieved.

### Step 5: Answer Generation
- Retrieved chunks are passed as context to a generative model (e.g., `flan-t5-base`).
- The model generates an answer conditioned on both the question and retrieved context.
