# Day 46: Retrieval-Augmented Generation (RAG)

## Learning Objectives

- Understand the RAG architecture and why combining retrieval with generation reduces hallucination
- Implement effective document chunking strategies (fixed-size, sentence-based, recursive)
- Build a retriever + generator pipeline that grounds LLM responses in source documents
- Evaluate RAG systems using faithfulness, relevance, and answer correctness metrics
- Identify common failure modes: retrieval misses, chunk boundary issues, and context window limits

## Key Concepts

Retrieval-Augmented Generation (RAG) addresses a fundamental limitation of large
language models: their knowledge is frozen at training time, and they can hallucinate
facts. RAG works by first retrieving relevant documents from an external knowledge base
using semantic search, then passing those documents as context to a language model that
generates a grounded answer. This architecture separates knowledge storage (the vector
database) from reasoning (the LLM), making it easy to update knowledge without
retraining and enabling citations back to source documents.

The quality of a RAG system depends heavily on the chunking strategy used to split
documents into retrievable pieces. Fixed-size chunks (e.g., 500 tokens with 50-token
overlap) are simple but may split sentences or ideas. Sentence-based chunking preserves
semantic boundaries. Recursive character splitting (used by LangChain) tries progressively
smaller separators (paragraphs, sentences, words) to hit a target chunk size. The chunk
size involves a trade-off: smaller chunks are more precise but may lack context, while
larger chunks provide more context but may dilute relevance and consume more of the
LLM's context window.

Evaluating RAG systems requires measuring both retrieval quality and generation quality.
Retrieval metrics include recall@k (did the relevant chunks appear in the top k results?)
and mean reciprocal rank. Generation metrics include faithfulness (is the answer
supported by the retrieved context?), answer relevance (does it address the question?),
and context utilization (did the model actually use the provided context?). Frameworks
like RAGAS automate these evaluations using LLM-as-judge approaches.

## Practical Example

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# Step 1: Chunk documents
documents = [
    "Machine learning is a field of AI that enables systems to learn from data. "
    "Supervised learning uses labeled examples to train models. Common algorithms "
    "include linear regression, decision trees, and neural networks. Deep learning "
    "is a subset that uses multi-layer neural networks for complex pattern recognition.",

    "Natural language processing (NLP) focuses on the interaction between computers "
    "and human language. Key tasks include sentiment analysis, named entity recognition, "
    "and machine translation. Modern NLP relies heavily on transformer architectures "
    "like BERT and GPT, which use self-attention mechanisms.",
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30, separators=["\n\n", ". ", " "]
)
chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:60]}...")

# Step 2: Embed and store in vector database
encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("knowledge_base")
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=encoder.encode(chunks).tolist(),
)

# Step 3: Retrieve relevant chunks for a query
query = "What is deep learning?"
results = collection.query(
    query_embeddings=encoder.encode([query]).tolist(),
    n_results=2,
)

print(f"\nQuery: '{query}'")
print("Retrieved context:")
for i, doc in enumerate(results["documents"][0]):
    print(f"  [{i}] {doc}")

# Step 4: Build prompt with retrieved context (for any LLM)
context = "\n".join(results["documents"][0])
prompt = f"""Answer the question based only on the following context.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

print(f"\nGenerated prompt ({len(prompt)} chars):")
print(prompt)
# Pass this prompt to your LLM of choice (OpenAI, Anthropic, local model, etc.)
```

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [RAGAS: evaluation framework for RAG](https://docs.ragas.io/)

## Next Day Preview

Day 47 begins Project 7: Financial News Risk Analyzer, applying NLP techniques to assess risk signals in financial text.
