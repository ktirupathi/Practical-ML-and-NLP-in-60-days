# Day 49: Project 9 -- Knowledge Base Chatbot (RAG)

## Learning Objectives

- Build a conversational chatbot that answers questions grounded in a custom knowledge base
- Implement conversation memory to handle multi-turn dialogue with context carryover
- Design an effective RAG pipeline with document loading, chunking, retrieval, and generation
- Handle edge cases: out-of-scope questions, conflicting information, and source attribution
- Evaluate chatbot responses for faithfulness and completeness against the source documents

## Key Concepts

A RAG-powered chatbot combines the conversational abilities of large language models
with the factual grounding of a retrieval system. Unlike a standalone LLM that may
hallucinate or rely on outdated training data, a RAG chatbot retrieves relevant passages
from a curated knowledge base before generating each response. This makes it ideal for
customer support, internal documentation Q&A, and any domain where accuracy and
traceability matter. The system must handle follow-up questions by maintaining
conversation history and reformulating queries when pronouns or ellipsis reference
earlier turns.

The architecture extends the basic RAG pipeline from Day 46 with three additions.
First, a document loader that supports multiple formats (PDF, Markdown, HTML, plain
text) and extracts clean text with metadata. Second, a conversation manager that
tracks dialogue history and uses it to contextualize each new query -- for example,
rewriting "What about its competitors?" to "What are the competitors of [entity from
previous turn]?" Third, a response generator that synthesizes retrieved passages into
a coherent answer and cites sources so users can verify claims.

Production RAG chatbots also need guardrails. When no relevant documents are retrieved
(low similarity scores), the system should acknowledge uncertainty rather than
fabricate an answer. When retrieved passages conflict, the system should present both
perspectives. Streaming responses improve perceived latency, and feedback buttons
enable continuous improvement through user signals on response quality.

## Practical Example

```python
"""
Knowledge Base Chatbot (RAG) -- Project skeleton
Full project code is in the project folder (see link below).
"""
from sentence_transformers import SentenceTransformer
import chromadb

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("knowledge_base")

# Load a sample knowledge base
knowledge_docs = [
    {"id": "policy-1", "text": "Employees get 20 days paid leave per year. Up to 5 days carry over.", "source": "HR Policy"},
    {"id": "policy-2", "text": "Remote work up to 3 days/week with manager approval.", "source": "HR Policy"},
    {"id": "policy-3", "text": "401(k) match up to 6% of salary. Immediate vesting.", "source": "Benefits Guide"},
    {"id": "it-1", "text": "Reset password at portal.company.com/reset or call ext. 4400.", "source": "IT FAQ"},
]

collection.add(
    ids=[d["id"] for d in knowledge_docs],
    documents=[d["text"] for d in knowledge_docs],
    embeddings=encoder.encode([d["text"] for d in knowledge_docs]).tolist(),
    metadatas=[{"source": d["source"]} for d in knowledge_docs],
)

class RAGChatbot:
    def __init__(self, collection, encoder):
        self.collection = collection
        self.encoder = encoder
        self.history = []

    def retrieve(self, query, top_k=2):
        results = self.collection.query(
            query_embeddings=self.encoder.encode([query]).tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "source": meta["source"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def build_prompt(self, query, passages):
        context = "\n".join(f"[{p['source']}]: {p['text']}" for p in passages)
        history_str = "\n".join(f"{t['role']}: {t['content']}" for t in self.history[-4:])
        return (
            f"Answer based only on the context. Cite sources in [brackets].\n\n"
            f"History:\n{history_str}\n\nContext:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

    def chat(self, query):
        passages = self.retrieve(query)
        prompt = self.build_prompt(query, passages)
        self.history.append({"role": "user", "content": query})
        # In production, send prompt to an LLM and capture the response
        print(f"Prompt for LLM ({len(prompt)} chars):\n{prompt}\n")
        return passages

bot = RAGChatbot(collection, encoder)
bot.chat("How many vacation days do I get?")
bot.chat("Can I carry unused days over?")
```

## Resources

- [LangChain conversational RAG tutorial](https://python.langchain.com/docs/tutorials/qa_chat_history/)
- [Building RAG applications (Anthropic cookbook)](https://github.com/anthropics/anthropic-cookbook)
- [RAGAS: evaluation framework for RAG pipelines](https://docs.ragas.io/)

## Next Day Preview

Day 50 covers advanced fine-tuning techniques -- LoRA, QLoRA, and PEFT -- for adapting large models with minimal compute.
