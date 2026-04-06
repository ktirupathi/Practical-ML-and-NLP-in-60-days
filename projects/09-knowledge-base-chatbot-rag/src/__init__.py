"""
Knowledge Base Chatbot with RAG (Retrieval-Augmented Generation).

This package implements an end-to-end RAG pipeline that:
1. Ingests SQuAD 2.0 Wikipedia passages as a knowledge base
2. Chunks and embeds documents into a ChromaDB vector store
3. Retrieves relevant passages for user queries
4. Generates answers using a HuggingFace language model
"""
