# RAG Demo (Groq + Llama + Vector DB)

This repo is a beginner-friendly but technically correct end-to-end RAG pipeline demo.

## What it shows
- Loading a knowledge base from `./internal_docs`
- Chunking
- Embeddings
- Vector database indexing (Chroma)
- Retrieval (top-K)
- Prompt assembly + grounded answering using Groq (Llama)

## Setup (uv)
```bash
uv sync
export GROQ_API_KEY="your_key"
python rag_demo.py
