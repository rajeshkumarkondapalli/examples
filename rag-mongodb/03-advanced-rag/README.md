# Example 3 — Advanced RAG with MongoDB Atlas

Production-grade RAG patterns: hybrid search, metadata filtering, and
multi-query retrieval for better answer quality.

## What You'll Learn

- **Hybrid search**: combine vector + full-text (BM25) search with RRF fusion
- **Metadata filtering**: scope retrieval to specific document categories
- **Multi-query retrieval**: generate query variations to reduce retrieval gaps
- **Conversation memory**: maintain chat history in a RAG chain

## Files

| File | Description |
|---|---|
| `hybrid_search.py` | Hybrid vector + full-text search via MongoDB aggregation |
| `filtered_retrieval.py` | Metadata-filtered vector search |
| `multi_query.py` | Multi-query retriever for robust retrieval |
| `conversational_rag.py` | RAG chain with chat history |
| `main.py` | Run all advanced examples |

## Run

```bash
# Requires documents already ingested (run example 01 or 02 first)
cd 03-advanced-rag && python main.py
```
