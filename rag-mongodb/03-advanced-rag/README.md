# Example 3 — Advanced RAG with MongoDB Atlas

Production-grade retrieval patterns that go beyond basic vector search.

## Patterns Covered

| Pattern | File | When to Use |
|---|---|---|
| Metadata filtering | `filtered_retrieval.py` | Scope retrieval to a subset of documents |
| Multi-query retrieval | `multi_query.py` | Queries that can be phrased in many ways |
| Hybrid search (vector + BM25) | `hybrid_search.py` | Queries with specific keywords or proper nouns |
| Conversational RAG | `conversational_rag.py` | Multi-turn chat applications |

---

## Pattern 1 — Metadata-Filtered Retrieval

Vector search + a **pre-filter** that scopes documents by metadata field before
similarity ranking. Atlas applies the filter on the server side (no Python
post-processing needed).

```python
retriever = vector_store.as_retriever(search_kwargs={
    "k": 3,
    "pre_filter": {"metadata.category": {"$eq": "database"}},
})
```

**Use cases:** multi-tenant apps, scoping to a user's documents, filtering by
date range, document type, or permission level.

---

## Pattern 2 — Multi-Query Retrieval

The LLM generates N alternative phrasings of the question; each is used for a
separate vector search. Results are deduplicated and combined.

```
Question: "How do I improve search relevance?"
     │
     ▼  LLM generates:
  ├─ "What techniques improve document retrieval quality?"
  ├─ "Best practices for semantic search ranking?"
  └─ "How to make vector search return better results?"
     │
     ▼  3 × vector search → deduplicated docs
```

**When to use:** open-ended questions that can be answered from many angles;
queries where a single phrasing misses relevant documents.

---

## Pattern 3 — Hybrid Search (Vector + BM25 with RRF)

Run vector search and full-text (Atlas Search / BM25) in parallel, then merge
rankings using **Reciprocal Rank Fusion (RRF)**:

```
RRF score = Σ  1 / (rank_i + k)
```

A document ranked #1 in vector search AND #1 in text search gets a much higher
combined score than one that only appears in one list.

**When to use:** short queries with specific keywords, product names, or
acronyms that semantic search may not capture well.

---

## Pattern 4 — Conversational RAG

Multi-turn chat that maintains history. Each question is:
1. **Condensed** with history → standalone question (via LLM)
2. **Retrieved** from MongoDB using the standalone question
3. **Answered** with the retrieved context + full chat history

```
Turn 1: "What is Atlas Vector Search?"
Turn 2: "What distance metrics does it support?"
         ↓ condensed to:
         "What distance metrics does MongoDB Atlas Vector Search support?"
         ↓ retrieves relevant docs
         ↓ answers with context
```

---

## Prerequisites

Documents must be ingested before running these examples:
```bash
cd ../01-basic-rag && python ingest.py
# Wait ~60s for Atlas to build the vector index
```

## How to Run

```bash
cd 03-advanced-rag

# Run all patterns
python main.py

# Or run individual patterns
python filtered_retrieval.py
python multi_query.py
python conversational_rag.py
```

## Files

| File | Description |
|---|---|
| `hybrid_search.py` | RRF fusion of vector + full-text search |
| `filtered_retrieval.py` | Pre-filter by metadata before vector ranking |
| `multi_query.py` | LLM-generated query variants for broader recall |
| `conversational_rag.py` | Multi-turn chat with history condensation |
| `main.py` | Runs all patterns sequentially |
