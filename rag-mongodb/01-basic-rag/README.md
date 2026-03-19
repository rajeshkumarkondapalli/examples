# Example 1 — Basic RAG Application

Build a complete question-answering pipeline that retrieves relevant context
from MongoDB Atlas and uses an LLM to generate grounded answers.

## What You'll Learn

- Index documents with embeddings into MongoDB Atlas
- Create an Atlas Vector Search index
- Run a semantic similarity search
- Build a LangChain RAG chain (retrieve → augment → generate)

## Files

| File | Description |
|---|---|
| `ingest.py` | Load and embed documents into MongoDB |
| `rag_chain.py` | Build the retrieval-augmented generation chain |
| `main.py` | Run the full demo end-to-end |

## Run

```bash
# From the rag-mongodb/ root
python -m 01-basic-rag.main
# or
cd 01-basic-rag && python main.py
```

## Architecture

```
ingest.py                         rag_chain.py
─────────                         ────────────
Documents                         User Question
    │                                   │
    ▼                                   ▼
Embedding Model              Embedding Model
    │                                   │
    ▼                                   ▼
MongoDB Atlas  ◀──────────  Vector Search (Top-K)
(Vector Store)                          │
                                        ▼
                               Context + Question
                                        │
                                        ▼
                                LLM (GPT-4o-mini)
                                        │
                                        ▼
                                    Answer
```
