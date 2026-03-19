# Example 1 — Basic RAG Application

Build a complete, end-to-end Retrieval-Augmented Generation (RAG) pipeline
using **LangChain** and **MongoDB Atlas Vector Search**.

## What You'll Learn

- How to embed documents using OpenAI's `text-embedding-3-small` model
- How to store embeddings alongside document text in MongoDB Atlas
- How to create and use an Atlas Vector Search index
- How to build a LangChain RAG chain using LCEL (LangChain Expression Language)

## RAG Architecture

```
INGESTION (ingest.py)                RETRIEVAL (rag_chain.py)
─────────────────────                ────────────────────────
Raw Documents                        User Question
      │                                     │
      ▼                                     ▼
OpenAI Embedding Model           OpenAI Embedding Model
(text-embedding-3-small)         (text-embedding-3-small)
      │                                     │
      ▼                                     ▼
MongoDB Atlas Collection  ◀──  $vectorSearch (Top-K docs)
  - text                                    │
  - embedding (1536-dim)         Context + Question
  - metadata                               │
                                           ▼
                                  ChatOpenAI (gpt-4o-mini)
                                           │
                                           ▼
                                       Answer
```

## Files

| File | Description |
|---|---|
| `ingest.py` | Embed documents and store in MongoDB Atlas with a vector index |
| `rag_chain.py` | LangChain LCEL RAG chain: retrieve → format → prompt → LLM → answer |
| `main.py` | End-to-end demo with sample questions |

## How to Run

```bash
# From the rag-mongodb/ root
cd 01-basic-rag

# Run the full demo (ingest + wait for index + question answering)
python main.py

# Or run steps individually:
python ingest.py      # Step 1: embed and store documents
# ... wait 60 seconds for Atlas to build the index ...
python -c "
from dotenv import load_dotenv
load_dotenv()
from shared.mongodb_utils import get_collection
from rag_chain import build_rag_chain
chain, _ = build_rag_chain(get_collection())
print(chain.invoke('What is MongoDB Atlas Vector Search?'))
"
```

## Key Concepts

### MongoDBAtlasVectorSearch
The `langchain-mongodb` package provides `MongoDBAtlasVectorSearch` which:
- Stores document text + embeddings in a MongoDB collection
- Creates/uses an Atlas Vector Search index
- Exposes `.as_retriever()` for plug-and-play use in chains

### Atlas Vector Search Index
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 1536,
    "similarity": "cosine"
  }]
}
```

### LangChain Expression Language (LCEL) Chain
```python
chain = (
    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

## Tuning Tips

| Parameter | Effect |
|---|---|
| `k` in `as_retriever` | More docs = more context but higher cost & latency |
| Embedding model | `text-embedding-3-large` gives higher accuracy at 2x cost |
| LLM `temperature` | Keep at `0` for factual Q&A |
| Prompt template | Always instruct the LLM to answer from context only |
