# Examples

Hands-on example projects for building AI-powered applications.

## Projects

### [RAG with MongoDB](./rag-mongodb/)

End-to-end workshop examples for building Retrieval-Augmented Generation (RAG)
applications using **MongoDB Atlas Vector Search** and **LangChain**.

| Example | Description |
|---|---|
| [01 — Basic RAG](./rag-mongodb/01-basic-rag/) | Full ingest → retrieve → generate pipeline |
| [02 — Chunking Strategies](./rag-mongodb/02-chunking-strategies/) | Compare fixed, recursive, sentence, and token chunking |
| [03 — Advanced RAG](./rag-mongodb/03-advanced-rag/) | Hybrid search, metadata filtering, multi-query, conversational RAG |

**Quick start:**
```bash
cd rag-mongodb
pip install -r requirements.txt
cp .env.example .env  # fill in MONGODB_URI and OPENAI_API_KEY
python -m 01-basic-rag.main
```
