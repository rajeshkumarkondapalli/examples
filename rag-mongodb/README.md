# RAG with MongoDB — Workshop Examples

Hands-on examples for building, optimizing, and deploying Retrieval-Augmented Generation (RAG)
applications using **MongoDB Atlas Vector Search** and **LangChain**.

## Workshop Goals

- Identify the architecture of a RAG application
- Describe and implement chunking strategies for data preparation
- Build a full RAG app powered by LangChain + MongoDB
- Deploy the application and MongoDB in Kubernetes

## Project Structure

```
rag-mongodb/
├── 01-basic-rag/           # End-to-end RAG pipeline
├── 02-chunking-strategies/ # Data preparation & chunking approaches
├── 03-advanced-rag/        # Hybrid search, metadata filtering, re-ranking, chat
├── 04-kubernetes/          # Kubernetes manifests: MongoDB pod + RAG app pod
├── shared/                 # Shared utilities and sample data
├── app.py                  # FastAPI server (used by the Docker container)
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| MongoDB Atlas account | Free tier works for examples 1–3 |
| OpenAI API key | Required for embeddings and LLM |
| kubectl + cluster | Required for example 4 |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env: add MONGODB_URI and OPENAI_API_KEY

# 3. Run the basic RAG demo
cd 01-basic-rag && python main.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `OPENAI_API_KEY` | OpenAI API key for embeddings + LLM |
| `DB_NAME` | Database name (default: `rag_workshop`) |
| `COLLECTION_NAME` | Collection name (default: `documents`) |

## Examples at a Glance

### [01 — Basic RAG](./01-basic-rag/)
Complete ingest → retrieve → generate pipeline.

```python
# Ingest documents into MongoDB Atlas
python 01-basic-rag/ingest.py

# Ask a question
chain.invoke("What is MongoDB Atlas Vector Search?")
```

### [02 — Chunking Strategies](./02-chunking-strategies/)
Compare four chunking approaches and understand their trade-offs.

```python
# See stats for all strategies on a sample text
python 02-chunking-strategies/compare.py
```

### [03 — Advanced RAG](./03-advanced-rag/)
Production patterns: hybrid search, metadata filtering, multi-query, conversational chat.

```python
# Metadata-filtered retrieval
retriever = build_filtered_retriever(category="database", k=3)

# Multi-turn conversation
chain.invoke({"question": "What distance metrics does it support?"}, ...)
```

### [04 — Kubernetes](./04-kubernetes/)
Deploy MongoDB as a pod and the RAG app in a Kubernetes cluster.

```bash
kubectl create namespace rag-workshop
kubectl apply -k 04-kubernetes/ -n rag-workshop
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Atlas Vector Search?"}'
```

## RAG Architecture Overview

```
  User Query
      │
      ▼
┌─────────────┐     ┌──────────────────────┐
│  Embedding  │────▶│  MongoDB Atlas       │
│  Model      │     │  Vector Search       │
└─────────────┘     └──────────┬───────────┘
                               │ Top-K docs
                               ▼
                    ┌──────────────────────┐
                    │  LLM (GPT-4o-mini)   │
                    │  + Retrieved Context │
                    └──────────┬───────────┘
                               │
                               ▼
                          Final Answer
```
