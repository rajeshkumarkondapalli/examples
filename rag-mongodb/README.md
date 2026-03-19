# RAG with MongoDB — Workshop Examples

Hands-on examples for building, optimizing, and deploying Retrieval-Augmented Generation (RAG)
applications using **MongoDB Atlas Vector Search** and **LangChain**.

## Workshop Goals

- Identify the architecture of a RAG application
- Describe and implement chunking strategies
- Build a full RAG app powered by LangChain + MongoDB

## Project Structure

```
rag-mongodb/
├── 01-basic-rag/           # End-to-end RAG pipeline
├── 02-chunking-strategies/ # Data preparation & chunking approaches
├── 03-advanced-rag/        # Hybrid search, metadata filtering, re-ranking
└── shared/                 # Shared utilities and sample data
```

## Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- OpenAI API key (or any supported LLM provider)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env

# Run the basic RAG example
cd 01-basic-rag
python main.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `OPENAI_API_KEY` | OpenAI API key for embeddings + LLM |
| `DB_NAME` | Database name (default: `rag_workshop`) |
| `COLLECTION_NAME` | Collection name (default: `documents`) |

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
                    │  LLM (GPT / Claude)  │
                    │  + Retrieved Context │
                    └──────────┬───────────┘
                               │
                               ▼
                          Final Answer
```
