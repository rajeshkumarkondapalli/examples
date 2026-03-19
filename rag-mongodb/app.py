"""
Minimal FastAPI server exposing the RAG chain as an HTTP API.

Endpoints:
  POST /ask   — { "question": "..." } → { "answer": "...", "sources": [...] }
  GET  /health — liveness check

Run locally:
    pip install fastapi uvicorn
    python app.py

In Kubernetes: this is the container entrypoint (see Dockerfile).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded chain so the container starts fast
_chain = None
_retriever = None


def _load_chain():
    global _chain, _retriever
    if _chain is None:
        from shared.mongodb_utils import get_collection
        from 01-basic-rag.rag_chain import build_rag_chain  # noqa: F401

        collection = get_collection()
        _chain, _retriever = build_rag_chain(collection, k=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_chain()
    yield


app = FastAPI(
    title="RAG with MongoDB",
    description="Question-answering over proprietary documents using MongoDB Atlas Vector Search",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    _load_chain()

    # Retrieve source documents for transparency
    retrieved_docs = _retriever.invoke(req.question)
    sources = [doc.metadata.get("title", "Unknown") for doc in retrieved_docs]

    answer = _chain.invoke(req.question)

    return AskResponse(question=req.question, answer=answer, sources=sources)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
