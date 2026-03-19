"""
End-to-end demo: ingest data then answer questions with RAG.

Usage:
    python main.py
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.mongodb_utils import get_collection
from ingest import ingest_documents
from rag_chain import build_rag_chain

load_dotenv()

SAMPLE_QUESTIONS = [
    "What is MongoDB Atlas Vector Search and what distance metrics does it support?",
    "How does RAG improve LLM responses compared to using a model alone?",
    "What is the recommended chunk overlap percentage when splitting documents?",
    "How do I perform hybrid search in MongoDB Atlas?",
]


def main():
    # ── 1. Ingest ───────────────────────────────────────────────────────────
    ingest_documents()

    # Wait for Atlas to build the index (60 s is usually sufficient in Atlas
    # free-tier; skip in environments where the index is pre-built).
    print("\nWaiting 60 seconds for Atlas to finish building the vector index...")
    time.sleep(60)

    # ── 2. Build RAG chain ──────────────────────────────────────────────────
    print("\n=== Step 2: Building RAG chain ===\n")
    collection = get_collection()
    chain, retriever = build_rag_chain(collection, k=3)

    # ── 3. Answer questions ─────────────────────────────────────────────────
    print("=== Step 3: Answering questions ===\n")
    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")

        # Show which documents were retrieved
        retrieved_docs = retriever.invoke(question)
        print("  Retrieved sources:")
        for doc in retrieved_docs:
            print(f"    - {doc.metadata.get('title')}")

        answer = chain.invoke(question)
        print(f"A: {answer}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()
