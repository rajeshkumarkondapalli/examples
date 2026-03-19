"""
Advanced RAG demo runner.

Runs all three advanced patterns in sequence:
  1. Metadata-filtered retrieval
  2. Multi-query retrieval
  3. Conversational RAG

Prerequisites:
  - Documents must be ingested first (run 01-basic-rag/main.py or
    02-chunking-strategies/main.py and wait for Atlas index to build).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from filtered_retrieval import demo_filtered_search
from multi_query import demo_multi_query
from conversational_rag import demo_conversation


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    section("PART 1 — Metadata-Filtered Retrieval")
    demo_filtered_search()

    section("PART 2 — Multi-Query Retrieval")
    demo_multi_query()

    section("PART 3 — Conversational RAG with History")
    demo_conversation()


if __name__ == "__main__":
    main()
