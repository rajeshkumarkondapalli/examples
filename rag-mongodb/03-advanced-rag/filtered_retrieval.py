"""
Metadata-filtered vector search.

Scope retrieval to a specific category, source, or any metadata field
without sacrificing semantic similarity ranking.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from shared.mongodb_utils import get_collection

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"


def build_filtered_retriever(
    category: str | None = None,
    source: str | None = None,
    k: int = 3,
):
    """
    Build a retriever that applies metadata pre-filters before vector search.

    Pre-filters run on the Atlas side before similarity scoring, so they are
    efficient and do not require post-processing in Python.

    Args:
        category: Filter to documents with this category (e.g. "ai", "database").
        source: Filter to documents from this source (e.g. "mongodb_docs").
        k: Number of results to return.
    """
    collection = get_collection()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        embedding_key="embedding",
        text_key="text",
    )

    # Build the pre_filter dict dynamically
    pre_filter: dict = {}
    if category:
        pre_filter["metadata.category"] = {"$eq": category}
    if source:
        pre_filter["metadata.source"] = {"$eq": source}

    search_kwargs: dict = {"k": k}
    if pre_filter:
        search_kwargs["pre_filter"] = pre_filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever


def demo_filtered_search():
    """Run example queries with and without metadata filters."""
    query = "How do I perform similarity search?"

    scenarios = [
        ("No filter (all documents)", None, None),
        ("AI category only", "ai", None),
        ("Database category only", "database", None),
        ("MongoDB docs source only", None, "mongodb_docs"),
    ]

    for label, category, source in scenarios:
        print(f"\n--- {label} ---")
        retriever = build_filtered_retriever(category=category, source=source, k=2)
        docs = retriever.invoke(query)
        for doc in docs:
            meta = doc.metadata
            print(f"  [{meta.get('category')} | {meta.get('source')}] {meta.get('title')}")


if __name__ == "__main__":
    demo_filtered_search()
