"""
Hybrid search: combine Atlas Vector Search + full-text search using
Reciprocal Rank Fusion (RRF) for better retrieval quality.

When to use hybrid search:
- Queries with specific technical terms or proper nouns
- Mixed queries (some semantic, some keyword-specific)
- When pure vector search misses exact-match results
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo.collection import Collection
from langchain_openai import OpenAIEmbeddings

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"
FULLTEXT_INDEX_NAME = "fulltext_index"


def create_fulltext_index(collection: Collection) -> None:
    """Create an Atlas Search (full-text) index for the text field."""
    from pymongo.operations import SearchIndexModel

    index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string"},
                    "metadata.title": {"type": "string"},
                },
            }
        },
        name=FULLTEXT_INDEX_NAME,
        type="search",
    )

    existing = [idx["name"] for idx in collection.list_search_indexes()]
    if FULLTEXT_INDEX_NAME not in existing:
        collection.create_search_index(model=index_model)
        print(f"Full-text index '{FULLTEXT_INDEX_NAME}' creation triggered.")
    else:
        print(f"Full-text index '{FULLTEXT_INDEX_NAME}' already exists.")


def hybrid_search(
    collection: Collection,
    query: str,
    k: int = 5,
    vector_weight: float = 0.5,
    text_weight: float = 0.5,
) -> list[dict]:
    """
    Perform hybrid search using MongoDB Atlas $vectorSearch + $search,
    then merge results with Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (rank + k_rrf)  for each result list
    Higher RRF score = better combined ranking.

    Args:
        collection: MongoDB collection with vector and full-text indexes.
        query: User's search query.
        k: Number of final results to return.
        vector_weight: Weight for vector search ranking (0–1).
        text_weight: Weight for text search ranking (0–1).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    query_vector = embeddings.embed_query(query)

    # ── Pipeline: run vector search and full-text search in parallel ───────
    pipeline = [
        # 1. Vector search branch
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": k * 10,
                "limit": k * 2,
            }
        },
        {
            "$group": {
                "_id": None,
                "vector_results": {
                    "$push": {
                        "_id": "$_id",
                        "text": "$text",
                        "metadata": "$metadata",
                        "vector_score": {"$meta": "vectorSearchScore"},
                    }
                },
            }
        },
        # 2. Add full-text search results via $lookup-style union
        {
            "$lookup": {
                "from": collection.name,
                "pipeline": [
                    {
                        "$search": {
                            "index": FULLTEXT_INDEX_NAME,
                            "text": {
                                "query": query,
                                "path": ["text", "metadata.title"],
                            },
                        }
                    },
                    {"$limit": k * 2},
                    {
                        "$project": {
                            "_id": 1,
                            "text": 1,
                            "metadata": 1,
                            "text_score": {"$meta": "searchScore"},
                        }
                    },
                ],
                "as": "text_results",
            }
        },
    ]

    # NOTE: The full $unionWith hybrid pipeline is available in MongoDB 6.0+.
    # For simplicity, we run vector and text search separately here and merge
    # results in Python using RRF.
    return _rrf_hybrid_search(collection, query, query_vector, k)


def _rrf_hybrid_search(
    collection: Collection,
    query: str,
    query_vector: list[float],
    k: int,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run vector search and text search separately, merge with RRF.
    This approach is compatible with all MongoDB Atlas tiers.
    """
    # Vector search
    vector_results = list(
        collection.aggregate([
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": k * 10,
                    "limit": k * 2,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "vector_score": {"$meta": "vectorSearchScore"},
                }
            },
        ])
    )

    # Full-text search
    text_results = list(
        collection.aggregate([
            {
                "$search": {
                    "index": FULLTEXT_INDEX_NAME,
                    "text": {"query": query, "path": ["text", "metadata.title"]},
                }
            },
            {"$limit": k * 2},
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "text_score": {"$meta": "searchScore"},
                }
            },
        ])
    )

    # RRF merge
    rrf_scores: dict = {}
    doc_map: dict = {}

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**doc_map[doc_id], "rrf_score": score}
        for doc_id, score in ranked[:k]
    ]


if __name__ == "__main__":
    from shared.mongodb_utils import get_collection

    collection = get_collection()
    create_fulltext_index(collection)

    query = "How does MongoDB Atlas handle vector similarity search?"
    print(f"Hybrid search: '{query}'\n")
    results = _rrf_hybrid_search(
        collection,
        query,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        ).embed_query(query),
        k=3,
    )
    for i, r in enumerate(results, 1):
        title = r.get("metadata", {}).get("title", "N/A")
        print(f"{i}. [{title}] RRF score: {r['rrf_score']:.4f}")
        print(f"   {r['text'][:120]}...\n")
