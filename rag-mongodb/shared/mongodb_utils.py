"""
Shared MongoDB utilities for the RAG workshop examples.
"""

import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel


def get_collection(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
) -> Collection:
    """Return a MongoDB collection, creating it if needed."""
    uri = uri or os.environ["MONGODB_URI"]
    db_name = db_name or os.getenv("DB_NAME", "rag_workshop")
    collection_name = collection_name or os.getenv("COLLECTION_NAME", "documents")

    client = MongoClient(uri)
    return client[db_name][collection_name]


def create_vector_search_index(
    collection: Collection,
    index_name: str = "vector_index",
    embedding_field: str = "embedding",
    dimensions: int = 1536,
    similarity: str = "cosine",
) -> None:
    """
    Create an Atlas Vector Search index on the collection.

    This only needs to be run once. Atlas takes ~60 seconds to build the index.
    """
    index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": embedding_field,
                    "numDimensions": dimensions,
                    "similarity": similarity,
                },
                # Optional: add filterable metadata fields
                {"type": "filter", "path": "source"},
                {"type": "filter", "path": "category"},
            ]
        },
        name=index_name,
        type="vectorSearch",
    )

    existing = [idx["name"] for idx in collection.list_search_indexes()]
    if index_name not in existing:
        collection.create_search_index(model=index_model)
        print(f"Vector search index '{index_name}' creation triggered.")
        print("Wait ~60 seconds for Atlas to build the index before querying.")
    else:
        print(f"Index '{index_name}' already exists — skipping creation.")


def drop_collection_data(collection: Collection) -> None:
    """Delete all documents from a collection (useful for re-running demos)."""
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from '{collection.name}'.")
