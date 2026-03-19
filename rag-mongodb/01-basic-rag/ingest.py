"""
Step 1 — Ingest documents into MongoDB Atlas with vector embeddings.

Run this once to populate the vector store before querying.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow imports from the shared package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

from shared.mongodb_utils import get_collection, create_vector_search_index, drop_collection_data
from shared.sample_data import SAMPLE_DOCUMENTS

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "text"


def ingest_documents() -> MongoDBAtlasVectorSearch:
    """
    Convert sample documents to LangChain Documents, embed them,
    and store in MongoDB Atlas.
    """
    print("=== Step 1: Ingesting documents ===\n")

    # 1. Connect to MongoDB
    collection = get_collection()
    drop_collection_data(collection)  # fresh start for the demo

    # 2. Prepare LangChain Documents
    docs = [
        Document(
            page_content=doc["content"],
            metadata={
                "title": doc["title"],
                "source": doc["source"],
                "category": doc["category"],
            },
        )
        for doc in SAMPLE_DOCUMENTS
    ]
    print(f"Prepared {len(docs)} documents for ingestion.")

    # 3. Create embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 1536 dimensions
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # 4. Ingest into MongoDB (this embeds and stores in one step)
    print("Embedding and storing documents in MongoDB Atlas...")
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
        embedding_key=EMBEDDING_FIELD,
        text_key=TEXT_FIELD,
    )
    print(f"Successfully stored {len(docs)} documents.\n")

    # 5. Create vector search index (Atlas builds it asynchronously)
    create_vector_search_index(
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
        embedding_field=EMBEDDING_FIELD,
        dimensions=1536,
    )

    return vector_store


if __name__ == "__main__":
    ingest_documents()
    print("\nIngestion complete. Wait ~60s for Atlas to build the vector index.")
