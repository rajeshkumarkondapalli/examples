"""
Ingest documents using a chosen chunking strategy.

Each chunk is stored with metadata indicating which strategy produced it,
enabling A/B comparison of retrieval quality in MongoDB.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

from shared.mongodb_utils import get_collection, create_vector_search_index, drop_collection_data
from shared.sample_data import SAMPLE_DOCUMENTS
from chunkers import recursive_chunks  # default strategy; swap as needed

load_dotenv()

STRATEGY_NAME = "recursive"  # tag stored in metadata for filtering
VECTOR_INDEX_NAME = "vector_index"


def ingest_with_strategy(strategy_fn=recursive_chunks, strategy_name: str = STRATEGY_NAME):
    """
    Chunk all sample documents using strategy_fn, embed, and store in MongoDB.
    """
    print(f"=== Ingesting with strategy: '{strategy_name}' ===\n")

    collection = get_collection()
    drop_collection_data(collection)

    # Build LangChain Documents from chunks
    all_docs: list[Document] = []
    for raw_doc in SAMPLE_DOCUMENTS:
        chunks = strategy_fn(raw_doc["content"])
        for i, chunk_text in enumerate(chunks):
            all_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "title": raw_doc["title"],
                        "source": raw_doc["source"],
                        "category": raw_doc["category"],
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "chunk_strategy": strategy_name,
                    },
                )
            )

    print(f"Total chunks to ingest: {len(all_docs)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
    )
    print("Ingestion complete.\n")

    create_vector_search_index(collection, index_name=VECTOR_INDEX_NAME)

    return vector_store


if __name__ == "__main__":
    ingest_with_strategy()
