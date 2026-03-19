"""
Sample documents used across workshop examples.
Replace or extend with your own proprietary data.
"""

SAMPLE_DOCUMENTS = [
    {
        "title": "MongoDB Atlas Vector Search Overview",
        "source": "mongodb_docs",
        "category": "database",
        "content": (
            "MongoDB Atlas Vector Search allows you to store vector embeddings alongside "
            "your operational data and perform semantic similarity searches using the $vectorSearch "
            "aggregation stage. It supports multiple distance metrics including cosine, euclidean, "
            "and dot product, and integrates natively with popular frameworks like LangChain and "
            "LlamaIndex. Vector indexes are managed directly in Atlas and scale automatically."
        ),
    },
    {
        "title": "Introduction to RAG",
        "source": "ai_concepts",
        "category": "ai",
        "content": (
            "Retrieval-Augmented Generation (RAG) is an AI technique that enhances large language "
            "model (LLM) responses by retrieving relevant external documents at inference time. "
            "Instead of relying solely on knowledge baked into model weights, RAG retrieves "
            "context from a vector store and passes it to the LLM as part of the prompt. "
            "This allows models to answer questions about private or up-to-date information "
            "without fine-tuning."
        ),
    },
    {
        "title": "Chunking Best Practices",
        "source": "ai_concepts",
        "category": "ai",
        "content": (
            "Chunking is the process of splitting large documents into smaller, semantically "
            "meaningful pieces before generating embeddings. Optimal chunk size balances "
            "context richness (larger chunks) vs. retrieval precision (smaller chunks). "
            "Common strategies include fixed-size chunking, sentence-based splitting, "
            "recursive character splitting, and semantic chunking. Chunk overlap (typically "
            "10–20%) helps preserve context across chunk boundaries."
        ),
    },
    {
        "title": "LangChain and MongoDB Integration",
        "source": "mongodb_docs",
        "category": "integration",
        "content": (
            "The langchain-mongodb package provides MongoDBAtlasVectorSearch, a ready-to-use "
            "vector store that stores documents and their embeddings in MongoDB Atlas. It "
            "exposes similarity_search(), similarity_search_with_score(), and as_retriever() "
            "methods that plug directly into LangChain chains and agents. Documents can be "
            "indexed with add_documents() and the store supports metadata filtering via the "
            "pre_filter parameter."
        ),
    },
    {
        "title": "Hybrid Search in MongoDB",
        "source": "mongodb_docs",
        "category": "database",
        "content": (
            "Hybrid search combines vector similarity search with full-text (BM25) search "
            "to improve retrieval quality. MongoDB Atlas supports hybrid search by running "
            "$vectorSearch and $search in parallel and merging results using Reciprocal Rank "
            "Fusion (RRF). This is especially useful when queries contain specific keywords "
            "that semantic search alone might miss, or when dealing with short, precise queries."
        ),
    },
    {
        "title": "Prompt Engineering for RAG",
        "source": "ai_concepts",
        "category": "ai",
        "content": (
            "Effective RAG prompts instruct the LLM to answer only from the provided context "
            "and to say 'I don't know' when the answer is not present. Including source "
            "citations in the prompt template improves answer grounding. Common pitfalls include "
            "context stuffing (too many retrieved chunks), context truncation, and hallucination "
            "when the model ignores context. Evaluation frameworks like RAGAS measure faithfulness, "
            "answer relevancy, and context precision."
        ),
    },
]
