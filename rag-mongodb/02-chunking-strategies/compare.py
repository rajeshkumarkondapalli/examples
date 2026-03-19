"""
Compare all four chunking strategies on the same sample text.
Prints a side-by-side summary showing chunk counts and sizes.
"""

from chunkers import (
    fixed_size_chunks,
    recursive_chunks,
    sentence_chunks,
    token_chunks,
    chunk_stats,
)

SAMPLE_TEXT = """
MongoDB Atlas Vector Search is a fully managed, multi-cloud database service
that makes it easy to build intelligent applications with semantic search capabilities.
It stores vector embeddings directly alongside your operational data, eliminating
the need for a separate vector database and reducing operational complexity.

Vector search uses high-dimensional embeddings—numerical representations of text,
images, or other data—to find semantically similar items. Unlike keyword search, which
matches exact words, vector search understands context and meaning. For example, a
search for "car" can also return documents about "automobile" or "vehicle."

Atlas Vector Search supports three distance metrics: cosine similarity (most common for
text), Euclidean distance, and dot product. You can create vector indexes directly from
the Atlas UI or programmatically using the Atlas Admin API or supported drivers.

The integration with LangChain via the langchain-mongodb package makes it straightforward
to build RAG pipelines. You instantiate MongoDBAtlasVectorSearch with your collection
and embedding function, then call as_retriever() to get a LangChain-compatible retriever
that plugs into any chain or agent. Metadata filtering is supported through the
pre_filter parameter, enabling you to scope retrieval to specific document subsets.

Chunking strategy significantly impacts retrieval quality. Chunks that are too small
lose context needed for meaningful answers. Chunks that are too large dilute relevance
and may exceed the embedding model's token limit. A chunk size of 300–600 characters
with 10–20% overlap works well for most general-purpose RAG applications. For technical
documentation, sentence-based splitting often outperforms character-based approaches
because it preserves complete ideas within each chunk.
""".strip()


def run_comparison():
    strategies = {
        "Fixed-size (500 chars, 50 overlap)": fixed_size_chunks(
            SAMPLE_TEXT, chunk_size=500, overlap=50
        ),
        "Recursive (500 chars, 100 overlap)": recursive_chunks(
            SAMPLE_TEXT, chunk_size=500, overlap=100
        ),
        "Sentence-based (3 sentences, 1 overlap)": sentence_chunks(
            SAMPLE_TEXT, sentences_per_chunk=3, overlap_sentences=1
        ),
        "Token-based (128 tokens, 20 overlap)": token_chunks(
            SAMPLE_TEXT, chunk_size=128, overlap=20
        ),
    }

    print("=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Input text: {len(SAMPLE_TEXT)} characters\n")

    for name, chunks in strategies.items():
        stats = chunk_stats(chunks)
        print(f"Strategy: {name}")
        print(f"  Chunks produced  : {stats['count']}")
        print(f"  Avg size (chars) : {stats['avg_chars']:.0f}")
        print(f"  Min / Max (chars): {stats['min_chars']} / {stats['max_chars']}")
        print(f"  Approx avg tokens: {stats['approx_avg_tokens']:.0f}")
        print()

    # Show first chunk of each strategy for qualitative comparison
    print("=" * 70)
    print("FIRST CHUNK OF EACH STRATEGY")
    print("=" * 70)
    for name, chunks in strategies.items():
        print(f"\n[{name}]")
        print(chunks[0] if chunks else "<empty>")


if __name__ == "__main__":
    run_comparison()
