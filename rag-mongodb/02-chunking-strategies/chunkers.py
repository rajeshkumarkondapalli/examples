"""
Four chunking strategies for preparing documents before embedding.

Each function takes a raw text string and returns a list of chunk strings.
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
)


# ── Strategy 1: Fixed-size character chunking ─────────────────────────────

def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size character chunks.

    Simple and fast. May split mid-sentence, losing semantic coherence.
    Use when text structure is uniform (e.g., log files, structured records).

    Args:
        text: Raw input text.
        chunk_size: Target characters per chunk.
        overlap: Characters shared between consecutive chunks.
    """
    splitter = CharacterTextSplitter(
        separator="",           # split anywhere
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


# ── Strategy 2: Recursive character chunking (recommended default) ────────

def recursive_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text recursively using a hierarchy of separators.

    Tries to split on paragraphs → sentences → words → characters.
    Preserves semantic boundaries better than fixed-size splitting.
    This is the default LangChain splitter and works well for most documents.

    Args:
        text: Raw input text.
        chunk_size: Target characters per chunk.
        overlap: Characters shared between consecutive chunks (10–20% of size).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ── Strategy 3: Sentence-based chunking ──────────────────────────────────

def sentence_chunks(
    text: str,
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Group sentences into chunks with a sliding window.

    Respects sentence boundaries for clean, readable chunks.
    Ideal for narrative text, articles, and documentation.

    Args:
        text: Raw input text.
        sentences_per_chunk: Number of sentences per chunk.
        overlap_sentences: Sentences shared between consecutive chunks.
    """
    import re

    # Simple sentence tokenizer (replace with NLTK/spaCy for production)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]

    chunks = []
    step = max(1, sentences_per_chunk - overlap_sentences)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + sentences_per_chunk]
        if window:
            chunks.append(" ".join(window))

    return chunks


# ── Strategy 4: Token-based chunking ─────────────────────────────────────

def token_chunks(text: str, chunk_size: int = 128, overlap: int = 20) -> list[str]:
    """
    Split text based on token count rather than characters.

    Critical when your embedding model has a token limit (e.g., 512 tokens).
    Uses SentenceTransformers tokenizer by default; swap for tiktoken when
    using OpenAI models.

    Args:
        text: Raw input text.
        chunk_size: Target tokens per chunk.
        overlap: Token overlap between consecutive chunks.
    """
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=overlap,
        tokens_per_chunk=chunk_size,
    )
    return splitter.split_text(text)


# ── Utility: token count estimator ───────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4


def chunk_stats(chunks: list[str]) -> dict:
    """Return basic statistics about a list of chunks."""
    if not chunks:
        return {"count": 0}
    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "approx_avg_tokens": sum(lengths) / len(lengths) / 4,
    }
