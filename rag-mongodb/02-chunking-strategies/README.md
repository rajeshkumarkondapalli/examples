# Example 2 — Chunking Strategies

High-quality RAG results start with well-structured inputs.
This example demonstrates **four chunking strategies** and lets you compare
their impact on chunk quality and downstream retrieval.

## Why Chunking Matters

LLMs and embedding models have **token limits**. Large documents must be split
into smaller pieces (chunks) before embedding. The chunking strategy directly
affects:

- **Retrieval precision** — overly large chunks dilute relevance scores
- **Context richness** — overly small chunks lose surrounding meaning
- **Token usage** — larger chunks cost more tokens at query time

## The Four Strategies

### 1. Fixed-Size Character Chunking
Split by a fixed number of characters regardless of content structure.

```
"...some text here..." → ["...some tex", "xt here..."]
```

- **Pros:** Simple, fast, predictable chunk sizes
- **Cons:** May split mid-word or mid-sentence
- **Best for:** Uniform structured data (logs, CSVs, structured records)

### 2. Recursive Character Chunking *(recommended default)*
Try to split on `\n\n` → `\n` → `. ` → ` ` → characters, recursively.

- **Pros:** Respects natural text boundaries, works on most documents
- **Cons:** Chunk sizes can still vary significantly
- **Best for:** General-purpose text (docs, articles, web pages)

### 3. Sentence-Based Chunking
Group N sentences into a sliding window chunk.

- **Pros:** Preserves complete thoughts, readable chunks
- **Cons:** Sentence lengths vary; technical text may have very long sentences
- **Best for:** Narrative text, articles, customer support transcripts

### 4. Token-Based Chunking
Split based on actual model tokens rather than characters.

- **Pros:** Guarantees compliance with embedding model token limits
- **Cons:** Requires tokenizer; slower than character-based approaches
- **Best for:** When you must respect strict token budgets

## Chunk Size Recommendations

| Document Type | Chunk Size | Overlap |
|---|---|---|
| Technical docs | 300–500 chars | 50–100 chars |
| Articles / books | 500–800 chars | 100–150 chars |
| FAQs / short Q&A | 150–300 chars | 20–50 chars |
| Code | 500–1000 chars | 100–200 chars |

## Files

| File | Description |
|---|---|
| `chunkers.py` | All four strategy implementations with docstrings |
| `compare.py` | Side-by-side stats comparison on a sample text |
| `ingest_with_chunks.py` | Ingest using a selected strategy; stores strategy in metadata |
| `main.py` | Run the full comparison and ingest demo |

## How to Run

```bash
cd 02-chunking-strategies

# Show the comparison (no MongoDB connection needed)
python compare.py

# Run the full demo with ingestion
python main.py

# Try a different strategy by editing ingest_with_chunks.py:
# Change: from chunkers import recursive_chunks
# To:     from chunkers import sentence_chunks
```

## Sample Output

```
======================================================================
CHUNKING STRATEGY COMPARISON
======================================================================
Input text: 2134 characters

Strategy: Fixed-size (500 chars, 50 overlap)
  Chunks produced  : 5
  Avg size (chars) : 500
  Min / Max (chars): 134 / 500
  Approx avg tokens: 125

Strategy: Recursive (500 chars, 100 overlap)
  Chunks produced  : 6
  Avg size (chars) : 437
  Min / Max (chars): 98 / 500
  Approx avg tokens: 109
...
```
