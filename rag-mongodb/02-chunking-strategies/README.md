# Example 2 — Chunking Strategies

High-quality RAG results start with well-structured inputs. This example
demonstrates four chunking approaches and lets you compare their impact on
retrieval quality.

## Strategies Covered

| Strategy | Best For |
|---|---|
| Fixed-size (character) | Simple, predictable splits |
| Recursive character | General-purpose text (default LangChain splitter) |
| Sentence-based | Narrative text, articles, documentation |
| Semantic (embedding-based) | High precision, respects topic boundaries |

## Files

| File | Description |
|---|---|
| `chunkers.py` | All four chunking strategy implementations |
| `compare.py` | Side-by-side comparison of strategies on the same text |
| `ingest_with_chunks.py` | Ingest chunks into MongoDB with strategy metadata |
| `main.py` | Run the full comparison demo |

## Run

```bash
cd 02-chunking-strategies && python main.py
```
