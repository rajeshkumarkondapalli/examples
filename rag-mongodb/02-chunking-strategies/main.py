"""
Demo: compare chunking strategies, then ingest using the best one.

Usage:
    python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare import run_comparison
from chunkers import recursive_chunks
from ingest_with_chunks import ingest_with_strategy


def main():
    # Show side-by-side stats for all strategies
    run_comparison()

    print("\n" + "=" * 70)
    print("Ingesting documents using Recursive strategy (recommended default)")
    print("=" * 70 + "\n")

    ingest_with_strategy(
        strategy_fn=recursive_chunks,
        strategy_name="recursive",
    )

    print(
        "\nDocuments are now stored in MongoDB Atlas.\n"
        "Switch the strategy_fn in ingest_with_chunks.py to experiment with\n"
        "different chunking approaches and compare retrieval quality."
    )


if __name__ == "__main__":
    main()
