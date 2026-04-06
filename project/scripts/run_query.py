"""
scripts/run_query.py

Interactive query runner for the Permission-Aware Vector QA System.

Loads the pre-built FAISS + BM25 indexes and runs a hybrid retrieval loop
so you can test queries against the handbook without starting the full API
server.

Usage
-----
  # Single query:
  python scripts/run_query.py --index-dir ./data/index --query "how do I request time off?"

  # Interactive REPL (prompts for queries until you type 'quit'):
  python scripts/run_query.py --index-dir ./data/index

  # With role filtering (only see chunks accessible to engineering):
  python scripts/run_query.py --index-dir ./data/index --role engineering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the handbook vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        type=Path,
        help="Directory containing index.faiss and metadata.json.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query string.  Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--role",
        default=None,
        metavar="ROLE",
        help="User role for permission filtering (e.g. engineering, finance, public). "
             "Omit to disable filtering and return all results.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model (must match the model used to build the index).",
    )
    return parser.parse_args()


def print_results(results, query: str) -> None:
    print(f'\nQuery: "{query}"')
    print(f"{'─' * 60}")
    if not results:
        print("  No results found.")
        return
    for rank, r in enumerate(results, 1):
        source = r.metadata.get("filename", r.metadata.get("source", "?"))
        section = r.metadata.get("section", "")
        roles = r.metadata.get("roles", [])
        chunk_idx = r.metadata.get("chunk_index", "?")
        print(
            f"[{rank}] score={r.score:.3f}  "
            f"file={source}  chunk={chunk_idx}  "
            f"roles={roles}"
        )
        if section:
            print(f"     section: {section}")
        # Print up to 3 lines of text
        preview_lines = r.text.strip().splitlines()[:3]
        for line in preview_lines:
            print(f"     {line}")
        print()


def run_query(retriever: HybridRetriever, query: str, user_role: str | None) -> None:
    results = retriever.retrieve(query, user_role=user_role)
    print_results(results, query)


def main() -> None:
    args = parse_args()

    print(f"Loading index from {args.index_dir} ...")
    faiss_store = FaissVectorStore.load(args.index_dir)
    bm25_store = BM25Store.load(args.index_dir)
    embedder = Embedder(model_name=args.model)
    retriever = HybridRetriever(
        embedder=embedder,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        top_k=args.top_k,
    )

    user_role: str | None = args.role
    if user_role is not None:
        print(f"Role filter active: {user_role!r}")
    print(f"Index ready — {len(faiss_store)} vectors  |  {len(bm25_store)} BM25 docs.\n")

    # ── Single query mode ──────────────────────────────────────────────────────
    if args.query:
        run_query(retriever, args.query, user_role)
        return

    # ── Interactive REPL ───────────────────────────────────────────────────────
    print('Enter a query (or "quit" to exit):\n')
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        run_query(retriever, query, user_role)


if __name__ == "__main__":
    main()
