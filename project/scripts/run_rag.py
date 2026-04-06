"""
scripts/run_rag.py

Run the full RAG pipeline against the handbook index.

Usage
-----
  python scripts/run_rag.py --index-dir ./data/index --query "how do I request time off?"
  python scripts/run_rag.py --index-dir ./data/index --query "..." --role finance
  python scripts/run_rag.py --index-dir ./data/index --stream  # streaming mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.retriever import Retriever
from src.permissions.access_control import AccessControl
from src.generation.claude_client import ClaudeClient
from src.pipeline.rag_pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--index-dir", required=True, type=Path)
    p.add_argument("--query", required=True)
    p.add_argument("--role", default=None, help="User role for permission filtering.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading index …")
    store = FaissVectorStore.load(args.index_dir)
    embedder = Embedder()
    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        access_control=AccessControl(),
        top_k=args.top_k,
    )
    claude = ClaudeClient(model=args.model)
    pipeline = RAGPipeline(retriever=retriever, claude_client=claude)

    print(f'\nQuery : "{args.query}"')
    if args.role:
        print(f"Role  : {args.role}")
    print("─" * 60)

    if args.stream:
        print("Answer (streaming):\n")
        result = pipeline.run_stream(
            args.query,
            user_role=args.role,
            on_token=lambda t: print(t, end="", flush=True),
        )
        print()  # newline after streamed output
    else:
        result = pipeline.run(args.query, user_role=args.role)
        print(f"Answer:\n\n{result.answer}")

    print("\n" + "─" * 60)
    print(f"Chunks used : {result.chunks_used}")
    print(f"Tokens      : {result.input_tokens} in / {result.output_tokens} out")
    print(f"TTFT        : {result.ttft_seconds:.2f}s")
    print(f"Total       : {result.total_seconds:.2f}s")

    if result.citations:
        print("\nCitations:")
        for c in result.citations:
            print(f"  [{c.index}] {c.filename}  |  {c.section}  (score={c.score:.3f})")


if __name__ == "__main__":
    main()
