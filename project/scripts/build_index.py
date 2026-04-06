"""
scripts/build_index.py

End-to-end index builder:

  1. Load all .md / .txt files from a source directory.
  2. Chunk each document (token-aware, with overlap).
  3. Attach metadata (roles, section title, doc/chunk IDs).
  4. Embed all chunk texts with sentence-transformers.
  5. Build a FAISS IndexFlatIP (cosine similarity) and save.
  6. Build a BM25 sparse index and save (for hybrid retrieval).

Usage
-----
  # From the project root:
  python scripts/build_index.py \\
      --source  ./data/handbook \\
      --index-dir ./data/index

  # Override embedding model or chunk parameters:
  python scripts/build_index.py \\
      --source     ./data/handbook \\
      --index-dir  ./data/index \\
      --model      all-mpnet-base-v2 \\
      --chunk-size 600 \\
      --max-tokens 800 \\
      --overlap    100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running as a script without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_directory
from src.ingestion.chunker import Chunker
from src.ingestion.metadata import attach_metadata
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.bm25_store import BM25Store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a handbook directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Root directory containing .md / .txt handbook files.",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        type=Path,
        help="Output directory for index.faiss and metadata.json.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        metavar="TOKENS",
        help="Target chunk size in tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        metavar="TOKENS",
        help="Hard ceiling for chunk size in tokens.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        metavar="TOKENS",
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (tune for available memory).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_index(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading documents from {args.source} ...")
    documents = load_directory(args.source)
    if not documents:
        print("  No documents found. Check --source path and file extensions.")
        sys.exit(1)
    print(f"  Loaded {len(documents)} documents.")

    # ── 2. Chunk + attach metadata ────────────────────────────────────────────
    print(f"\n[2/5] Chunking  (target={args.chunk_size}, "
          f"max={args.max_tokens}, overlap={args.overlap} tokens) ...")

    chunker = Chunker(
        target_tokens=args.chunk_size,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
    )

    nested_records: list[dict] = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        records = attach_metadata(chunks, document_content=doc.content)
        nested_records.extend(records)

    # Flatten {"text": ..., "metadata": {...}} → {"text": ..., **metadata_fields}
    # so the vector store can access chunk_id, roles, etc. at the top level.
    all_records = [{"text": r["text"], **r["metadata"]} for r in nested_records]

    print(f"  Produced {len(all_records)} chunks "
          f"from {len(documents)} documents.")

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    print(f"\n[3/5] Embedding with '{args.model}' ...")
    embedder = Embedder(model_name=args.model, batch_size=args.batch_size)

    texts = [r["text"] for r in all_records]
    embeddings = embedder.embed_texts(texts)

    print(f"  Embedded {len(embeddings)} chunks  "
          f"→  shape {embeddings.shape}  dtype {embeddings.dtype}")

    # ── 4. Build + save FAISS index ───────────────────────────────────────────
    print(f"\n[4/5] Building FAISS index and saving to {args.index_dir} ...")
    store = FaissVectorStore(embedding_dim=embedder.embedding_dim)
    store.add(all_records, embeddings)
    store.save(args.index_dir)

    print(f"  index.faiss   → {args.index_dir / 'index.faiss'}")
    print(f"  metadata.json → {args.index_dir / 'metadata.json'}")

    # ── 5. Build + save BM25 index ────────────────────────────────────────────
    print(f"\n[5/5] Building BM25 sparse index ...")
    bm25_store = BM25Store()
    bm25_store.add(all_records)
    bm25_store.save(args.index_dir)

    print(f"  bm25.pkl      → {args.index_dir / 'bm25.pkl'}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s.  "
          f"FAISS: {len(store)} vectors  |  BM25: {len(bm25_store)} documents.")

    # ── Quick sanity-check search ─────────────────────────────────────────────
    _smoke_test(embedder, store)


def _smoke_test(embedder: Embedder, store: FaissVectorStore) -> None:
    """Run a single test query to confirm the index is working."""
    query = "how do I set up my development environment?"
    q_vec = embedder.embed_query(query)
    results = store.search(q_vec, top_k=3)

    print(f'\n── Smoke-test query: "{query}"')
    if not results:
        print("  (no results — index may be empty)")
        return
    for rank, r in enumerate(results, 1):
        print(
            f"  [{rank}] score={r.score:.3f}  roles={r.roles}  "
            f"chunk={r.metadata.get('chunk_index')}  "
            f"file={r.metadata.get('filename', '?')}"
        )
        print(f"       {r.text[:100].replace(chr(10), ' ')!r}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_index(parse_args())
