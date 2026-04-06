"""
scripts/ingest.py

CLI script to trigger a document ingestion run.
Reads documents from a source directory (or S3 prefix), chunks them,
embeds them, and upserts them into the vector store.

Usage:
    python scripts/ingest.py --source ./data/raw --recursive
    python scripts/ingest.py --source s3://my-bucket/docs --prefix finance/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--source", required=True, help="Local path or S3 URI to ingest from.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--prefix", default="", help="S3 key prefix filter (for S3 sources).")
    parser.add_argument(
        "--permission-tags",
        nargs="*",
        metavar="KEY=VALUE",
        help="Permission tags to attach to all ingested documents (e.g. department=finance).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the ingestion CLI."""
    args = parse_args()

    # TODO: Build components from settings
    # settings = get_settings()
    # loader = DocumentLoader(...)
    # chunker = Chunker(...)
    # embedder = Embedder(...)
    # vector_store = VectorStore(...)

    # TODO: Load → chunk → embed → upsert
    raise NotImplementedError("Ingestion pipeline not yet implemented.")


if __name__ == "__main__":
    main()
