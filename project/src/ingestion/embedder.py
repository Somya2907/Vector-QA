"""
embedder.py

Generates dense vector embeddings for Chunk objects using a configurable
embedding model (e.g. sentence-transformers, OpenAI text-embedding-3-small,
or Anthropic voyage-* models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunker import Chunk


@dataclass
class EmbeddedChunk:
    """A Chunk paired with its dense vector embedding."""

    chunk: Chunk
    embedding: list[float]
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder:
    """Converts text chunks into vector embeddings."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        """
        Args:
            model_name: Identifier of the embedding model to use.
            batch_size: Number of chunks to embed in a single API call.
        """
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Generate an embedding for a single Chunk.

        Args:
            chunk: The Chunk to embed.

        Returns:
            EmbeddedChunk with the dense vector attached.
        """
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of Chunks, respecting batch_size limits.

        Args:
            chunks: List of Chunks to embed.

        Returns:
            List of EmbeddedChunk objects in the same order.
        """
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Embed a raw query string for similarity search.

        Args:
            query: The user's question or search string.

        Returns:
            Dense embedding vector.
        """
        raise NotImplementedError
