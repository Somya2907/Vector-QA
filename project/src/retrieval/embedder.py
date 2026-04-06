"""
embedder.py

Generates dense, L2-normalised vector embeddings using sentence-transformers.

Because vectors are L2-normalised, inner-product similarity (used by
FaissVectorStore) is mathematically equivalent to cosine similarity —
no post-processing needed at search time.

Default model: all-MiniLM-L6-v2
  - 384 dimensions
  - ~22 MB on disk
  - Strong quality/speed trade-off for semantic search
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: int = 64


class Embedder:
    """Wraps a SentenceTransformer model and exposes a minimal embed interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: Any sentence-transformers model ID or local path.
            batch_size: Sentences per forward pass.  Tune for GPU/CPU memory.
            device:     "cpu", "cuda", "mps", or None (auto-detect).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        # Exposed so callers can size the FAISS index correctly
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into L2-normalised float32 vectors.

        Args:
            texts: Strings to embed.  Empty list returns a (0, dim) array.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
            each row has unit L2 norm.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,     # L2-normalise in-place
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Convenience wrapper around embed_texts; avoids the caller having to
        unpack a (1, dim) array.

        Args:
            query: Raw query string from the user.

        Returns:
            np.ndarray of shape (embedding_dim,), dtype float32, unit L2 norm.
        """
        return self.embed_texts([query])[0]
