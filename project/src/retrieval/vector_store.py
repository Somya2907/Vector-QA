"""
vector_store.py

FAISS-backed vector store with persistent metadata.

Design
------
- Uses IndexFlatIP (exact inner-product search).  Because the Embedder
  returns L2-normalised vectors, inner product == cosine similarity.
- Metadata is stored in a parallel list (one dict per vector) and persisted
  as a JSON file alongside the FAISS index binary.
- Deletion is supported by rebuilding the index from the surviving records.
  This is acceptable for handbook-scale datasets (< 1 M vectors).

Persistence layout
------------------
  <index_dir>/
    index.faiss     — FAISS binary index
    metadata.json   — JSON array, one object per vector (text + metadata fields)

Thread safety
-------------
Not thread-safe.  Add an external lock if multiple threads write concurrently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result returned by a similarity search."""

    chunk_id: str
    doc_id: str
    text: str
    score: float           # Cosine similarity in [−1, 1]; higher is more similar
    roles: list[str]       # Permission roles attached to this chunk
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class FaissVectorStore:
    """Exact cosine-similarity vector store backed by a FAISS IndexFlatIP."""

    def __init__(self, embedding_dim: int) -> None:
        """
        Args:
            embedding_dim: Dimensionality of the embedding vectors.
                           Must match the Embedder used to generate them.
        """
        self.embedding_dim = embedding_dim
        # IndexFlatIP: brute-force inner product (= cosine sim for unit vectors)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        # Parallel metadata list — index i in _records corresponds to FAISS
        # vector i.  Each record is the full {"text": ..., ...metadata...} dict
        # produced by src.ingestion.metadata.attach_metadata().
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add chunk records with their pre-computed embeddings.

        Args:
            records:    List of {"text": str, ...metadata fields...} dicts.
                        Typically the output of attach_metadata().
            embeddings: Float32 array of shape (len(records), embedding_dim).
                        Vectors must already be L2-normalised (Embedder does this).

        Raises:
            ValueError: If len(records) != len(embeddings) or dim mismatch.
        """
        if len(records) != len(embeddings):
            raise ValueError(
                f"records ({len(records)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}."
            )

        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vecs)
        self._records.extend(records)

    def delete(self, chunk_ids: set[str]) -> int:
        """Remove chunks by chunk_id.  Rebuilds the index in place.

        This is O(n) but correct for handbook-scale data.  For large corpora
        switch to IndexIDMap which supports O(1) deletion.

        Args:
            chunk_ids: Set of chunk_id strings to remove.

        Returns:
            Number of records actually deleted.
        """
        surviving = [
            (rec, i) for i, rec in enumerate(self._records)
            if rec.get("chunk_id") not in chunk_ids
        ]

        deleted = len(self._records) - len(surviving)
        if deleted == 0:
            return 0

        # Rebuild
        surviving_records = [rec for rec, _ in surviving]
        surviving_indices = [i for _, i in surviving]

        if surviving_indices:
            old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
            surviving_vecs = old_vecs[surviving_indices]
        else:
            surviving_vecs = np.empty((0, self.embedding_dim), dtype=np.float32)

        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._records = []
        if len(surviving_records):
            self.add(surviving_records, surviving_vecs)

        return deleted

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query vector.

        Args:
            query_vector: Shape (embedding_dim,) or (1, embedding_dim), float32,
                          L2-normalised.  Use Embedder.embed_query() to produce it.
            top_k:        Maximum number of results.

        Returns:
            Ranked SearchResult list, best match first.
            Returns fewer than top_k results if the index has fewer vectors.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS pads with -1 when fewer than k results exist
                continue
            rec = self._records[idx]
            results.append(
                SearchResult(
                    chunk_id=rec.get("chunk_id", ""),
                    doc_id=rec.get("doc_id", ""),
                    text=rec.get("text", ""),
                    score=float(score),
                    roles=rec.get("roles", []),
                    metadata=rec,
                )
            )
        return results

    def get_by_doc_id(self, doc_id: str) -> list[SearchResult]:
        """Return all stored chunks belonging to a document (no score).

        Useful for deletion and document-level audits.
        """
        return [
            SearchResult(
                chunk_id=rec.get("chunk_id", ""),
                doc_id=rec.get("doc_id", ""),
                text=rec.get("text", ""),
                score=1.0,
                roles=rec.get("roles", []),
                metadata=rec,
            )
            for rec in self._records
            if rec.get("doc_id") == doc_id
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        """Persist the FAISS index and metadata to disk.

        Creates the directory if it doesn't exist.

        Args:
            index_dir: Directory to write index.faiss and metadata.json into.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_dir / "index.faiss"))

        with open(index_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, ensure_ascii=False, indent=2)

        print(
            f"[vector_store] Saved {self._index.ntotal} vectors → {index_dir}"
        )

    @classmethod
    def load(cls, index_dir: Path) -> FaissVectorStore:
        """Load a previously saved store from disk.

        Args:
            index_dir: Directory containing index.faiss and metadata.json.

        Returns:
            Populated FaissVectorStore ready for search.

        Raises:
            FileNotFoundError: If the expected files are missing.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"

        for p in (index_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing index file: {p}")

        index = faiss.read_index(str(index_path))

        with open(meta_path, encoding="utf-8") as fh:
            records = json.load(fh)

        store = cls(embedding_dim=index.d)
        store._index = index
        store._records = records

        print(
            f"[vector_store] Loaded {index.ntotal} vectors from {index_dir}"
        )
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"FaissVectorStore(dim={self.embedding_dim}, "
            f"vectors={self._index.ntotal})"
        )
