"""
retriever.py

Retrieves relevant chunks for a query using the FAISS vector store, with
permission filtering delegated to AccessControl.

Pipeline
--------
  1. Embed the raw query string into a dense vector.
  2. Search FAISS for the top (k × CANDIDATE_MULTIPLIER) nearest neighbours.
  3. Filter candidates through AccessControl.filter_search_results().
  4. Return the top-k permitted results with text, metadata, and score.

The over-fetch in step 2 (default ×3) ensures we still surface top_k results
even when some candidates are filtered out by access control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.permissions.access_control import AccessControl
from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import FaissVectorStore

DEFAULT_TOP_K = 5
_CANDIDATE_MULTIPLIER = 3   # Over-fetch factor to absorb filtered-out chunks


@dataclass
class RetrievalResult:
    """A single retrieval result returned to the caller."""

    text: str
    score: float          # Cosine similarity in [−1, 1]; higher = more relevant
    metadata: dict[str, Any]

    def __str__(self) -> str:
        source = self.metadata.get("filename", self.metadata.get("source", "?"))
        section = self.metadata.get("section", "")
        roles = self.metadata.get("roles", [])
        header = f"[score={self.score:.3f}  file={source}  section={section!r}  roles={roles}]"
        preview = self.text[:200].replace("\n", " ")
        return f"{header}\n  {preview}..."


class Retriever:
    """Embeds a query and returns permission-filtered top-k chunks."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        access_control: AccessControl | None = None,
        reranker: Reranker | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Args:
            embedder:       Converts query strings to dense vectors.
            vector_store:   Loaded FAISS index to search against.
            access_control: Permission filter.  Pass None to disable filtering.
            reranker:       Optional reranker applied after access control.
                            Pass None to skip reranking (FAISS order preserved).
            top_k:          Maximum results to return after filtering/reranking.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.access_control = access_control or AccessControl()
        self.reranker = reranker
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        user_role: str | None = None,
    ) -> list[RetrievalResult]:
        """Embed a query and return permitted top-k chunks.

        Args:
            query:     Raw query string from the user.
            user_role: The role the requesting user holds (e.g. "engineering",
                       "finance", "public").  Pass None to skip filtering.

        Returns:
            Up to top_k RetrievalResult objects ranked by cosine similarity.
            Returns an empty list if the index is empty or nothing passes
            the role filter.
        """
        query_vector = self.embedder.embed_query(query)

        candidates = self.vector_store.search(
            query_vector,
            top_k=self.top_k * _CANDIDATE_MULTIPLIER,
        )

        if user_role is not None:
            candidates = self.access_control.filter_search_results(
                candidates, user_role
            )

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        return [
            RetrievalResult(text=r.text, score=r.score, metadata=r.metadata)
            for r in candidates[: self.top_k]
        ]
