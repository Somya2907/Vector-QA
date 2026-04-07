"""
hybrid_retriever.py

Hybrid retriever that fuses dense (FAISS) and sparse (BM25) results using
Reciprocal Rank Fusion (RRF).

Why hybrid?
-----------
Dense retrieval (bi-encoder) captures semantic similarity but misses exact
keyword matches and rare terms.  BM25 is strong for exact keywords but fails
on paraphrasing and synonyms.  RRF combines ranked lists from both so each
method covers the other's blind spots.

Retrieval pipeline
------------------
  query
    ├─ Embedder ──────────────────► FaissVectorStore.search()  [dense]
    └─ BM25Store.search()                                       [sparse]
         │
         ▼
  _rrf_fuse(dense_results, sparse_results, k=60)
         │
         ▼
  AccessControl.filter_search_results()    [permission gate]
         │
         ▼
  Reranker.rerank()                        [optional cross-encoder]
         │
         ▼
  top-k RetrievalResult objects

RRF formula
-----------
  RRF(d) = Σ_i  1 / (k + rank_i(d))

  k = 60  (standard; penalises top ranks less aggressively)
  rank_i  = 1-based position of document d in retrieval list i

A document that appears in both lists accumulates scores from each.
A document that appears in only one still contributes via that list's score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.permissions.access_control import AccessControl
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import RetrievalResult
from src.retrieval.vector_store import FaissVectorStore, SearchResult

DEFAULT_TOP_K = 5
_CANDIDATE_MULTIPLIER = 3   # over-fetch before fusion; absorbed by filtering
_RRF_K = 60                 # standard RRF constant


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    k: int = _RRF_K,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense:  Results from FAISS (ranked by cosine similarity).
        sparse: Results from BM25 (ranked by BM25 score).
        k:      RRF constant.  Higher values dampen the influence of top ranks.

    Returns:
        Unified list sorted by RRF score, highest first.
        result.score is set to the RRF score for each document.
    """
    rrf_scores: dict[str, float] = {}          # chunk_id → accumulated RRF score
    result_map: dict[str, SearchResult] = {}   # chunk_id → SearchResult object

    for rank, r in enumerate(dense, start=1):
        key = r.chunk_id or r.text[:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        result_map[key] = r

    for rank, r in enumerate(sparse, start=1):
        key = r.chunk_id or r.text[:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in result_map:
            result_map[key] = r

    # Normalise by the theoretical maximum (rank-1 in both lists) so scores
    # are in [0, 1]: 1.0 = top of both retrievers, 0.5 = top of one only.
    max_possible = 2.0 / (k + 1)

    fused: list[SearchResult] = []
    for key, score in sorted(rrf_scores.items(), key=lambda t: t[1], reverse=True):
        r = result_map[key]
        r.score = score / max_possible
        fused.append(r)

    return fused


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Retrieves chunks via RRF fusion of dense + sparse signals.

    Drop-in replacement for Retriever: exposes the same
    retrieve(query, user_role) → list[RetrievalResult] interface so that
    RAGPipeline and the Streamlit app need no changes.
    """

    def __init__(
        self,
        embedder: Embedder,
        faiss_store: FaissVectorStore,
        bm25_store: BM25Store,
        access_control: AccessControl | None = None,
        reranker: Reranker | None = None,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = _RRF_K,
        debug: bool = False,
    ) -> None:
        """
        Args:
            embedder:       Encodes the query string into a dense vector.
            faiss_store:    Pre-built FAISS index for dense retrieval.
            bm25_store:     Pre-built BM25 index for sparse retrieval.
            access_control: Permission filter (default: allow all).
            reranker:       Optional cross-encoder reranker applied after fusion.
            top_k:          Final number of results to return.
            rrf_k:          RRF constant (default 60).
            debug:          Print per-step result tables to stdout.
        """
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.access_control = access_control or AccessControl()
        self.reranker = reranker
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.debug = debug

    def retrieve(
        self,
        query: str,
        user_role: str | None = None,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval and return the top-k permitted chunks.

        Args:
            query:     Raw query string from the user.
            user_role: Role for permission filtering.  None = no filter.

        Returns:
            Up to top_k RetrievalResult objects sorted by final score.
        """
        n_candidates = self.top_k * _CANDIDATE_MULTIPLIER

        # ── 1. Dense retrieval ────────────────────────────────────────────────
        query_vector = self.embedder.embed_query(query)
        dense = self.faiss_store.search(query_vector, top_k=n_candidates)

        # ── 2. Sparse retrieval ───────────────────────────────────────────────
        sparse = self.bm25_store.search(query, top_k=n_candidates)

        if self.debug:
            _print_list("Dense (FAISS)", dense[:5])
            _print_list("Sparse (BM25)", sparse[:5])

        # ── 3. RRF fusion ─────────────────────────────────────────────────────
        candidates = _rrf_fuse(dense, sparse, k=self.rrf_k)

        if self.debug:
            _print_list("Fused (RRF)", candidates[:5])

        # ── 4. Permission filter ──────────────────────────────────────────────
        if user_role is not None:
            candidates = self.access_control.filter_search_results(
                candidates, user_role
            )

        # ── 5. Optional cross-encoder reranking ───────────────────────────────
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        # ── 6. Return top-k ───────────────────────────────────────────────────
        return [
            RetrievalResult(text=r.text, score=r.score, metadata=r.metadata)
            for r in candidates[: self.top_k]
        ]


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _print_list(label: str, results: list[SearchResult]) -> None:
    print(f"\n  {label}:")
    for i, r in enumerate(results, 1):
        filename = r.metadata.get("filename", "?")
        print(f"    [{i}] {filename:<35}  score={r.score:.4f}")
