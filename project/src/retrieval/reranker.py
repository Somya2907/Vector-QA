"""
reranker.py

Neural reranker using BAAI/bge-reranker-base (cross-encoder architecture).

How a cross-encoder works
-------------------------
Unlike the bi-encoder (all-MiniLM-L6-v2) that embeds query and passage
independently and then computes cosine similarity, a cross-encoder receives
the (query, passage) pair concatenated and scores them jointly.  This is
slower but captures fine-grained semantic interactions that bi-encoders miss.

Pipeline position
-----------------
  FAISS retrieval (bi-encoder, fast, broad)
      ↓  top k × 3 candidates
  CrossEncoder reranker (slower, precise)
      ↓  rescored + sorted
  top-k returned to caller

Scores
------
The model outputs a raw logit for each (query, passage) pair.
We apply sigmoid to map logits → [0, 1] relevance probabilities, then
update each result's .score so downstream consumers see the reranked value.

Public API
----------
  Reranker(model_name, debug)        — class for Retriever injection
  rerank_results(query, results, …)  — standalone dict interface
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-base"
DEFAULT_MAX_LENGTH = 512   # token budget per (query, passage) pair


# ---------------------------------------------------------------------------
# Reranker class — injected into Retriever
# ---------------------------------------------------------------------------

class Reranker:
    """Cross-encoder reranker for use inside the Retriever pipeline.

    Loads the model once on construction.  Thread-safe for read-only inference.
    Works on any objects with .text, .score, .metadata attributes
    (SearchResult, RetrievalResult, or compatible duck types).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        debug: bool = False,
    ) -> None:
        """
        Args:
            model_name:  HuggingFace model ID or local path.
            max_length:  Token budget for each (query, passage) pair.
                         Passages longer than this are truncated.
            debug:       Print before/after ranking tables to stdout.
        """
        self.model_name = model_name
        self.debug = debug
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, results: list) -> list:
        """Score every (query, result.text) pair and return results re-sorted.

        Updates result.score to the cross-encoder probability so downstream
        consumers (Retriever, UI) see the reranked value.

        Args:
            query:   Raw query string from the user.
            results: Objects with .text, .score, .metadata attributes.

        Returns:
            Same objects sorted by cross-encoder score, highest first.
            Returns an empty list when results is empty.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = _sigmoid(self._model.predict(pairs))

        if self.debug:
            _print_debug(query, results, scores, attr_mode=True)

        ranked = sorted(
            zip(scores, results),
            key=lambda t: t[0],
            reverse=True,
        )

        reranked = []
        for score, r in ranked:
            r.score = float(score)
            reranked.append(r)

        return reranked


# ---------------------------------------------------------------------------
# Standalone function — dict interface
# ---------------------------------------------------------------------------

# Module-level lazy singleton so the model loads only once per process
_default_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _default_model
    if _default_model is None:
        _default_model = CrossEncoder(DEFAULT_MODEL, max_length=DEFAULT_MAX_LENGTH)
    return _default_model


def rerank_results(
    query: str,
    results: list[dict],
    debug: bool = False,
) -> list[dict]:
    """Rerank a list of chunk dicts with the cross-encoder.

    Each result dict must have the shape:
        {"text": str, "metadata": {...}, "score": float}

    Returns a new list sorted by "final_score" (descending), with that
    key added to every dict.

    Args:
        query:   The user's raw query string.
        results: Chunk dicts from the retrieval pipeline.
        debug:   If True, print before/after ranking to stdout.

    Returns:
        Sorted list of dicts with "final_score" added.
        Empty list when results is empty.
    """
    if not results:
        return []

    model = _get_model()
    pairs = [(query, r["text"]) for r in results]
    scores = _sigmoid(model.predict(pairs))

    if debug:
        _print_debug(query, results, scores, attr_mode=False)

    ranked = sorted(
        zip(scores, results),
        key=lambda t: t[0],
        reverse=True,
    )
    return [{**r, "final_score": float(s)} for s, r in ranked]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Map raw cross-encoder logits to [0, 1] relevance probabilities."""
    return 1.0 / (1.0 + np.exp(-logits))


def _print_debug(
    query: str,
    results: list,
    scores: np.ndarray,
    attr_mode: bool,
) -> None:
    """Print before/after ranking tables to stdout."""

    def filename(r) -> str:
        meta = r.metadata if attr_mode else r.get("metadata", {})
        return meta.get("filename", "?")

    def orig_score(r) -> float:
        return r.score if attr_mode else r.get("score", 0.0)

    print(f'\nReranker [{DEFAULT_MODEL}] — Query: "{query}"')
    print("Before rerank:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] file={filename(r):<35}  embed_score={orig_score(r):.4f}")

    print("After rerank:")
    order = np.argsort(scores)[::-1]
    for rank, idx in enumerate(order, 1):
        print(
            f"  [{rank}] file={filename(results[idx]):<35}"
            f"  rerank_score={scores[idx]:.4f}"
        )
    print()
