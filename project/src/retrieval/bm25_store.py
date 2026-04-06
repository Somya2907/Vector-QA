"""
bm25_store.py

Sparse BM25 index over chunk texts.

BM25 (Best Match 25) is a probabilistic ranking function that scores
documents against a query using term frequency, inverse document frequency,
and document length normalisation.  It excels at exact keyword matching —
the complementary strength to dense vector search.

Persistence layout
------------------
  <index_dir>/bm25.pkl   — pickled BM25Okapi object + parallel record list

Thread safety
-------------
Read-only search is safe to call concurrently.  Do not call add() while
any thread is reading.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import SearchResult

_INDEX_FILE = "bm25.pkl"

# Simple tokeniser: lowercase alphanumeric tokens of length ≥ 2
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens (len ≥ 2)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Sparse keyword index backed by BM25Okapi.

    Stores a parallel list of flat chunk records (same format as
    FaissVectorStore) so that search results carry the full metadata.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, records: list[dict[str, Any]]) -> None:
        """Build the BM25 index over a list of chunk records.

        Calling add() again replaces the existing index entirely.

        Args:
            records: Flat chunk dicts with at least a "text" key.
                     Typically the same list passed to FaissVectorStore.add().
        """
        corpus = [_tokenize(r["text"]) for r in records]
        self._index = BM25Okapi(corpus)
        self._records = records
        print(f"[bm25_store] Built BM25 index over {len(records)} documents.")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return the top-k chunks with the highest BM25 scores.

        Chunks with a BM25 score of 0 (no token overlap with the query)
        are excluded from results.

        Args:
            query:  Raw query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by BM25 score, descending.
            BM25 scores are raw (not normalised) — use only for ranking.
        """
        if self._index is None:
            raise RuntimeError("BM25Store is empty. Call add() or load() first.")

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Pair (score, index), keep only positive scores, take top_k
        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0),
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for score, idx in ranked:
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        """Pickle the BM25 index and records to <index_dir>/bm25.pkl.

        Args:
            index_dir: Directory that already contains the FAISS index files.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / _INDEX_FILE
        with open(path, "wb") as fh:
            pickle.dump({"index": self._index, "records": self._records}, fh)
        print(f"[bm25_store] Saved BM25 index → {path}")

    @classmethod
    def load(cls, index_dir: Path) -> BM25Store:
        """Load a previously saved BM25Store from disk.

        Args:
            index_dir: Directory containing bm25.pkl.

        Returns:
            Populated BM25Store ready for search.

        Raises:
            FileNotFoundError: If bm25.pkl is missing.
        """
        path = Path(index_dir) / _INDEX_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {path}\n"
                "Run scripts/build_index.py to build it."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        store = cls()
        store._index = data["index"]
        store._records = data["records"]
        print(f"[bm25_store] Loaded BM25 index ({len(store._records)} documents) from {path}")
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"BM25Store(documents={len(self._records)})"
