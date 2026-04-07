"""
src/utils/cache.py

In-memory query cache with optional disk persistence for the RAG pipeline.

Cache key
---------
  normalize(query) + "::" + (user_role or "none")

Normalisation: lowercase → strip whitespace → remove punctuation.
Including the role means the same query under different roles produces
separate entries, which is required for correct permission semantics.

Persistence
-----------
Pass cache_file=Path("data/cache.pkl") to persist the cache across
process restarts.  Pickle is used so that RAGResponse dataclass objects
are stored and restored without any manual serialisation logic.

TTL
---
Each entry is stamped with a unix timestamp.  Entries older than
ttl_seconds (default 24 h) are treated as expired on read and evicted
on load.
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TTL_SECONDS: int = 86_400   # 24 hours

_PUNCT_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _normalize(query: str) -> str:
    """Lowercase, strip, remove punctuation."""
    return _PUNCT_RE.sub("", query.lower().strip()).strip()


def _make_key(query: str, user_role: str | None) -> str:
    return f"{_normalize(query)}::{user_role or 'none'}"


# ---------------------------------------------------------------------------
# QueryCache
# ---------------------------------------------------------------------------

class QueryCache:
    """In-memory LRU-style cache keyed on (query, user_role) pairs.

    Args:
        ttl_seconds: How long an entry remains valid (default 24 h).
        cache_file:  If given, the cache is persisted to this path as a
                     pickle file and loaded back on init.
    """

    def __init__(
        self,
        ttl_seconds: int = TTL_SECONDS,
        cache_file: Path | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._cache_file = Path(cache_file) if cache_file else None
        # key → (stored_at: float, value: Any)
        self._store: dict[str, tuple[float, Any]] = {}

        if self._cache_file and self._cache_file.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, user_role: str | None = None) -> Any | None:
        """Return the cached value, or None on miss / expiry.

        Prints a one-line status line (cache hit / miss / expired).
        """
        key = _make_key(query, user_role)
        entry = self._store.get(key)

        if entry is None:
            print("[cache] miss")
            return None

        stored_at, value = entry
        age = time.time() - stored_at

        if age > self._ttl:
            del self._store[key]
            print("[cache] expired")
            return None

        print(f"[cache] hit  (age {age:.0f}s  —  saved full pipeline run)")
        return value

    def set(
        self,
        query: str,
        user_role: str | None,
        value: Any,
    ) -> None:
        """Store value under the (query, user_role) key and persist if configured."""
        key = _make_key(query, user_role)
        self._store[key] = (time.time(), value)
        if self._cache_file:
            self._save()

    def clear(self) -> None:
        """Remove all entries from memory and delete the cache file if present."""
        self._store.clear()
        if self._cache_file and self._cache_file.exists():
            self._cache_file.unlink()

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "wb") as fh:
                pickle.dump(self._store, fh)
        except Exception as exc:
            print(f"[cache] save failed: {exc}")

    def _load(self) -> None:
        try:
            with open(self._cache_file, "rb") as fh:
                raw: dict = pickle.load(fh)

            # Evict entries that have already expired
            now = time.time()
            self._store = {
                k: v for k, v in raw.items()
                if now - v[0] <= self._ttl
            }
            print(
                f"[cache] loaded {len(self._store)} valid entries "
                f"from {self._cache_file}"
            )
        except Exception as exc:
            print(f"[cache] load failed: {exc}")
            self._store = {}
