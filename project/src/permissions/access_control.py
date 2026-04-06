"""
access_control.py

Role-based access control for chunk filtering.

Design
------
The atomic primitive is can_access(chunk_roles, user_role):
  - "public" chunks are visible to everyone.
  - A private chunk is visible only when the user holds a matching role.

filter_by_role() builds on that to filter a list of chunks, and returns
an empty list (never raises) when nothing passes — callers must handle that.

The AccessControl class wraps the two functions for dependency-injection
into higher-level components (Retriever, QAPipeline), making it easy to
swap in a richer policy engine later without touching call sites.
"""

from __future__ import annotations

PUBLIC_ROLE = "public"


# ---------------------------------------------------------------------------
# Pure functions  (stateless, easily unit-tested)
# ---------------------------------------------------------------------------

def can_access(chunk_roles: list[str], user_role: str) -> bool:
    """Return True if user_role permits reading a chunk with these roles.

    Access is granted when:
      - The chunk is tagged "public" (visible to everyone), OR
      - The user's role appears in the chunk's role list.

    Args:
        chunk_roles: Roles attached to the chunk (from metadata["roles"]).
        user_role:   The single role the requesting user holds.

    Returns:
        True if access is permitted, False otherwise.

    Examples:
        >>> can_access(["public"], "finance")
        True
        >>> can_access(["finance"], "finance")
        True
        >>> can_access(["finance"], "engineering")
        False
        >>> can_access([], "engineering")
        False
    """
    return PUBLIC_ROLE in chunk_roles or user_role in chunk_roles


def filter_by_role(chunks: list[dict], user_role: str) -> list[dict]:
    """Return only the chunks the user is permitted to read.

    Each chunk must follow the canonical format produced by the ingestion
    pipeline:
        {"text": str, "metadata": {"roles": list[str], ...}}

    Args:
        chunks:    List of chunk dicts to filter.
        user_role: The single role the requesting user holds.

    Returns:
        Ordered subset of chunks that pass the access check.
        Returns an empty list if no chunks are permitted — never raises.

    Examples:
        >>> chunks = [
        ...     {"text": "a", "metadata": {"roles": ["public"]}},
        ...     {"text": "b", "metadata": {"roles": ["finance"]}},
        ...     {"text": "c", "metadata": {"roles": ["engineering"]}},
        ... ]
        >>> [c["text"] for c in filter_by_role(chunks, "finance")]
        ['a', 'b']
        >>> filter_by_role(chunks, "hr")
        [{'text': 'a', 'metadata': {'roles': ['public']}}]
        >>> filter_by_role([], "finance")
        []
    """
    allowed: list[dict] = []
    for chunk in chunks:
        roles: list[str] = chunk.get("metadata", {}).get("roles", [])
        if can_access(roles, user_role):
            allowed.append(chunk)
    return allowed


# ---------------------------------------------------------------------------
# Class wrapper  (for dependency injection and future extensibility)
# ---------------------------------------------------------------------------

class AccessControl:
    """Stateless access-control service wrapping the role-based filter functions.

    Using a class rather than bare functions lets callers receive an
    AccessControl instance through dependency injection and swap it for a
    richer implementation (e.g. ABAC, OPA) without changing call sites.
    """

    def can_access(self, chunk_roles: list[str], user_role: str) -> bool:
        """Check whether user_role permits access to a chunk.

        Delegates to the module-level can_access() function.
        """
        return can_access(chunk_roles, user_role)

    def filter_by_role(self, chunks: list[dict], user_role: str) -> list[dict]:
        """Filter a list of chunk dicts to those the user may read.

        Delegates to the module-level filter_by_role() function.
        Returns an empty list if nothing passes — never raises.
        """
        return filter_by_role(chunks, user_role)

    def filter_search_results(
        self,
        results: list,
        user_role: str,
    ) -> list:
        """Filter SearchResult objects (which carry a .roles attribute).

        Convenience method for the retrieval layer, which works with
        SearchResult dataclasses rather than raw dicts.

        Args:
            results:   List of SearchResult objects (must have .roles: list[str]).
            user_role: The single role the requesting user holds.

        Returns:
            Filtered list of SearchResult objects.
        """
        return [r for r in results if can_access(r.roles, user_role)]
