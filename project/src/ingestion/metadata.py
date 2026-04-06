"""
metadata.py

Attaches structured metadata to Chunk objects and produces the canonical
output record format: {"text": "...", "metadata": {...}}.

Responsibilities
----------------
- Infer role-based permissions from the source file path.
- Extract a human-readable section/title from the document content.
- Assemble the final flat metadata dict attached to every chunk record.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .chunker import Chunk

# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------

# Ordered list of (path-substring, role) mappings.
# A file path may match multiple rules — all matching roles are returned.
_ROLE_RULES: list[tuple[str, str]] = [
    ("finance", "finance"),
    ("engineering", "engineering"),
]

_DEFAULT_ROLE: str = "public"


def infer_roles(source: str) -> list[str]:
    """Infer access roles from the source file path.

    Rules (applied to the lowercased path):
      - Contains "finance"     → role "finance"
      - Contains "engineering" → role "engineering"
      - No match               → role "public"

    A path can match multiple rules (e.g. "finance/engineering-tools").

    Args:
        source: File path string (absolute or relative).

    Returns:
        Non-empty list of role strings.

    Examples:
        >>> infer_roles("handbook/finance/expenses.md")
        ['finance']
        >>> infer_roles("handbook/engineering/onboarding.md")
        ['engineering']
        >>> infer_roles("handbook/people/values.md")
        ['public']
        >>> infer_roles("handbook/finance/engineering-contracts.md")
        ['finance', 'engineering']
    """
    lower = source.lower()
    roles = [role for keyword, role in _ROLE_RULES if keyword in lower]
    return roles if roles else [_DEFAULT_ROLE]


# ---------------------------------------------------------------------------
# Section / title extraction
# ---------------------------------------------------------------------------

# Matches ATX-style markdown headings: # Title, ## Title, ### Title
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)


def extract_section(content: str) -> str:
    """Extract the first Markdown heading from the document content.

    Falls back to the first non-empty line if no heading is found.

    Args:
        content: Cleaned document text.

    Returns:
        Section title string, or empty string if content is blank.

    Examples:
        >>> extract_section("# Getting Started\\n\\nSome text here.")
        'Getting Started'
        >>> extract_section("No heading here\\njust prose.")
        'No heading here'
    """
    match = _HEADING_RE.search(content)
    if match:
        return match.group(1).strip()

    # Fallback: first non-empty line
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]   # Cap to avoid absurdly long titles

    return ""


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------

def build_metadata(
    chunk: Chunk,
    *,
    roles: list[str] | None = None,
    section: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the metadata dict for a chunk record.

    Args:
        chunk:   The Chunk object to extract base metadata from.
        roles:   Pre-computed role list; if None, inferred from chunk.metadata["source"].
        section: Pre-computed section title; if None, kept as empty string.
        extra:   Additional key-value pairs merged into the metadata.

    Returns:
        Flat metadata dict suitable for storage in a vector database.
    """
    source = chunk.metadata.get("source", chunk.doc_id)

    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "source": source,
        "filename": chunk.metadata.get("filename", Path(source).name),
        "section": section if section is not None else "",
        "roles": roles if roles is not None else infer_roles(source),
        **(extra or {}),
    }


def attach_metadata(
    chunks: list[Chunk],
    document_content: str = "",
) -> list[dict[str, Any]]:
    """Convert a list of Chunks into canonical output records.

    For each chunk this function:
      1. Infers roles from the source path.
      2. Extracts the section title from the parent document content
         (the same title is applied to all chunks of that document).
      3. Assembles the final {"text": ..., "metadata": ...} record.

    Args:
        chunks:           Chunks produced by Chunker.chunk_document().
        document_content: Full original document text used for title extraction.
                          Pass the Document.content value here.

    Returns:
        List of records in the format {"text": "...", "metadata": {...}}.
    """
    if not chunks:
        return []

    # Extract section once per document (shared across all chunks)
    section = extract_section(document_content)

    # Infer roles once per document using the first chunk's source path
    source = chunks[0].metadata.get("source", chunks[0].doc_id)
    roles = infer_roles(source)

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = build_metadata(chunk, roles=roles, section=section)
        records.append({"text": chunk.content, "metadata": metadata})

    return records


# ---------------------------------------------------------------------------
# Quick demo — run with:  python -m src.ingestion.metadata
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    from .loader import load_directory
    from .chunker import Chunker

    # ── Create a tiny synthetic handbook in a temp directory ──────────────────
    SAMPLE_FILES = {
        "handbook/engineering/onboarding.md": """\
# Engineering Onboarding

Welcome to the engineering team. This guide covers everything you need to
get up and running in your first week.

## Development Environment

Install the required tools using the bootstrap script:

```bash
./scripts/bootstrap.sh
```

The script installs Python 3.11, Docker, and the internal CLI tools.

## Code Review Process

All changes require at least one approval from a senior engineer before
merging. Use the PR template located at `.github/pull_request_template.md`.
""",
        "handbook/finance/expenses.md": """\
# Expense Reimbursement Policy

This document describes the process for submitting business expenses.

## Eligible Expenses

- Travel: flights, hotels, and ground transport for approved business trips.
- Meals: up to $75 per person for client entertainment.
- Equipment: pre-approved purchases only.

## Submission Process

Submit all receipts within 30 days via the Finance Portal. Attach receipts
as PDF or image files. Expenses submitted after 30 days will not be reimbursed.
""",
        "handbook/values.md": """\
# GitLab Values

Our six core values guide every decision we make.

## Collaboration

We work together across teams, time zones, and disciplines to achieve
outcomes that no single individual could accomplish alone.

## Results

We measure success by outcomes, not outputs. Every initiative should
map to a measurable impact on our customers or the business.
""",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write sample files
        for rel_path, content in SAMPLE_FILES.items():
            full_path = tmp / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # ── Run the full ingestion pipeline ───────────────────────────────────
        print("Loading documents...")
        documents = load_directory(tmp)
        print(f"  Loaded {len(documents)} documents\n")

        chunker = Chunker(target_tokens=200, max_tokens=300, overlap_tokens=30)
        all_records: list[dict] = []

        for doc in documents:
            chunks = chunker.chunk_document(doc)
            records = attach_metadata(chunks, document_content=doc.content)
            all_records.extend(records)

        # ── Print results ──────────────────────────────────────────────────────
        print(f"Total chunks produced: {len(all_records)}\n")
        print("=" * 60)
        for record in all_records:
            meta = record["metadata"]
            print(f"[{meta['source'].split('/')[-1]}]  "
                  f"section={meta['section']!r}  "
                  f"roles={meta['roles']}  "
                  f"tokens={meta['token_count']}  "
                  f"chunk={meta['chunk_index']}")
            print(f"  text preview: {record['text'][:80].replace(chr(10), ' ')!r}...")
            print()

        # ── Show one full record as JSON ───────────────────────────────────────
        print("=" * 60)
        print("Sample full record (JSON):")
        print(json.dumps(all_records[0], indent=2))
