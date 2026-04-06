"""
loader.py

Concrete loader for GitLab handbook files (.md / .txt).
Reads files recursively from a local directory, strips YAML front matter and
HTML comments, and returns normalised Document objects ready for chunking.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from .document_loader import Document

# File extensions this loader supports
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})

# Matches YAML front matter blocks at the very start of a file:  --- ... ---
_FRONT_MATTER_RE = re.compile(r"^\s*---.*?---\s*", re.DOTALL)

# Matches HTML comments: <!-- ... -->
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Collapses 3+ consecutive blank lines into exactly two
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean_content(raw: str) -> str:
    """Remove boilerplate and normalise whitespace in raw file content.

    Steps applied in order:
      1. Strip YAML front matter.
      2. Remove HTML comments.
      3. Collapse runs of blank lines.
      4. Strip leading/trailing whitespace.

    Args:
        raw: Raw string read directly from a file.

    Returns:
        Cleaned text ready for chunking.
    """
    text = _FRONT_MATTER_RE.sub("", raw)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def load_file(path: Path) -> Document:
    """Load and clean a single .md or .txt file.

    Args:
        path: Path to the file (must exist).

    Returns:
        Document with cleaned content and basic metadata.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the path does not exist.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Expected one of: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    content = _clean_content(raw)

    return Document(
        doc_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()))),
        content=content,
        source=str(path),
        metadata={
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
        },
    )


def load_directory(
    directory: Path,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> list[Document]:
    """Recursively load all supported files from a directory.

    Files are returned in a stable, sorted order (by resolved path) so that
    repeated runs produce identical doc_id sequences.

    Args:
        directory: Root directory to scan.
        extensions: Set of lowercase file extensions to include.

    Returns:
        List of Document objects, one per file found.

    Raises:
        NotADirectoryError: If `directory` does not exist or is not a directory.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    documents: list[Document] = []
    for path in paths:
        try:
            documents.append(load_file(path))
        except Exception as exc:  # noqa: BLE001
            # Log and skip unreadable files rather than aborting the whole run
            print(f"[loader] Skipping {path}: {exc}")

    return documents
