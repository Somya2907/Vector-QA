"""
document_loader.py

Responsible for loading raw documents from various sources (filesystem, S3, databases)
and normalizing them into a standard internal Document format before ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Represents a single loaded document with its content and metadata."""

    doc_id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Permission tags attached at load time (e.g. owner, access_level, department)
    permission_tags: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads documents from one or more sources and returns normalized Document objects."""

    def __init__(self, source_config: dict[str, Any]) -> None:
        """
        Args:
            source_config: Configuration dict describing the data source
                           (e.g. path, bucket name, DB connection string).
        """
        self.source_config = source_config

    def load_from_file(self, path: Path) -> Document:
        """Load a single document from a local file path.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            A normalized Document instance.
        """
        raise NotImplementedError

    def load_from_directory(self, directory: Path) -> list[Document]:
        """Recursively load all supported documents from a directory.

        Args:
            directory: Root directory to scan.

        Returns:
            List of Document instances.
        """
        raise NotImplementedError

    def load_from_s3(self, bucket: str, prefix: str) -> list[Document]:
        """Load documents from an S3 bucket with the given key prefix.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix to filter objects.

        Returns:
            List of Document instances.
        """
        raise NotImplementedError
