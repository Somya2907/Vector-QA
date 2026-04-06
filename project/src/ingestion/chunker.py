"""
chunker.py

Token-aware chunker for handbook documents.

Strategy
--------
1.  Split the document into paragraphs (blank-line boundaries).
2.  Greedily accumulate paragraphs until the running token count reaches the
    target size (600 tokens, within the 500–800 range).
3.  When the next paragraph would push the chunk over max_tokens (800), flush
    the current chunk and start a new one.
4.  Carry the last `overlap` tokens (100) of each chunk forward as a prefix for
    the next chunk so that context is not lost at boundaries.
5.  If a single paragraph exceeds max_tokens on its own, it is split at sentence
    boundaries before being accumulated.

Token counting uses tiktoken's cl100k_base encoding, which is a close
approximation to Claude's tokeniser.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .document_loader import Document

# ---------------------------------------------------------------------------
# Tokeniser (module-level singleton — loading it is expensive)
# ---------------------------------------------------------------------------
_ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TARGET_TOKENS: int = 600   # Aim for this chunk size
DEFAULT_MAX_TOKENS: int = 800      # Hard ceiling before forcing a break
DEFAULT_OVERLAP_TOKENS: int = 100  # Tokens carried over to the next chunk

# Sentence boundary: period/question/exclamation followed by whitespace
_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A sub-document chunk derived from a parent Document."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    permission_tags: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return the canonical output format: {"text": ..., "metadata": ...}."""
        return {"text": self.content, "metadata": {**self.metadata, **self.permission_tags}}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[int]:
    return _ENC.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return _ENC.decode(tokens)


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, discard empty strings."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split a long paragraph into sentences as a fallback."""
    parts = _SENTENCE_END_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _split_into_units(text: str, max_tokens: int) -> list[str]:
    """
    Return a list of text units (paragraphs, or sentences when a paragraph
    is too long) where each unit individually fits within max_tokens.
    """
    units: list[str] = []
    for para in _split_paragraphs(text):
        if len(_tokenize(para)) <= max_tokens:
            units.append(para)
        else:
            # Paragraph too large — fall back to sentence-level units
            for sentence in _split_sentences(para):
                if len(_tokenize(sentence)) <= max_tokens:
                    units.append(sentence)
                else:
                    # Single sentence still too long — hard-slice by tokens
                    toks = _tokenize(sentence)
                    for start in range(0, len(toks), max_tokens):
                        units.append(_detokenize(toks[start : start + max_tokens]))
    return units


def _build_raw_chunks(
    text: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Core chunking logic. Returns a list of raw text strings.

    Each chunk satisfies:  overlap_tokens < len(chunk) <= max_tokens
    (except possibly the final chunk, which may be smaller).
    """
    units = _split_into_units(text, max_tokens)
    if not units:
        return []

    chunks: list[str] = []
    current_tokens: list[int] = []

    for unit in units:
        unit_tokens = _tokenize(unit + "\n\n")

        # If adding this unit would exceed the hard ceiling, flush first
        if current_tokens and len(current_tokens) + len(unit_tokens) > max_tokens:
            chunks.append(_detokenize(current_tokens).strip())
            # Carry the tail of the current chunk into the next one
            current_tokens = current_tokens[-overlap_tokens:] + unit_tokens
        else:
            current_tokens.extend(unit_tokens)

        # Reached the target — flush eagerly to keep chunks near the target size
        if len(current_tokens) >= target_tokens:
            chunks.append(_detokenize(current_tokens).strip())
            current_tokens = current_tokens[-overlap_tokens:]

    # Flush whatever is left
    remainder = _detokenize(current_tokens).strip()
    if remainder:
        chunks.append(remainder)

    return chunks


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class Chunker:
    """Splits Documents into token-bounded Chunks with overlap."""

    def __init__(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        """
        Args:
            target_tokens:  Preferred chunk size; chunker flushes when reached.
            max_tokens:     Hard ceiling — a chunk never exceeds this.
            overlap_tokens: Tokens carried over from the previous chunk.
        """
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks, propagating metadata and permission tags.

        Args:
            document: Source document to split.

        Returns:
            Ordered list of Chunk objects.
        """
        raw_chunks = _build_raw_chunks(
            document.content,
            self.target_tokens,
            self.max_tokens,
            self.overlap_tokens,
        )

        chunks: list[Chunk] = []
        for index, text in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.doc_id}::{index}"))
            token_count = len(_tokenize(text))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=text,
                    chunk_index=index,
                    token_count=token_count,
                    metadata=dict(document.metadata),
                    permission_tags=dict(document.permission_tags),
                )
            )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a batch of Documents, returning a flat list.

        Args:
            documents: Documents to process.

        Returns:
            Flat list of all Chunks across all documents.
        """
        result: list[Chunk] = []
        for doc in documents:
            result.extend(self.chunk_document(doc))
        return result
